# %%

"""
makelogits - jos extension of makemore by Andrej Karpathy - details in ./README.md

you give this script some words (one per line) and it will generate more things like it.
uses super state of the art Transformer AI tech
this code is intended to be super hackable. tune it to your needs.

Changes from minGPT:
- I removed the from_pretrained function where we init with GPT2 weights
- I removed dropout layers because the models we train here are small,
  it's not necessary to understand at this stage and at this scale.
- I removed weight decay and all of the complexity around what parameters are
  and are not weight decayed. I don't believe this should make a massive
  difference at the scale that we operate on here.
"""

import pdb

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

# -----------------------------------------------------------------------------
# JOS:

import mambaminimal as mm # mambaminimal.py
# defines class Mamba

import cProfile
import random

from data_utilities import DataMode, DistanceMode, create_datasets, InfiniteDataLoader, ascii_plot

def setSeed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# %%

#traceTensors = True
traceTensors = False

#traceTensorsXY = True
traceTensorsXY = False

# None of these worked, but nnviz did, after creating defaultConfig below to use in "default constructors"
# Perhaps one or more of these can work now:
# N: from torch.utils.tensorboard import SummaryWriter
# N: import hiddenlayer as hl
# N: from ann_visualizer.visualize import ann_viz;

# To try:
# https://www.youtube.com/watch?v=ChfEO8l-fas&t=1139s

# See also:
# Works: ./tdot.py
# Works so far: ./tkerasviz.py

# -----------------------------------------------------------------------------

# %%

@dataclass
class ModelConfig:
    vocab_size: int = None # number of output logits / output classes
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64  # input embedding
    n_embd2: int = 64 # hidden-state embedding (GRU et al.)
    n_head: int = 4
    # extension of makemore to new types of data (beyond just words to make more of)
    data_mode: DataMode = DataMode.WORDS # data modes are WORDS (original), QA (Question/Answer), and DISTANCE
    block_size: int = 16 # input sequence length, originally max_word_length+1 == max chars/word + <start>
    logits_size: int = None # output logits length, originally same as block_size
    output_size: int = 128 # number of output logits == number of chars for WORDS, desired memory length for QA or DISTANCE

    # block_size is important for Transformer and any model that works only on one input buffer at a time
    # (no recurrence).

# For visualizations which need a "default constructor":
defaultConfig = ModelConfig(vocab_size=27, block_size=32, logits_size=27, n_layer=4, n_head=4, n_embd=16, n_embd2=16, data_mode=DataMode.WORDS)

# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        #print(f"Transformer: idx for wte embedding =\n{idx.transpose(0,1)=}")
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # print(f"Transformer: Given {len(idx)} in idx: {idx.transpose(0,1)=}")
            # print(f"\thave {len(targets)} targets: {targets.transpose(0,1)=}")
            assert idx.shape == targets.shape, f"Transformer: {idx.shape=} != {targets.shape=}"
            logits_view = logits.view(-1, logits.size(-1))
            targets_view = targets.view(-1)
            ascii_plot(logits_view, targets_view, title="Transformer: Logits and Targets")
            loss = F.cross_entropy(logits_view, targets_view, ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# Bag of Words (BoW) language model

class CausalBoW(nn.Module):
    """
    Causal bag of words. Averages the preceding elements and looks suspiciously like
    a CausalAttention module you'd find in a transformer, for no apparent reason at all ;)
    """
    def __init__(self, config):
        super().__init__()

        # used to mask out vectors and preserve autoregressive property
        self.block_size = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, n_embd

        # do the weighted average of all preceeding token features
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ x # (B, T, T) x (B, T, C) -> (B, T, C)

        return y

class BoWBlock(nn.Module):
    """ collects BoW features and adds an MLP """

    def __init__(self, config):
        super().__init__()

        # Causal BoW module
        self.cbow = CausalBoW(config)
        # MLP assembler
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, config.n_embd2),
            c_proj  = nn.Linear(config.n_embd2, config.n_embd),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.cbow(x)
        x = x + self.mlpf(x)
        return x

class BoW(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    also encodes their positions with lookup table, then averages all of those
    embeddings up and uses that to predict the next token.
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # position embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # context block
        self.context_block = BoWBlock(config)
        # language model head decoder layer
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):

        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"BoW: Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the token and position embedding layers
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        # add and run through the decoder MLP
        x = tok_emb + pos_emb
        # run the bag of words context module
        x = self.context_block(x)
        # decode to next token probability
        logits = self.lm_head(x) # vocab_size

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            logits_view = logits.view(-1, logits.size(-1))
            targets_view = targets.view(-1)
            loss = F.cross_entropy(logits_view, targets_view, ignore_index=-1)
            ascii_plot(logits_view, targets_view, title="BoW: Logits and Targets")

        return logits, loss

# -----------------------------------------------------------------------------
"""
Recurrent Neural Net language model: either a vanilla RNN recurrence or a GRU.
Did not implement an LSTM because its API is a bit more annoying as it has
both a hidden state and a cell state, but it's very similar to GRU and in
practice works just as well.
"""

class RNNCell(nn.Module):
    """
    the job of a 'Cell' is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state
    h_{t} at the current timestep
    """
    def __init__(self, config=defaultConfig):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        ht = F.tanh(self.xh_to_h(xh))
        return ht

class GRUCell(nn.Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.
    """
    def __init__(self, config):
        super().__init__()
        # input, forget, output, gate
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        # first use the reset gate to wipe some channels of the hidden state to zero
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev
        # calculate the candidate new hidden state hbar
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr))
        # calculate the switch gate that determines if each channel should be updated at all
        z = F.sigmoid(self.xh_to_z(xh))
        # blend the previous hidden state and the new candidate hidden state
        ht = (1 - z) * hprev + z * hbar
        return ht

class RNN(nn.Module):

    def __init__(self, config=defaultConfig, cell_type='gru'):
        super().__init__()
        self.block_size = config.block_size # in
        self.logits_size = config.logits_size # out
        self.vocab_size = config.vocab_size # embedding size
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2)) # the starting hidden state
        self.wte = nn.Embedding(config.vocab_size+1, config.n_embd) # token embeddings table, +1 for NULL
        if traceTensors:
            print(f"\nRNN: token embedding shape is num_tokens x n_embd: {(config.vocab_size+1)=} by {config.n_embd=}")
        if cell_type == 'rnn':
            self.cell = RNNCell(config)
        elif cell_type == 'gru':
            self.cell = GRUCell(config)
        self.lm_head = nn.Linear(config.n_embd2, self.logits_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, tokens, targets=None):
        device = tokens.device
        b, t = tokens.size()

        # print(f"RNN: tokens shape is {tokens.shape}")
        # Not true for last block: assert t == self.block_size, f"RNN: {t=} != {self.block_size=}"

        # pdb.set_trace()

        # embed all the integers up front and all at once for efficiency
        if traceTensors:
            print(f"\nRNN: === AT DATA EMBEDDING BREAKPOINT:\n{tokens=}")
        emb = self.wte(tokens) # (b, t, n_embd)

        # sequentially iterate over the inputs and update the RNN state each tick
        hprev = self.start.expand((b, -1)) # expand out the batch dimension
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :] # (b, n_embd)
            ht = self.cell(xt, hprev) # (b, n_embd2)
            hprev = ht
            hiddens.append(ht)

        # decode the outputs
        hidden = torch.stack(hiddens, 1) # (b, t, n_embd2)
        logits = self.lm_head(hidden)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # Not very interesting since RNNs must be called one sample at a time
            # print(f"RNN: Given {len(tokens)} in tokens: {tokens.transpose(0,1)=}")
            # print(f"\thave {len(targets)} targets: {targets.transpose(0,1)=}")
            assert tokens.shape == targets.shape, f"RNN: {tokens.shape=} != {targets.shape=}"
            # print(f"{config.logits_size=}")
            # print(f"{logits.shape=}")
            # Clip targets to ensure they are within the valid range [0, C-1]
            num_classes = logits.size(-1)
            targets_clipped = torch.clamp(targets, 0, num_classes - 1)
            logits_view = logits.view(-1, logits.size(-1))
            targets_view = targets_clipped.view(-1)
            ascii_plot(logits_view, targets_view, title="RNN: Logits and Targets")
            loss = F.cross_entropy(logits_view, targets_view, ignore_index=-1)
            # print(f"RNN: loss={loss}, {logits.shape=}")

        return logits, loss

# -----------------------------------------------------------------------------
# MLP language model

class MLP(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    concatenates the vectors and predicts the next token with an MLP.

    Reference:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, config=defaultConfig):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd) # token embeddings table
        # +1 in the line above for a special <BLANK> token that gets inserted if encoding a token
        # before the beginning of the input sequence
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):

        # gather the word embeddings of the previous 3 words
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size # special <BLANK> token
            embs.append(tok_emb)

        # concat all of the embeddings together and pass through an MLP
        x = torch.cat(embs, -1) # (b, t, n_embd * block_size)
        logits = self.mlp(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# Bigram language model

class Bigram(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, config=defaultConfig):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1 # this model only needs one previous character to predict the next

    def forward(self, idx, targets=None):

         # 'forward pass', lol
        logits = self.logits[idx]

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    A list of max_new_tokens tokens is returned.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        if traceTensors:
            print(f"generate:\n\t{idx_cond.shape=}\n\t{logits.shape=}")
        # pluck the logits (b, t, d) at the final step and scale by desired temperature
        logits = logits[:, -1, :] / max(temperature, 1.0e-7)
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

@torch.no_grad()
def print_word_samples(num=10):
    """ samples from the model and pretty prints the decoded samples """
    setSeed(43)
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device) # generate num examples in parallel as one batch
    top_k = args.top_k if args.top_k != -1 else None
    steps = config.output_size - 1 # == max_word_length (-1 because we added 1 for <START> token (index 0))
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    assert X_samp.size(0) == num, f"I thought {num=} would equal {X_samp.size(0)=}"
    for i in range(X_samp.size(0)): # loop over generated samples == batch size
        # get the i'th row of sampled integers, as a python list:
        row = X_samp[i, 1:].tolist() # initial <START> token omitted
        crop_index = row.index(0) if 0 in row else len(row) # find the <STOP> token
        row = row[:crop_index] # take everything up to but not including <STOP> token
        word_samp = train_dataset.decode(row) # convert the list of integers to a string
        # separately track samples that we have and have not seen before
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print('-'*80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print('-'*80)

@torch.inference_mode()
def evaluate(model, dataset, data_mode, batch_size=50, max_batches=None, make_graphs=False, num_print=0):
    model.eval() # set evaluation mode

    # Output model diagram if requested:
    if make_graphs:
        # nnviz on the command line works. These don't:
        # ann_viz(model)
        trySummaryWriter = False
        if trySummaryWriter:
            writer = SummaryWriter()
            dummy_input = torch.randn(1, len(dataset), batch_size)
            writer.add_graph(model, dummy_input)
            writer.close()
        # tryHiddenLayer = False
        # if tryHiddenLayer:
        #     hl_graph = hl.build_graph(model, test_dataset)  # Adjust the input shape
        #     hl_graph.theme = hl.graph.THEMES["blue"].copy()
        #     gfname = "model_visualization"
        #     hl_graph.save(gfname, format="png")
        #     print(f"Written: {gfname}.png")

    # original: loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    doShuffle = (data_mode != DataMode.DISTANCE) # this is a memory task that shuffling would destroy
    loader = DataLoader(dataset, shuffle=doShuffle, batch_size=batch_size, num_workers=0) # , batch_sampler=doShuffle)
    loader = DataLoader(dataset, shuffle=doShuffle, batch_size=batch_size, num_workers=0) # , batch_sampler=doShuffle)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        if traceTensorsXY:
            print(f"evaluate: {X.shape=}\n{Y.shape=}")
        logits, loss = model(X, Y) # ********************** MAIN EVENT **********************
        if traceTensorsXY:
            print(f"evaluate: {logits.shape=}")
        losses.append(loss.item())
        if num_print>0:
            final_logits = logits[:, -1, :] # b t v -> b v where v = vocab_size or num-logits
            probs = F.softmax(final_logits, dim=-1) # logits to probabilities
            _, idx_best = torch.topk(probs, k=1, dim=-1) # find the index of maximum probability in each batch element
            Yh = idx_best
            for p in range(min(num_print,batch_size)):
                bp = p + i*batch_size
                if traceTensorsXY:
                    print(f"\nevaluate:\nX[{bp}]:\n{X}\nY[{bp}]:\n{Y}\nYh[{bp}]:{Yh}")
            num_print -= batch_size
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()

    model.train() # reset model back to training mode
    return mean_loss

# -----------------------------------------------------------------------------

# %%

# We are not a module, so either in Jupyter or standalone Python

print(f"=== {__name__}({__file__}):")

# failed to work: assume_jupyter = __name__ != '__main__'

if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default=None, help="input text file, where .txt suffix => one word per line to make more of, while .tsv => <answer><tab><prompt> each line (e.g., ListOps data)")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--data-mode', type=str, default="words", help="data type: (words|qa|distance|distance-exp)")
    # input/output sizes and dimensionalities
    parser.add_argument('--block-size', type=int, default=-1, help="input block size [default = vocab_size measured from data]: (max word length + 1 | max Q+A length + 2 | max short-term memory")
    parser.add_argument('--embedding-size', type=int, default=32, help="embedding size: (max word length + 1 | number of Answers | max_distance + 1)")
    parser.add_argument('--logits-size', type=int, default=None, help="logits size: defaults to embedding-size")
    # training
    parser.add_argument('--device', type=str, default='cpu', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=-1, help="max number of optimization steps to run for, or -1 for infinite.")
    # sampling
    parser.add_argument('--seed', type=int, default=3407, help="seed")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer|mamba")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels elsewhere in the model")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=1, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")

    if False: # was the failed flag assume_jupyter:
        args_list = "--data-mode words".split() # test example - EDIT THIS
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    def str2dm(s):
        if s == "words":
            dm = DataMode.WORDS
        elif s == "qa":
            dm = DataMode.QA
        elif s == "distance":
            dm = DataMode.DISTANCE
        else:
            assert False, f"Unrecognized --data-mode {s}"
        return dm

    data_mode = str2dm(args.data_mode)
    print(f"{data_mode=}")

    input_file = args.input_file

    if input_file == None:
        match data_mode:
            case DataMode.WORDS:
                input_file = './data/words/names.txt'
            case DataMode.QA:
                input_file = './data/listops/data.txt'
            case DataMode.DISTANCE:
                input_file = './data/distance/dist1.txt'

    # init datasets

    block_size = None if args.block_size <= 0 else args.block_size
    print(f"{block_size=}")

    train_dataset, test_dataset, block_size = create_datasets(input_file, data_mode, block_size)

    embedding_size = args.embedding_size
    logits_size = args.logits_size
    vocab_size = train_dataset.get_vocab_size()
    print(f"train_dataset says {vocab_size=}")
    if logits_size == None:
        logits_size = vocab_size
        print(f"main: logits_size set to {vocab_size=}")

#    if data_mode == DataMode.DISTANCE:
#        maxLastOccurrence = max(Y)
#        if max(train_dataset(:,

    print(f"+++ dataset determined that: {vocab_size=}")
    print(f"test_dataset: {test_dataset[0]=}")
    # init model - FIXME: For DataMode.DISTANCE, output an unsigned int instead of (too many) logits
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size, logits_size=logits_size,
                         n_layer=args.n_layer, n_head=args.n_head,
                         n_embd=args.n_embd, n_embd2=args.n_embd2, data_mode=data_mode)

    print(f"main: {vocab_size=}")

    if args.type == 'transformer':
        model = Transformer(config)
    elif args.type == 'bigram':
        model = Bigram(config)
    elif args.type == 'mlp':
        model = MLP(config)
    elif args.type == 'rnn':
        model = RNN(config, cell_type='rnn')
    elif args.type == 'gru':
        model = RNN(config, cell_type='gru')
    elif args.type == 'bow':
        model = BoW(config)
    elif args.type == 'mamba':
        mambaConfig = mm.ModelArgs(
            d_model=args.n_embd,
            n_layer=args.n_layer,
            vocab_size=vocab_size,
            block_size=block_size,
            # Mamba output size == block_size because it is a sequence to sequence map:
            # Mamba logits size == vocab_size
            d_state=args.n_head, # too janky?
            expand=2, # FIXME: bring out state-expansion-factor parameter
            dt_rank='auto', # auto => d_model/16
            d_conv=4, # Conv1d kernel size
            pad_vocab_size_multiple=1, # Forces vocab_size to be a multiple of this
            conv_bias=True,
            bias=False)
        model = mm.Mamba(mambaConfig)
    else:
        raise ValueError(f'model type {args.type} is not recognized')
    model.to(args.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_word_samples(block_size,data_mode)
        sys.exit()

    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

    # init dataloader
    shuffle = (data_mode != DataMode.DISTANCE) # this is a memory task that shuffling would destroy
    batch_loader = InfiniteDataLoader(train_dataset, shuffle, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    # training loop
    best_loss = None
    step = 0
    while True:

        t0 = time.time()

        # print("=== AT TRAINING BREAKPOINT ===")
        # pdb.set_trace()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch] # for each tensor t in batch
        X, Y = batch
        assert X.shape == Y.shape, f"{X.shape=} != {Y.shape=}"
        if traceTensors:
            print(f"\nbatch is type {type(batch)} with length {len(batch)}")
            print(f"{X.shape=}, {Y.shape=}")
            print(f"X:\n\t{X=}")
            print(f"Y:\n\t{Y=}")

        # feed into the model
        logits, loss = model(X, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # graph the model: evaluate(model, test_dataset, data_mode, batch_size=args.batch_size, max_batches=1, make_graphs=True)

        # evaluate the model
        if step > 0 and step % 200 == 0:
            # print("\n"+'-'*30+" TRAIN "+'-'*30)
            train_loss = evaluate(model, train_dataset, data_mode, batch_size=args.batch_size, max_batches=10, num_print=10)
            # print("\n"+'-'*30+" TEST "+'-'*30)
            test_loss  = evaluate(model, test_dataset, data_mode, batch_size=args.batch_size, max_batches=10, num_print=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        if data_mode == DataMode.WORDS:
            # sample words from the model
            if step > 0 and step % 200 == 0:
                print_word_samples(block_size)

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break
