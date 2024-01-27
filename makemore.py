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
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------
# JOS:

import mambaminimal as mm # mamba-minimal.py
# defines class Mamba

from enum import Enum
import cProfile
import random

def setSeed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

from torch.utils.tensorboard import SummaryWriter
import hiddenlayer as hl

# -----------------------------------------------------------------------------

DataMode = Enum('DataMode', ['WORDS', 'QA', 'DISTANCE'])

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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

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
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the token and position embedding layers
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        # add and run through the decoder MLP
        x = tok_emb + pos_emb
        # run the bag of words context module
        x = self.context_block(x)
        # decode to next token probability
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

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
    def __init__(self, config):
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

    def __init__(self, config, cell_type):
        super().__init__()
        self.block_size = config.block_size # in
        self.vocab_size = config.vocab_size # out, but also embedding size
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2)) # the starting hidden state
        self.wte = nn.Embedding(config.vocab_size+1, config.n_embd) # token embeddings table, +1 for NULL
        print(f"RNN: token embedding shape is {(config.vocab_size+1)=} by {config.n_embd=}")
        if cell_type == 'rnn':
            self.cell = RNNCell(config)
        elif cell_type == 'gru':
            self.cell = GRUCell(config)
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        print(f"RNN: idx shape is {idx.shape}")
        # Not true for last block: assert t == self.block_size, f"RNN: {t=} != {self.block_size=}"

        print("\n=== AT DATA EMBEDDING BREAKPOINT ===\n")
        print(f"idx == {idx}\n")
        # pdb.set_trace()

        # embed all the integers up front and all at once for efficiency
        emb = self.wte(idx) # (b, t, n_embd)

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
            # print(f"RNN: Given {len(idx)} in idx: {idx.transpose(0,1)=}")
            # print(f"\thave {len(targets)} targets: {targets.transpose(0,1)=}")
            assert idx.shape == targets.shape, f"RNN: {idx.shape=} != {targets.shape=}"
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

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

    def __init__(self, config):
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

    def __init__(self, config):
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
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
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
def print_samples(num=10):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    print(f"print_samples: {data_mode=}")
    setSeed(42)
    if data_mode == DataMode.DISTANCE:
        nBlocks = 1 + num // block_size
        for k in range(nBlocks):
            X_init = torch.tensor([random.randint(0, num/2) for _ in range(num)]).expand(1,num) # num/2 to get some repeats
            print(f"X_init.shape: {X_init.shape=}")
            print(f"X_init[{k}]:\n\t{X_init=}")
            X_true, _ = lastOccurrenceDistance(torch.flatten(X_init).tolist(), block_size)
            print(f"X_true:\n\t{X_true=}")
            # Initialize with one block, and then generate another
            X_samp = generate(model, X_init, block_size, temperature=0.0, do_sample=False, top_k=None).to('cpu')
            print(f"X_samp.shape (two blocks: initial + predicted): {X_samp.shape=}")
            print(f"X_samp[{k}]:\n\t{X_samp=}")
            print(f"X_samp.shape: {X_samp.shape=}")
            max_values, max_indices = torch.max(X_samp, dim=1)
            print(f"X_samp max indices:\n\t{max_indices=}")
            steps = config.output_size - 1 # -1 because we already start with <START> token (index 0)
            X_samp = generate(model, X_init, steps, do_sample=True).to('cpu')
            distances = []
            for i in range(X_samp.size(0)):
                # get the i'th row of sampled integers, as python list
                row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
                # token 0 is the <STOP> token, so we crop the output sequence at that point
                crop_index = row.index(0) if 0 in row else len(row)
                row = row[:crop_index]
                dist_samp = train_dataset.max_index(row)
                distances.append(dist_samp)
            print('-'*80)
            print(f"{len(distances)} distances:")
            for dist in distances:
                print(dist)
            print('-'*80)
        return
    elif data_mode == DataMode.QA:
        print(f"print_samples: Write model samples for QA case")
        return
    elif data_mode == DataMode.WORDS:
        X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
        # processing below:
    else:
        assert False, f"Unrecognized data mode {data_mode=}"
    top_k = args.top_k if args.top_k != -1 else None
    steps = config.output_size - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
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
def evaluate(model, dataset, data_mode, batch_size=50, max_batches=None):
    model.eval()
    doShuffle = data_mode != DataMode.DISTANCE
    loader = DataLoader(dataset, shuffle=doShuffle, batch_size=batch_size, num_workers=0)
    loader = DataLoader(dataset, shuffle=doShuffle, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        print(f"evaluate: {X.shape=}\n{Y.shape=}")
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

# -----------------------------------------------------------------------------
# helper functions for creating the training and test Datasets that emit words

    for i, itm in enumerate(ints):
        if itm in lastSeen:
            # Calculate distance from the last occurrence, capped at max_distance - 1
            dist = min(i - lastSeen[itm], max_distance - 1)
            lastOccurrence[i] = dist
        # Update the last seen index for the current item
        lastSeen[itm] = i

    # print(f"lastOccurrenceDistance: Returning {len(lastOccurrence)} distances:\n{lastOccurrence=}")
    return lastOccurrence

# my version:
def lastOccurrenceDistance(ints, max_distance):
    # # Example usage:
    # ints = [1, 2, 3, 2, 4, 1, 2, 3, 4, 2]
    # max_distance = 5
    # distances = lastOccurrenceDistance(ints, max_distance)
    # print("Distances:", distances)
    nints = len(ints)
    # print(f"lastOccurrenceDistance: Received {nints} ints:\n{ints=}")
    lastOccurrence = [0] * nints
    # New version by ChatGPT-4T:
    lastSeen = {}  # Dictionary to track the last seen index of each item
    for i, itm in enumerate(ints):
        if itm in lastSeen:
            # Calculate distance from the last occurrence, capped at max_distance - 1
            dist = min(i - lastSeen[itm], max_distance - 1)
            lastOccurrence[i] = dist
            ints[i-dist] = itm # make it true even when clipped - ASSUMES MUTABILITY OF ints
        # Update the last seen index for the current item
        lastSeen[itm] = i
    # Previous version by me;
    # for i in range(nints):
    #     if i == 0:
    #         continue
    #     itm = ints[i]
    #     for ir in reversed(range(i)):
    #         if ints[ir] == itm:
    #             dist = i - ir
    #             if dist >= max_distance:
    #                 dist = max_distance - 1
    #                 ints[i-dist] = itm # just whack it to be true - ASSUMES MUTABILITY OF ints
    #             lastOccurrence[i] = dist
    #             # print(f"lastOccurrence[{i}] of {itm} is {dist} samples ago at index {ir}")
    #             break
    # print(f"lastOccurrenceDistance: Returning {len(lastOccurrence)} distances:\n{lastOccurrence=}")
    return lastOccurrence, ints


class CharDataset(Dataset):

    def __init__(self, mode, words, chars, block_size):
        self.data_mode = mode  # DataMode.(WORDS|QA|DISTANCE)
        self.words = words     # List of strings: names (WORDS) | ListOps examples (QA) | ints (DISTANCE)
        if mode == DataMode.DISTANCE:
            self.ints = [int(w) for w in words]
            # print(f"ints = {self.ints}\n")
            # self.lastOccurrence = {value: index for index, value in enumerate(self.ints)}
            #  == dictionary mapping each int to its last occurrence (index)
            self.lastOccurrence, self.ints = lastOccurrenceDistance(self.ints, block_size)
            # print(f"lastOccurrence = {self.lastOccurrence}\n")
            print(f"maximum lastOccurrence for {len(self.ints)} ints = {max(self.lastOccurrence)}")
        self.chars = chars     # Set of all chars used in words
        self.block_size = block_size # number of inputs (typically 1 for RNNs, max_word_length+1 for transformers (WORDS), etc.
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)} # +1 to reserve 0 for padding char
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping
        self.unknown_index = -1

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self): # INPUT vocabulary = number of symbols in input
        if self.data_mode == DataMode.DISTANCE:
            vocab_size = max(self.ints) + 1 # number of tokens we need to be able to embed
            print(f"CharDataset: DISTANCE: {vocab_size=}")
            return vocab_size
        elif self.data_mode == DataMode.WORDS:
            return len(self.chars) + 1 # all the possible characters and special 0 token
        elif self.data_mode == DataMode.QA:
            return len(self.chars) + 1 # all the possible characters and special 0 token
        else:
            assert False, f"Unknown data_mode {self.data_mode=}"

    def encode(self, word):
        """
        encode - convert str word into a list of ints, one per character and return them in a type long tensor.
        """
        # original: ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)

        # JOS 1: ix = torch.tensor([self.stoi.get(w, self.unknown_index) for w in word], dtype=torch.long)
        # for i in range(len(ix)):
        #     if ix[i] == self.unknown_index:
        #         print(f"*** unknown char at index {i} == {word[i]}")
        # assert all(ixi != self.unknown_index for ixi in ix), "Unknown chars in word " + word

        ix = []
        for w in word:
            # print(f"\nencode: word == {word}\n")
            iw = self.stoi.get(w, self.unknown_index)
            if iw == self.unknown_index:
                print(f"*** unknown char `{w}'\n")
                assert False
            assert iw != 0 # reserved for padding char
            ix.append(iw)
        ixt = torch.tensor(ix,dtype=torch.long)
        ixt = torch.tensor(ix,dtype=torch.long)
        return ixt

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def max_index(self, ix):
        print(f"max_index: {ix=}")
        if len(ix)>0:
            max_index, max_value = max(enumerate(ix), key=lambda pair: pair[1])
            return max_index
        else:
            return -1

    # def samplesToLastOccurrence(self, ix, idx):
    #     stlo = 0
    #     iolo = 0
    #     for ixi in reversed(range(idx-1)):
    #         if ix == self.ints[ixi]:
    #             stlo = idx - ixi
    #             iolo = ixi
    #             break
    #     # print(f"Last occurrence of ix = {ix} from index {idx} is at index {iolo} which is {stlo} samples earlier\n")
    #     return stlo

    def __getitem__(self, idx): # CharDataset.__getitem__: idx is an int addressing one word (line) in input:
        # print (f"__getitem__: idx = {idx}, word == {self.words[idx]}, data_mode = {self.data_mode}")
        if self.data_mode == DataMode.WORDS:
            word = self.words[idx].strip()
            assert word[0] != '|', f"ListOps input format not supported by data-mode WORDS"
            ix = self.encode(word) # tensor of type long
            N = self.block_size
            assert len(ix) <= N, f"getitem: input WORD of length {len(ix)} overflows input buffer {self.block_size=}"
            x = torch.zeros(N, dtype=torch.long)
            y = torch.zeros(N, dtype=torch.long)
            Nix = len(ix)
            x[1:1+Nix] = ix  # Copy 'ix' into 'x' starting at index 1: [0,   ix0, ix1, ..., ixNM2, ixNM1, 0, 0, ... 0]
            y[:Nix] = ix     # Copy 'ix' into 'y' starting at index 0: [ix0, ix1, ix2, ..., ixNM1,     0, 0, 0, ... 0]
            y[Nix+1:] = -1   # index -1 will mask the loss at the inactive locations
        elif self.data_mode == DataMode.DISTANCE: # randomly ordered ints, right-justified in the block
            N = self.block_size
            idx0 = max(0,idx - N + 1) # include as much history as we can fit into the block
            ix = self.ints[idx0:idx+1]  # all ints up to and including the latest at idx
            Nix = len(ix)
            iy0 = self.lastOccurrence[ix[Nix-1]]
            assert ix[-1]==self.ints[idx], f"BF"
            x = -torch.ones(N, dtype=torch.long)
            ixt = torch.tensor(ix, dtype=x.dtype)
            assert Nix <= N, f"ix is longer ({len(ix)}) than the specified length {N=} of the tensor x."
            xs = N-Nix # starting index for ixt in x
            x[xs:] = ixt
            y = -torch.ones(N, dtype=torch.long)
            assert iy0 < self.block_size, f"FIXME: Must embed out to MAX MEMORY for DISTANCE"
            y[N-1] = iy0 # last element contains the answer (distance back to the last occurrence of latest input int)
        elif self.data_mode == DataMode.DISTANCE_LEFT_JUSTIFIED: # randomly ordered ints
            N = self.block_size
            idx0 = max(0,idx - N + 1) # include as much history as we can fit into the block
            ix = self.ints[idx0:idx+1]  # all ints up to and including the latest at idx
            iy0 = self.lastOccurrence[ix[-1]]  # Using ix[-1] to safely get the last element
            x = -torch.ones(N, dtype=torch.long)
            ix_tensor = torch.tensor(ix, dtype=x.dtype)
            assert len(ix) <= N, f"ix is longer ({len(ix)}) than the specified length {N=} of the tensor x."
            x[:len(ix)] = ix_tensor
            y = -torch.ones(N, dtype=torch.long)
            y[min(len(ix)-1,N-1)] = iy0 # last element contains the answer (distance back to the last occurrence of latest input int)
        elif self.data_mode == DataMode.QA:
            word = self.words[idx].strip()
            assert word[0] == '|', f"QA data-mode requires ListOps input format (.tsv), found word[0] = {word[0]}"
            _, target, test = word.split('|')  # example received as "|target|test"
            # print(f"\ntest == {test}\n\ntarget == {target}\n")
            # Create this format:
            # x: test ......
            # y: .... target

            # We seem to be in a multithreaded callback that cannot do this:
            # print("=== AT DATA ENCODING BREAKPOINT ===")
            # pdb.set_trace()

            # print(f"\nEncoding test == {test}\n")
            ix = self.encode(test)  # each char converted to an integer
            # print(f"\nEncoding target == {target}\n")
            iy = self.encode(target)  # each char of target converted to an integer
            nx = len(ix)
            ny = len(iy)
            assert nx+ny < self.block_size, f"getitem: block_size {self.block_size=} must equal or exceed {nx=} + {ny=}"
            x = torch.zeros(self.block_size, dtype=torch.long)
            y = torch.zeros(self.block_size, dtype=torch.long)

            # Convert ix to a torch.LongTensor if it's not already
            ix_tensor = torch.tensor(ix, dtype=torch.long) if not isinstance(ix, torch.Tensor) else ix.long()

            # Assign ix_tensor to the beginning of x
            x[:nx] = ix_tensor

            # Fill the rest of x with -1
            x[nx:] = 0

            # Fill the beginning of y with -1
            y[:nx] = 0

            # Convert iy to a torch.LongTensor if it's not already
            iy_tensor = torch.tensor(iy, dtype=torch.long) if not isinstance(iy, torch.Tensor) else iy.long()

            # Assign iy_tensor to the appropriate slice of y
            y[nx:nx+ny] = iy_tensor

        return x, y

def create_datasets(input_file, data_mode, block_size):
    """
    Create training and test datasets based on the input file, data mode, and block size.

    Args:
        input_file (str): The path to the input file.
        data_mode (DataMode): The mode of the data, which can be DataMode.WORDS, DataMode.DISTANCE, or DataMode.QA.
        block_size (int): The size of the data block.  Set to None to have it computed automatically and retrieve using get_vocab_size

    Returns:
        train_dataset (CharDataset): The training dataset.
        test_dataset (CharDataset): The test dataset.
    """
    # code implementation...

    # print("=== AT DATA LOADING BREAKPOINT ===")
    # pdb.set_trace()

    # preprocessing of the input text file
    with open(input_file, 'r') as f:
        data = f.read() # read whole file into a single str

    words = data.splitlines() # words[i] can be a word, ListOps example, or int
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    chars = sorted(list(set(''.join(words)))) # all the possible characters
    vocab_size = len(chars) + 1 # add one for special separation token
    if block_size == None:
        block_size = vocab_size

    basename = os.path.basename(input_file)
    name,ext = os.path.splitext(basename)

    if data_mode == DataMode.WORDS: # original makemore case - input = list of words such as names
        assert ext == '.txt', f"DataMode.WORDS requires .txt input format, got {ext}"
        max_word_length = max(len(w) for w in words)
    elif data_mode == DataMode.DISTANCE: # distance ints
        assert ext == '.txt', "DataMode.DISTANCE requires .txt input format"
        if name == 'names': # original makemore default
            print(f"Generating DISTANCE DATA AUTOMATICALLY since no input file-name specified")

            # numExamples = 32033 # same as original names.txt why not
            print(f"*** USING FOUR-BLOCK DATA SET FOR TESTING ***")
            numExamples = 4*block_size

            numInts = 27 # same as original vocab_size in names.txt
            ints = [random.randint(0, numInts-1) for _ in range(numExamples)]
            words = [str(i) for i in ints] # FIXME: should not need this - revise data structures
            chars = sorted(list(set(''.join(words)))) # gross - not used - just to eliminate misleading printouts
        else:
            ints = [int(w) for w in words]
        max_int = max(ints)
        max_word_length = block_size # number of samples delay supported
        # print(f"DISTANCE data_mode:\n\twords = {words}\n\tints = {ints}\n\tmax_int = {max_int}\n")
    elif data_mode == DataMode.QA: # ListOps case [added to makemore]
        assert ext == '.tsv', "DataMode.QA requires .tsv input format"
        lines = words # Each line is a complete ListOps example in the format "|solution|problem"
        # targets, tests = lines.toString().split('\t',1)
        # GPT-4's fancy method: targets, tests = zip(*[line.split('\t', 1) for line in lines])

        targets = []
        tests = []
        words = [] # hacky pack for CharDataset

        for line in lines:

            target, test = line.split('\t', 1)

            trgs = target.strip()
            assert trgs != '', "Target is empty after stripping"
            targets.append(trgs)

            ts = test.strip()
            assert ts != '', "Test is empty after stripping"
            tests.append(ts)

            # My version: words.append('|' + trgs + '|' + ts)
            words.append('|' + '|'.join([trgs, ts])) # GPT-4 wins again

        chars_set = set(''.join(words))
        chars_set.discard('|')
        chars = sorted(list(chars_set))
        max_target_length = max(len(w) for w in targets)
        max_test_length   = max(len(w) for w in tests)
        max_word_length   = max_target_length + max_test_length # format is test...\n ...target, nonoverlapping

        # print("=== AT DATA LOADING BREAKPOINT ===")
        # pdb.set_trace()


    print(f"number of examples in the dataset: {len(words)}")
    print(f"input block size: {block_size}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print(f"chars: {''.join(chars)}")

    # create_datasets: partition the input data into a training and the test set
    nWords = len(words)
    test_set_size = max(block_size, int(nWords * 0.1)) # 10% of the training set, or up to 1000 examples
    assert nWords > 2*test_set_size, f"Only {nWords} words of data for {block_size=}"
    if data_mode != DataMode.DISTANCE:
        rp = torch.randperm(len(words)).tolist()
    else:
        rp = range(len(words)) # cannot permute this memory task
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(data_mode, train_words, chars, block_size)
    test_dataset = CharDataset(data_mode, test_words, chars, block_size)

    return train_dataset, test_dataset, block_size

class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default='./data/words/names.txt', help="input text file, where .txt suffix => one word per line to make more of, while .tsv => <answer><tab><prompt> each line (e.g., ListOps data)")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--data-mode', type=str, default="words", help="data type: (words|qa|distance|distance-exp)")
    # input/output sizes and dimensionalities
    parser.add_argument('--block-size', type=int, default=None, help="input block size [default = vocab_size measured from data]: (max word length + 1 | max Q+A length + 2 | max short-term memory")
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
    parser.add_argument('--batch-size', '-b', type=int, default=8, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
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
    print(f"dataset determined that: {data_mode=}")

    # init datasets

    block_size = args.block_size

    train_dataset, test_dataset, block_size = create_datasets(args.input_file, data_mode, args.block_size)

    embedding_size = args.embedding_size
    logits_size = args.logits_size
    vocab_size = train_dataset.get_vocab_size()
    if logits_size == None:
        logits_size = vocab_size
        
    print(f"+++ dataset determined that: {vocab_size=}")
    print(f"test_dataset: {test_dataset=}")
    # init model - FIXME: For DataMode.DISTANCE, output an unsigned int instead of (too many) logits
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size, logits_size=logits_size,
                         n_layer=args.n_layer, n_head=args.n_head,
                         n_embd=args.n_embd, n_embd2=args.n_embd2, data_mode=data_mode)

    mambaConfig = mm.ModelArgs(
        d_model=args.n_embd,
        n_layer=args.n_layer,
        vocab_size=vocab_size,
        block_size=block_size,
        # Mamba output size == block_size because it is a sequence to sequence map => no
        # logits_size=logits_size,
        d_state=args.n_head, # too janky?
        expand=2, # FIXME: bring out state-expansion-factor parameter
        dt_rank='auto', # auto => d_model/16
        d_conv=4, # Conv1d kernel size
        pad_vocab_size_multiple=8, # Forces vocab_size to be a multiple of this
        conv_bias=True,
        bias=False)

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
        model = mm.Mamba(mambaConfig)
    else:
        raise ValueError(f'model type {args.type} is not recognized')
    model.to(args.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_samples(block_size)
        sys.exit()

    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

    # init dataloader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    # training loop
    best_loss = None
    step = 0
    while True:

        t0 = time.time()

        # print("=== AT TRAINING BREAKPOINT ===")
        # pdb.set_trace()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        assert X.shape == Y.shape, f"{X.shape=} != {Y.shape=}"
        print(f"{X.shape=}, {Y.shape=}")
        print(f"X:\n\t{X=}")
        print(f"Y:\n\t{Y=}")

        # Output model diagram:
        make_graphs = 0
        if step == 0 and make_graphs == 1:

            writer = SummaryWriter()
            writer.add_graph(model, X)
            writer.close()

            hl_graph = hl.build_graph(model, X)  # Adjust the input shape
            hl_graph.theme = hl.graph.THEMES["blue"].copy()
            hl_graph.save("model_visualization", format="png")

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

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, data_mode, batch_size=args.batch_size, max_batches=10)
            test_loss  = evaluate(model, test_dataset, data_mode, batch_size=args.batch_size, max_batches=10)
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

        # sample from the model
        if step > 0 and step % 200 == 0:
            print_samples(block_size)

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break
