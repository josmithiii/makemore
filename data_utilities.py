import os
import torch
from torch.utils.data import Dataset
from enum import Enum
from itertools import chain
from torch.utils.data.dataloader import DataLoader

DataMode = Enum('DataMode', ['WORDS', 'QA', 'DISTANCE', 'DISTANCE_LEFT_JUSTIFIED'])
DistanceMode = Enum('DistanceMode', ['LoopingInts', 'LastOccurrence', 'ReservedIntsRandomlyPlaced'])

traceTensors = False
traceTensorsXY = False

# -----------------------------------------------------------------------------

class CharDataset(Dataset): # original makemore case

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars     # Set of all chars used in words
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)} # +1 to reserve 0 for padding char
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self): # INPUT vocabulary = number of symbols in input
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def encode(self, word):
        """
        encode - convert str word into a list of ints, one per character and return them in a type long tensor.
        """
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def __getitem__(self, idx): # CharDataset.__getitem__: idx is an int addressing one word (line) in input data file:
        # Return inputs and targets for one line of the input file (one training example).
        # print (f"__getitem__: idx = {idx}, word == {self.words[idx]}, data_mode = {self.data_mode}")
        if traceTensors:
            print(f"CharDataset: getitem: {idx=}") # randomly jumps among batches, but data builds ok below
        word = self.words[idx].strip()
        assert word[0] != '|', f"ListOps input format not supported by data-mode WORDS"
        ix = self.encode(word) # tensor of type long
        assert len(ix) <= N, f"getitem: input WORD of length {len(ix)} overflows input buffer {self.block_size=}"
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        Nix = len(ix)
        x[1:1+Nix] = ix  # Copy 'ix' into 'x' starting at index 1: [0,   ix0, ix1, ..., ixNM2, ixNM1, 0, 0, ... 0]
        y[:Nix] = ix     # Copy 'ix' into 'y' starting at index 0: [ix0, ix1, ix2, ..., ixNM1,     0, 0, 0, ... 0]
        y[Nix+1:] = -1   # index -1 will mask the loss at the inactive locations
        return x, y

class ListOpsDataset(Dataset):

    def __init__(self, mode, problems, chars, block_size):
        self.problems = problems     # List of strings: ListOps examples (DataMode.QA)
        self.chars = chars     # Set of all chars used in problems
        self.block_size = block_size # number of inputs (typically 1 for RNNs, max_problem_length+1 for transformers (WORDS), etc.
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)} # +1 to reserve 0 for padding char
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping
        self.unknown_index = -1

    def __len__(self):
        return len(self.problems)

    def contains(self, problem):
        return problem in self.problems

    def get_vocab_size(self): # INPUT vocabulary = number of symbols in input
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def encode(self, problem):
        """
        encode - convert str problem into a list of ints, one per character and return them in a type long tensor.
        """
        # original (for problems): ix = torch.tensor([self.stoi[w] for w in problem], dtype=torch.long)
        ix = []
        for p in problem:
            # print(f"\nListOpsDataset: encode: problem == {problem}\n")
            ip = self.stoi.get(p, self.unknown_index)
            if ip == self.unknown_index:
                print(f"*** unknown char `{p}'\n")
                assert False
            assert ip != 0 # reserved for padding char
            ix.append(ip)
        ixt = torch.tensor(ix,dtype=torch.long)
        ixt = torch.tensor(ix,dtype=torch.long)
        return ixt

    def decode_problem(self, ix):
        problem = ''.join(self.itos[i] for i in ix)
        return problem

    def max_index(self, ix):
        print(f"max_index: {ix=}")
        if len(ix)>0:
            max_index, max_value = max(enumerate(ix), key=lambda pair: pair[1])
            return max_index
        else:
            return -1

    def __getitem__(self, idx): # ListOpsDataset.__getitem__: idx is an int addressing one problem (line) in input:
        # Return inputs and targets for one line of the input file (one training example).
        # print (f"ListOpsDataset.__getitem__: idx = {idx}, problem == {self.problems[idx]}, data_mode = {self.data_mode}")
        if traceTensors:
            print(f"ListOpsDataset: getitem: {idx=}") # randomly jumps among batches, but data builds ok below
        N = self.block_size
        problem = self.problems[idx].strip()
        assert problem[0] == '|', f"QA data-mode requires ListOps input format (.tsv), found problem[0] = {problem[0]}"
        _, target, test = problem.split('|')  # example received as "|target|test"
        # print(f"\ntest == {test}\n\ntarget == {target}\n")
        # Create this format:
        # x: test ......
        # y: .... target

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

        return x, y # inputs and targets

class DistanceDataset(Dataset):

    def __init__(self, mode, ints, occurrences, block_size):
        print(f"DistanceDataset: {ints=}")
        assert isinstance(ints[0][0], int)
        print(f"DistanceDataset: {occurrences=}")
        assert isinstance(occurrences[0][0], int)
        self.distance_mode = mode
        self.ints = ints               # List of lists of ints
        self.occurrences = occurrences # same
        self.block_size = block_size   # number of inputs (typically 1 for RNNs, max_word_length+1 for transformers (WORDS), etc.
        self.unknown_index = -1

    def __len__(self):
        return len(self.ints)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self): # INPUT vocabulary = number of symbols in input
        vocab_size = max(chain.from_iterable(self.ints)) + 1 # number of tokens we need to be able to embed - add '0'
        print(f"CharDataset: DISTANCE: {vocab_size=}")
        return vocab_size

    def encode(self, word):
        """
        encode - convert str word into a list of ints, one per character and return them in a type long tensor.
        """
        # original: ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
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

    def decode_word(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def max_index(self, ix):
        print(f"max_index: {ix=}")
        if len(ix)>0:
            max_index, max_value = max(enumerate(ix), key=lambda pair: pair[1])
            return max_index
        else:
            return -1

    def __getitem__(self, idx): # DistanceDataset.__getitem__: idx is an int addressing one word (line) in input:
        # Return inputs and targets for one block of the input file (one training example).
        if traceTensors:
            print(f"DistanceDataset: getitem: {idx=}") # randomly jumps among batches
            print(f"getitem: self.ints[{idx}] == {ix0}")
            print(f"getitem: self.occurrences[{idx}] == {iy0}")

        N = self.block_size

        # Fetch the block directly since each index corresponds to a full block
        ints_block = self.ints[idx]
        occurrences_block = self.occurrences[idx]

        # Convert lists to tensors
        x = torch.tensor(ints_block, dtype=torch.long)
        y = torch.tensor(occurrences_block, dtype=torch.long)

        # Ensure x and y are the correct shape (N,)
        assert x.size(0) == N and y.size(0) == N, "Blocks must be of size N"

        return x, y # inputs and targets

# ------------------------------------- Helpers ---------------------------------------------------

def read_input_file(input_file):
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words if w]
    return words

def split_dataset(lines, block_size, shuffle=True):
    nLines = len(lines) # strings
    test_set_size = max(1, int(nLines * 0.1))
    assert nLines > 1, f"Only {nLines} lines of data.  Need at least 2 for both training and testing"
    if shuffle:
        rp = torch.randperm(len(lines)).tolist()
    else:
        rp = range(len(lines))
    train_lines = [lines[i] for i in rp[:-test_set_size]]
    test_lines = [lines[i] for i in rp[-test_set_size:]]
    return train_lines, test_lines

# ---------------------------------- Synthesize Datasets -------------------------------------------------
# helper functions first:

def lastOccurrenceDistances(ints):
    """
    Calculate the distance from the last occurrence of each element in the list:
    
    Args:
    ints (list of int): A list of integers.

    Returns:
    lastOccurrenceDistances (list of int):
      lastOccurrenceDistances[i] = distance backwards to last occurrence of i, or 0 if i is new.

    Example usage:
    ints = [1, 2, 3, 2, 4, 1, 2, 3, 4, 2]
    distances = lastOccurrenceDistances(ints)
    print("ints:", ints)
    print("Distances:", distances)
    """
    nints = len(ints)
    print(f"lastOccurrenceDistance: Received {nints} ints")
    if traceTensors:
        print(f"\t{ints=}")

    lastSeen = {}  # Dictionary to track the last seen index of each item
    distances = []  # List to store distances
    for i, itm in enumerate(ints):
        # Calculate distance from the last occurrence
        lasti = lastSeen.get(itm, -1)
        dist = i-lasti if (lasti >= 0) else 0
        distances.append(dist)
        lastSeen[itm] = i

    if traceTensors:
        print(f"lastOccurrenceDistance: Returning {len(distances)} distances:\n{distances}")

    return distances

# --------------------------------------------------------------------------------------------
# DISTANCE

def synth_distance_datasets(distance_mode=DistanceMode.ReservedIntsRandomlyPlaced, num_ints=27, num_target_ints=5, num_examples=128, block_size=64):
    print(f"synth_distance_datasets: Generating {num_examples} ints between 1 and {num_ints} in blocks of size {block_size} with {num_target_ints} target ints")
    ints = [0] * num_examples * block_size
    tests = []
    targs = []
    for bi in range(0, num_examples):
        block = [random.randint(1, num_ints) for _ in range(block_size)] # avoid 0 which means "no input"
        occurrences = [0]*block_size
        match distance_mode:
            case DistanceMode.LoopingInts:
                assert False, "DistanceMode.LoopingInts not written"
            case DistanceMode.LastOccurrence:
                # Original DISTANCE benchmark: recall distance to last occurrence (unbounded)
                occurrences.append(lastOccurrenceDistances(block)) # FIXME: Memory does not span beyond a block
                # print(f"lastOccurrence [within one block] = {self.lastOccurrenceDistances}\n")
                maxLastOccurrence = max(self.lastOccurrenceDistances) # must create this many logits (possibly downsampled)
                print(f"maximum lastOccurrence for {len(ints)} ints = {maxLastOccurrence}")
            case DistanceMode.ReservedIntsRandomlyPlaced:
                assert False, "DistanceDataset: Move this calculation to a separate data-generation script"
                # Write 1:num_target_ints in randomized locations in a list of different random ints:
                block = [random.randint(1+num_target_ints, numInts) for _ in range(numExamples)] # avoid 0 and 1:num_target_ints
                target_indices = random.sample(range(1, block_size + 1), num_target_ints) # random locations
                print(f"DistanceDataset: {target_indices=}")
                # N: ints[target_indices] = range(1, num_target_ints + 1) # I want to say this, but have to iterate:
                for k, target_index in enumerate(target_indices):
                    block[target_index] = k + 1 # exactly one occurrence at a random location
                    print(f"\t: block[{target_index}] = {k+1}")
                    occurrences[target_index] = 1 # logit is 1 to indicate "here is the occurrence"
        # N: ints[i:i+block_size] = block
        for j in range(block_size):
            ints[bi + j] = block[j] # exactly one occurrence at a random location
        tests.append(block)
        targs.append(occurrences)
    return tests, targs
# ---------------------------------- Create Datasets -------------------------------------------------

def create_words_datasets(input_file, block_size=None):
    words = read_input_file(input_file)
    chars = sorted(list(set(''.join(words))))
    vocab_size = len(chars) + 1
    if block_size is None:
        block_size = vocab_size
    train_words, test_words = split_dataset(words, block_size)
    train_dataset = CharDataset(train_words, chars, block_size)
    test_dataset = CharDataset(test_words, chars, block_size)
    return train_dataset, test_dataset, block_size

def create_distance_datasets(input_file, block_size, distance_mode):
    print(f"create_distance_datasets: Reading {input_file=}")
    lines = read_input_file(input_file)
    print(f"{lines=}")
    trgs, ints = zip(*[line.split('\t', 1) for line in lines])
    print(f"{trgs=}")
    print(f"{ints=}")
    if block_size is None:
        block_size = 1 + max(len(ln) for ln in lines)
    trgsx = []
    for ln in range(len(lines)):
        trgsx.append([0] * (block_size-1) + [int(trgs[ln])])
    print(f"{trgsx=}")
    intsx = []
    for line in ints:
        intsx.append([int(x) for x in line.split()])
    print(f"{trgsx=}")
    # train_lines, test_lines = split_dataset(lines, block_size, shuffle=False)
    train_trgsx, test_trgsx = split_dataset(trgsx, block_size, shuffle=False)
    train_lines, test_lines = split_dataset(ints, block_size, shuffle=False)
    train_ints = []
    for ln in train_lines:
        train_ints.append([int(x) for x in ln.split()])
    test_ints = []
    for ln in test_lines:
        test_ints.append([int(x) for x in ln.split()])
    print(f"{test_ints=}")
    print(f"{test_trgsx=}")
    print(f"{train_ints=}")
    print(f"{train_trgsx=}")
    # Assuming DistanceDataset requires some specific initialization
    train_dataset = DistanceDataset(distance_mode, train_ints, train_trgsx, block_size)
    test_dataset  = DistanceDataset(distance_mode, test_ints,  test_trgsx,  block_size)
    return train_dataset, test_dataset, block_size

def create_qa_datasets(input_file, block_size=None):
    words = read_input_file(input_file)
    # Process QA data
    # Splitting into targets and tests might be specific to ListopsDataset
    targets, tests = zip(*[line.split('\t', 1) for line in words])
    if block_size is None:
        block_size = 1 + max(len(w) for w in words)
    train_words, test_words = split_dataset(words, block_size)
    # Assuming ListopsDataset or a similar dataset class is used for QA
    train_dataset = ListopsDataset(train_words, block_size)
    test_dataset = ListopsDataset(test_words, block_size)
    return train_dataset, test_dataset, block_size

def create_datasets(input_file, data_mode, block_size=None):
    if data_mode == DataMode.WORDS:
        train_dataset, test_dataset, block_size = create_words_datasets(input_file, block_size)
    elif data_mode == DataMode.DISTANCE:
        distance_mode = DistanceMode.ReservedIntsRandomlyPlaced # maybe bring this out as a CL option
        print(f"Setting DISTANCE mode to: {distance_mode=}")
        train_dataset, test_dataset, block_size = create_distance_datasets(input_file, block_size, distance_mode)
    elif data_mode == DataMode.QA:
        train_dataset, test_dataset, block_size = create_qa_datasets(input_file, block_size)
    return train_dataset, test_dataset, block_size

# --------------------------------------------------------
# Data loading

from torch.utils.data import Sampler

class LoopingSequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        while True:  # Loop indefinitely
            for i in range(len(self.data_source)):
                yield i

    def __len__(self):
        return float('inf')  # Technically, the length is infinite


class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, shuffle, **kwargs):
        print(f"{shuffle=}")
        if shuffle:
            train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        else:
            train_sampler = LoopingSequentialSampler(dataset)
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch
