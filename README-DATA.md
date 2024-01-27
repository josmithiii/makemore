# Data Input and Output

Here are the supported data types at present and how they work

## Common

All data formats are provided by one text file from [./data/](./data/) containing one training example per line.

If the input `--block-size` is not specified, a minimal adequate
buffer size is calculated automatically from the input data file.

If the output `--logits-size` is not specified, a minimal adequate
number of logits size is calculated automatically from the input data
file.

## Words

The original `makemore` data file is `names.txt`, containing one name
per line, such as `emma`.  Each character is embedded separately, so
that the input to each neural-net model looks like
`<0><e><m><m><a><0>`, where each character is a row-vector of size
`n_embd` (64 elements by default), and `<0>` refers to the embedding
of a start/stop token.  Thus, the `emma` example requires the input
buffer size to be at least 5, and the actual number of float32s is
`5*n_embd`, or 320 numbers. The input buffer-size is specified in
units of tokens, so `5` would suffice for `emma`.  

The task is to predict the next char (or <0>) in a name given the
chars before it in the word. Thus, the neural weights implicitly learn
the conditional probability distribution p(char|previousChars).

### Outputs

Since the prediction task is the next character in a name or <0> to
terminate the name, only 27 logits are needed at the output of any
model. This is determined automatically from the input data file if
`--logits-size` is not specified.

## ListOps

The ListOps data type is a "Question-Answer" data type in the format `Answer<tab>Question`,
where the question is a mathematical expression involving many parentheses, e.g.,
```
7	( ( ( ( [MAX 1 ) 7 ) 2 ) ] )
```
(Square brackets seem to be interchangeable with parens.)
This format gets translated internally to `|7|( ( ( [MAX 1 ) 7 ) 2 ) ] )`, etc.

The entire question part of the line is treated as a single "word" in original `makemore`.
Thus, each char is embedded separately, etc.

The task is to predict the answer from the question.  This too learns
the conditional probability distribution, but it can also solve the
task by learning algorithms to balance parens and do elementary
operations on ints.

### Outputs

The input data file is examined to determine the maximum number of
logits needed to express the answer (typically one digit).

## Distance

The synthetic `Distance` benchmark takes a list of integers, one per
line, from [./data/distance/](./data/distance/), or it will generate
data synthetically (random ints) if no input file is specified.

The distance to last repetition of each int is computed automatically to create the "answers".

The task is to predict the answer given a sequence of ints.  Ideally,
the input buffer contains all ints up to "now", and for transformers,
this is the only way they can work.  For RNNs, on the other hand, even
one input int is enough, since all past inputs are summed into the
hidden state vector at some amplitude level.

If the input buffer size is set to 1, then only a pure RNN can be
used.  Anything more than 1 allows an "FIR part" for RNNs, and/or
transformer usage.

### Outputs

The output logits indicate where the last int occurred in the input,
and this could be anywhere all the way back to the start of the file.
Thus, a 1000 line file needs 999 logits to express that the int on the
last line last occurred on the first line.  (We could alternatively
learn a binary encoding of the distance.)

The datatype `--data-mode distance-exp` is like `--type distance`
(just described) except that the last-occurrence logits are
exponentially distributed, starting with single-sample-step resolution
and ending with a much lower resolution.  It could make sense to learn
this exponent.

---

#### [Back to `README`.md](README.md)
