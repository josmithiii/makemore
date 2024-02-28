#!/bin/sh -x

# USAGE:
# In PARENT DIRECTORY, say
#   sh tests/tests.sh

MAKEMORE=makemore.py

DATADIR=data
LISTOPSDIR=$DATADIR/listops
WORDSDIR=$DATADIR/words
DISTANCEDIR=$DATADIR/distance

WORDS_DATA="$WORDSDIR/names.txt"
LISTOPS_DATA="$LISTOPSDIR/train_d8.tsv"
DISTANCE_DATA="$DISTANCEDIR/dist1.txt"

echo TEST 1 && python $MAKEMORE --input "$WORDS_DATA" --data-mode "words" --type transformer --max-steps 210 && \
echo TEST 2 && python $MAKEMORE --input "$WORDS_DATA" --type gru --max-steps 210 && \
echo TEST 3 && python $MAKEMORE --input "$WORDS_DATA" --type mamba --max-steps 210 && \
echo TEST 4 && python $MAKEMORE --input "$LISTOPS_DATA" --data-mode "qa" --block-size -1 --type gru --max-steps 210 && \
echo TEST 5 && python $MAKEMORE --input "$LISTOPS_DATA" --data-mode "qa" --block-size -1 --type transformer --max-steps 210 && \
echo TEST 6 && python $MAKEMORE --input "$DISTANCE_DATA" --data-mode "distance" --batch-size 8 --type gru --max-steps 210 && \
say All tests completed with error code 0 && \
say complete || \
say failed
echo DONE (kill tail with ^C)
