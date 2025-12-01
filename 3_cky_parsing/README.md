# Saar CoLi Assignment 2: POS tagging with HMMs

## Author

Juangui Xu

## Directory Structure

```
│   dataset.py                  -- grammar and sentences loading
│   main.ipynb                  -- main Notebook
│   measure.py                  -- F1 score calculation function
│   parser.py                   -- Parser class, the main part of CKY algorithm
│   parsing_results.txt         -- Parses counts of sentences
│   README.md                   -- This README file
│
└───chart
        backpointer.py          -- BackpointerChar class, used for building trees
        base.py                 -- ChartBase, base class of chart classes
        counting.py             -- CountingChart class, used for counting without building trees
        prob_backpointer.py     -- ProbBackpointerChart class, used for ViterbiCKY
        __init__.py             -- This README file
```

## Environments

### Device

- CPU: Snapdragon (TM) 8cx Gen 3 @ 2.69 GHz (8 Cores)
- RAM: 16 GB
- OS: Windows 11 Pro ARM 25H2

### Packages

```
Python==3.13.9
ipykernel==7.0.1
matplotlib==3.10.7
nltk==3.9.2
svgling==0.5.0
```

## Runtime

- It take shorter than 10s on my computer to parse / count all 98 sentences in the dataset.
- However, running NLTK parser to get ground truths will take ~1min.

## Extra Points

- I implemented a method to convert the nltk format CFG into Chomsky Normal Form.
- I implemented the method to count the number of parse trees without actually computing these parse trees.
- I plotted a Sentence Length vs. Speed & Number of Trees curve to compare the efficiency.
- I implemented the labeled and unlabeled F1 score calculation.
- I implemented the Viterbi CKY parser, it achieved `F1=1.0` when compared with `nltk.ViterbiParser` on all sentences.
