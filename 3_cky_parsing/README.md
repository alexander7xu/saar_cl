# Saar CoLi Assignment 2: POS tagging with HMMs

## Author

Juangui Xu

## Directory Structure

```
    dataset.py              -- Dataset-related functions
    main.ipynb              -- Main assignment notebook
    parser.py               -- Definition of Parser class
    parsing_results.txt     -- number of parses per sentence
    README.md               -- This README file
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
- However, running NLTK parser to get ground truth counts will take ~3min. Therefore, I suggest those who reproduce my expriment to save the ground truth counts in the first run, and read it directly from the file latter.

## Extra Points

- I implemented the method to count the number of parse trees without actually computing these parse trees.
- I plotted a Sentence Length vs. Speed & Number of Trees curve to compare the efficiency.
