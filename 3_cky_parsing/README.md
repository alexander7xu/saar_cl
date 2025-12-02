# Saar CoLi Assignment 3: CKY parsing

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
- However, running NLTK parser to get ground truths will take ~1min. **My algorithm is faster than NLTK**.

## Extra Points

- I implemented a method to convert the nltk format CFG into Chomsky Normal Form.
- I implemented a method to count the number of parse trees without actually building them.
- I plotted a Sentence Length vs. Speed & Number of Trees curve to compare the efficiency (**in nanosecond precision**).
- I implemented the labeled and unlabeled F1 score calculation and tested it with the example parse trees of sentence `But the concept is workable`.
- I implemented an algorithm to build PCFG based on all the parse trees from my parser.
- I implemented a Viterbi CKY parser, and compared it against `nltk.ViterbiParser` on all sentences.

### The Idea of Grammar Normalization

Only rules in `A → B C`, `A -> a` is allowed by Chomsky Normal Form. However, the nltk.CFG contains rules that `A -> NT` or `A -> B C D ...`. To normalize the grammar, we need to:
- Convert multi-branch trees `A -> B C D ...` into binary trees.
- Reduce rules `A -> B -> C -> a` as `A -> a`.

See `convert_cfg_to_chomsky_normal_form()` in `./dataset.py`.

### The Idea of Counting without Building Trees

In the code of building trees using backpointer (Line 66-71 in `./chart/backpointer.py`):

```python
trees = trees_buffer[key] = list()
for rec in records:
    left_trees = recur(left_idx, rec.mid_idx, rec.left_nonterminal)
    right_trees = recur(rec.mid_idx + 1, right_idx, rec.right_nonterminal)
    trees.extend(
        nltk.Tree(nonterminal, children_trees)
        for children_trees in product(left_trees, right_trees)
    )
```

We can see that every time after building the left subtrees collection $L$ and right subtrees collection $R$, we use each the pair in the Cartesian product $L\times R$ to build a new tree. Therefore, the number of new trees would be $|L\times R| = |L|\times |R|$. What we need to do is replacing the backpointers with the counts, and calculating them on-the-fly when reducing in CKY.

See `CountingChart.reduce()` in `./chart/counting.py`

### The Idea of F1-score Calculation

It's hard to compare two trees directly node-by-node.
1. We need to convert the trees into spans firstly, so that each span can represents one node with their structural and label information. (Although a span may unable to uniquely represent a node, this is tolerated by the F1 calculation rules.)
2. Then we can compute the precision and recall, and then F1-score by comparing the spans corresponding to the two trees.

See `tree_to_spans()` and `tree_f1_score()` `./measure.py`

### The Idea of building PCFG and Viterbi CKY Parser

- It's extremely easy to implement the Viterbi CKY Parser, just modify the chart used in standard CKY parser to record only the backpointer with max probability.
- In order to evaulate the CKY Parser, we need a PCFG. The PCFG can be obtained based on all the parse trees from the standard CKY parser. Just count the number of rules on every trees, then calculate the probabilities.

See `ProbBackpointerChart.reduce()` in `./chart/prob_backpointer.py` and the main notebook.

## Engineering Design Ideas

We can see that severals variants of CKY parser share the same main logic. The only differences are on the charts. For detail in the algorithm: 

```
Data structure:
    # Ch(i,k) eventually contains {A | A ⇒* wi ... wk-1}, initially all empty. 
    Recording(i,j,s), abstract structure records the data in tree span at [i...,j] with nonterminal s, initially with Recording.default()

for each i from 1 to n:
    for each production rule A → w_i:
        # add A to Ch(i, i+1)
        Recording(i,i+1,A).init()

for each width b from 2 to n:
    for each start position i from 1 to n-b+1:
        for each left width k from 1 to b-1:
            for each B in Ch(i,i+k) and C in Ch(i+k,i+b):
                for each production rule A -> B C:
                    # add A to Ch(i, i+b)
                    Recording(i,i+b,A).add((i,i+k,B), (i+k,i+b,C))

Output: Recording(1,n,sigma).output()
```

For example, the structure of `Recording` could be counting for CKY counter, backpointers for standard parser, and backpointers with probability for viterbi parser.

Therefore, we can design the chart as an abstract data structure class and implement it as different subclasses. By providing the same CKY algorithm backbone with these different chart object, we are able to avoid rewriting the backbone for every algorithm variants, thereby achieving the goal of efficient code reuse. As we seen in the pseudo-code, the interface of a chart class should contain `add()`, `init()`, `default()`, and `output()` methods, which will be implemented by the subclasses.

## The Core Algorithm Parts in the Code

Chart classes: I use OOP to implement a base class with several chart subclasses in `./chart/`.
- The base class implemented the access abstraction of chart. Note that its compact structure **saves 50% of the space**.
- The base class define abstract method `ChartBase.reduce()` that need to be implemented by subclasses for main chart actions when reduce in CKY algorithm.
- The base class define abstract method `ChartBase.output()` that need to be implemented by subclasses for producing the final result based on the whole chart.
- The base class define abstract method `ChartBase._init_leaf_record()` that need to be implemented by subclasses for the initialization in CKY algorithm. The method is proteced and will be called in `ChartBase.__init__()`.
- The base class define abstract method `ChartBase._make_default_record()` that need to be implemented by subclasses for the default value in CKY algorithm. The method is protected and will be use as factory function of `defaultdict`.
- Subclasses are highly correlated with CKY algorithm variants, hence I will note them in the parser.

CKY recognizing: `CkyParser.recognize()` in `./parser.py`. I did not implement a independent recognizer because **it is unnecessary**. Given the fact that it share the same time and space complexity with counting, it is wiser to directly use the result of CKY counting.

CKY parsing: `CkyParser.parse()` in `./parser.py`
- `CkyParser._cky_parse_one_sentence()` in `./parser.py` implements the core logic (the 3 loops for Dynamic Programming).
- `CkyParser._reduce()` in `./parser.py` implements the recude action of CKY. I separated it from `CkyParser._cky_parse_one_sentence` because the depth of loops was too large to make the code ugly.
- `BackpointerChart.reduce()` in `./chart/backpointer.py` implements the chart record method.
- `BackpointerChart.output()` in `./chart/backpointer.py` build trees according to the final chart.

CKY counting: `CkyParser.count()` in `./parser.py`. The main logic is same as CKY parsing. The only difference is the chart, using `CountingChart` in `./chart/counting.py`, in which we directly obtain the count by calculating the product of the subtrees.

ViterbiCKY: `CkyParser.viterbi()` in `./parser.py`. The main logic is same as CKY parsing. The only difference is the chart, using `ProbBackpointerChart` in `./chart/prob_backpointer.py`, in which we only record the reduce with max probability.

Labeled and unlabeled F1 score: `tree_f1_score()` in `./measure.py`.
