from dataset import load_grammar, load_sentences
from parser import CkyParser


gt_cnt = 2085
grammar = load_grammar()
senten = load_sentences()[0]
parser = CkyParser(grammar)

assert parser.count(senten) == gt_cnt
