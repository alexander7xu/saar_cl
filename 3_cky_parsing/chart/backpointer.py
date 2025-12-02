import nltk

from typing import override
from dataclasses import dataclass
from itertools import product

from .base import ChartBase


@dataclass
class BackpointerRecord:
    mid_idx: int
    left_symbol: str
    right_symbol: str


class BackpointerChart(ChartBase[list[BackpointerRecord]]):
    @override
    def add(
        self,
        left_idx: int,
        right_idx: int,
        mid_idx: int,
        left_symbol: str,
        right_symbol: str,
        parent_symbol: str,
    ) -> None:
        """
        When reduction, record the backpointer (middle_index, left_nonterminal, right_nonterminal).

        Note that we record left_nonterminal and right_nonterminal in additional.
        This allows us to avoid recalculating productions during trace.
        """
        new_rec = BackpointerRecord(
            mid_idx=mid_idx, left_symbol=left_symbol, right_symbol=right_symbol
        )
        self.get(left_idx, right_idx)[parent_symbol].append(new_rec)

    @override
    def output(self, root_symbol: str) -> list[nltk.Tree]:
        """
        Find out all parse trees with $root_symbol as root in the records.
        """
        if root_symbol not in self.get(0, self.sentence_length - 1).keys():
            return list()

        # Avoid repeated calculation, beacuse a subtree may occur in different trees.
        trees_buffer = dict[tuple[int, int, str], list[nltk.Tree]]()

        def recur(left: int, right: int, symbol: str) -> list[nltk.Tree]:
            key = (left, right, symbol)
            if key in trees_buffer:
                return trees_buffer[key]
            records = self.get(left, right)[symbol]

            if left == right:  # For leaf, the record is [(index, word, word)]
                assert len(records) == 1
                trees = trees_buffer[key] = [
                    nltk.Tree(symbol, [records[0].left_symbol])
                ]
                return trees

            # Recursively build the left and right subtrees to construct the tree.
            trees = trees_buffer[key] = list()
            for rec in records:
                left_trees = recur(left, rec.mid_idx, rec.left_symbol)
                right_trees = recur(rec.mid_idx + 1, right, rec.right_symbol)
                trees.extend(
                    nltk.Tree(symbol, children_trees)
                    for children_trees in product(left_trees, right_trees)
                )
            return trees

        results = recur(0, self.sentence_length - 1, root_symbol)
        return results

    @override
    def _init_terminal_record(
        self, idx: int, word: str, parent: str
    ) -> list[BackpointerRecord]:
        """
        Record the index and word in sentence corresponding to a leaf.
        """
        return [BackpointerRecord(mid_idx=idx, left_symbol=word, right_symbol=word)]

    @staticmethod
    @override
    def _make_default_record() -> list[BackpointerRecord]:
        return list()
