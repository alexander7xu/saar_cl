import nltk

from typing import override
from dataclasses import dataclass
from itertools import product

from .base import ChartBase


@dataclass(frozen=True)
class BackpointerRecord:
    mid_idx: int
    left_nonterminal: str
    right_nonterminal: str


class BackpointerChart(ChartBase[list[BackpointerRecord]]):
    @override
    def reduce(
        self,
        left_idx: int,
        right_idx: int,
        mid_idx: int,
        left_nonterminal: str,
        right_nonterminal: str,
        parent_nonterminal: str,
    ) -> None:
        # Record the backpointer (middle_index, left_nonterminal, right_nonterminal).
        # Note that we record left_nonterminal and right_nonterminal in additional.
        # This allows us to avoid recalculating productions during trace.
        new_rec = BackpointerRecord(
            mid_idx=mid_idx,
            left_nonterminal=left_nonterminal,
            right_nonterminal=right_nonterminal,
        )
        self.get(left_idx, right_idx)[parent_nonterminal].append(new_rec)

    @override
    def output(self, root_nonterminal: str) -> list[nltk.Tree]:
        """
        Find out all parse trees with `root_nonterminal` as root in the records.
        """
        if root_nonterminal not in self.get(0, self._sentence_length - 1).keys():
            return list()

        # Avoid repeated calculation, beacuse a subtree may occur in different trees.
        trees_buffer = dict[tuple[int, int, str], list[nltk.Tree]]()

        def recur(left_idx: int, right_idx: int, nonterminal: str) -> list[nltk.Tree]:
            key = (left_idx, right_idx, nonterminal)
            if key in trees_buffer:
                return trees_buffer[key]
            records = self.get(left_idx, right_idx)[nonterminal]

            # For leaf, left_nonterminal=right_nonterminal=word
            if left_idx == right_idx:
                assert len(records) == 1
                trees = trees_buffer[key] = [
                    nltk.Tree(nonterminal, [records[0].left_nonterminal])
                ]
                return trees

            # Recursively build the left and right subtrees to construct the tree.
            trees = trees_buffer[key] = list()
            for rec in records:
                left_trees = recur(left_idx, rec.mid_idx, rec.left_nonterminal)
                right_trees = recur(rec.mid_idx + 1, right_idx, rec.right_nonterminal)
                trees.extend(
                    nltk.Tree(nonterminal, children_trees)
                    for children_trees in product(left_trees, right_trees)
                )
            return trees

        results = recur(0, self._sentence_length - 1, root_nonterminal)
        return results

    @override
    def _init_leaf_record(
        self, idx: int, word: str, nonterminal: str
    ) -> list[BackpointerRecord]:
        """
        Record the `idx` and `word` in sentence corresponding to a leaf.
        """
        rec = BackpointerRecord(
            mid_idx=idx, left_nonterminal=word, right_nonterminal=word
        )
        return [rec]

    @staticmethod
    @override
    def _make_default_record() -> list[BackpointerRecord]:
        return list()
