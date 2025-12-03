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
    """
    Chart class for CKY parser.

    The value type of record `dict` is `list[BackpointerRecord]`,
    recording all the middle index, left nonterminal, and right nonterminal reduced to the node.

    Note that we record left nonterminal and right nonterminal in additional.
    This allows us to avoid recalculating the production rules during trace.
    """

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
        records = self.get(left_idx, right_idx)
        if parent_nonterminal not in records:
            records[parent_nonterminal] = list()

        # record the mid_idx, left_nonterminal, and right_nonterminal reduced to parent_nonterminal.
        new_rec = BackpointerRecord(
            mid_idx=mid_idx,
            left_nonterminal=left_nonterminal,
            right_nonterminal=right_nonterminal,
        )
        records[parent_nonterminal].append(new_rec)

    @override
    def output(self, root_nonterminal: str) -> list[nltk.Tree]:
        """
        Find out all parse trees with `root_nonterminal` as root in the records.
        """
        if root_nonterminal not in self.get(0, self._sentence_length - 1):
            return list()

        # Avoid repeated calculation, beacuse a subtree may occur in different trees.
        trees_buffer = dict[tuple[int, int, str], list[nltk.Tree]]()

        def recur(left_idx: int, right_idx: int, nonterminal: str) -> list[nltk.Tree]:
            key = (left_idx, right_idx, nonterminal)
            if key in trees_buffer:
                return trees_buffer[key]
            backpointers = self.get(left_idx, right_idx)[nonterminal]

            # For leaf, left_nonterminal=right_nonterminal=word
            if left_idx == right_idx:
                assert len(backpointers) == 1
                trees = trees_buffer[key] = [
                    nltk.Tree(nonterminal, [backpointers[0].left_nonterminal])
                ]
                return trees

            # Recursively build the left and right subtrees to construct the tree.
            trees = trees_buffer[key] = list()
            for bp in backpointers:
                left_trees = recur(left_idx, bp.mid_idx, bp.left_nonterminal)
                right_trees = recur(bp.mid_idx + 1, right_idx, bp.right_nonterminal)
                trees.extend(
                    nltk.Tree(nonterminal, children)
                    for children in product(left_trees, right_trees)
                )
            return trees

        results = recur(0, self._sentence_length - 1, root_nonterminal)
        return results

    @override
    def _calc_leaf_value(
        self, idx: int, word: str, nonterminal: str
    ) -> list[BackpointerRecord]:
        """
        Record the `idx` and `word` in sentence corresponding to a leaf.
        """
        rec = BackpointerRecord(
            mid_idx=idx, left_nonterminal=word, right_nonterminal=word
        )
        return [rec]
