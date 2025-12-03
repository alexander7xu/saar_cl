from typing import override

from .base import ChartBase


class CountingChart(ChartBase[int]):
    """
    Chart class for CKY counter.

    The value type of record `dict` is `int`, record the counts of trees.
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
        # sum up the product of children counts.
        records = self.get(left_idx, right_idx)  # NT => countings
        left_count = self.get(left_idx, mid_idx)[left_nonterminal]
        right_count = self.get(mid_idx + 1, right_idx)[right_nonterminal]
        prod = left_count * right_count
        records[parent_nonterminal] = records.get(parent_nonterminal, 0) + prod

    @override
    def output(self, root_nonterminal: str) -> int:
        """
        Get the counting on the chart at span (0, N-1) with `root_nonterminal`
        """
        records = self.get(0, self._sentence_length - 1)  # NT => countings
        cnt = records.get(root_nonterminal, 0)
        return cnt

    @override
    def _calc_leaf_value(self, idx: int, word: str, nonterminal: str) -> int:
        return 1
