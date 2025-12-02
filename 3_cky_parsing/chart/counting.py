from typing import override

from .base import ChartBase


class CountingChart(ChartBase[int]):
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
        left_count = self.get(left_idx, mid_idx)[left_nonterminal]
        right_count = self.get(mid_idx + 1, right_idx)[right_nonterminal]
        self.get(left_idx, right_idx)[parent_nonterminal] += left_count * right_count

    @override
    def output(self, root_nonterminal: str) -> int:
        """
        Get the counting on the chart at span (0, N-1) with `root_nonterminal`
        """
        cnt = self.get(0, self._sentence_length - 1)[root_nonterminal]
        return cnt

    @override
    def _init_leaf_record(self, idx: int, word: str, nonterminal: str) -> int:
        return 1

    @staticmethod
    @override
    def _make_default_record() -> int:
        return 0
