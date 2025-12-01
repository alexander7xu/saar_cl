from typing import override

from .base import ChartBase


class CountingChart(ChartBase[int]):
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
        When reduction, sum up the product of children counts.
        """
        left_count = self.get(left_idx, mid_idx)[left_symbol]
        right_count = self.get(mid_idx + 1, right_idx)[right_symbol]
        self.get(left_idx, right_idx)[parent_symbol] += left_count * right_count

    @override
    def _init_terminal_record(self, idx: int, word: str, parent: str) -> int:
        return 1

    @staticmethod
    @override
    def _make_default_record() -> int:
        return 0
