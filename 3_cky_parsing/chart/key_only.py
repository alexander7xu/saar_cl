from typing import override

from .base import ChartBase


class KeyOnlyChart(ChartBase[None]):
    """
    Chart class for CKY recognizer.

    The value type of record `dict` is `None`, hence it will behave like `set`.
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
        records = self.get(left_idx, right_idx)  # NT => None
        assert left_nonterminal in self.get(left_idx, mid_idx)
        assert right_nonterminal in self.get(mid_idx + 1, right_idx)
        records[parent_nonterminal] = None  # Just use the dict as a set

    @override
    def output(self, root_nonterminal: str) -> bool:
        """
        Judge whether the sentence is grammatical
        """
        accept = root_nonterminal in self.get(0, self._sentence_length - 1)
        return accept

    @override
    def _calc_leaf_value(self, idx: int, word: str, nonterminal: str) -> None:
        return None
