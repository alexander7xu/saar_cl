import abc
from typing import Generic, TypeVar, Any
from collections import defaultdict


RecordedT = TypeVar("RecordedT")


class ChartBase(Generic[RecordedT], abc.ABC):
    """
    The CKY Parser chart data structure with features:
    - **Compact storage**: 1D-list with size=(N+1)*N//2, ~50% smaller than 2D-list with N*N.
    - **Abstract induction interface**: Subclasses only need to implement this method
        to perform the core calculations of different algorithms.
    """

    def __init__(self, sentence: list[str], terminal: dict[str, set[str]]) -> None:
        """
        Args:
        - sentence: list of words.
        - terminal: dict with structure (Terminal) => {set of Nonterminal Parents}
        """
        self._sentence_length = len(sentence)
        assert self.sentence_length > 0

        self._data = [
            defaultdict[str, RecordedT](self._make_default_record)
            for _ in range((self.sentence_length + 1) * self.sentence_length // 2)
        ]
        for idx, word in enumerate(sentence):
            for termi in terminal.get(word, ()):
                value = self._init_terminal_record(idx, word, termi)
                self._data[self._calc_offset(idx, idx)][termi] = value

    @property
    def sentence_length(self):
        return self._sentence_length

    def _calc_offset(self, left: int, right: int) -> int:
        """
        Structure:
        ```
        0,1,2,3,...,N-1
          1,2,3,...,N-1
            2,3,...,N-1
                    ...
                N-2,N-1
                    N-1
        ```

        When point to `left` row `right` column, the offset is:
        - Sum up the number of elements in rows before `left`. It's a trapezoid, with:
            - base-length `N`
            - top-length `N-(left-1)`
            - number of rows `left`
            - The formula is `(N + N-(left-1)) * left // 2`
        - Plus the offset in current row, aka. the column index:
            - First `left` elements would be "void"
            - So the real offset in current row is `right - left`
        """
        assert 0 <= left <= right < self.sentence_length
        offset = left * (self.sentence_length * 2 - (left - 1)) // 2 + right - left
        return offset

    def get(self, left: int, right: int) -> defaultdict[str, RecordedT]:
        return self._data[self._calc_offset(left, right)]

    @abc.abstractmethod
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
        Add a parent to form a new branch of tree:

        parent_symbol -> (left_symbol, right_symbol).

        The range of left is [left_idx, mid_idx], and [mid_idx+1, right_idx] for right.
        """
        pass

    @abc.abstractmethod
    def output(self, root_symbol: str) -> Any:
        """
        Build final output based on the whole chart
        """
        pass

    @abc.abstractmethod
    def _init_terminal_record(self, idx: int, word: str, parent: str) -> RecordedT:
        """
        Args:
        - idx: index of word in the sentence.
        - word: of the sentence.
        - parent: nonterminal in the production `nonterminal -> word`
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _make_default_record() -> RecordedT:
        pass
