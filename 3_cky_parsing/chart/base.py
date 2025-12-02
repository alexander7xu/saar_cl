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

    Structure:
    ```
    idx | 0 1 2 3 ... N-1 left (row index)
    ---------------------
    0   | D D D D ... D
    1   |   D D D ... D
    2   |     D D ... D
    ... |
    N-1 |             D
    right (column index)

    D is a dict that map nonterminal => record (so this is a 3D chart in actual)

    Note that there is empty in the cell[l, r] with l > r.
    An exception would be raised given such an access location.
    ```
    """

    def __init__(
        self, sentence: list[str], inv_terminal_productions: dict[str, set[str]]
    ) -> None:
        """
        :param sentence: list of words.
        :param inv_terminal_productions: dict that map terminal => {nt in all (nt -> terminal)}
        """
        self._sentence_length = len(sentence)
        assert self._sentence_length > 0

        self._data = [
            defaultdict[str, RecordedT](self._make_default_record)
            for _ in range((self._sentence_length + 1) * self._sentence_length // 2)
        ]
        for idx, word in enumerate(sentence):
            for nt in inv_terminal_productions.get(word, ()):
                init_rec = self._init_leaf_record(idx, word, nt)
                self._data[self._calc_offset(idx, idx)][nt] = init_rec

    def _calc_offset(self, left: int, right: int) -> int:
        """
        Calculate the real offset in the chart given tree span (`left`, `right`)

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
        assert 0 <= left <= right < self._sentence_length
        offset = left * (self._sentence_length * 2 - (left - 1)) // 2 + right - left
        return offset

    def get(self, left: int, right: int) -> defaultdict[str, RecordedT]:
        """
        Get the records at tree span (`left`, `right`)

        :return: a dict at tree span (`left`, `right`) that map nonterminal => record value
        :rtype: defaultdict[str, RecordedT]
        """
        return self._data[self._calc_offset(left, right)]

    @abc.abstractmethod
    def reduce(
        self,
        left_idx: int,
        right_idx: int,
        mid_idx: int,
        left_nonterminal: str,
        right_nonterminal: str,
        parent_nonterminal: str,
    ) -> None:
        """
        Given two subtrees:
        - left: span (`left_idx`, `mid_idx`) with `left_nonterminal`
        - right: span (`mid_idx+1`, `right_idx`) with `right_nonterminal`

        Record the parent tree with span (`left_idx`, `right_idx`), reduced by:
        (`left_nonterminal`, `right_nonterminal`) <- `parent_nonterminal`.
        """
        pass

    @abc.abstractmethod
    def output(self, root_nonterminal: str) -> Any:
        """
        Build final output based on the whole chart given the `root_nonterminal`
        """
        pass

    @abc.abstractmethod
    def _init_leaf_record(self, idx: int, word: str, nonterminal: str) -> RecordedT:
        """
        Perform the initialization of leaf record in CKY algorithm.

        :param idx: index of `word` in the sentence.
        :param word: word of the sentence at `idx`.
        :param nonterminal: in the production that `nonterminal -> word`
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _make_default_record() -> RecordedT:
        pass
