from itertools import product
from typing import Generic, TypeVar, override
from collections import defaultdict
import abc

import nltk


def grammar_to_dict(
    grammar: nltk.CFG,
) -> tuple[dict[tuple[str, str], set[str]], dict[str, set[str]]]:
    nonterminal = dict[tuple[str, str], set[str]]()
    terminal = dict[str, set[str]]()

    for item in grammar.productions():
        assert isinstance(item, nltk.grammar.Production)
        parent, children = item.lhs(), item.rhs()
        assert isinstance(parent, nltk.grammar.Nonterminal)
        assert 1 <= len(children) <= 2
        if len(children) == 2:
            assert isinstance(children[0], nltk.grammar.Nonterminal)
            assert isinstance(children[1], nltk.grammar.Nonterminal)
            children = (children[0].symbol(), children[1].symbol())
            nt_children = nonterminal.get(children, None)
            if nt_children is None:
                nt_children = nonterminal[children] = set()
            nt_children.add(parent.symbol())
        else:
            assert isinstance(children[0], str)
            t_children = terminal.get(children[0], None)
            if t_children is None:
                t_children = terminal[children[0]] = set()
            t_children.add(parent.symbol())
    return nonterminal, terminal


RecordedT = TypeVar("RecordedT")


class RecordBase(Generic[RecordedT], abc.ABC):
    def __init__(self, sentence: list[str], terminal: dict[str, set[str]]) -> None:
        self._sentence_length = len(sentence)
        assert self.sentence_length > 0

        self._data = [
            defaultdict[str, RecordedT](self._make_default_value)
            for _ in range((self.sentence_length + 1) * self.sentence_length // 2)
        ]
        for idx, word in enumerate(sentence):
            for termi in terminal.get(word, ()):
                value = self._init_terminal(idx, word, termi)
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
            - So `right - left`
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
        pass

    @staticmethod
    @abc.abstractmethod
    def _init_terminal(idx: int, word: str, parent: str) -> RecordedT:
        pass

    @staticmethod
    @abc.abstractmethod
    def _make_default_value() -> RecordedT:
        pass


class CountingRecord(RecordBase[int]):
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
        record = self.get(left_idx, right_idx)
        left_value = self.get(left_idx, mid_idx)[left_symbol]
        right_value = self.get(mid_idx + 1, right_idx)[right_symbol]
        record[parent_symbol] += left_value * right_value

    @staticmethod
    @override
    def _init_terminal(idx: int, word: str, parent: str) -> int:
        return 1

    @staticmethod
    @override
    def _make_default_value() -> int:
        return 0


class TraceRecord(RecordBase[list[tuple[int, str, str]]]):
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
        record = self.get(left_idx, right_idx)
        record[parent_symbol].append((mid_idx, left_symbol, right_symbol))

    @staticmethod
    @override
    def _init_terminal(idx: int, word: str, parent: str) -> list[tuple[int, str, str]]:
        return [(idx, word, word)]

    @staticmethod
    @override
    def _make_default_value() -> list[tuple[int, str, str]]:
        return list()

    def trace(self, root_symbol: str) -> list[nltk.Tree]:
        if root_symbol not in self.get(0, self.sentence_length - 1).keys():
            return list()

        trees_buffer = dict[tuple[int, int, str], list[nltk.Tree]]()

        def recur(left: int, right: int, symbol: str) -> list[nltk.Tree]:
            key = (left, right, symbol)
            if key in trees_buffer:
                return trees_buffer[key]
            rec = self.get(left, right)[symbol]

            if left == right:
                assert len(rec) == 1
                trees = trees_buffer[key] = [nltk.Tree(symbol, [rec[0][-1]])]
                return trees

            trees = trees_buffer[key] = list()
            for mid, left_symbol, right_symbol in rec:
                left_trees = recur(left, mid, left_symbol)
                right_trees = recur(mid + 1, right, right_symbol)
                trees.extend(
                    nltk.Tree(symbol, children_trees)
                    for children_trees in product(left_trees, right_trees)
                )
            return trees

        results = recur(0, self.sentence_length - 1, root_symbol)
        return results


class CkyParser:
    def __init__(self, grammar: nltk.CFG) -> None:
        self._nonterminal, self._terminal = grammar_to_dict(grammar)
        self._start_symbol: str = grammar.start().symbol()

    def _induce(self, records: RecordBase, left: int, right: int, mid: int) -> None:
        for left_symbol, right_symbol in product(
            records.get(left, mid).keys(),
            records.get(mid + 1, right).keys(),
        ):
            for nt in self._nonterminal.get((left_symbol, right_symbol), ()):
                records.add(left, right, mid, left_symbol, right_symbol, nt)

    def _cky_parse_one_sentence(
        self, sentence: list[str], record_cls: type
    ) -> RecordBase:
        assert issubclass(record_cls, RecordBase)
        record = record_cls(sentence, self._terminal)
        for length in range(1, len(sentence)):
            for left in range(0, len(sentence) - length):
                right = left + length
                for mid in range(left, right):
                    self._induce(record, left, right, mid)
        return record

    def parse(self, sentence: list[str]) -> list[nltk.Tree]:
        record = self._cky_parse_one_sentence(sentence, TraceRecord)
        assert isinstance(record, TraceRecord)
        trees = record.trace(self._start_symbol)
        return trees

    def count(self, sentence: list[str]) -> int:
        record = self._cky_parse_one_sentence(sentence, CountingRecord)
        cnt = record.get(0, len(sentence) - 1)[self._start_symbol]
        return cnt
