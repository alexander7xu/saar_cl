import nltk

from typing import override
from dataclasses import dataclass
from collections import defaultdict
import math

from .base import ChartBase


@dataclass
class ProbBackpointerRecord:
    logprob: float
    mid_idx: int
    left_symbol: str
    right_symbol: str


class ProbBackpointerChart(ChartBase[ProbBackpointerRecord]):
    def __init__(
        self,
        sentence: list[str],
        terminal: dict[str, set[str]],
        terminal_probs: dict[tuple[str, str], float],
        nonterminal_probs: dict[tuple[str, str, str], float],
    ) -> None:
        """
        Args:
        - sentence: list of words.
        - terminal: dict with structure (Terminal) => {set of Nonterminal Parents}
        - terminal_probs: dict with the strcuture (NT, T) => probability
        - nonterminal_probs: dict with the strcuture (Left, Right, Parent) => probability
        """

        neginf = lambda: -float("inf")
        self._terminal_logprobs = defaultdict[tuple[str, str], float](neginf)
        self._nonterminal_logprobs = defaultdict[tuple[str, str, str], float](neginf)

        self._terminal_logprobs.update(
            {k: math.log(v) for k, v in terminal_probs.items()}
        )
        self._nonterminal_logprobs.update(
            {k: math.log(v) for k, v in nonterminal_probs.items()}
        )
        super().__init__(sentence, terminal)

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
        When reduction with $parent_symbol at (left_idx, right_idx),
        only record the one with max probability.
        """
        left_logprob = self.get(left_idx, mid_idx)[left_symbol].logprob
        right_logprob = self.get(mid_idx + 1, right_idx)[right_symbol].logprob
        trans = self._nonterminal_logprobs[(left_symbol, right_symbol, parent_symbol)]
        new_logprob = left_logprob + right_logprob + trans

        record = self.get(left_idx, right_idx)
        best_logprob = record[parent_symbol].logprob
        if new_logprob > best_logprob:
            record[parent_symbol] = ProbBackpointerRecord(
                logprob=new_logprob,
                mid_idx=mid_idx,
                left_symbol=left_symbol,
                right_symbol=right_symbol,
            )

    @override
    def _init_terminal_record(
        self, idx: int, word: str, parent: str
    ) -> ProbBackpointerRecord:
        return ProbBackpointerRecord(
            logprob=self._terminal_logprobs[(word, parent)],
            mid_idx=idx,
            left_symbol=word,
            right_symbol=word,
        )

    @staticmethod
    @override
    def _make_default_record() -> ProbBackpointerRecord:
        return ProbBackpointerRecord(
            logprob=-float("inf"), mid_idx=-1, left_symbol="", right_symbol=""
        )

    @staticmethod
    def _dict_argmax(data: dict[str, ProbBackpointerRecord]) -> str:
        """
        Find out the key so that the probability `data[key][0]` >= `data[any_other_key][0]`.
        """

        best_logprob, best_key = -float("inf"), None
        for k, rec in data.items():
            if rec.logprob > best_logprob:
                best_prob, best_key = rec.logprob, k
        assert best_key is not None
        return best_key

    def trace(self, root_symbol: str) -> nltk.ProbabilisticTree | None:
        """
        Find out the parse tree with max probability with $root_symbol as root in the records.
        """
        if root_symbol not in self.get(0, self.sentence_length - 1).keys():
            return None

        def recur(left: int, right: int, symbol: str | None = None):
            record = self.get(left, right)
            # $symbol should be current node with max probability, or `start` for root node.
            if symbol is None:
                symbol = self._dict_argmax(record)
            record = record[symbol]

            if left == right:  # For leaf, left_symbol=right_symbol=word
                return nltk.ProbabilisticTree(
                    symbol, [record.left_symbol], prob=math.exp(record.logprob)
                )

            # Find out left and right subtrees and construct current tree.
            left_tree = recur(left, record.mid_idx, record.left_symbol)
            right_tree = recur(record.mid_idx + 1, right, record.right_symbol)
            result = nltk.ProbabilisticTree(
                symbol, [left_tree, right_tree], prob=math.exp(record.logprob)
            )
            return result

        result = recur(0, self.sentence_length - 1, root_symbol)
        return result
