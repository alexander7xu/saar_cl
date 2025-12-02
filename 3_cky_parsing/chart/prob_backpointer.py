import nltk

from typing import override
from dataclasses import dataclass
from collections import defaultdict
import math

from .base import ChartBase


@dataclass(frozen=True)
class ProbBackpointerRecord:
    logprob: float
    mid_idx: int
    left_nonterminal: str
    right_nonterminal: str


class ProbBackpointerChart(ChartBase[ProbBackpointerRecord]):
    def __init__(
        self,
        sentence: list[str],
        inv_terminal_productions: dict[str, set[str]],
        leaf_probs: dict[tuple[str, str], float],
        nonleaf_probs: dict[tuple[str, str, str], float],
    ) -> None:
        """
        :param sentence: list of words.
        :param inv_terminal_productions: dict that map terminal => {nt in all (nt -> terminal)}
        :param leaf_probs: dict with the strcuture (word, NT) => probability, given rule (NT -> word)
        :param nonleaf_probs: dict with the strcuture (NT_left, NT_right, NT_parent) => probability,
               given rule (NT_parent -> NT_left NT_right)
        """

        neginf = lambda: -float("inf")
        self._leaf_probs = leaf_probs
        self._nonleaf_logprobs = defaultdict[tuple[str, str, str], float](neginf)
        self._nonleaf_logprobs.update(
            {k: math.log(v) for k, v in nonleaf_probs.items()}
        )
        super().__init__(sentence, inv_terminal_productions=inv_terminal_productions)

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
        left_logprob = self.get(left_idx, mid_idx)[left_nonterminal].logprob
        right_logprob = self.get(mid_idx + 1, right_idx)[right_nonterminal].logprob
        trans_logprob = self._nonleaf_logprobs[  # (left, right, parent) => logprob
            (left_nonterminal, right_nonterminal, parent_nonterminal)
        ]
        new_logprob = left_logprob + right_logprob + trans_logprob
        assert not math.isinf(new_logprob)

        # only record the one with max probability.
        record = self.get(left_idx, right_idx)
        best_logprob = record[parent_nonterminal].logprob
        if new_logprob > best_logprob:
            record[parent_nonterminal] = ProbBackpointerRecord(
                logprob=new_logprob,
                mid_idx=mid_idx,
                left_nonterminal=left_nonterminal,
                right_nonterminal=right_nonterminal,
            )

    @override
    def output(self, root_nonterminal: str) -> nltk.ProbabilisticTree | None:
        """
        Find out the parse tree with max probability with `root_nonterminal` as root in the records.

        return `None` if no such a tree.
        """
        if root_nonterminal not in self.get(0, self._sentence_length - 1).keys():
            return None

        def recur(left_idx: int, right_idx: int, nonterminal: str | None = None):
            record = self.get(left_idx, right_idx)
            # $nonterminal should be current node with max probability, or `start` for root node.
            if nonterminal is None:
                nonterminal = self._dict_argmax(record)
            record = record[nonterminal]

            # For leaf, left_nonterminal=right_nonterminal=word
            if left_idx == right_idx:
                return nltk.ProbabilisticTree(
                    nonterminal,
                    [record.left_nonterminal],
                    prob=math.exp(record.logprob),
                )

            # Find out left and right subtrees.
            left_tree = recur(left_idx, record.mid_idx, record.left_nonterminal)
            right_tree = recur(record.mid_idx + 1, right_idx, record.right_nonterminal)

            # DO NOT use logprob kwarg here, because nltk will
            # use 2**logprob to obtain prob, rather than exp(logprob)
            result = nltk.ProbabilisticTree(
                nonterminal, [left_tree, right_tree], prob=math.exp(record.logprob)
            )
            return result

        result = recur(0, self._sentence_length - 1, root_nonterminal)
        return result

    @override
    def _init_leaf_record(
        self, idx: int, word: str, nonterminal: str
    ) -> ProbBackpointerRecord:
        # Prob must be in the dict
        prob = self._leaf_probs[(word, nonterminal)]
        return ProbBackpointerRecord(
            logprob=math.log(prob),
            mid_idx=idx,
            left_nonterminal=word,
            right_nonterminal=word,
        )

    @staticmethod
    @override
    def _make_default_record() -> ProbBackpointerRecord:
        return ProbBackpointerRecord(
            logprob=-float("inf"), mid_idx=-1, left_nonterminal="", right_nonterminal=""
        )

    @staticmethod
    def _dict_argmax(data: dict[str, ProbBackpointerRecord]) -> str:
        """
        Return the best_key so that the probability
        `data[best_key].logprob` >= `data[key].logprob` for all `key` in `data`.
        """

        best_logprob, best_key = -float("inf"), None
        for k, rec in data.items():
            if rec.logprob > best_logprob:
                best_prob, best_key = rec.logprob, k
        assert best_key is not None
        return best_key
