import nltk

from typing import override
from dataclasses import dataclass
import math

from .base import ChartBase


@dataclass(frozen=True)
class ProbBackpointerRecord:
    logprob: float
    mid_idx: int
    left_nonterminal: str
    right_nonterminal: str


class ProbBackpointerChart(ChartBase[ProbBackpointerRecord]):
    """
    Chart class for Viterbi CKY parser.

    The value type of record `dict` is `ProbBackpointerRecord`,
    recording the log probability, middle index, left nonterminal, and right nonterminal reduced to the node.
    Only the one with max log probability would be recorded.

    Note that we record left nonterminal and right nonterminal in additional.
    This allows us to avoid recalculating the production rules during trace.
    """

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
        :param leaf_probs: dict that map (word, NT) => probability, given rule (NT -> word)
        :param nonleaf_probs: dict that map (NT_left, NT_right, NT_parent) => probability,
               given rule (NT_parent -> NT_left NT_right)
        """
        self._leaf_probs = leaf_probs
        self._nonleaf_logprobs = {k: math.log(v) for k, v in nonleaf_probs.items()}
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
        assert math.isfinite(new_logprob)

        records = self.get(left_idx, right_idx)  # NT => record value
        if parent_nonterminal not in records:
            records[parent_nonterminal] = ProbBackpointerRecord(
                logprob=new_logprob,
                mid_idx=mid_idx,
                left_nonterminal=left_nonterminal,
                right_nonterminal=right_nonterminal,
            )
        # only record the one with max probability.
        elif new_logprob > records[parent_nonterminal].logprob:
            records[parent_nonterminal] = ProbBackpointerRecord(
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
        if root_nonterminal not in self.get(0, self._sentence_length - 1):
            return None

        def recur(left_idx: int, right_idx: int, nonterminal: str | None = None):
            records = self.get(left_idx, right_idx)
            # nonterminal should be current node with max probability, or `start` for root node.
            if nonterminal is None:
                nonterminal = self._dict_argmax(records)
            backpointer = records[nonterminal]

            # For leaf, left_nonterminal=right_nonterminal=word
            if left_idx == right_idx:
                return nltk.ProbabilisticTree(
                    nonterminal,
                    [backpointer.left_nonterminal],
                    prob=math.exp(backpointer.logprob),
                )

            # Find out left subtree and right subtree.
            left_tree = recur(
                left_idx, backpointer.mid_idx, backpointer.left_nonterminal
            )
            right_tree = recur(
                backpointer.mid_idx + 1, right_idx, backpointer.right_nonterminal
            )

            # DO NOT use logprob kwarg here, because nltk will
            # use 2**logprob to obtain prob, rather than exp(logprob)
            result = nltk.ProbabilisticTree(
                nonterminal, [left_tree, right_tree], prob=math.exp(backpointer.logprob)
            )
            return result

        result = recur(0, self._sentence_length - 1, root_nonterminal)
        return result

    @override
    def _calc_leaf_value(
        self, idx: int, word: str, nonterminal: str
    ) -> ProbBackpointerRecord:
        # Prob must be in the dict that map (T, NT) => prob
        prob = self._leaf_probs[(word, nonterminal)]
        assert 0.0 < prob <= 1.0
        return ProbBackpointerRecord(
            logprob=math.log(prob),
            mid_idx=idx,
            left_nonterminal=word,
            right_nonterminal=word,
        )

    @staticmethod
    def _dict_argmax(data: dict[str, ProbBackpointerRecord]) -> str:
        """
        Return the `best_key` so that the probability
        `data[best_key].logprob >= data[key].logprob` for all `key` in `data`.
        """
        best_logprob, best_key = -math.inf, None
        for k, v in data.items():
            if v.logprob > best_logprob:
                best_logprob, best_key = v.logprob, k
        assert best_key is not None
        return best_key
