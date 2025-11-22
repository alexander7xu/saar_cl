import torch
from torchtyping import TensorType as T

from typing import Collection, Self, override
from itertools import groupby

from .base import HmmPosTaggerSupervisedBase, HmmPosTaggerTensorBase
from .utils import Counter, log_sum


class HmmNgramPosTagger(HmmPosTaggerSupervisedBase):
    """
    N-gram version of HMM POS Tagger (consider previous N states).

    Use dict as the data structure of parameters.
    """

    def __init__(self, n_previous_states: int = 1) -> None:
        self._n_previous_states = n_previous_states
        probs = {
            "transition": dict[tuple[str, ...], dict[str, float]](),
            "emission": dict[tuple[str, ...], dict[str, float]](),
        }
        counters = {
            "state": Counter[tuple[str, ...]](),
            "prev_state": Counter[tuple[str, ...]](),
            "transition": Counter[tuple[tuple[str, ...], str]](),
            "emission": Counter[tuple[tuple[str, ...], str]](),
        }
        self._init_states_probs = {
            (self.TAG_START,) * self._n_previous_states: (0.0, None)
        }
        super().__init__(probs, counters)

    @override
    def _viterbi_one_timestep(
        self,
        word: str,
        prev_states_probs: dict[tuple[str, ...], tuple[float, tuple[str, ...]]] | None,
    ) -> dict[tuple[str, ...], tuple[float, tuple[str, ...]]]:
        if prev_states_probs is None:
            prev_states_probs = self._init_states_probs

        results, seen = dict[tuple[str, ...], tuple[float, tuple[str, ...]]](), set()
        for prev_stat, (prob_v, _) in prev_states_probs.items():
            transition = self._probs["transition"].get(prev_stat, None)
            if transition is None:  # Only occurs when the model is not trained
                raise RuntimeError("please train the model before evaluation")

            for trans, prob_a in transition.items():
                next_state = prev_stat[1:] + (trans,)
                emission = self._probs["emission"].get(next_state, None)
                if emission is None:  # Only occurs when the model is not trained
                    raise RuntimeError("please train the model before evaluation")

                prob_b, prob = emission.get(word, None), None
                if prob_b is None:
                    if next_state not in seen:
                        prob = prob_v + prob_a
                else:
                    seen.add(next_state)
                    prob = prob_v + prob_a + prob_b

                if prob is not None:
                    if prob > results.get(next_state, (-torch.inf, ""))[0]:
                        results[next_state] = (prob, prev_stat)

        if len(seen) > 0:
            results = {k: v for k, v in results.items() if k in seen}
        assert len(results) > 0
        return results

    @staticmethod
    @override
    def _trace_best_states(
        states_probs_list: list[dict[tuple[str, ...], tuple[float, tuple[str, ...]]]],
    ) -> list[tuple[str, float]]:
        results = list()

        state = max(states_probs_list[-1].items(), key=lambda x: x[1][0])[0]
        for states_probs in states_probs_list[-1::-1]:
            prob = states_probs[state][0]
            results.append((state[-1], prob))
            state = states_probs[state][1]
        results.reverse()
        return results

    @override
    def _fit_one_sentence(
        self, sentence: Collection[str], pos_tags: Collection[str]
    ) -> Self:
        prev_state = (self.TAG_START,) * self._n_previous_states
        for word, tag in zip(sentence, pos_tags):
            self._counters["prev_state"].inc(prev_state)
            self._counters["transition"].inc((prev_state, tag))
            prev_state = prev_state[1:] + (tag,)
            self._counters["state"].inc(prev_state)
            self._counters["emission"].inc((prev_state, word))
        return self

    @override
    def _parameterize(self) -> Self:
        def param(x: float):
            return torch.log(torch.tensor(x))

        for v in self._probs.values():
            v.clear()
        for prev_state in self._counters["prev_state"].keys():
            self._probs["transition"][prev_state] = dict()
        for state in self._counters["state"].keys():
            self._probs["emission"][state] = dict()

        for (prev_state, state), cnt in self._counters["transition"].items():
            self._probs["transition"][prev_state][state] = param(
                cnt / self._counters["prev_state"][prev_state]
            )
        for (state, word), cnt in self._counters["emission"].items():
            self._probs["emission"][state][word] = param(
                cnt / self._counters["state"][state]
            )
        return self


class HmmPosTagger(HmmPosTaggerSupervisedBase, HmmPosTaggerTensorBase):
    """
    Standard HMM POS Tagger.
    """

    def __init__(
        self,
        possible_states: Collection[str],
        possible_words: Collection[str],
    ) -> None:
        self._counters: dict[str, Counter] = {
            "state": Counter[str](),
            "prev_state": Counter[str](),
            "transition": Counter[tuple[str, str]](),
            "emission": Counter[tuple[str, str]](),
        }
        HmmPosTaggerTensorBase.__init__(self, possible_states, possible_words)
        # HmmPosTaggerSupervisedBase.__init__(self, self._probs, counters)
        assert self.TAG_START not in self._state_to_idx

    @override
    def _fit_one_sentence(
        self, sentence: Collection[str], pos_tags: Collection[str]
    ) -> Self:
        prev_state = self.TAG_START
        for word, tag in zip(sentence, pos_tags):
            self._counters["prev_state"].inc(prev_state)
            self._counters["transition"].inc((prev_state, tag))
            prev_state = tag
            self._counters["state"].inc(prev_state)
            self._counters["emission"].inc((prev_state, word))
        return self

    @override
    def _parameterize(self) -> Self:
        # Cache-optimized matrix writing, utilizing the Locality of Reference principle

        def group_counter(counter: Counter, map_y: dict = self._state_to_idx):
            def parse_grouped(item: list[tuple[int, int, int]]):
                u = item[0]
                _, v, cnt = list(zip(*item[1]))
                return (u, torch.tensor(v), torch.tensor(cnt))

            sorted_items = sorted(
                (self._state_to_idx.get(x, -1), map_y[y], z)
                for (x, y), z in counter.items()
            )
            return (parse_grouped(x) for x in groupby(sorted_items, key=lambda x: x[0]))

        for v in self._probs.values():
            v[:] = -torch.inf
        transition, emission = self._probs["transition"], self._probs["emission"]

        counter_state = self._counters["state"]
        counter_prev_state = self._counters["prev_state"]
        counter_transition = group_counter(self._counters["transition"])
        counter_emission = group_counter(self._counters["emission"], self._word_to_idx)

        u, v, cnt = next(counter_transition)
        assert u == -1
        self._probs["init"][v] = torch.log(cnt / counter_prev_state[self.TAG_START])
        for u, v, cnt in counter_transition:
            transition[u][v] = torch.log(cnt / counter_prev_state[self._states_name[u]])

        for v, w, cnt in counter_emission:
            emission[v][w] = torch.log(cnt / counter_state[self._states_name[v]])

        if False:
            assert (
                (torch.exp(self._probs["transition"]).sum(1) - 1).abs() < 1e-6
            ).all()
            assert ((torch.exp(self._probs["emission"]).sum(1) - 1).abs() < 1e-6).all()
        return self


class HmmPosTaggerDeprecated(HmmPosTagger):
    @override
    def _parameterize(self) -> Self:
        # A deprecated implementation, slower than the version with cache-optimization

        def param(x: float):
            return torch.log(torch.tensor(x))

        for v in self._probs.values():
            v[:] = -torch.inf

        for (prev_state, state), cnt in self._counters["transition"].items():
            v = self._state_to_idx[state]
            if prev_state == self.TAG_START:
                self._probs["init"][v] = param(
                    cnt / self._counters["prev_state"][self.TAG_START]
                )
                continue
            u = self._state_to_idx[prev_state]
            self._probs["transition"][u, v] = param(
                cnt / self._counters["prev_state"][prev_state]
            )
        for (state, word), cnt in self._counters["emission"].items():
            v, w = self._state_to_idx[state], self._word_to_idx[word]
            self._probs["emission"][v, w] = param(cnt / self._counters["state"][state])

        if False:
            assert (
                (torch.exp(self._probs["transition"]).sum(1) - 1).abs() < 1e-6
            ).all()
            assert ((torch.exp(self._probs["emission"]).sum(1) - 1).abs() < 1e-6).all()
        return self


class HmmMaskedPosTagger(HmmPosTagger):
    WORD_UNSEEN = "__UNSEEN__"

    def __init__(
        self,
        possible_states: Collection[str],
        possible_words: Collection[str],
    ) -> None:
        possible_words = set(possible_words)
        assert self.WORD_UNSEEN not in possible_words
        possible_words.add(self.WORD_UNSEEN)
        super().__init__(possible_states, possible_words)

    @override
    def _viterbi_one_timestep(
        self, word: str, prev_states_probs: tuple[T["N"], T["N", int]] | None
    ) -> tuple[T["N"], T["N", int]]:
        return super()._viterbi_one_timestep(
            word if word in self._word_to_idx else self.WORD_UNSEEN, prev_states_probs
        )


class UnsupervisedHmmPosTagger(HmmPosTaggerTensorBase):
    """
    Unsupervised HMM POS Tagger.
    """

    def __init__(
        self,
        num_states: int,
        possible_words: Collection[str],
        init_seed: int = 42,
    ) -> None:
        possible_states = map(str, range(num_states))
        super().__init__(possible_states, possible_words, probs_init_seed=init_seed)
        N, M = len(self._state_to_idx), len(self._word_to_idx)
        self._exp_probs: dict[str, T] = {
            "xi": torch.zeros(N, N),
            "gamma": torch.zeros(N),
            "gamma_0": torch.zeros(N),
            "gamma_t-1": torch.zeros(N),
            "emission": torch.zeros(N, M),
        }
        self._py = list()
        self._tmp_exp_emission = torch.zeros_like(self._probs["emission"])

    def _forward(self, y_idx: T["T", int]) -> T["T", "N"]:
        alpha: T["T", "N"] = torch.zeros(y_idx.shape[0], len(self._state_to_idx))
        alpha[0, :] = self._probs["init"] + self._probs["emission"][:, y_idx[0]]
        for t in range(1, y_idx.shape[0]):
            b: T["N"] = self._probs["emission"][:, y_idx[t]]
            a: T["N", "N"] = self._probs["transition"]
            # \sum_i\alpha_{t-1}(i)a_{iJ}b_J(y_t)
            alpha[t, :] = b + log_sum(alpha[t - 1, :, None] + a, 0)
        return alpha

    def _backward(self, y_idx: T["T", int]) -> T["T", "N"]:
        beta: T["T", "N"] = torch.zeros(y_idx.shape[0], len(self._state_to_idx))
        for t in range(y_idx.shape[0] - 2, -1, -1):
            b: T["N"] = self._probs["emission"][:, y_idx[t + 1]]
            a: T["N", "N"] = self._probs["transition"]
            # \sum_j\beta_{t+1}(j)a_{Ij}b_j(y_{t+1})
            beta[t, :] = log_sum(a + b[None, :] + beta[t + 1, None, :], 1)
        return beta

    def _e_step(self, y_idx: T["T", int]):
        alpha: T["T", "N"] = self._forward(y_idx)
        beta: T["T", "N"] = self._backward(y_idx)

        py = log_sum(alpha[-1])
        # print(py)
        self._py.append(py)
        assert py.isfinite()
        gamma: T["T", "N"] = alpha + beta - py

        # \alpha_T(I)a_{IJ}
        u = alpha[:-1, :, None] + self._probs["transition"][None, :, :]
        # \b_J(y_{T+1})\beta_{T+1}(J)
        v = beta[1:, :] + self._probs["emission"][:, y_idx[1:]].T
        xi: T["T-1", "N", "N"] = u + v[:, None, :] - py
        return gamma, xi

    def _m_step(self, y_idx: T["T", int], gamma: T["T", "N"], xi: T["T-1", "N", "N"]):
        exp_gamma, exp_xi = torch.exp(gamma), torch.exp(xi)
        del gamma, xi

        exp_gamma_t_1 = exp_gamma[:-1].sum(0)
        self._exp_probs["gamma_0"] += exp_gamma[0]
        self._exp_probs["gamma_t-1"] += exp_gamma_t_1
        self._exp_probs["gamma"] += exp_gamma_t_1 + exp_gamma[-1]
        self._exp_probs["xi"] += exp_xi.sum(0)

        unique_index = dict[int, list[int]]()
        for i, y in enumerate(y_idx):
            indices = unique_index.get(y, None)
            if indices is None:
                indices = unique_index[y] = [i]
            else:
                indices.append(i)
        self._tmp_exp_emission[:, :] = 0
        for uni, idx in unique_index.items():
            self._tmp_exp_emission[:, uni] = exp_gamma[idx].sum(0)
        self._exp_probs["emission"] += self._tmp_exp_emission

    @override
    def _fit_one_sentence(
        self,
        sentence: Collection[str],
        pos_tags: Collection[str] | None = None,
    ) -> Self:
        y_idx = torch.tensor([self._word_to_idx[x] for x in sentence])
        gamma, xi = self._e_step(y_idx)
        self._m_step(y_idx, gamma, xi)
        return self

    @override
    def fit(
        self,
        sentences: Collection[Collection[str]],
        pos_tags_list: Collection[Collection[str]] | None = None,
    ) -> Self:
        super().fit(sentences, pos_tags_list)
        tiny = torch.finfo(self._probs["init"].dtype).tiny

        self._probs["init"] = torch.log(
            self._exp_probs["gamma_0"] / len(sentences) + tiny
        )
        self._probs["transition"] = torch.log(
            self._exp_probs["xi"] / (self._exp_probs["gamma_t-1"][:, None] + tiny)
            + tiny
        )
        self._probs["emission"] = torch.log(
            self._exp_probs["emission"] / (self._exp_probs["gamma"][:, None] + tiny)
            + tiny
        )

        print(f"Log likelyhood: {sum(self._py) / len(self._py)}")
        self._py.clear()
        for v in self._exp_probs.values():
            v[:] = 0
