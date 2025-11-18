import torch
from torchtyping import TensorType as T

import abc
from typing import Iterable, Self, override
from copy import deepcopy

from .utils import Counter


class HmmPosTaggerInterface(abc.ABC):
    @abc.abstractmethod
    def export(self) -> dict[str, object]:
        """
        Export the model parameters as dict.
        - Return: dict of model parameters.
        """

    @abc.abstractmethod
    def load(self, data: dict[str, object]) -> Self:
        """
        Load model parameters from dict.
        """

    @abc.abstractmethod
    def tag(self, sentences: Iterable[Iterable[str]]) -> list[list[tuple[str, float]]]:
        """
        Perform pos-tagging on the given sentences.
        - Args:
            - sentences: list of sentences (as list of words) to be tagged.
        - Return:
            - list of pos-tagged sentences with log proberbality.
        """

    @abc.abstractmethod
    def fit(
        self,
        sentences: Iterable[Iterable[str]],
        pos_tags_list: Iterable[Iterable[str]] | None = None,
    ) -> Self:
        """
        Train the model on given sentences.
        - Args:
            - sentences: list of sentences (as list of words).
            - pos_tag_list: pos-tags corresponding to the sentences, only for supervised learning.
        """


class HmmPosTaggerBase(HmmPosTaggerInterface):
    """
    Base class of HMM POS Taggers.

    Children classes must call super().__init__() and implement the following methods:
    - `_viterbi_one_timestep(self, word: str, prev_states_probs: object) -> object`
    - `_trace_best_states(self, states_probs_list: list) -> list[tuple[str, float]]`
    - `_fit_one_sentence(self, sentence: list[str], pos_tags: list[str] | None = None) -> Self`
    """

    def __init__(self, probs: dict[str, object]) -> None:
        super().__init__()
        self._probs = probs

    @override
    def export(self) -> dict[str, object]:
        return deepcopy(self._probs)

    @override
    def load(self, data: dict[str, object]) -> Self:
        assert data.keys() == self._probs.keys()
        data = deepcopy(data)
        self._probs = data
        return self

    @override
    def tag(self, sentences: Iterable[Iterable[str]]) -> list[list[tuple[str, float]]]:
        results = list()
        for words_list in sentences:
            assert len(words_list) > 0
            states_list = list()
            states_probs = None
            for word in words_list:
                states_probs = self._viterbi_one_timestep(word, states_probs)
                states_list.append(states_probs)
            results.append(self._trace_best_states(states_list))
        return results

    @override
    def fit(
        self,
        sentences: Iterable[Iterable[str]],
        pos_tags_list: Iterable[Iterable[str]] | None = None,
    ) -> Self:
        if pos_tags_list is not None:
            for sentence, pos_tags in zip(sentences, pos_tags_list):
                assert len(sentence) == len(pos_tags) > 0
                self._fit_one_sentence(sentence, pos_tags)
        else:
            for sentence in sentences:
                assert len(sentence) > 0
                self._fit_one_sentence(sentence)
        return self

    @abc.abstractmethod
    def _viterbi_one_timestep(
        self, word: str, prev_states_probs: object | None
    ) -> object:
        """
        Perform Viterbi algorithm to predict pos-tag given word and previous states.
        - Args:
            - word: to be tagged.
            - prev_states_probs: previous states and probs.
        - Return:
            - states that can be fed into next time step.
        """

    @abc.abstractmethod
    def _trace_best_states(self, states_probs_list: list) -> list[tuple[str, float]]:
        """
        Find the best path at the terminal of Viterbi algorithm.
        """

    @abc.abstractmethod
    def _fit_one_sentence(
        self,
        sentence: Iterable[str],
        pos_tags: Iterable[str] | None = None,
    ) -> Self:
        """
        Train the model on the given sentence.
        - Args:
            - sentence: list of words.
            - pos_tag: pos-tags corresponding to the sentence, only for supervised learning.
        """


class HmmPosTaggerTensorBase(HmmPosTaggerBase):
    """
    Base class of HMM POS Taggers.

    Use torch.Tensor as the data structure of parameters, with CUDA support.

    Children classes must call super().__init__() and implement the following methods:
    - `_fit_one_sentence(self, sentence: list[str], pos_tags: list[str] | None = None) -> Self`
    """

    def __init__(
        self,
        possible_states: Iterable[str],
        possible_words: Iterable[str],
        probs_init_seed: int | None = None,
    ) -> None:
        """
        - probs_init_seed
            - None: initial probs will be log(0) aka. -inf
            - int number: uniform random with probs_init_seed as generator seed
        """
        states_name, state_to_idx, word_to_idx = list(), dict(), dict()
        for state in possible_states:
            if state in state_to_idx:
                continue
            state_to_idx[state] = len(state_to_idx)
            states_name.append(state)
        for word in possible_words:
            word_to_idx[word] = word_to_idx.get(word, len(word_to_idx))

        self._states_name = tuple(states_name)
        self._state_to_idx = state_to_idx
        self._word_to_idx = word_to_idx
        N, M = len(self._state_to_idx), len(self._word_to_idx)

        neginf = torch.tensor(-torch.inf)
        if probs_init_seed is None:
            probs = {
                "transition": neginf.reshape(1, 1).expand(N, N).clone(),
                "emission": neginf.reshape(1, 1).expand(N, M).clone(),
                "init": neginf.reshape(1).expand(N).clone(),
            }

        else:
            generator = torch.Generator(device=neginf.device)
            generator.manual_seed(probs_init_seed)

            def param(*shape):
                data = torch.rand(*shape, generator=generator)
                return data / data.sum(-1, keepdim=True)

            probs = {
                "transition": param(N, N),
                "emission": param(N, M),
                "init": param(N),
            }

        super().__init__(probs)

    @override
    def export(self) -> dict[str, object]:
        data = {
            "probs": self._probs,
            "vocabulary": {
                "state": self._state_to_idx.keys(),
                "word": self._word_to_idx.keys(),
            },
        }
        return deepcopy(data)

    @override
    def load(self, data: dict[str, object]) -> Self:
        assert data.keys() == {"probs", "vocabulary"}
        assert data["probs"].keys() == self._probs.keys()
        assert data["vocabulary"].keys() == {"state", "word"}
        data = deepcopy(data)
        self._probs = data["probs"]
        self._states_name = tuple(data["vocabulary"]["state"])
        self._state_to_idx = {x: i for i, x in enumerate(self._states_name)}
        self._word_to_idx = {x: i for i, x in enumerate(data["vocabulary"]["word"])}
        return self

    @override
    def _viterbi_one_timestep(
        self, word: str, prev_states_probs: tuple[T["N"], T["N", int]] | None
    ) -> tuple[T["N"], T["N", int]]:
        if prev_states_probs is not None:
            prev_states_probs: T["N"] = prev_states_probs[0]
            N = prev_states_probs.shape[0]
            prev_states_probs = prev_states_probs.reshape(N, 1)
            probs_wo_emission: T[N, N] = prev_states_probs + self._probs["transition"]
        else:
            N = self._probs["init"].shape[0]
            probs_wo_emission: T[1, N] = self._probs["init"].reshape(1, N).expand(N, N)

        word_idx = self._word_to_idx.get(word, None)
        if word_idx is not None:
            emission: T[N] = self._probs["emission"][:, word_idx]
            emission = emission.reshape(1, N)
            probs: T[N, N] = probs_wo_emission + emission
            next_states_probs, prev_states = probs.max(0)
        if word_idx is None or torch.isneginf(next_states_probs).all():
            next_states_probs, prev_states = probs_wo_emission.max(0)

        assert not torch.isneginf(next_states_probs).all()
        return next_states_probs, prev_states

    @override
    def _trace_best_states(
        self,
        states_probs_list: list[tuple[T["N"], T["N", int]]],
    ) -> list[tuple[str, float]]:
        results = list()

        state = states_probs_list[-1][0].argmax()
        for states_probs, prev_states in states_probs_list[::-1]:
            prob = states_probs[state]
            results.append((self._states_name[state], prob))
            state = prev_states[state]

        results.reverse()
        return results


class HmmPosTaggerSupervisedBase(HmmPosTaggerBase):
    """
    Base class of supervised HMM POS Taggers.

    export / load counters rather than probs for continuous learning

    Children classes must call super().__init__() and implement the following methods:
    - `_viterbi_one_timestep(self, word: str, prev_states_probs: object) -> object`
    - `_trace_best_states(self, states_probs_list: list) -> list[tuple[str, float]]`
    - `_fit_one_sentence(self, sentence: list[str], pos_tags: list[str]) -> Self`
    - `_parameterize(self) -> Self`
    """

    TAG_START = "__START__"

    def __init__(self, probs: dict[str, object], counters: dict[str, Counter]) -> None:
        super().__init__(probs)
        self._counters = counters

    @override
    def export(self) -> dict[str, object]:
        return deepcopy(self._counters)

    @override
    def load(self, data: dict[str, object]) -> Self:
        assert data.keys() == self._counters.keys()
        data = deepcopy(data)
        self._counters = data
        return self._parameterize()

    @override
    def fit(
        self,
        sentences: Iterable[Iterable[str]],
        pos_tags_list: Iterable[Iterable[str]] | None = None,
    ) -> Self:
        super().fit(sentences, pos_tags_list)
        return self._parameterize()

    @abc.abstractmethod
    def _parameterize(self) -> Self:
        """
        Calculate log probabilities from counters.
        """
