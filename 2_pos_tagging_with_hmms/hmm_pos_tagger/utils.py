import torch

from typing import TypeVar, Hashable, Self
from collections import UserDict


CountedObjectT = TypeVar("CountedObjectT", bound=Hashable)


class Counter(UserDict[CountedObjectT, int]):
    def inc(self, item: CountedObjectT, cnt: int = 1) -> Self:
        self.data[item] = self.data.get(item, 0) + cnt
        return self

    def merge(self, other: "Counter") -> Self:
        for item, cnt in other.items():
            self.inc(item, cnt)
        return self

    def __getitem__(self, item: CountedObjectT) -> int:
        return self.data[item]

    def __setitem__(self, key, item):
        raise NotImplementedError("Do NOT use .__setitem__(), only use .inc()")


def log_sum(log_probs: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """
    Calculate log(x1 + x2 + x3 + ...)

    **IMPORTANT**: exp(log_probs) must have same order of magnitude.

    ---

    log(a*b) = log_a + log_b

    log(a+b)
        = log( exp(log_a) + exp(log_b) )
        = log( (exp(log_a) + exp(log_b)) / exp(log_b) ) + log_b
        = log( exp(log_a) / exp(log_b) + 1 ) + log_b
        = log( exp(log_a - log_b) + exp(log_b - log_b) ) + log_b

    As a and b generally have same order of magnitude,
    a/b = exp(log_a - log_b) will not be extremely small or large.
    """
    tiny = torch.finfo(log_probs.dtype).tiny
    log_max = log_probs.max()
    result = torch.exp(log_probs - log_max)
    if dim is not None:
        result = result.sum(dim=dim)
    else:
        result = result.sum()
    result = torch.log(result + tiny) + log_max
    return result
