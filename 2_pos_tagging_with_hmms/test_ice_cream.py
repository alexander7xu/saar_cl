import torch

from hmm_pos_tagger import HmmPosTaggerBase, HmmNgramPosTagger, HmmPosTagger


DATA = ["3", "1", "3"]
TRUTH_TAG = ["H", "H", "H"]
TRUTH_PROB = [0.32, 0.0448, 0.012544]


def pa(x) -> torch.Tensor:
    return torch.log(torch.tensor(x))


def test(model: HmmPosTaggerBase) -> None:
    results = model.tag([DATA])
    assert len(results) == 1
    tags = [x[0] for x in results[0]]
    probs = torch.exp(torch.tensor([x[1] for x in results[0]]))

    print(type(model).__name__, "output:", tags, probs.tolist())
    assert tags == TRUTH_TAG, tags
    assert (torch.abs(probs - torch.tensor(TRUTH_PROB)) < 1e-7).all(), probs


def load_HmmNgramPosTagger() -> HmmPosTaggerBase:
    model = HmmNgramPosTagger(1)
    model._probs["transition"].update(
        {
            (model.TAG_START,): {"H": pa(0.8), "C": pa(0.2)},
            ("H",): {"H": pa(0.7), "C": pa(0.3)},
            ("C",): {"H": pa(0.4), "C": pa(0.6)},
        }
    )
    model._probs["emission"].update(
        {
            ("H",): {"1": pa(0.2), "2": pa(0.4), "3": pa(0.4)},
            ("C",): {"1": pa(0.5), "2": pa(0.4), "3": pa(0.1)},
        }
    )
    return model


def load_HmmPosTagger() -> HmmPosTaggerBase:
    states, words = ["H", "C"], ["1", "2", "3"]
    model = HmmPosTagger(states, words)
    idx_h, idx_c = map(lambda x: model._state_to_idx[x], states)
    idx_1, idx_2, idx_3 = map(lambda x: model._word_to_idx[x], words)

    init = torch.empty(2)
    init[idx_h] = pa(0.8)
    init[idx_c] = pa(0.2)

    transition = torch.empty(2, 2)
    transition[idx_h, idx_h] = pa(0.7)
    transition[idx_h, idx_c] = pa(0.3)
    transition[idx_c, idx_h] = pa(0.4)
    transition[idx_c, idx_c] = pa(0.6)

    emission = torch.empty(2, 3)
    emission[idx_h, idx_1] = pa(0.2)
    emission[idx_h, idx_2] = pa(0.4)
    emission[idx_h, idx_3] = pa(0.4)
    emission[idx_c, idx_1] = pa(0.5)
    emission[idx_c, idx_2] = pa(0.4)
    emission[idx_c, idx_3] = pa(0.1)

    model._probs["init"][:] = init
    model._probs["transition"][:] = transition
    model._probs["emission"][:] = emission
    return model


if __name__ == "__main__":
    test(load_HmmNgramPosTagger())
    test(load_HmmPosTagger())
    print("Pass Test")
