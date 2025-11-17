import argparse
import time

from hmm_pos_tagger import HmmPosTaggerMultithreadingWrap, HmmPosTaggerBase
from hmm_pos_tagger import HmmPosTagger, HmmNgramPosTagger, HmmPosTaggerDeprecated
from dataset import load_dataset_ud_german_gsd, ALL_POS_TAGS


def accuracy(
    predicts: list[list[tuple[str, float]]],
    targets: list[list[str]],
) -> float:
    predicts = [x[0] for xx in predicts for x in xx]
    targets = [x for xx in targets for x in xx]
    assert len(predicts) == len(targets)

    same_cnt = 0
    for pred, targ in zip(predicts, targets):
        same_cnt += targ == pred
    return same_cnt / len(predicts)


def test_time(
    base_model: HmmPosTaggerBase,
    train_set: list[list[tuple[str, str]]],
    test_set: list[list[tuple[str, str]]],
    num_threads: int,
) -> None:
    prep = time.time()
    if num_threads > 0:
        model = HmmPosTaggerMultithreadingWrap(base_model, num_threads=num_threads)
    else:
        model = base_model
    prep = time.time() - prep

    train = time.time()
    model.fit(train_set["sentences"], train_set["labels"])
    train = time.time() - train

    tagging = time.time()
    predicts = model.tag(test_set["sentences"])
    tagging = time.time() - tagging

    clean = time.time()
    del model
    clean = time.time() - clean

    acc = accuracy(predicts, test_set["labels"])
    total = prep + train + tagging + clean
    return total, prep, train, tagging, clean, acc


def main(args: argparse.Namespace) -> argparse.Namespace:
    train_dataset = load_dataset_ud_german_gsd(
        "train", return_labels=True, local_cache=args.local_cache
    )
    test_dataset = load_dataset_ud_german_gsd(
        "test", return_labels=True, local_cache=args.local_cache
    )

    words_set = {x for xx in train_dataset["sentences"] for x in xx}
    tags_set = set(ALL_POS_TAGS)

    print(f"|  N |    Total |    Prep. | Training |  Tagging | Cleaning | Accuracy |")
    print(f"|----|----------|----------|----------|----------|----------|----------|")

    for name, cls in zip(("DP", "PT"), [HmmPosTaggerDeprecated, HmmPosTagger]):
        data = test_time(cls(tags_set, words_set), train_dataset, test_dataset, 0)
        print(f"| {name}", end="")
        for item in data[:-1]:
            print(f" | {item*1000:>8.2f}", end="")
        print(f" | {data[-1]:>8.4%} |")

    base_model = HmmNgramPosTagger()
    for num_threads in range(0, args.max_num_threads + 1):
        data = test_time(base_model, train_dataset, test_dataset, num_threads)
        print(f"| {num_threads:>2}", end="")
        for item in data[:-1]:
            print(f" | {item*1000:>8.2f}", end="")
        print(f" | {data[-1]:>8.4%} |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_threads", type=int, default=10)
    parser.add_argument("--local_cache", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
