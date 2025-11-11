import datasets
import requests

from typing import Hashable, Iterable
from pathlib import Path
import string


def load_dataset(
    name: str, cache_dir: str | Path | None = None
) -> str | datasets.Dataset:
    """
    Load different kind (e.g. from URL on HuggingFace) of dataset by name
    - Arguments:
        - name: name of dataset
        - cache_root: path to save. Could be None, then:
            - HF datasets would be save to their default cache folder.
            - URL datasets would be downloaded without saving in local.
    """

    NAME_TO_URL = {
        "King James Bible": "https://raw.githubusercontent.com/coli-saar/cl/main/a1/kingjamesbible_tokenized.txt",
        "The Jungle Book": "https://raw.githubusercontent.com/coli-saar/cl/main/a1/junglebook.txt",
    }
    NAME_TO_HF = {"SETIMES": ["community-datasets/setimes", "bg-tr"]}

    if name in NAME_TO_URL:
        path = cache_dir and (Path(cache_dir) / f"{name}.txt")
        if path is not None and path.exists():
            with open(path, encoding="utf-8") as file:
                return file.read()
        content = requests.get(NAME_TO_URL[name]).content.decode("utf-8")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                file.write(content)
        return content
    elif name in NAME_TO_HF:
        return datasets.load_dataset(*NAME_TO_HF[name], cache_dir=cache_dir)
    else:
        raise NotImplementedError(name)


def simple_tokenize(text: str, remove_symbols: bool = False) -> list[str]:
    """
    - Actions:
        - Convert to lowercase
        - Replace invisible chars with whitespace
        - Splitting at whitespace(s)
        - (Optional) remove symbols
    """
    set_to_remove = set(string.whitespace)
    if remove_symbols:
        set_to_remove |= set(string.punctuation)
    text = "".join([" " if c in set_to_remove else c for c in text])
    result = text.lower().split(" ")
    result = [x for x in result if len(x) > 0]
    return result


def counter(items: Iterable[Hashable]) -> list[tuple[Hashable, int]]:
    """
    - Return:
        - unique items
        - corresponding counts
    """
    obj_to_cnt: dict[str, int] = dict()
    for obj in items:
        obj_to_cnt[obj] = obj_to_cnt.get(obj, 0) + 1

    obj_cnt = sorted(obj_to_cnt.items(), key=lambda x: -x[-1])
    return obj_cnt
