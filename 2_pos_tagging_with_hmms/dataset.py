from pathlib import Path
import requests
import conllu


# https://universaldependencies.org/u/pos/index.html
ALL_POS_TAGS = [
    "ADJ",  # adjective
    "ADP",  # adposition
    "ADV",  # adverb
    "AUX",  # auxiliary
    "CCONJ",  # coordinating conjunction
    "DET",  # determiner
    "INTJ",  # interjection
    "NOUN",  # noun
    "NUM",  # numeral
    "PART",  # particle
    "PRON",  # pronoun
    "PROPN",  # proper noun
    "PUNCT",  # punctuation
    "SCONJ",  # subordinating conjunction
    "SYM",  # symbol
    "VERB",  # verb
    "X",  # other
    "_",  # NOTE: I'm not sure this occurs in the dataset
]


DATASET_URL = {
    "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/refs/heads/master/de_gsd-ud-train.conllu",
    "dev": "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/refs/heads/master/de_gsd-ud-dev.conllu",
    "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/refs/heads/master/de_gsd-ud-test.conllu",
}


def load_dataset_ud_german_gsd(
    split: str,
    return_labels: bool = False,
    local_cache: str | Path | None = None,
) -> dict[str, object]:
    """
    - Args:
        - split: train, dev, or test.
    - Return:
        - sentences: `list[list[str]]` list of sentences (as list of words).
        - labels: `list[list[str]]` pos tags corresponding to sentences (if `return_labels`)
    """
    if local_cache is not None:
        local_cache = Path(local_cache) / f"{split}.conllu"

    if local_cache is None or not local_cache.exists():
        url = DATASET_URL[split]
        resp = requests.get(url)
        resp.raise_for_status()
        content = resp.content.decode("utf-8")
        if local_cache is not None:
            local_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(local_cache, "w", encoding="utf-8") as file:
                file.write(content)
    else:
        with open(local_cache, encoding="utf-8") as file:
            content = file.read()

    conllu_sentences = conllu.parse(content)
    result_sentences, result_labels = list(), list()
    for sentence in conllu_sentences:
        result_sentences.append(list())
        result_labels.append(list())
        for word in sentence:
            result_sentences[-1].append(word["form"])
            result_labels[-1].append(word["upos"])

    results = {"sentences": result_sentences}
    if return_labels:
        results["labels"] = result_labels
    return results
