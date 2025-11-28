import nltk
import nltk.parse.util


def load_sentences(
    path: str = "nltk:/grammars/large_grammars/atis_sentences.txt",
) -> list[list[str]]:
    #nltk.download("large_grammars")
    raw = nltk.data.load(path)
    extract = nltk.parse.util.extract_test_sentences(raw)
    sentences, _ = map(list, zip(*extract))
    return sentences


def load_grammar(path: str = "file://../datasets/atis-grammar-cnf.cfg") -> nltk.CFG:
    grammar = nltk.data.load(path)
    assert isinstance(grammar, nltk.CFG)
    assert grammar.is_chomsky_normal_form()
    return grammar
