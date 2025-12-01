import nltk
import nltk.parse.util


def load_sentences(
    path: str = "nltk:/grammars/large_grammars/atis_sentences.txt",
) -> list[list[str]]:
    # nltk.download("large_grammars")
    raw = nltk.data.load(path)
    extract = nltk.parse.util.extract_test_sentences(raw)
    sentences, _ = map(list, zip(*extract))
    return sentences


def load_grammar(path: str = "nltk:/grammars/large_grammars/atis.cfg") -> nltk.CFG:
    grammar = nltk.data.load(path)
    assert isinstance(grammar, nltk.CFG)
    grammar = convert_cfg_to_chomsky_normal_form(grammar, True)
    return grammar


def convert_cfg_to_chomsky_normal_form(
    grammar: nltk.CFG, remove_unitary_rules: bool
) -> nltk.CFG:
    """
    Actions:
    1. Convert multi-branch tree into binary tree:

        Given `A -> (B,C,D,E)`, yield:
        `A -> (__B__C__D__, E) ; __B__C__D__ -> (__B__C__, D) ; __B__C__ -> (B, C)`

    2. (Optional) Remove unary lexical productions use nltk.
    """

    if grammar.is_chomsky_normal_form():
        return grammar
    # return grammar.chomsky_normal_form()

    start = grammar.start()
    productions = set[nltk.Production]()

    for prod in grammar.productions():
        assert isinstance(prod, nltk.Production)
        rhs, lhs = prod.rhs(), prod.lhs()
        assert isinstance(lhs, nltk.Nonterminal)

        # Handle unary productions
        if len(rhs) == 1:
            rhs = rhs[0]
            assert type(rhs) in (str, nltk.Nonterminal)
            productions.add(prod)
            continue

        # Convert multi-branch tree into binary tree
        all_rhs = list[nltk.Nonterminal](rhs)
        root = lhs.symbol()
        all_symbols = [root] + list(map(lambda x: x.symbol(), all_rhs))
        while len(all_rhs) > 2:
            child_right = all_rhs.pop()
            all_symbols.pop()
            child_left = nltk.Nonterminal(f"__{'__'.join(all_symbols)}__")
            productions.add(nltk.Production(lhs, (child_left, child_right)))
            lhs = child_left
        productions.add(nltk.Production(lhs, all_rhs))

    res = nltk.CFG(start, productions)

    if remove_unitary_rules:
        res = nltk.CFG.remove_unitary_rules(res)
        assert res.is_chomsky_normal_form()
    return res
