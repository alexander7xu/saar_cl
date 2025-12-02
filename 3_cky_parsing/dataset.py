import nltk
import nltk.parse.util

from collections import defaultdict


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
    grammar = convert_cfg_to_chomsky_normal_form(grammar)
    return grammar


def convert_cfg_to_chomsky_normal_form(grammar: nltk.CFG) -> nltk.CFG:
    """
    Actions:
    1. Convert multi-branch tree into binary tree:

        Given `A -> (B,C,D,E)`, yield:
        `A -> (__B__C__D__, E) ; __B__C__D__ -> (__B__C__, D) ; __B__C__ -> (B, C)`

    2. Remove unary lexical productions:

        Given `A -> B -> C -> D -> T`, yield: `A -> T`
    """

    if grammar.is_chomsky_normal_form():
        return grammar
    # return grammar.chomsky_normal_form()

    start = grammar.start()
    lhs_to_productions = defaultdict[nltk.Nonterminal, set[nltk.Production]](set)
    unary_lexical = set[nltk.Production]()

    for prod in grammar.productions():
        assert isinstance(prod, nltk.Production)
        rhs, lhs = prod.rhs(), prod.lhs()
        assert isinstance(lhs, nltk.Nonterminal)

        # Handle unary productions
        if len(rhs) == 1:
            rhs = rhs[0]
            if isinstance(rhs, str):  # A -> T, valid
                lhs_to_productions[prod.lhs()].add(prod)
            else:  # A -> NT, invalid
                assert isinstance(rhs, nltk.Nonterminal)
                unary_lexical.add(prod)
            continue

        # Convert multi-branch tree into binary tree
        all_rhs = list[nltk.Nonterminal](rhs)
        root = lhs.symbol()
        all_symbols = [root] + list(map(lambda x: x.symbol(), all_rhs))
        while len(all_rhs) > 2:
            child_right = all_rhs.pop()
            all_symbols.pop()
            child_left = nltk.Nonterminal(f"__{'__'.join(all_symbols)}__")
            lhs_to_productions[lhs].add(nltk.Production(lhs, (child_left, child_right)))
            lhs = child_left
        lhs_to_productions[lhs].add(nltk.Production(lhs, all_rhs))

    # Remove unary lexical productions
    productions = set.union(*lhs_to_productions.values())  # All valid productions
    for prod in unary_lexical:  # All invalid productions, aka. A -> NT
        lhs_to_productions[prod.lhs()].add(prod)

    while len(unary_lexical) > 0:
        rule = unary_lexical.pop()  # A -> NT
        lhs = rule.lhs()
        for old_rule in lhs_to_productions[rule.rhs()[0]]:
            # Given A -> NT -> X, reduced to A -> X
            rhs = old_rule.rhs()
            new_rule = nltk.Production(lhs, rhs)

            # A -> (C,D) or A -> T, valid
            if len(rhs) != 1 or isinstance(rhs[0], str):
                productions.add(new_rule)
            # A -> NT, still invalid
            else:
                unary_lexical.add(new_rule)

    res = nltk.CFG(start, productions)
    assert res.is_chomsky_normal_form()
    return res
