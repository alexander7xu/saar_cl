import nltk
import nltk.parse.util

from collections import defaultdict


def load_sentences(
    path: str = "nltk:/grammars/large_grammars/atis_sentences.txt",
) -> list[list[str]]:
    # nltk.download("large_grammars")
    raw = nltk.data.load(path)
    extract = nltk.parse.util.extract_test_sentences(raw)
    # Ignore the counts here. They are useless because they are based on another grammar.
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
    1. Convert rules like `A -> B word` into `A -> B __word__` and `__word__ -> word`

    2. Convert multi-branch tree into binary tree:

        Given `A -> (B,C,D,E)`, replace it by:
        `A -> (__A__B__C__D__, E) ; __A__B__C__D__ -> (__A__B__C__, D) ; __A__B__C__ -> (B, C)`

    3. Remove unary lexical productions:

        Given `A -> B -> C -> D -> word`, replace it by `A -> word`.

        Given `A -> B -> C -> D E`, replace it by `A -> D E`
    """

    assert not grammar.is_chomsky_normal_form()
    # return grammar.chomsky_normal_form()

    start = grammar.start()
    lhs_to_productions = defaultdict[nltk.Nonterminal, set[nltk.Production]](set)
    unary_lexical = set[nltk.Production]()

    for prod in grammar.productions():
        assert isinstance(prod, nltk.Production)
        rhs = list[nltk.Nonterminal | str](prod.rhs())

        # Handle unary productions
        if len(rhs) == 1:
            rhs = rhs[0]
            if isinstance(rhs, str):  # A -> word, valid
                lhs_to_productions[prod.lhs()].add(prod)
            else:  # A -> NT, invalid
                assert isinstance(rhs, nltk.Nonterminal)
                unary_lexical.add(prod)
            continue
        assert len(rhs) >= 2

        """
        1. Convert rules like `A -> B word` into `A -> B __word__` and `__word__ -> word`
        """
        all_rhs = list[nltk.Nonterminal]()
        for item in rhs:
            if isinstance(item, str):  # A -> ... word ...
                vitrual_nt = nltk.Nonterminal(f"__{item}__")
                # Add "__word__ -> word" as valid rule
                lhs_to_productions[vitrual_nt].add(nltk.Production(vitrual_nt, item))
                # Replace original A -> ... word ... by A -> ... __word__ ...
                item = vitrual_nt
            assert isinstance(item, nltk.Nonterminal)
            all_rhs.append(item)

        """
        2. Convert multi-branch tree into binary tree
        Given `A -> (B,C,D,E)`, replace it by:
        `A -> (__A__B__C__D__, E) ; __A__B__C__D__ -> (__A__B__C__, D) ; __A__B__C__ -> (B, C)`
        """
        lhs = prod.lhs()
        assert isinstance(lhs, nltk.Nonterminal)
        all_names: list[str] = list(map(lambda x: x.symbol(), [lhs] + all_rhs))

        while len(all_rhs) > 2:
            # pop the rightmost one on the right-hand-side, this is a "real" node
            real_right = all_rhs.pop()
            all_names.pop()  # Also remove its name

            # create a vitrual node for the vitrual subtree
            virtual_left = nltk.Nonterminal(f"__{'__'.join(all_names)}__")
            lhs_to_productions[lhs].add(
                nltk.Production(lhs, (virtual_left, real_right))
            )

            # process the new vitrual node in next loop
            lhs = virtual_left
        lhs_to_productions[lhs].add(nltk.Production(lhs, all_rhs))  # Last 2 "real" node

    """
    3. Remove unary lexical productions:
    Given `A -> B -> C -> D -> word`, replace it by `A -> word`.
    Given `A -> B -> C -> D E`, replace it by `A -> D E`
    """
    # All valid productions
    productions: set[nltk.Production] = set.union(*lhs_to_productions.values())

    # Add all invalid productions into the map, so that it can map to all possible productions
    for prod in unary_lexical:
        lhs_to_productions[prod.lhs()].add(prod)

    while len(unary_lexical) > 0:
        rule = unary_lexical.pop()  # A -> B
        lhs = rule.lhs()

        # B -> X ; X could be C or word or (C, D)
        for old_rule in lhs_to_productions[rule.rhs()[0]]:
            # Given A -> B and B -> X, reduced to A -> X
            rhs = old_rule.rhs()
            new_rule = nltk.Production(lhs, rhs)  # A -> X

            # A -> (C, D) or A -> word, valid
            if len(rhs) != 1 or isinstance(rhs[0], str):
                productions.add(new_rule)
            # A -> C, still invalid
            else:
                unary_lexical.add(new_rule)

    res = nltk.CFG(start, productions)
    assert res.is_chomsky_normal_form()
    return res
