import nltk

from itertools import product

from chart import ChartBase, CountingChart, BackpointerChart, ProbBackpointerChart


def grammar_to_dict(
    grammar: nltk.CFG,
) -> tuple[dict[tuple[str, str], set[str]], dict[str, set[str]]]:
    """
    Convert grammar into dict.

    Return:
    - inv_nonterminal_production: dict that map (left, right) => {set of nonterminal parents}
    - inv_terminal_production: dict that map terminal => {set of nonterminal Parents}
    """

    inv_nonterminal_production = dict[tuple[str, str], set[str]]()
    inv_terminal_production = dict[str, set[str]]()

    for item in grammar.productions():
        assert isinstance(item, nltk.Production)
        parent, children = item.lhs(), item.rhs()
        assert isinstance(parent, nltk.Nonterminal)
        assert 1 <= len(children) <= 2
        if len(children) == 2:  # Nonterminal
            assert isinstance(children[0], nltk.Nonterminal)
            assert isinstance(children[1], nltk.Nonterminal)
            children = (children[0].symbol(), children[1].symbol())
            nt_children = inv_nonterminal_production.get(children, None)
            if nt_children is None:
                nt_children = inv_nonterminal_production[children] = set[str]()
            nt_children.add(parent.symbol())
        else:  # Terminal
            assert isinstance(children[0], str)
            t_children = inv_terminal_production.get(children[0], None)
            if t_children is None:
                t_children = inv_terminal_production[children[0]] = set[str]()
            t_children.add(parent.symbol())
    return inv_nonterminal_production, inv_terminal_production


class CkyParser:
    """
    CKY parser class with different detailed algorithms.

    - `parse`: standard CKY algorithm to find out all parse trees.
    - `count`: Find out counts of all parse trees without building them.
    - `viterbi`: Viterbi-CKY algorithm to find out the parse tree with max probability.
    """

    def __init__(self, grammar: nltk.CFG) -> None:
        (
            self._inv_nonterminal_production,  # (left, right) => {NT for all NT -> left right}
            self._inv_terminal_production,  # word => {NT for all NT -> word}
        ) = grammar_to_dict(grammar)
        self._start_symbol: str = grammar.start().symbol()

    def _cky_parse_one_sentence(self, sentence: list[str], chart: ChartBase) -> None:
        """
        Main logic of CKY algorithm.
        """
        # Terminal records are initialized by the Chart class.
        assert isinstance(chart, ChartBase)
        # for each width b from 2 to n:
        for length in range(1, len(sentence)):
            # for each start position i from 1 to n-b+1:
            for left in range(0, len(sentence) - length):
                right = left + length
                # for each left width k from 1 to b-1:
                for mid in range(left, right):
                    self._reduce(chart, left, right, mid)

    def _reduce(self, chart: ChartBase, left: int, right: int, mid: int) -> None:
        """
        Try to reduce (left, mid) (mid+1, right) -> (left, right) with all possible rules
        """
        # for each B in Ch(i,i+k) and C in Ch(i+k,i+b):
        for left_symbol, right_symbol in product(
            chart.get(left, mid).keys(), chart.get(mid + 1, right).keys()
        ):
            # for each production rule A -> B C:
            for nt in self._inv_nonterminal_production.get(
                (left_symbol, right_symbol), ()
            ):
                chart.reduce(left, right, mid, left_symbol, right_symbol, nt)

    def parse(self, sentence: list[str]) -> list[nltk.Tree]:
        """
        Return all parse trees of the given sentence.
        """
        chart = BackpointerChart(sentence, self._inv_terminal_production)
        self._cky_parse_one_sentence(sentence, chart)
        trees = chart.output(self._start_symbol)
        return trees

    def count(self, sentence: list[str]) -> int:
        """
        Return the **count** of all parse trees of the given sentence.
        """
        chart = CountingChart(sentence, self._inv_terminal_production)
        self._cky_parse_one_sentence(sentence, chart)
        cnt = chart.output(self._start_symbol)
        return cnt

    def recognize(self, sentence: list[str]) -> bool:
        """
        Return whether the given sentence is grammatical.
        """
        # Just used a counter, because they are same in terms of time complexity.
        return self.count(sentence) > 0

    def viterbi(
        self,
        sentence: list[str],
        terminal_production_probs: dict[tuple[str, str], float],
        nonterminal_production_probs: dict[tuple[str, str, str], float],
    ) -> nltk.ProbabilisticTree | None:
        """
        Return the parse tree with max probability of the given sentence.
        Return None if no such a tree.

        :param terminal_production_probs: dict that map (NT, T) => probs
        :param nonterminal_production_probs: dict that map (Left, Right, Parent) => probs
        """
        chart = ProbBackpointerChart(
            sentence,
            self._inv_terminal_production,
            leaf_probs=terminal_production_probs,
            nonleaf_probs=nonterminal_production_probs,
        )
        self._cky_parse_one_sentence(sentence, chart)
        tree = chart.output(self._start_symbol)
        return tree
