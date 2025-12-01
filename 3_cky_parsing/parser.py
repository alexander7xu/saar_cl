import nltk

from itertools import product

from chart import ChartBase, CountingChart, BackpointerChart, ProbBackpointerChart


def grammar_to_dict(
    grammar: nltk.CFG,
) -> tuple[dict[tuple[str, str], set[str]], dict[str, set[str]]]:
    """
    Convert grammar into dict.

    Return:
    - nonterminal: dict with structure (Left, Right) => {set of Parents}
    - terminal: dict with structure (Terminal) => {set of Nonterminal Parents}
    """

    nonterminal = dict[tuple[str, str], set[str]]()
    terminal = dict[str, set[str]]()

    for item in grammar.productions():
        assert isinstance(item, nltk.Production)
        parent, children = item.lhs(), item.rhs()
        assert isinstance(parent, nltk.Nonterminal)
        assert 1 <= len(children) <= 2
        if len(children) == 2:  # Nonterminal
            assert isinstance(children[0], nltk.Nonterminal)
            assert isinstance(children[1], nltk.Nonterminal)
            children = (children[0].symbol(), children[1].symbol())
            nt_children = nonterminal.get(children, None)
            if nt_children is None:
                nt_children = nonterminal[children] = set()
            nt_children.add(parent.symbol())
        else:  # Terminal
            assert isinstance(children[0], str)
            t_children = terminal.get(children[0], None)
            if t_children is None:
                t_children = terminal[children[0]] = set()
            t_children.add(parent.symbol())
    return nonterminal, terminal


class CkyParser:
    """
    CKY parser class with different detailed algorithms.

    - `parse`: standard CKY algorithm to find out all parse trees.
    - `count`: Find out counts of all parse trees without building them.
    - `viterbi`: Viterbi-CKY algorithm to find out the parse tree with max probability.
    """

    def __init__(self, grammar: nltk.CFG) -> None:
        self._nonterminal, self._terminal = grammar_to_dict(grammar)
        self._start_symbol: str = grammar.start().symbol()

    def _induce(self, chart: ChartBase, left: int, right: int, mid: int) -> None:
        """
        Try to induce (left, mid) (mid+1, right) -> (left, right) with all possible rules
        """
        # for each B in Ch(i,i+k) and C in Ch(i+k,i+b):
        for left_symbol, right_symbol in product(
            chart.get(left, mid).keys(), chart.get(mid + 1, right).keys()
        ):
            # for each production rule A -> B C:
            for nt in self._nonterminal.get((left_symbol, right_symbol), ()):
                chart.add(left, right, mid, left_symbol, right_symbol, nt)

    def _cky_parse_one_sentence(
        self, sentence: list[str], chart: ChartBase
    ) -> ChartBase:
        # Terminal records are initialized by the Chart class.
        assert isinstance(chart, ChartBase)
        # for each width b from 2 to n:
        for length in range(1, len(sentence)):
            # for each start position i from 1 to n-b+1:
            for left in range(0, len(sentence) - length):
                right = left + length
                # for each left width k from 1 to b-1:
                for mid in range(left, right):
                    self._induce(chart, left, right, mid)
        return chart

    def parse(self, sentence: list[str]) -> list[nltk.Tree]:
        """
        Return all parse trees of the given sentence.
        """
        chart = BackpointerChart(sentence, self._terminal)
        record = self._cky_parse_one_sentence(sentence, chart)
        assert isinstance(record, BackpointerChart)
        trees = record.trace(self._start_symbol)
        return trees

    def count(self, sentence: list[str]) -> int:
        """
        Return the **count** of all parse trees of the given sentence.
        """
        record = CountingChart(sentence, self._terminal)
        record = self._cky_parse_one_sentence(sentence, record)
        cnt = record.get(0, len(sentence) - 1)[self._start_symbol]
        return cnt

    def viterbi(
        self,
        sentence: list[str],
        terminal_probs: dict[tuple[str, str], float],
        nonterminal_probs: dict[tuple[str, str, str], float],
    ) -> nltk.ProbabilisticTree | None:
        """
        Return the parse tree with max probability of the given sentence.
        Return None if no such a tree.

        Args:
        - terminal_probs: dict with the strcuture (NT, T) => probs
        - nonterminal_probs: dict with the strcuture (Left, Right, Parent) => probs
        """

        record = ProbBackpointerChart(
            sentence, self._terminal, terminal_probs, nonterminal_probs
        )
        record = self._cky_parse_one_sentence(sentence, record)
        assert isinstance(record, ProbBackpointerChart)
        tree = record.trace(self._start_symbol)
        return tree
