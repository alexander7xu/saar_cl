import nltk


def tree_span(tree: nltk.Tree) -> set[tuple[str, int, int]]:
    r"""
    Convert a tree into span with format (label, left, right)

    Example:
    ```
    given a tree:
            A
           / \
          B   C
         / \
        D   E

    produce:
        (D, 0, 0)
        (E, 1, 1)
        (B, 0, 1)
        (C, 2, 2)
        (A, 0, 2)
    ``` 
    """

    results = set[tuple[str, int, int]]()

    def recur(root: nltk.Tree, index: int) -> int:
        assert 1 <= len(root) <= 2
        if len(root) == 1:  # Leaf
            assert type(root[0]) in (nltk.Nonterminal, str)
            results.add((root.label(), index, index))
            return index

        left_tree, right_tree = root
        assert isinstance(left_tree, nltk.Tree)
        assert isinstance(right_tree, nltk.Tree)
        end_index = recur(left_tree, index)
        end_index = recur(right_tree, end_index)
        results.add((root.label(), index, end_index))
        return end_index

    recur(tree, 0)
    return results


def set_f1_score(gt: set, pred: set) -> float:
    ins = gt & pred
    """
    precision = len(ins) / len(gt)
    recall = len(ins) / len(pred)
    f1 = 2 * precision * recall / (precision + recall)

      2 * u/x * u/y     /   (u/x + u/y)
    = 2*u*u / (x*y)     /   (u * (x+y) / (x*y))
    = 2*u / (x*y)       /   ((x+y) / (x*y))
    = 2*u               /   (x+y)
    Calculate with integers for better result precision
    """
    f1 = 2 * len(ins) / (len(gt) + len(pred))
    return f1


def tree_f1_score(tree_gt: nltk.Tree, tree_pred: nltk.Tree, unlabeled: bool) -> float:
    """
    Calculate the f1 score of two nltk.Tree.
    """
    span_gt = tree_span(tree_gt)
    span_pred = tree_span(tree_pred)

    if unlabeled:  # Remove labels
        span_gt = {x[1:] for x in span_gt}
        span_pred = {x[1:] for x in span_pred}

    f1 = set_f1_score(span_gt, span_pred)
    return f1
