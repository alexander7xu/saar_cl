import nltk


def tree_to_spans(tree: nltk.Tree) -> list[tuple[str, int, int]]:
    r"""
    Convert a tree into span with format (label, left, right)

    The tree can be multi-branch. Words aka. terminals in the tree would be ignored.

    Example:
    ```
    given a tree:
            A
           / \
          B   C
         / \  |
        D   E |
        |   | |
        w1 w2 w3

    produce:
        (D, 0, 0)
        (E, 1, 1)
        (B, 0, 1)
        (C, 2, 2)
        (A, 0, 2)
    ```
    """

    results = list[tuple[str, int, int]]()

    def recur(root: nltk.Tree | str, index: int) -> int:
        if isinstance(root, str):  # Ignore words
            return index
        assert isinstance(root, nltk.Tree)
        assert len(root) > 0
        end_index = index - 1  # it will >= index because have at least one subtree
        for subtree in root:
            end_index = recur(subtree, end_index + 1)
        results.append((root.label(), index, end_index))
        return end_index

    recur(tree, 0)
    return results


def tree_f1_score(tree_gt: nltk.Tree, tree_pred: nltk.Tree, unlabeled: bool) -> float:
    """
    Calculate the f1 score of two nltk.Tree.
    """

    def _spans_same_rate(spans: list[tuple], same: set[tuple]) -> float:
        # calculate the rate that items in spans occurs in same
        correct = 0
        for item in spans:
            correct += int(item in same)
        return correct / len(spans)

    spans_gt = tree_to_spans(tree_gt)
    spans_pred = tree_to_spans(tree_pred)

    if unlabeled:  # Remove labels
        spans_gt = [x[1:] for x in spans_gt]
        spans_pred = [x[1:] for x in spans_pred]

    same = set(spans_gt) & set(spans_pred)
    precision = _spans_same_rate(spans_pred, same)
    recall = _spans_same_rate(spans_gt, same)

    f1 = 2 * precision * recall / (precision + recall)
    return f1
