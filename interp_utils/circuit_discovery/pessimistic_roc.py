import numpy as np
import matplotlib.pyplot as plt

def pessimistic_auc(xs, ys):
    # Sort indices based on 'x' and 'y'
    i = np.lexsort(
        (ys, xs)
    )  # lexsort sorts by the last column first, then the second last, etc., i.e we firstly sort by x and then y to break ties

    xs = np.array(xs, dtype=np.float64)[i]
    ys = np.array(ys, dtype=np.float64)[i]
    xs = np.concatenate([[0], xs])
    ys = np.concatenate([[0], ys])

    dys = np.diff(ys)
    assert np.all(np.diff(xs) >= 0), "not sorted"
    assert np.all(dys >= 0), "not monotonically increasing"

    # The slabs of the stairs
    area = np.sum((1 - xs)[1:] * dys)
    return area

def pessimistic_roc(curves: list[tuple[list[float], list[float]]], labels: list[str], ax: plt.Axes | None = None):
    if ax is None:
        ax = plt.figure(figsize=(10, 6)).add_subplot(111)
    
    for i, (xs, ys) in enumerate(curves):
        # Sort indices based on 'x' and 'y'
        j = np.lexsort((ys, xs))  # Sort by x and then by y to break ties

        xs = np.array(xs, dtype=np.float64)[j]
        ys = np.array(ys, dtype=np.float64)[j]

        auc = pessimistic_auc(xs, ys)

        # Plot the ROC curve with markers at x, y
        ax.step(xs, ys, where='post', alpha=0.7, label=f'Pessimistic ROC {labels[i]} (AUC = {auc:.2f})',
                marker='o', markersize=3, linewidth=1.5)
        ax.fill_between(xs, ys, step='post', alpha=0.1)

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Pessimistic ROC Curves')
    ax.legend(loc='lower right')
    ax.grid()
    
