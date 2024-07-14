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
    xs, ys, _, _ = make_monotonically_increasing(xs, ys)

    # The slabs of the stairs
    area = np.sum((1 - xs)[1:] * dys)
    return area

def make_monotonically_increasing(xs, ys) -> tuple[list[float], list[float]]:
    for i in range(len(ys)):
        if ys[i] == 'N/A':
            ys[i] = 0
        if xs[i] == 'N/A':
            xs[i] = 0
    # Sort indices based on 'x' and 'y'
    i = np.lexsort(
        (ys, xs)
    )  # lexsort sorts by the last column first, then the second last, etc., i.e we firstly sort by x and then y to break ties

    new_xs = list(np.array(xs, dtype=np.float64)[i])
    new_ys = list(np.array(ys, dtype=np.float64)[i])

    current_idx = 0
    removed_xs = []
    removed_ys = []
    
    while True:
        if current_idx == len(new_ys) - 1:
            break
        if new_ys[current_idx] > new_ys[current_idx + 1]:
            removed_xs.append(new_xs[current_idx+1])
            removed_ys.append(new_ys[current_idx+1])
            # remove the index which is smaller than the current index
            new_xs = list(new_xs[:current_idx+1]) + list(new_xs[current_idx + 2:])
            new_ys = list(new_ys[:current_idx+1]) + list(new_ys[current_idx + 2:])
        else:
            current_idx += 1

        
    # to_remove_indices = []
    # removed_xs = []
    # removed_ys = []

    # while True:
    #     monotonic = True
    #     for i in range(1, len(ys)):
    #         if ys[i] < ys[i-1]:
    #             to_remove_indices.append(i)
    #             removed_xs.append(xs[i])
    #             removed_ys.append(ys[i])
    #             monotonic = False
    #     if monotonic:
    #         break
            
    
    # new_xs = []
    # new_ys = []
    # for i in range(len(ys)):
    #     if i not in to_remove_indices:
    #         new_xs.append(xs[i])
    #         new_ys.append(ys[i])
    assert np.all(np.diff(new_xs) >= 0), "not sorted"
    assert np.all(np.diff(new_ys) >= 0), "not sorted"
    return np.array(new_xs), np.array(new_ys), np.array(removed_xs), np.array(removed_ys)

def pessimistic_roc(curves: list[tuple[list[float], list[float]]], labels: list[str], ax: plt.Axes | None = None):
    if ax is None:
        ax = plt.figure(figsize=(10, 6)).add_subplot(111)
    
    for i, (xs, ys) in enumerate(curves):
        xs, ys, removed_xs, removed_ys = make_monotonically_increasing(xs, ys)
        xs = np.concatenate([[0], xs, [1]])
        ys = np.concatenate([[0], ys, [1]])
        auc = pessimistic_auc(xs, ys)

        # Plot the ROC curve with markers at x, y
        ax.step(xs, ys, where='post', alpha=0.7, label=f'Pessimistic ROC {labels[i]} (AUC = {auc:.2f})',
                marker='o', markersize=3, linewidth=1.5)
        ax.fill_between(xs, ys, step='post', alpha=0.1)
        # scatter with the same color as the curve
        ax.scatter(removed_xs, removed_ys, color=ax.get_lines()[-1].get_color(), s=80, alpha=1, marker='x')

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Pessimistic ROC Curves')
    ax.legend(loc='lower right')
    ax.grid()
    
