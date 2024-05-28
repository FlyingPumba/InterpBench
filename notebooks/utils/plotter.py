from sklearn import metrics
import matplotlib.pyplot as plt

def plot_roc_curve(tprs, fprs, title, labels=None):
    plt.figure()
    lw = 2
    for i, rates in enumerate(zip(tprs, fprs)):
        i += 1
        tpr_list, fpr_list = rates
        try:
            auc = metrics.auc(fpr_list, tpr_list)
        except ValueError:
            # make the list non-increasing by removing elements that are less than the previous
            # get indices where the list is decreasing
            tpr_indices = [idx for idx in range(1, len(tpr_list)) if tpr_list[idx] > tpr_list[idx-1]]
            fpr_indices = [idx for idx in range(1, len(fpr_list)) if fpr_list[idx] > fpr_list[idx-1]]
            all_indices = list(set(tpr_indices + fpr_indices))
            # sort descending so we can remove from the end
            all_indices.sort(reverse=True)
            for idx in all_indices:
                del tpr_list[idx]
                del fpr_list[idx]
            auc = metrics.auc(fpr_list, tpr_list)
            auc = metrics.auc(fpr_list, tpr_list)
        # plot with colorscheme viridis
        cmap = plt.get_cmap('OrRd')
        label = labels[i-1] if labels is not None else f"ROC curve {i}"
        plt.plot(fpr_list, tpr_list, lw=lw, label=f'{label} (area = {auc:.2f})', color=cmap(i*80))
    # auc = metrics.auc(fprs, tprs)

    # plt.plot(fprs, tprs, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.3f)' % auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} ROC curve')
    plt.legend(loc="lower right")
    # increase font size of legend
    plt.legend(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    # increase font size of axis labels
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.show()

def get_roc_curves_from_df(df, title, nodes=True):
    tprs = []
    fprs = []
    labels = []
    for run, entries in df.groupby("run"):
        entries = entries.sort_values("threshold", ascending=False)
        tpr = entries["node_tpr"].tolist().copy() if nodes else entries["edge_tpr"].tolist().copy()
        fpr = entries["node_fpr"].tolist().copy() if nodes else entries["edge_fpr"].tolist().copy()
        if len(tpr) < 2:
            continue
        assert len(tpr) == len(fpr)
        tprs.append(tpr)
        fprs.append(fpr)
        labels.append(run)
    plot_roc_curve(tprs, fprs, title, labels)
