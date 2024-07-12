from interp_utils.circuit_discovery.pessimistic_roc import pessimistic_roc


def get_roc_curves_from_df(df, title, nodes=True, algorithm="acdc"):
    if algorithm in ["acdc", "eap", "integrated_grads"]:
        sort_key = "threshold"
        ascending = False
    elif "sp" in algorithm:
        sort_key = "lambda"
        ascending = True
    tprs = []
    fprs = []
    labels = []
    for run, entries in df.groupby("run"):
        entries = entries.sort_values(sort_key, ascending=ascending)
        tpr = (
            entries["node_tpr"].tolist().copy()
            if nodes
            else entries["edge_tpr"].tolist().copy()
        )
        fpr = (
            entries["node_fpr"].tolist().copy()
            if nodes
            else entries["edge_fpr"].tolist().copy()
        )
        if len(tpr) < 2:
            continue
        assert len(tpr) == len(fpr)
        tprs.append(tpr)
        fprs.append(fpr)
        labels.append(run)
    pessimistic_roc(curves=[(fprs, tprs),], labels=labels)