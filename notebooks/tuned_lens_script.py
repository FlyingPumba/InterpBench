# %%
import pickle
import torch
from transformer_lens import HookedTransformerConfig, HookedTransformer
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.utils.iit import make_iit_hl_model
import circuits_benchmark.utils.iit.correspondence as correspondence
import iit.model_pairs as mp
import matplotlib.pyplot as plt
import pandas as pd
from iit.utils.node_picker import get_all_individual_nodes_in_circuit
from circuits_benchmark.utils.iit.dataset import get_unique_data
import plotly.express as px
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from argparse import ArgumentParser
from utils.tuned_lens import do_tuned_lens, TunedLensConfig
import os

parser = ArgumentParser()

parser.add_argument(
    "-i",
    "--task_idx",
    dest="task_idx",
    type=int,
    default=3,
    choices=[11, 13, 18, 19, 20, 21, 26, 29, 3, 33, 34, 35, 36, 37, 4, 8],
)

args = parser.parse_args()
task_idx = str(args.task_idx)
out_dir_name = f"./tuned_lens_results/{task_idx}"
os.makedirs(out_dir_name, exist_ok=True)
# %%
task = get_cases(indices=[task_idx])[0]
dir_name = f"../InterpBench/{task_idx}"
cfg_dict = pickle.load(open(f"{dir_name}/ll_model_cfg.pkl", "rb"))
cfg = HookedTransformerConfig.from_dict(cfg_dict)
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer(cfg)
weights = torch.load(f"{dir_name}/ll_model.pth", map_location=cfg.device)
model.load_state_dict(weights)
# turn off grads
model.eval()
model.requires_grad_(False)


# load high level model
def make_model_pair(benchmark_case):
    hl_model = benchmark_case.build_transformer_lens_model()
    hl_model = make_iit_hl_model(hl_model, eval_mode=True)
    tracr_output = benchmark_case.get_tracr_output()
    hl_ll_corr = correspondence.TracrCorrespondence.from_output(
        case=benchmark_case, tracr_output=tracr_output
    )
    model_pair = mp.StrictIITModelPair(hl_model, model, hl_ll_corr)
    return model_pair


# %%
max_len = 1000
model_pair = make_model_pair(task)
unique_test_data = get_unique_data(task, max_len=max_len)


def collate_fn(batch):
    encoded_x = model_pair.hl_model.map_tracr_input_to_tl_input(list(zip(*batch))[0])
    return encoded_x


loader = torch.utils.data.DataLoader(
    unique_test_data,
    batch_size=256,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
)

# %%

tuned_lens_results, labels, train_metrics = do_tuned_lens(
    model_pair,
    loader,
    TunedLensConfig(
        num_epochs=50,
        lr=1e-2,
    ),
    return_train_metrics=True,
)

# %%

metric_df = pd.DataFrame(train_metrics)
metric_df.plot()
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"{out_dir_name}/train_metrics.pdf", bbox_inches="tight", dpi=600)

# %%

nodes = get_all_individual_nodes_in_circuit(model, model_pair.corr)


def convert_ll_node_to_str(node: mp.LLNode):
    if "attn" in node.name:
        block = node.name.split(".")[1]
        head = node.index.as_index[2]
        return f"L{block}H{head}"
    if "mlp" in node.name:
        block = node.name.split(".")[1]
        return f"{block}_mlp_out"


nodes = [convert_ll_node_to_str(node) for node in nodes]
nodes


# %%
def plot_pearson(key, tuned_lens_results, labels, model_pair, nodes):
    in_circuit_str = "in circuit" if key in nodes else "not in circuit"
    fig = go.Figure()

    for i in range(tuned_lens_results[key].shape[1]):
        y = labels[:, i].squeeze().detach().cpu().numpy()
        x = tuned_lens_results[key][:, i].detach().cpu().numpy().squeeze()
        if model_pair.hl_model.is_categorical():
            y = y.argmax(axis=-1)
            x = x.argmax(axis=-1)

        pearson_corr = stats.pearsonr(x, y)
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers", name=f"pos {i}, corr: {pearson_corr[0]:.2f}"
            )
        )

    fig.update_layout(
        title=f"Logit Lens Results for {key} ({in_circuit_str})",
        yaxis_title="True Logits",
        xaxis_title="Logit Lens Results",
    )
    fig.write_image(f"{out_dir_name}/{key}_tuned_lens_plot.png")
    # fig.show()


# %%


def plot_combined_pearson(tuned_lens_results, labels, model_pair, nodes, abs_corr=True):
    pearson_corrs = {}
    for k in tuned_lens_results.keys():
        x = tuned_lens_results[k].detach().cpu().numpy().squeeze()
        y = labels.detach().cpu().numpy().squeeze()
        if model_pair.hl_model.is_categorical():
            y = y.argmax(axis=-1)
            x = x.argmax(axis=-1)
        for i in range(x.shape[1]):
            pearson_corr = stats.pearsonr(x[:, i], y[:, i])
            k_ = k + "(IC)" if k in nodes else k
            if k_ not in pearson_corrs:
                pearson_corrs[k_] = {}
            if np.isnan(pearson_corr.correlation):
                pearson_corrs[k_][str(i)] = 0
            elif abs_corr:
                pearson_corrs[k_][str(i)] = abs(pearson_corr.correlation)
            else:
                pearson_corrs[k_][str(i)] = pearson_corr.correlation

    pearson_corrs = pd.DataFrame(pearson_corrs)
    fig = px.imshow(
        pearson_corrs,
        # set color map
        color_continuous_scale="Viridis",
        # set axis labels
        labels=dict(y="Position", x="Layer/Head", color="Pearson Correlation"),
    )
    # remove margins around plot
    fig.update_layout(margin=dict(l=0, r=0, t=1, b=0))

    # make xticks bigger
    fig.update_xaxes(tickfont=dict(size=15))
    fig.write_image(f"{out_dir_name}/combined_tuned_lens_plot.png")

    # fig.show()

# %%

plot_combined_pearson(tuned_lens_results, labels, model_pair, nodes, abs_corr=False)
plot_combined_pearson(tuned_lens_results, labels, model_pair, nodes, abs_corr=True)
for key in tuned_lens_results.keys():
    plot_pearson(key, tuned_lens_results, labels, model_pair, nodes)

