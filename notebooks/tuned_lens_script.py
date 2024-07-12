# %%
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import torch

import iit.model_pairs as mp
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import (
    get_ll_model_loader,
)
from interp_utils.lens import TunedLensConfig
from interp_utils.lens.plot_utils import (
    get_formatted_node_names_in_circuit,
    plot_combined_pearson,
    plot_pearson,
)
from interp_utils.lens.tuned_lens import do_tuned_lens

parser = ArgumentParser()
parser.add_argument("--task", type=str, default="3")
parser.add_argument("--max_len", type=int, default=1000)
task_idx = parser.parse_args().task
max_len = parser.parse_args().max_len
out_dir = f"./interp_results/{task_idx}/tuned_lens"
os.makedirs(out_dir, exist_ok=True)

task: BenchmarkCase = get_cases(indices=[task_idx])[0]

ll_model_loader = get_ll_model_loader(task, interp_bench=True)
hl_ll_corr, model = ll_model_loader.load_ll_model_and_correspondence(
    device="cuda" if torch.cuda.is_available() else "cpu"
)
# turn off grads
model.eval()
model.requires_grad_(False)

hl_model = task.get_hl_model()
model_pair = mp.StrictIITModelPair(hl_model, model, hl_ll_corr)

# %%
unique_test_data = task.get_clean_data(max_samples=max_len, unique_data=True)

loader = torch.utils.data.DataLoader(
    unique_test_data, batch_size=256, shuffle=False, drop_last=False
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
plt.savefig(f"{out_dir}/tuned_lens_loss.pdf", bbox_inches="tight")

# %%

nodes = get_formatted_node_names_in_circuit(model_pair=model_pair)
nodes

# %%

k = "L1H0"
plot_pearson(
    key=k,
    lens_results=tuned_lens_results,
    labels=labels,
    in_circuit=k in nodes,
    is_categorical=task.is_categorical(),
    tuned_lens=True,
    case_name=task.get_name(),
    show=True,
)

# %%
plot_combined_pearson(
    lens_results=tuned_lens_results,
    labels=labels,
    nodes_in_circuit=nodes,
    is_categorical=task.is_categorical(),
    tuned_lens=True,
    case_name=task.get_name(),
)
