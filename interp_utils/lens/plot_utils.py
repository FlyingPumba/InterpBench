import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from scipy import stats
import os

import iit.model_pairs as mp
from iit.utils.node_picker import get_all_individual_nodes_in_circuit
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
import wandb


def write_all_pearson_plots(
    lens_results: dict[str, torch.Tensor],
    labels: torch.Tensor,
    case: BenchmarkCase,
    model_pair: mp.BaseModelPair,
    tuned_lens: bool,
    per_vocab_lens_results: dict[str, dict[int, torch.Tensor]] | None = None,
    per_vocab_labels: torch.Tensor | None = None,
    save_to_wandb: bool = False,
):
    if save_to_wandb:
        lens_str = "tuned_lens" if tuned_lens else "logit_lens"
        wandb.init(
            project="lens_interpretability",
            name=f"{lens_str}_{case.get_case_name()}_pearson",
            config={
                "case_name": case.get_case_name(),
                "tuned_lens": tuned_lens,
            }
        )

    is_categorical = case.is_categorical()
    case_name = case.get_case_name()
    nodes_in_circuit = get_formatted_node_names_in_circuit(model_pair)
    for key in lens_results.keys():
        file = plot_pearson(
            key,
            lens_results,
            labels,
            is_categorical,
            in_circuit=False,
            tuned_lens=tuned_lens,
            case_name=case_name,
        )
        if save_to_wandb:
            wandb.log({f"{key}_pearson": wandb.Image(file)})
        
        if per_vocab_lens_results is None or per_vocab_labels is None:
            continue
        for vocab_idx in per_vocab_lens_results[key].keys():
            file = plot_pearson_at_vocab_idx(
                key,
                vocab_idx,
                per_vocab_lens_results,
                per_vocab_labels,
                in_circuit=False,
                tuned_lens=tuned_lens,
                case_name=case_name,
            )
            if save_to_wandb:
                wandb.log({f"{key}_pearson_vocab_{vocab_idx}": wandb.Image(file)})

    file = plot_combined_pearson(
        lens_results,
        labels,
        is_categorical,
        nodes_in_circuit,
        tuned_lens,
        case_name=case_name,
    )
    if save_to_wandb:
        wandb.log({"combined_pearson": wandb.Image(file)})
        wandb.finish()



def get_formatted_node_names_in_circuit(model_pair: mp.BaseModelPair):
    nodes_in_circuit = get_all_individual_nodes_in_circuit(
        model_pair.ll_model, model_pair.corr
    )
    return [convert_ll_node_to_str(node) for node in nodes_in_circuit]


def convert_ll_node_to_str(node: mp.LLNode):
    if "attn" in node.name:
        block = node.name.split(".")[1]
        head = node.index.as_index[2]
        return f"L{block}H{head}"
    if "mlp" in node.name:
        block = node.name.split(".")[1]
        return f"{block}_mlp_out"

    return node.name


def plot_pearson(
    key: str,
    lens_results: dict[str, torch.Tensor],
    labels: torch.Tensor,
    is_categorical: bool,
    in_circuit: bool,
    tuned_lens: bool,
    case_name: str,
    out_dir: str = "./interp_results/",
    show=False,
) -> str:
    in_circuit_str = "in circuit" if in_circuit else "not in circuit"
    lens_str = "tuned_lens" if tuned_lens else "logit_lens"
    out_dir = f"{out_dir}/{case_name}/{lens_str}/{key}"
    os.makedirs(out_dir, exist_ok=True)
    fig = go.Figure()

    for i in range(lens_results[key].shape[1]):
        y = labels[:, i].squeeze().detach().cpu().numpy()
        x = lens_results[key][:, i].detach().cpu().numpy().squeeze()
        if is_categorical and tuned_lens:
            y = y.argmax(axis=-1)
            x = x.argmax(axis=-1)

        pearson_corr = stats.pearsonr(x, y)
        # normalize x and y by variance and plot
        x = (x - np.mean(x)) / (np.std(x) + 1e-6)
        y = (y - np.mean(y)) / (np.std(y) + 1e-6)
        # assert stats.pearsonr(x, y)[0] - pearson_corr[0] < 1e-6, RuntimeError("Pearson correlation is not preserved after normalization!")
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers", name=f"pos {i}, corr: {pearson_corr[0]:.2f}, p-value: {pearson_corr[1]:.2f}"
            )
        )
        # print(f"p-value for position {i}: {pearson_corr[1]}")

    fig.update_layout(
        title=f"Logit Lens Results for {key} ({in_circuit_str})",
        yaxis_title="True Logits",
        xaxis_title="Logit Lens Results",
    )
    if show:
        fig.show()
    file = f"{out_dir}/pearson.png"
    fig.write_image(file)
    return file


def plot_combined_pearson(
    lens_results: dict[str, torch.Tensor],
    labels: torch.Tensor,
    is_categorical: bool,
    nodes_in_circuit: list[str],
    tuned_lens: bool,
    case_name: str,
    out_dir: str = "./interp_results/",
    abs_corr: bool = True,
    return_data: bool = False,
    show=False,
) -> str:
    lens_str = "tuned_lens" if tuned_lens else "logit_lens"
    out_dir = f"{out_dir}/{case_name}/{lens_str}"
    os.makedirs(out_dir, exist_ok=True)
    pearson_corrs = {}
    p_values = {}
    for k in lens_results.keys():
        x = lens_results[k].detach().cpu().numpy().squeeze()
        y = labels.detach().cpu().numpy().squeeze()
        if is_categorical and tuned_lens:
            y = y.argmax(axis=-1)
            x = x.argmax(axis=-1)
        for i in range(x.shape[1]):
            pearson_corr = stats.pearsonr(x[:, i], y[:, i])
            k_ = k + "(IC)" if k in nodes_in_circuit else k
            if k_ not in pearson_corrs:
                pearson_corrs[k_] = {}
                p_values[k_] = {}
            if np.isnan(pearson_corr.correlation):
                pearson_corrs[k_][str(i)] = 0
                p_values[k_][str(i)] = 1
            elif abs_corr:
                pearson_corrs[k_][str(i)] = abs(pearson_corr.correlation)
                p_values[k_][str(i)] = pearson_corr.pvalue
            else:
                pearson_corrs[k_][str(i)] = pearson_corr.correlation
                p_values[k_][str(i)] = pearson_corr.pvalue

    pearson_corrs = pd.DataFrame(pearson_corrs)
    p_values = pd.DataFrame(p_values)
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

    if show:
        fig.show()
    file = f"{out_dir}/combined_pearson.png"
    fig.write_image(file)
    pearson_file = f"{out_dir}/combined_pearson.csv"
    p_values_file = f"{out_dir}/combined_p_values.csv"
    pearson_corrs.to_csv(pearson_file)
    p_values.to_csv(p_values_file)
    
    if return_data:
        return file, pearson_file, p_values_file
    return file


def plot_pearson_at_vocab_idx(
    key: str,
    vocab_idx: int,
    lens_results_per_vocab: dict[str, torch.Tensor],
    per_vocab_labels: torch.Tensor,
    in_circuit: bool,
    tuned_lens: bool,
    case_name: str,
    out_dir: str = "./interp_results/",
    show=False,
) -> str:
    lens_str = "tuned_lens" if tuned_lens else "logit_lens"
    in_circuit_str = "in circuit" if in_circuit else "not in circuit"
    out_dir = f"{out_dir}/{case_name}/{lens_str}/{key}"
    os.makedirs(out_dir, exist_ok=True)
    fig = go.Figure()

    for i in range(lens_results_per_vocab[key][vocab_idx].shape[1]):
        y = per_vocab_labels[vocab_idx][:, i].squeeze().detach().cpu().numpy()
        x = lens_results_per_vocab[key][vocab_idx][:, i].detach().cpu().numpy()
        pearson_corr = stats.pearsonr(x, y)
        x = (x - np.mean(x)) / (np.std(x) + 1e-6)
        y = (y - np.mean(y)) / (np.std(y) + 1e-6)
        # assert stats.pearsonr(x, y)[0] - pearson_corr[0] < 1e-6, RuntimeError("Pearson correlation is not preserved after normalization!")
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
    if show:
        fig.show()
    file = f"{out_dir}/pearson_at_{vocab_idx}.png"
    fig.write_image(file)
    return file
