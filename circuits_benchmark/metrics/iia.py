import typing
from functools import partial
from typing import Set, Optional, Literal, Dict

import torch as t
from iit.model_pairs.ll_model import LLModel
from iit.utils import IITDataset
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.metrics.resampling_ablation_loss.intervention import regular_intervention_hook_fn
from circuits_benchmark.utils.circuit.circuit_eval import get_full_circuit
from circuits_benchmark.utils.circuit.circuit_node import CircuitNode
from circuits_benchmark.utils.iit.iit_dataset_batch import IITDatasetBatch

AblationType = Literal["zero", "mean", "resample"]
ablation_types = list(typing.get_args(AblationType))

IIAGranularity = Literal["qkv", "head"]
iia_granularity_options = list(typing.get_args(IIAGranularity))


def regular_intervention_hook_fn(
    activation: Float[Tensor, ""],
    hook: HookPoint,
    corrupted_cache: ActivationCache = None,
    head_index: int = None
):
    """This hook just replaces the output with a corrupted output."""
    if head_index is None:
        return corrupted_cache[hook.name]
    else:
        activation[:, :, head_index] = corrupted_cache[hook.name][:, :, head_index]
        return activation


def evaluate_iia_on_all_ablation_types(
    case: BenchmarkCase,
    base_model: LLModel,
    hypothesis_model: LLModel,
    data: IITDataset,
    iia_granularity: Optional[IIAGranularity] = "head",
    accuracy_atol: Optional[float] = 1e-2):
    iia_evaluation_results = {}

    data_loader = data.make_loader(batch_size=len(data), num_workers=0)
    clean_data, corrupted_data = next(iter(data_loader))
    clean_inputs = clean_data[0]
    corrupted_inputs = corrupted_data[0]

    # run corrupted data on both models
    _, base_model_corrupted_cache = base_model.run_with_cache(corrupted_inputs)
    _, hypothesis_model_corrupted_cache = hypothesis_model.run_with_cache(corrupted_inputs)

    # run clean data on both models
    _, base_model_clean_cache = base_model.run_with_cache(clean_inputs)
    _, hypothesis_model_clean_cache = hypothesis_model.run_with_cache(clean_inputs)

    full_circuit = get_full_circuit(base_model.cfg.n_layers, base_model.cfg.n_heads)
    ll_circuit = case.get_ll_gt_circuit(granularity="acdc_hooks")

    for node in set(full_circuit.nodes):
        node_str = str(node)

        if "mlp_in" in node.name:
            continue

        if iia_granularity != "qkv" and is_qkv_granularity_hook(node.name):
            continue

        iia_evaluation_results[node_str] = {
            "node": node_str,
            "hook_name": node.name,
            "head_index": node.index,
            "in_circuit": node in ll_circuit.nodes,
        }

    for ablation_type in ablation_types:
        results_by_node = evaluate_iia(
            case,
            base_model,
            hypothesis_model,
            clean_data,
            corrupted_data,
            base_model_corrupted_cache,
            hypothesis_model_corrupted_cache,
            base_model_clean_cache,
            hypothesis_model_clean_cache,
            iia_granularity=iia_granularity,
            ablation_type=ablation_type,
            accuracy_atol=accuracy_atol
        )

        for node_str, result_dict in results_by_node.items():
            for key, result in result_dict.items():
                iia_evaluation_results[node_str][f"{key}_{ablation_type}_ablation"] = result

    return iia_evaluation_results


def is_qkv_granularity_hook(hook_name):
    return "_q" in hook_name or "_k" in hook_name or "_v" in hook_name


def evaluate_iia(case: BenchmarkCase,
                 base_model: LLModel,
                 hypothesis_model: LLModel,
                 clean_data: IITDatasetBatch,
                 corrupted_data: IITDatasetBatch,
                 base_model_corrupted_cache: ActivationCache,
                 hypothesis_model_corrupted_cache: ActivationCache,
                 base_model_clean_cache: ActivationCache,
                 hypothesis_model_clean_cache: ActivationCache,
                 iia_granularity: Optional[IIAGranularity] = "head",
                 ablation_type: Optional[AblationType] = "resample",
                 accuracy_atol: Optional[float] = 1e-2) -> Dict[str, Dict[str, float]]:
    """Run Interchange Intervention Accuracy to measure if a hypothesis model has the same circuit as a base model."""
    print(f"Running IIA evaluation for case {case.get_name()} using ablation type \"{ablation_type}\".")
    full_circuit = get_full_circuit(base_model.cfg.n_layers, base_model.cfg.n_heads)

    # evaluate all nodes in the full circuit
    results_by_node = {}
    all_nodes: Set[CircuitNode] = set(full_circuit.nodes)
    for node in tqdm(all_nodes):
        node_str = str(node)
        hook_name = node.name
        head_index = node.index

        if "mlp_in" in hook_name:
            continue

        if iia_granularity != "qkv" and is_qkv_granularity_hook(hook_name):
            continue

        clean_inputs = clean_data[0]
        clean_targets = clean_data[1]
        corrupted_targets = corrupted_data[1]

        base_model_original_logits = base_model(clean_inputs)
        hypothesis_model_original_logits = hypothesis_model(clean_inputs)

        # run clean data on both models, patching corrupted data where necessary
        base_model_hook_fn, hypothesis_model_hook_fn = build_hook_fns(hook_name, head_index,
                                                                      base_model_clean_cache,
                                                                      hypothesis_model_clean_cache,
                                                                      base_model_corrupted_cache,
                                                                      hypothesis_model_corrupted_cache,
                                                                      ablation_type=ablation_type)

        with base_model.hooks([(hook_name, base_model_hook_fn)]):
            base_model_intervened_logits = base_model(clean_inputs)

        with hypothesis_model.hooks([(hook_name, hypothesis_model_hook_fn)]):
            hypothesis_model_intervened_logits = hypothesis_model(clean_inputs)

        # Remove BOS from logits
        base_model_original_logits = base_model_original_logits[:, 1:]
        hypothesis_model_original_logits = hypothesis_model_original_logits[:, 1:]
        base_model_intervened_logits = base_model_intervened_logits[:, 1:]
        hypothesis_model_intervened_logits = hypothesis_model_intervened_logits[:, 1:]

        # compare the outputs of the two models
        if base_model.is_categorical():
            # apply log softmax to the logits
            base_model_original_logits: Float[Tensor, "batch pos vocab"] = t.nn.functional.log_softmax(
                base_model_original_logits, dim=-1)
            hypothesis_model_original_logits: Float[Tensor, "batch pos vocab"] = t.nn.functional.log_softmax(
                hypothesis_model_original_logits, dim=-1)
            base_model_intervened_logits: Float[Tensor, "batch pos vocab"] = t.nn.functional.log_softmax(
                base_model_intervened_logits, dim=-1)
            hypothesis_model_intervened_logits: Float[Tensor, "batch pos vocab"] = t.nn.functional.log_softmax(
                hypothesis_model_intervened_logits, dim=-1)

            # calculate labels for each position
            base_original_labels: Int[Tensor, "batch pos"] = t.argmax(base_model_original_logits, dim=-1)
            hypothesis_original_labels: Int[Tensor, "batch pos"] = t.argmax(hypothesis_model_original_logits, dim=-1)
            base_intervened_labels: Int[Tensor, "batch pos"] = t.argmax(base_model_intervened_logits, dim=-1)
            hypothesis_intervened_labels: Int[Tensor, "batch pos"] = t.argmax(hypothesis_model_intervened_logits,
                                                                              dim=-1)

            # calculate kl divergence between intervened logits
            kl_div = t.nn.functional.kl_div(
                hypothesis_model_intervened_logits,  # the output of our model
                base_model_intervened_logits,  # the target distribution
                reduction="none",
                log_target=True  # because we already applied log_softmax to the base_model_logits
            ).sum(dim=-1).mean().item()

            # calculate accuracy, checking for each input in batch dimension if all labels are the same across positions
            same_outputs_between_both_models_after_intervention = (
                    base_intervened_labels == hypothesis_intervened_labels).all(dim=-1).float()
            accuracy = same_outputs_between_both_models_after_intervention.mean().item()

            # calculate effect of node on the output: how many labels change between the intervened and non-intervened models
            base_model_effect = (base_original_labels != base_intervened_labels).float().mean().item()
            hypothesis_model_effect = (hypothesis_original_labels != hypothesis_intervened_labels).float().mean().item()

            results_by_node[node_str] = {
                "kl_div": kl_div,
                "accuracy": accuracy,
                "base_model_effect": base_model_effect,
                "hypothesis_model_effect": hypothesis_model_effect
            }

        else:
            # calculate accuracy
            same_outputs_between_both_models_after_intervention = t.isclose(base_model_intervened_logits,
                                                                            hypothesis_model_intervened_logits,
                                                                            atol=accuracy_atol).float()
            accuracy = same_outputs_between_both_models_after_intervention.mean().item()

            # calculate effect of node on the output: how much change there is between the intervened and non-intervened models
            base_model_effect = t.abs(base_model_original_logits - base_model_intervened_logits).mean().item()
            hypothesis_model_effect = t.abs(
                hypothesis_model_original_logits - hypothesis_model_intervened_logits).mean().item()

            results_by_node[node_str] = {
                "accuracy": accuracy,
                "base_model_effect": base_model_effect,
                "hypothesis_model_effect": hypothesis_model_effect
            }

    return results_by_node


def build_hook_fns(hook_name: str,
                   head_index: int,
                   base_model_clean_cache: ActivationCache,
                   hypothesis_model_clean_cache: ActivationCache,
                   base_model_corrupted_cache: ActivationCache,
                   hypothesis_model_corrupted_cache: ActivationCache,
                   ablation_type: Optional[AblationType] = "resample"):
    # decide which data we are going to use for the patching
    base_model_patching_data = {}
    hypothesis_model_patching_data = {}

    if ablation_type == "resample":
        base_model_patching_data[hook_name] = base_model_corrupted_cache[hook_name]
        hypothesis_model_patching_data[hook_name] = hypothesis_model_corrupted_cache[hook_name]

    elif ablation_type == "mean":
        # take mean over all inputs
        base_model_orig_shape = base_model_clean_cache[hook_name].shape
        hypothesis_model_orig_shape = hypothesis_model_clean_cache[hook_name].shape

        if len(base_model_orig_shape) == 3:
            base_model_patching_data[hook_name] = base_model_clean_cache[hook_name].mean(dim=0).repeat(
                base_model_orig_shape[0], 1, 1)
            hypothesis_model_patching_data[hook_name] = hypothesis_model_clean_cache[hook_name].mean(dim=0).repeat(
                hypothesis_model_orig_shape[0], 1, 1)
        else:
            base_model_patching_data[hook_name] = base_model_clean_cache[hook_name].mean(dim=0).repeat(
                base_model_orig_shape[0], 1, 1, 1)
            hypothesis_model_patching_data[hook_name] = hypothesis_model_clean_cache[hook_name].mean(dim=0).repeat(
                hypothesis_model_orig_shape[0], 1, 1, 1)

    elif ablation_type == "zero":
        base_model_patching_data[hook_name] = t.zeros_like(base_model_clean_cache[hook_name])
        hypothesis_model_patching_data[hook_name] = t.zeros_like(hypothesis_model_clean_cache[hook_name])

    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    # build the hook functions
    base_model_hook_fn = partial(regular_intervention_hook_fn, corrupted_cache=base_model_patching_data,
                                 head_index=head_index)
    hypothesis_model_hook_fn = partial(regular_intervention_hook_fn, corrupted_cache=hypothesis_model_patching_data,
                                       head_index=head_index)
    return base_model_hook_fn, hypothesis_model_hook_fn
