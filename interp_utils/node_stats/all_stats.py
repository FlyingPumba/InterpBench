import pandas as pd
import torch
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
import iit.model_pairs as mp
from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import get_ll_model_loader
from circuits_benchmark.utils.iit.iit_hl_model import IITHLModel
from interp_utils.node_stats import get_node_norm_stats
import circuits_benchmark.commands.evaluation.iit.iit_eval as eval_node_effect
from interp_utils.node_stats.node_grads import get_grad_norms_by_node

def get_all_stats_for_model_pair(model_pair, loader, task, max_len=100, kl=False):
    node_norms = get_node_norm_stats(model_pair, loader, return_cache_dict=False)
    node_effects, _ = eval_node_effect.get_node_effects(
        case=task, model_pair=model_pair, use_mean_cache=False, max_len=max_len, categorical_metric="kl_div" if kl else "accuracy"
    )
    combined_df = pd.merge(
        node_effects, node_norms, left_on="node", right_on="name", how="inner"
    )
    combined_df.drop(columns=["name", "in_circuit"], inplace=True)
    grad_stats = get_grad_norms_by_node(model_pair, loader, loss_fn=model_pair.loss_fn)
    combined_df = pd.merge(
        combined_df, grad_stats, left_on="node", right_on="name", how="inner"
    )
    combined_df.drop(columns=["name", "in_circuit"], inplace=True)
    return combined_df


def make_model_pair(case: BenchmarkCase, 
                    natural: bool, iit: bool = False) -> mp.StrictIITModelPair:
    assert not (natural and iit)
    interp_bench = not natural and not iit
    if iit: 
        ll_model_loader = get_ll_model_loader(case, siit_weights="110", load_from_wandb=True)
    else:
        ll_model_loader = get_ll_model_loader(case, interp_bench=interp_bench, natural=natural, load_from_wandb=True)
    hl_ll_corr, model = ll_model_loader.load_ll_model_and_correspondence(device='cuda' if torch.cuda.is_available() else 'cpu')
    # turn off grads
    model.eval()
    model.requires_grad_(False)

    hl_model = case.get_hl_model()
    hl_model = IITHLModel(hl_model, eval_mode=True)
    model_pair = mp.StrictIITModelPair(hl_model, model, hl_ll_corr)
    return model_pair


def make_all_stats(
    cases: list[BenchmarkCase],
    kl=False,
    only_siit = False
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Generates statistics for all cases in the list.

    Parameters:
        kl (bool): Flag indicating whether to calculate Kullback-Leibler (KL) divergence. Default is False.
        cases (list[BenchmarkCase]): A list of BenchmarkCase objects.

    Returns:
        dict[str, dict[str, pd.DataFrame]]: A dictionary containing statistics for each case. The keys are the case names,
        and the values are dictionaries containing the statistics for different models (siit, natural, iit). Each model's
        statistics are stored as a pandas DataFrame.
    """
    all_stats = {}
    for case in cases:
        if kl and not case.is_categorical():
            continue
        try:
            siit_model_pair = make_model_pair(case, natural=False)
            if not only_siit:
                natural_model_pair = make_model_pair(case, natural=True)
                iit_model_pair = make_model_pair(case, natural=False, iit=True)
        except Exception as e:
            print(f"Failed to load {case.get_name()}")
            print(e)
            continue
        dataset = case.get_clean_data(max_samples=2000)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)
        
        siit_combined_df = get_all_stats_for_model_pair(siit_model_pair, loader, case, kl=kl)
        if only_siit:
            all_stats[case.get_name()] = {
                "siit": siit_combined_df
            }
            continue
        natural_model_combined_df = get_all_stats_for_model_pair(natural_model_pair, loader, case, kl=kl)
        natural_model_combined_df["status"] = "not_in_circuit"
        iit_combined_df = get_all_stats_for_model_pair(iit_model_pair, loader, case, kl=kl)
        all_stats[case.get_name()] = {
            "siit": siit_combined_df,
            "natural": natural_model_combined_df,
            "iit": iit_combined_df
        }
    return all_stats