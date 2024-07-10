from iit.model_pairs.nodes import LLNode
from typing import Callable
from torch import Tensor
from transformer_lens.hook_points import HookPoint
import iit.utils.eval_ablations as eval_ablations
from iit.utils.iit_dataset import IITDataset
import iit.model_pairs as mp
import pandas as pd


def get_ablation_effects_for_scales(
    model_pair,
    unique_test_data,
    hook_maker: Callable[
        [mp.BaseModelPair, LLNode, float], Callable[[Tensor, HookPoint], Tensor]
    ],
    scales=[0.1, 1.0],
):
    combined_scales_df = pd.DataFrame(
        columns=["node", "status"] + [f"scale {scale}" for scale in scales]
    )

    for scale in scales:
        print(f"Running scale {scale}\n")
        test_set = IITDataset(
            unique_test_data,
            unique_test_data,
            model_pair.hl_model,
            every_combination=True,
        )

        hook_maker_for_node = lambda ll_node: hook_maker(model_pair=model_pair, ll_node=ll_node, scale=scale)  # noqa: E731

        causal_effects_not_in_circuit = eval_ablations.check_causal_effect(
            model_pair=model_pair,
            dataset=test_set,
            hook_maker=hook_maker_for_node,
            node_type="n",
        )

        causal_effects_in_circuit = eval_ablations.check_causal_effect(
            model_pair=model_pair,
            dataset=test_set,
            hook_maker=hook_maker_for_node,
            node_type="individual_c",
        )

        causal_effects = eval_ablations.make_dataframe_of_results(
            causal_effects_not_in_circuit, causal_effects_in_circuit
        )

        # change column name causal effect to scale
        causal_effects.rename(columns={"causal effect": f"scale {scale}"}, inplace=True)
        combined_scales_df = pd.merge(
            combined_scales_df, causal_effects, on=["node", "status"], how="outer"
        )
        # drop columns with nan
        combined_scales_df.dropna(axis=1, how="all", inplace=True)
    return combined_scales_df