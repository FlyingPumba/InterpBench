import torch as t
import numpy as np
import wandb
from transformer_lens import HookedTransformer
import iit.model_pairs as mp
import iit.utils.index as index
from utils.get_cases import get_cases
from commands.build_main_parser import build_main_parser
from iit_utils import make_iit_hl_model, create_dataset
import iit_utils.correspondence as correspondence
import random

# seed everything
t.manual_seed(0)
np.random.seed(0)
# t.use_deterministic_algorithms(True)
random.seed(0)

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
WANDB_ENTITY = "cybershiptrooper"  # TODO make this an env var
args, _ = build_main_parser().parse_known_args(
    [
        "compile",
        "-i=3",
        "-f",
    ]
)
cases = get_cases(args)
case = cases[0]

tracr_output = case.build_tracr_model()
hl_model = case.build_transformer_lens_model()
# this is the graph node -> hl node correspondence
tracr_hl_corr = correspondence.TracrCorrespondence.from_output(tracr_output)


train_data, test_data = create_dataset(case, hl_model)


def config_is_bad(config):
    iit_weight = config.iit_weight
    behavior_weight = config.behavior_weight
    strict_weight = config.strict_weight

    if iit_weight == behavior_weight and behavior_weight == strict_weight:
        return True
    # reject any combination of [0, x, x]
    weights_tuple = (iit_weight, behavior_weight, strict_weight)
    if weights_tuple in [(0, x, x) for x in [0, 0.5, 1.5]] + [(x, 0, x) for x in [0, 0.5, 1.5]] + [
        (x, x, 0) for x in [0, 0.5, 1.5]
    ]:
        return True

    # reject any combination of [x, 0, 0]
    if weights_tuple in [(x, 0, 0) for x in [0, 0.5, 1.5]] + [(0, x, 0) for x in [0, 0.5, 1.5]] + [
        (0, 0, x) for x in [0, 0.5, 1.5]
    ]:
        return True

    return False


def train_model(config, use_wandb=False):
    if config_is_bad(config):
        return  # skip this config
    cfg_dict = {
        "n_layers": 2,
        "n_heads": 4,
        "d_head": 4,
        "d_model": 8,
        "d_mlp": 16,
        "seed": 0,
        "act_fn": "gelu",
    }
    ll_cfg = hl_model.cfg.to_dict().copy()
    ll_cfg.update(cfg_dict)
    print(ll_cfg)
    model = HookedTransformer(ll_cfg)

    training_args = {
        "lr": config.lr,
        "batch_size": 512,
        "atol": config.atol,
        "use_single_loss": config.use_single_loss,
        "iit_weight": config.iit_weight,
        "behavior_weight": config.behavior_weight,
        "strict_weight": config.strict_weight,
    }
    num_heads = config.num_heads
    head_idx = index.Ix[:, :, :num_heads, :]
    tracr_ll_corr = {
        ("is_x_3", None): {(0, "mlp", index.Ix[[None]])},
        ("frac_prevs_1", None): {(1, "attn", head_idx)},
    }
    hl_ll_corr = correspondence.make_hl_ll_corr(tracr_hl_corr, tracr_ll_corr)
    iit_hl_model = make_iit_hl_model(hl_model)
    model_pair = mp.StrictIITModelPair(
        hl_model=iit_hl_model, ll_model=model, corr=hl_ll_corr, training_args=training_args
    )

    model_pair.train(
        train_data,
        test_data,
        epochs=config.epochs,
        use_wandb=use_wandb,
    )


def main():
    wandb.init()
    train_model(wandb.config, use_wandb=True)


use_wandb = False

if use_wandb:
    sweep_config = {
        "name": "tracr_iit",
        "method": "grid",
        "parameters": {
            "num_heads": {"values": [1, 2, 3, 4]},
            "atol": {"values": [0.05]},
            "lr": {"values": [1e-3, 1e-4, 1e-5]},
            "use_single_loss": {"values": [True, False]},
            "iit_weight": {"values": [0.5, 1.0, 1.5]},
            "behavior_weight": {"values": [0.5, 1.0, 1.5]},
            "strict_weight": {"values": [0.0, 0.5, 1.0, 1.5]},
            "epochs": {"values": [50]},
            "act_fn": {"values": ["relu", "gelu"]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="iit", entity=WANDB_ENTITY)
    wandb.agent(sweep_id, main)
else:
    print("Not using wandb")
    config = {
        "num_heads": 1,
        "atol": 0.05,
        "lr": 1e-2,
        "use_single_loss": False,
        "iit_weight": 1.0,
        "behavior_weight": 1.0,
        "strict_weight": 0.0,
        "epochs": 50,
        "act_fn": "gelu",
    }
    import argparse

    args = argparse.Namespace(**config)
    train_model(config=args, use_wandb=False)
