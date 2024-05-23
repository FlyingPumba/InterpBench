import torch as t
import numpy as np
from transformer_lens import HookedTransformer
import iit.model_pairs as mp
from circuits_benchmark.utils.iit import make_iit_hl_model, create_dataset
import circuits_benchmark.utils.iit.correspondence as correspondence
import random


def train_model(config, case, tracr_output, hl_model, use_wandb=False):
    # seed everything
    t.manual_seed(0)
    np.random.seed(0)
    # t.use_deterministic_algorithms(True)
    random.seed(0)

    train_data, test_data = create_dataset(case, hl_model)  # , 500, 100)
    ll_cfg = hl_model.cfg.to_dict().copy()
    n_heads = max(4, ll_cfg["n_heads"])
    d_head = ll_cfg["d_head"] // 2
    d_model = n_heads * d_head
    d_mlp = d_model * 4
    cfg_dict = {
        "n_layers": max(2, ll_cfg["n_layers"]),
        "n_heads": n_heads,
        "d_head": d_head,
        "d_model": d_model,
        "d_mlp": d_mlp,
        "seed": 0,
        "act_fn": "gelu",
        # "initializer_range": 0.02,
    }
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
    hl_ll_corr = correspondence.TracrCorrespondence.from_output(
        case, tracr_output
    )
    iit_hl_model = make_iit_hl_model(hl_model)
    model_pair = mp.StrictIITModelPair(
        hl_model=iit_hl_model,
        ll_model=model,
        corr=hl_ll_corr,
        training_args=training_args,
    )

    model_pair.train(
        train_data,
        test_data,
        epochs=config.epochs,
        use_wandb=use_wandb,
        wandb_name_suffix=config.wandb_suffix,
    )
    return model_pair
