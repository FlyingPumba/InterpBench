def make_ll_cfg(hl_model):
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
    return ll_cfg