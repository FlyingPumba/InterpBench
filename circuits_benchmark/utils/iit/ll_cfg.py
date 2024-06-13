compression_ratio_map = {
    "5": 1.8,
    "18": 1.6,
    "21": 1.3,
    "25": 1.5,
    "34": 1.5,
    "35": 3,
    "36": 2.5,
    "37": 2.2,
    "default": 2,
}

cases_with_resid_compression = ["5", "18", "21", "25", "26", "29", "34", "35", "36", "37"]


def make_ll_cfg_for_case(
    hl_model, case_index: str, compression_ratio: float | None = None
):
    compress_resid = case_index in cases_with_resid_compression
    if compression_ratio is None:
        compression_ratio = compression_ratio_map.get(
            case_index, compression_ratio_map["default"]
        )
    return make_ll_cfg(
        hl_model, compress_resid=compress_resid, compression_ratio=compression_ratio
    )


def make_ll_cfg(hl_model, compress_resid: bool, compression_ratio: float):
    ll_cfg = hl_model.cfg.to_dict().copy()
    n_heads = max(4, ll_cfg["n_heads"])
    if compress_resid:
        d_model = int(hl_model.cfg.d_model // compression_ratio)
        d_model = max(2, d_model)
        d_head = max(1, d_model // n_heads)
        d_mlp = d_model * 4
    else:
        d_head = int(max(1, ll_cfg["d_head"] // compression_ratio))
        d_model = n_heads * d_head
        d_mlp = d_model * 4
    assert d_model > 0
    assert d_head > 0
    assert d_mlp > 0
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
