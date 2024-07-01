import pickle
from enum import Enum
from typing import Optional

from huggingface_hub import hf_hub_download
from iit.utils.correspondence import Correspondence
from transformer_lens import HookedTransformerConfig

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from circuits_benchmark.transformers.hooked_tracr_transformer import (
    HookedTracrTransformer,
)
from circuits_benchmark.utils.iit.best_weights import get_best_weight
from circuits_benchmark.utils.iit.correspondence import TracrCorrespondence
from circuits_benchmark.utils.iit.wandb_loader import load_model_from_wandb


class ModelType(str, Enum):
    NATURAL = "naturally_trained"
    TRACR = "tracr"
    INTERP_BENCH = "InterpBench"
    BEST = "best_model"

    @classmethod
    def make_model_type(
        cls, natural: bool, tracr: bool, interp_bench: bool
    ) -> "ModelType":
        assert (
            not (natural and tracr)
            and not (natural and interp_bench)
            and not (tracr and interp_bench)
        ), "Only one of natural, tracr, interp_bench can be set"

        if natural:
            return ModelType.NATURAL
        if tracr:
            return ModelType.TRACR
        if interp_bench:
            return ModelType.INTERP_BENCH
        # default to best model
        return ModelType.BEST
    
    @staticmethod
    def get_weight_for_model_type(model_type: "ModelType", task: str) -> str:
        if model_type == ModelType.BEST:
            return get_best_weight(task)
        elif model_type == ModelType.NATURAL:
            return "100"
        elif model_type == ModelType.INTERP_BENCH:
            return "interp_bench"
        elif model_type == ModelType.TRACR:
            return "tracr"
        else:
            raise ValueError(f"Model type {model_type} not supported")
        
    def __repr__(self) -> str:
        return self.value
    
    def __str__(self) -> str:
        return self.value



def load_ll_model_and_correspondence(
    case: BenchmarkCase,
    model_type: ModelType,
    load_from_wandb: bool,
    device: str,
    output_dir: Optional[str] = None,
    same_size: bool = False,
) -> tuple[Correspondence, HookedTracrTransformer]:
    if model_type == ModelType.TRACR:
        assert isinstance(case, TracrBenchmarkCase)
        assert not load_from_wandb, "Tracr models cannot loaded from wandb"
        return get_tracr_model(case, device)
    if model_type == ModelType.INTERP_BENCH:
        assert not same_size, "InterpBench models are never same size"
        assert not load_from_wandb, "InterpBench models cannot loaded from wandb"
        return get_interp_bench_model(case, device)

    return get_siit_model(
        case, model_type, device, load_from_wandb, output_dir, same_size
    )


def get_interp_bench_model(
    case: BenchmarkCase, device: str
) -> tuple[Correspondence, HookedTracrTransformer]:
    
    case_idx = case.get_name()
    model_file = hf_hub_download("cybershiptrooper/InterpBench", subfolder=case_idx, filename="ll_model.pth")
    cfg_file = hf_hub_download("cybershiptrooper/InterpBench", subfolder=case_idx, filename="ll_model_cfg.pkl")
    
    hl_model = case.get_hl_model()
    try:
        cfg_dict = pickle.load(open(cfg_file, "rb"))
        cfg = HookedTransformerConfig.from_dict(cfg_dict)
        cfg.device = device
        ll_model = HookedTracrTransformer(
            cfg,
            hl_model.tracr_input_encoder,
            hl_model.tracr_output_encoder,
            hl_model.residual_stream_labels,
        )
        ll_model.load_weights_from_file(model_file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find InterpBench model for case {case.get_name()}"
        )

    hl_ll_corr = case.get_correspondence()
    return hl_ll_corr, ll_model


def get_tracr_model(
    case: TracrBenchmarkCase, device: str
) -> tuple[TracrCorrespondence, HookedTracrTransformer]:
    hl_model = case.get_hl_model(device=device)
    tracr_corr = case.get_correspondence()
    assert isinstance(tracr_corr, TracrCorrespondence)
    return tracr_corr, hl_model


def get_siit_model(
    case: BenchmarkCase,
    model_type: ModelType,
    device: str,
    load_from_wandb: bool,
    output_dir: str,
    same_size: bool = False,
) -> tuple[Correspondence, HookedTracrTransformer]:
    weights = ModelType.get_weight_for_model_type(model_type, task=case.get_name())
    hl_model = case.get_hl_model(device=device)
    try:
        ll_cfg = pickle.load(
            open(
                f"{output_dir}/ll_models/{case.get_name()}/ll_model_cfg_{weights}.pkl",
                "rb",
            )
        )
    except FileNotFoundError:
        ll_cfg = case.get_ll_model_cfg(same_size=same_size)

    ll_model = HookedTracrTransformer(
        ll_cfg,
        hl_model.tracr_input_encoder,
        hl_model.tracr_output_encoder,
        hl_model.residual_stream_labels,
    )

    hl_ll_corr = case.get_correspondence()

    if load_from_wandb:
        try:
            load_model_from_wandb(
                case.get_name(), weights, output_dir, same_size=same_size
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find model {model_type} for case {case.get_name()} in wandb"
            )
    ll_model.load_weights_from_file(
        f"{output_dir}/ll_models/{case.get_name()}/ll_model_{weights}.pth"
    )
    ll_model.to(device)

    return hl_ll_corr, ll_model
