from argparse import Namespace

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.ll_model_loader.ground_truth_model_loader import GroundTruthModelLoader
from circuits_benchmark.utils.ll_model_loader.interp_bench_model_loader import InterpBenchModelLoader
from circuits_benchmark.utils.ll_model_loader.ll_model_loader import LLModelLoader
from circuits_benchmark.utils.ll_model_loader.natural_model_loader import NaturalModelLoader
from circuits_benchmark.utils.ll_model_loader.siit_model_loader import SIITModelLoader


def get_ll_model_loader_from_args(case: BenchmarkCase, args: Namespace) -> LLModelLoader:
    return get_ll_model_loader(
        case,
        args.natural,
        args.tracr,
        args.interp_bench,
        args.siit_weights,
        args.load_from_wandb,
        args.load_wandb_project,
        args.load_wandb_name,
    )


def get_ll_model_loader(
    case: BenchmarkCase,
    natural: bool = False,
    tracr: bool = False,
    interp_bench: bool = False,
    siit_weights: str | None = None,
    load_from_wandb: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> LLModelLoader:
    assert (
        not (natural and tracr)
        and not (natural and interp_bench)
        and not (tracr and interp_bench)
    ), "Only one of natural, tracr, interp_bench can be set"

    if natural:
        return NaturalModelLoader(
            case,
            load_from_wandb=load_from_wandb,
            wandb_project=wandb_project,
            wandb_name=wandb_name
        )

    if tracr:
        return GroundTruthModelLoader(case)

    if interp_bench:
        return InterpBenchModelLoader(case)

    return SIITModelLoader(
        case,
        weights=siit_weights,
        load_from_wandb=load_from_wandb,
        wandb_project=wandb_project,
        wandb_name=wandb_name
    )
