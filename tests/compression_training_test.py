import dataclasses
import math

import pytest
import torch as t
from iit.model_pairs.ll_model import LLModel
from iit.utils import node_picker

from circuits_benchmark.benchmark.cases.case_1 import Case1
from circuits_benchmark.benchmark.cases.case_19 import Case19
from circuits_benchmark.benchmark.cases.case_32 import Case32
from circuits_benchmark.commands.build_main_parser import build_main_parser
from circuits_benchmark.commands.train import train
from circuits_benchmark.training.compression.autencoder import AutoEncoder
from circuits_benchmark.training.compression.non_linear_compressed_tracr_transformer_trainer import \
    NonLinearCompressedTracrTransformerTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.init_functions import wang_init_method


class TestCompressionTraining:
    def test_linear_compression(self):
        args, _ = build_main_parser().parse_known_args(["train",
                                                        "linear-compression",
                                                        "-i=2,3",
                                                        "--d-model=5",
                                                        "--epochs=1",
                                                        "--max-train-samples=10",
                                                        "--min-train-samples=10",
                                                        "--test-data-ratio=0.3",
                                                        "--batch-size=2",
                                                        "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
        train.run(args)

    def test_non_linear_compression_works(self):
        args, _ = build_main_parser().parse_known_args(["train",
                                                        "non-linear-compression",
                                                        "-i=2,3",
                                                        "--d-model=8",
                                                        "--max-train-samples=10",
                                                        "--min-train-samples=10",
                                                        "--test-data-ratio=0.3",
                                                        "--epochs=1",
                                                        "--ae-epochs=2",
                                                        "--ae-max-train-samples=5",
                                                        "--resample-ablation-test-loss=True",
                                                        "--resample-ablation-max-interventions=1",
                                                        "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
        train.run(args)

    @pytest.mark.parametrize("case", [Case1(), Case32(), Case19()])
    def test_siia_is_not_nan_for_models_that_have_all_nodes_in_circuit(self, case):
        hl_model: HookedTracrTransformer = case.get_hl_model()

        original_d_model_size = hl_model.cfg.d_model
        original_d_head_size = hl_model.cfg.d_head

        training_args = TrainingArgs(epochs=2, batch_size=10, lr_start=0.01, max_train_samples=100)

        compressed_d_model_size = 12
        compressed_d_head_size = 11

        # Get LL model with compressed dimensions
        ll_model = case.get_ll_model(
            overwrite_cfg_dict={
                "d_model": compressed_d_model_size,
                "d_head": compressed_d_head_size,
                "d_mlp": compressed_d_model_size * 4
            },
            same_size=True
        )

        # Assert all nodes in this case are in the circuit
        hl_ll_corr = case.get_correspondence(same_size=True)
        nodes_not_in_circuit = node_picker.get_nodes_not_in_circuit(
            ll_model, hl_ll_corr
        )
        assert len(nodes_not_in_circuit) == 0

        # reset params
        init_fn = wang_init_method(hl_model.cfg.n_layers, compressed_d_model_size)
        for name, param in ll_model.named_parameters():
            init_fn(param)

        # Set up autoencoders
        autoencoders_dict = {}
        autoencoders_dict["blocks.*.hook_mlp_out"] = AutoEncoder(original_d_model_size,
                                                                 compressed_d_model_size,
                                                                 2,
                                                                 "wide")
        for layer in range(hl_model.cfg.n_layers):
            for head in range(hl_model.cfg.n_heads):
                autoencoders_dict[f"blocks.{layer}.attn.hook_result[{head}]"] = AutoEncoder(original_d_model_size,
                                                                                            compressed_d_model_size,
                                                                                            2,
                                                                                            "wide")

        ae_training_args = dataclasses.replace(training_args,
                                               wandb_project=None,
                                               wandb_name=None,
                                               epochs=2,
                                               batch_size=10,
                                               lr_start=0.01)

        trainer = NonLinearCompressedTracrTransformerTrainer(case,
                                                             LLModel(model=hl_model),
                                                             ll_model,
                                                             autoencoders_dict,
                                                             training_args,
                                                             output_dir=None,
                                                             ae_training_args=ae_training_args,
                                                             ae_desired_test_mse=0.1,
                                                             ae_train_loss_weight=0.5)
        final_metrics = trainer.train()

        assert not math.isnan(final_metrics["siia"])
        assert final_metrics["siia"] == 1
