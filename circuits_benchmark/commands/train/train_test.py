import os
import unittest

import torch as t

from circuits_benchmark.commands.build_main_parser import build_main_parser
from circuits_benchmark.commands.train import train
from circuits_benchmark.utils.project_paths import get_default_output_dir


class TrainTest(unittest.TestCase):

  def test_natural_compression(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "natural-compression",
                                                    "-i=1,2,3",
                                                    "--residual-stream-compression-size=8",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--epochs=2",
                                                    "--resample-ablation-test-loss=True",
                                                    "--resample-ablation-max-interventions=5",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_linear_compression(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "linear-compression",
                                                    "-i=1,2,3",
                                                    "--residual-stream-compression-size=5",
                                                    "--epochs=2",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--batch-size=2",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_linear_compression_with_component_level_train_loss(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "linear-compression",
                                                    "-i=1,2,3",
                                                    "--residual-stream-compression-size=5",
                                                    "--epochs=2",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--batch-size=2",
                                                    "--train-loss=component",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_linear_compression_with_intervention_level_train_loss(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "linear-compression",
                                                    "-i=1,2,3",
                                                    "--residual-stream-compression-size=5",
                                                    "--epochs=2",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--batch-size=2",
                                                    "--train-loss=intervention",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_auto_linear_compression_works_for_case_2(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "linear-compression",
                                                    "-i=2",
                                                    "--residual-stream-compression-size=auto",
                                                    "--epochs=2",
                                                    "--train-data-size=10",
                                                    "--auto-compression-accuracy=0.01",
                                                    "--early-stop-test-accuracy=0.01",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_autoencoder_training_works_for_case_3(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "autoencoder",
                                                    "-i=3",
                                                    "--residual-stream-compression-size=8",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--epochs=2",
                                                    "--resample-ablation-test-loss=True",
                                                    "--resample-ablation-max-interventions=5",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_non_linear_compression_works_for_case_3(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "non-linear-compression",
                                                    "-i=3",
                                                    "--residual-stream-compression-size=8",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--ae-epochs=2",
                                                    "--freeze-ae-weights",
                                                    "--epochs=2",
                                                    "--resample-ablation-test-loss=True",
                                                    "--resample-ablation-max-interventions=5",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_non_linear_compression_with_component_level_train_loss_works_for_case_3(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "non-linear-compression",
                                                    "-i=3",
                                                    "--residual-stream-compression-size=8",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--ae-epochs=2",
                                                    "--freeze-ae-weights",
                                                    "--epochs=2",
                                                    "--train-loss=component",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_non_linear_compression_with_intervention_level_train_loss_works_for_case_3(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "non-linear-compression",
                                                    "-i=5",
                                                    "--residual-stream-compression-size=8",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--ae-epochs=2",
                                                    "--freeze-ae-weights",
                                                    "--epochs=2",
                                                    "--resample-ablation-test-loss=True",
                                                    "--resample-ablation-max-interventions=3",
                                                    "--train-loss=intervention",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_non_linear_compression_works_for_case_3_when_loading_ae_weights(self):
    output_dir = get_default_output_dir()
    ae_weights_path = os.path.join(output_dir, "case-3-resid-8-autoencoder-weights.pt")

    if not os.path.exists(ae_weights_path):
      # retrieve the autoencoder
      self.test_autoencoder_training_works_for_case_3()

    args, _ = build_main_parser().parse_known_args(["train",
                                                    "non-linear-compression",
                                                    "-i=3",
                                                    "--residual-stream-compression-size=8",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--freeze-ae-weights",
                                                    "--epochs=2",
                                                    "--resample-ablation-test-loss=True",
                                                    "--resample-ablation-max-interventions=5",
                                                    ("--ae-path=" + ae_weights_path),
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_non_linear_compression_works_when_unfreezing_ae_weights(self):
    output_dir = get_default_output_dir()
    ae_weights_path = os.path.join(output_dir, "case-3-resid-8-autoencoder-weights.pt")

    if not os.path.exists(ae_weights_path):
      # retrieve the autoencoder
      self.test_autoencoder_training_works_for_case_3()

    args, _ = build_main_parser().parse_known_args(["train",
                                                    "non-linear-compression",
                                                    "-i=3",
                                                    "--residual-stream-compression-size=8",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--epochs=2",
                                                    "--resample-ablation-test-loss=True",
                                                    "--resample-ablation-max-interventions=5",
                                                    ("--ae-path=" + ae_weights_path),
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)
