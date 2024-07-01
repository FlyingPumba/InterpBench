import os
import unittest

import torch as t

from circuits_benchmark.commands.build_main_parser import build_main_parser
from circuits_benchmark.commands.train import train
from circuits_benchmark.utils.project_paths import get_default_output_dir


class TrainTest(unittest.TestCase):
  def test_linear_compression(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "linear-compression",
                                                    "-i=1,2,3",
                                                    "--d-model=5",
                                                    "--epochs=2",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--batch-size=2",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_non_linear_compression_works_for_case_3(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "non-linear-compression",
                                                    "-i=3",
                                                    "--d-model=8",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--ae-epochs=2",
                                                    "--epochs=2",
                                                    "--resample-ablation-test-loss=True",
                                                    "--resample-ablation-max-interventions=5",
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
                                                    "--d-model=8",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
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
                                                    "--d-model=8",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--epochs=2",
                                                    "--resample-ablation-test-loss=True",
                                                    "--resample-ablation-max-interventions=5",
                                                    ("--ae-path=" + ae_weights_path),
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)
