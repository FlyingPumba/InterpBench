import unittest

import torch as t

from commands.build_main_parser import build_main_parser
from commands.train import train


class TrainTest(unittest.TestCase):

  def test_linear_compression_does_not_throw_exceptions(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "linear-compression",
                                                    "-i=2,3",
                                                    "--residual-stream-compression-size=5",
                                                    "--epochs=2",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--batch-size=2",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_auto_linear_compression_works_for_case_2(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "linear-compression",
                                                    "-i=2",
                                                    "--residual-stream-compression-size=auto",
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
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_non_linear_compression_works_for_case_3(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "non-linear-compression",
                                                    "-i=3",
                                                    "--residual-stream-compression-size=8",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--epochs=2",
                                                    "--ae-path=results/case-3-resid-8-ae.pt",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_natural_compression_works_for_cases_2_and_3(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "natural-compression",
                                                    "-i=2,3",
                                                    "--residual-stream-compression-size=8",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--epochs=2",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)
