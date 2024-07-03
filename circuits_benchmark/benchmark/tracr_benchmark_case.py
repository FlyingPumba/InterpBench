import itertools
import random
from functools import partial
from typing import Optional, Sequence, Set, Callable

import numpy as np
import torch as t
from jaxtyping import Float
from torch import Tensor
from tracr.rasp.rasp import RASPExpr
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookedRootModule

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.tracr_dataset import TracrDataset
from circuits_benchmark.benchmark.vocabs import TRACR_BOS, TRACR_PAD
from circuits_benchmark.metrics.validation_metrics import l2_metric, kl_metric
from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_granularity import CircuitGranularity
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer, \
  HookedTracrTransformerBatchInput
from circuits_benchmark.transformers.tracr_circuits_builder import build_tracr_circuits
from circuits_benchmark.utils.iit import make_ll_cfg_for_case
from circuits_benchmark.utils.iit.correspondence import TracrCorrespondence
from iit.model_pairs.base_model_pair import BaseModelPair
from iit.model_pairs.freeze_model_pair import FreezedModelPair
from iit.model_pairs.stop_grad_pair import StopGradModelPair
from iit.model_pairs.strict_iit_model_pair import StrictIITModelPair
from iit.utils.correspondence import Correspondence
from tracr.compiler import compiling
from tracr.compiler.compiling import TracrOutput
from tracr.rasp import rasp


class TracrBenchmarkCase(BenchmarkCase):

  def __init__(self):
    super().__init__()
    self.tracr_output: TracrOutput | None = None

  def get_program(self) -> rasp.SOp:
    """Returns the RASP program to be compiled by Tracr."""
    raise NotImplementedError()

  def get_vocab(self) -> Set:
    """Returns the vocabulary to be used by Tracr."""
    raise NotImplementedError()

  def supports_causal_masking(self) -> bool:
    """Returns whether the case supports causal masking. True by default, since it is more restrictive.
    If the case does not support causal masking, it should override this method and return False.
    """
    return True

  def build_model_pair(
      self,
      model_pair_name: str | None = None,
      training_args: dict | None = None,
      ll_model: HookedTransformer | None = None,
      hl_model: HookedRootModule | None = None,
      hl_ll_corr: Correspondence | None = None,
      *args, **kwargs
  ) -> BaseModelPair:
    """Returns a model pair for training the LL model."""
    mp_map = {
      "freeze": FreezedModelPair,
      "strict": StrictIITModelPair,
      "stop_grad": StopGradModelPair,
    }

    if model_pair_name is None:
      model_pair_name = "strict"

    if training_args is None:
      training_args = {}

    if ll_model is None:
      ll_model = self.get_ll_model()

    if hl_model is None:
      hl_model = self.get_hl_model()

    if hl_ll_corr is None:
      hl_ll_corr = self.get_correspondence()

    return mp_map[model_pair_name](
      ll_model=ll_model,
      hl_model=hl_model,
      corr=hl_ll_corr,
      training_args=training_args,
    )

  def get_ll_model_cfg(self,
                       overwrite_cfg_dict: dict | None = None,
                       same_size: bool = False,
                       *args, **kwargs) -> dict:
    """Returns the configuration for the LL model for this benchmark case."""
    hl_model = self.get_hl_model()
    cfg_dict = make_ll_cfg_for_case(hl_model, self.get_name(), same_size=same_size)

    if overwrite_cfg_dict is not None:
      cfg_dict.update(overwrite_cfg_dict)

    return cfg_dict

  def get_ll_model(
      self,
      device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
      overwrite_cfg_dict: dict | None = None,
      same_size: bool = False,
      *args, **kwargs
  ) -> HookedTransformer:
    """Returns the untrained transformer_lens model for this benchmark case.
    In IIT terminology, this is the LL model before training."""
    ll_cfg = self.get_ll_model_cfg(same_size=same_size,
                                   overwrite_cfg_dict=overwrite_cfg_dict,
                                   *args, **kwargs)
    ll_model = HookedTransformer(ll_cfg)
    ll_model.to(device)
    return ll_model

  def get_hl_model(
      self,
      device: str | t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
      *args, **kwargs
  ) -> HookedTracrTransformer:
    """Returns the transformer_lens reference model for this benchmark case.
    In IIT terminology, this is the HL model."""
    tracr_output = self.get_tracr_output()
    return HookedTracrTransformer.from_tracr_model(tracr_output.model, device=device, *args, **kwargs)

  def get_correspondence(self, same_size: bool = False, *args, **kwargs) -> Correspondence:
    """Returns the correspondence between the reference and the benchmark model."""
    tracr_output = self.get_tracr_output()
    if same_size:
      return TracrCorrespondence.make_identity_corr(tracr_output=tracr_output)
    else:
      return TracrCorrespondence.from_output(self, tracr_output)

  def is_categorical(self) -> bool:
    """Returns whether the benchmark case is categorical."""
    return self.get_hl_model().is_categorical()

  def get_clean_data(self,
                     min_samples: Optional[int] = 10,
                     max_samples: Optional[int] = 10,
                     seed: Optional[int] = 42,
                     unique_data: Optional[bool] = False,
                     variable_length_seqs: Optional[bool] = False,
                     encoded_dataset: bool = True) -> TracrDataset:
    """Returns clean data for the benchmark case.
    If the number of unique datapoints is between min_samples and max_samples, returns all possible unique datapoints.
    Otherwise, returns a random sample of max_samples datapoints."""
    max_seq_len = self.get_max_seq_len()

    if variable_length_seqs:
      min_seq_len = self.get_min_seq_len()
    else:
      min_seq_len = max_seq_len

    # assert min_seq_len is at least 2 elements to account for BOS
    assert min_seq_len >= 2, "min_seq_len must be at least 2 to account for BOS"

    # set numpy seed and sort vocab to ensure reproducibility
    if seed is not None:
      t.random.manual_seed(seed)
      np.random.seed(seed)
      random.seed(seed)

    input_data = None
    output_data = None
    if min_samples is not None and max_samples is not None and min_samples < self.get_total_data_len() < max_samples:
      # the unique data is between min_samples and max_samples, produce all possible sequences for this vocab
      input_data, output_data = self.gen_all_data(min_seq_len, max_seq_len)
    elif min_samples is None and max_samples is None:
      # we didn't get max_samples nor min_samples, produce all possible sequences for this vocab
      input_data, output_data = self.gen_all_data(min_seq_len, max_seq_len)
    elif min_samples is not None and max_samples is None:
      if self.get_total_data_len() < min_samples:
        # we have fewer data than the min_samples, produce at least min_samples, with repeating sequences
        input_data, output_data = self.sample_data(min_samples, min_seq_len, max_seq_len)
      else:
        input_data, output_data = self.gen_all_data(min_seq_len, max_seq_len)
    elif max_samples is not None:
      # produce at most max_samples
      input_data, output_data = self.sample_data(max_samples, min_seq_len, max_seq_len)

    assert len(set([tuple(o) for o in output_data])) > 1, "All outputs are the same for this case"

    unique_inputs = set()
    if unique_data:
      # remove duplicates from input_data
      unique_input_data = []
      unique_output_data = []
      for i in range(len(input_data)):
        input = input_data[i]
        if str(input) not in unique_inputs:
          unique_inputs.add(tuple(input))
          unique_input_data.append(input)
          unique_output_data.append(output_data[i])
      input_data = unique_input_data
      output_data = unique_output_data

    # shuffle input_data and output_data maintaining the correspondence between input and output
    indices = np.arange(len(input_data))
    np.random.shuffle(indices)
    input_data = [input_data[i] for i in indices]
    output_data = [output_data[i] for i in indices]

    tracr_dataset = TracrDataset(np.array(input_data), np.array(output_data), self.get_hl_model())

    if encoded_dataset:
      return tracr_dataset.get_encoded_dataset()
    else:
      return tracr_dataset

  def get_total_data_len(self):
    """Returns the total number of possible sequences for the vocab and sequence lengths."""
    vals = sorted(list(self.get_vocab()))
    max_len = self.get_max_seq_len()
    min_len = self.get_min_seq_len()

    total_len = 0
    for l in range(min_len, max_len + 1):
      total_len += len(vals) ** l

    return total_len

  def get_corrupted_data(self,
                         min_samples: Optional[int] = 10,
                         max_samples: Optional[int] = 10,
                         seed: Optional[int] = 43,
                         unique_data: Optional[bool] = False) -> TracrDataset:
    """Returns the corrupted data for the benchmark case.
    Default implementation: re-generate clean data with a different seed."""
    return self.get_clean_data(min_samples=min_samples, max_samples=max_samples, seed=seed, unique_data=unique_data)

  def sample_data(self, n_samples: int, min_seq_len: int, max_seq_len: int):
    """Samples random data for the benchmark case."""
    vals = sorted(list(self.get_vocab()))

    input_data: HookedTracrTransformerBatchInput = []
    output_data: HookedTracrTransformerBatchInput = []

    for _ in range(n_samples):
      input, output = self.gen_random_input_output(vals, min_seq_len, max_seq_len)
      input_data.append(input)
      output_data.append(output)

    return input_data, output_data

  def gen_random_input_output(self, vals, min_seq_len, max_seq_len) -> (Sequence, Sequence):
    seq_len = random.randint(min_seq_len, max_seq_len)

    # figure out padding
    pad_len = max_seq_len - seq_len
    pad = [TRACR_PAD] * pad_len

    sample = np.random.choice(vals, size=seq_len - 1).tolist()  # sample with replacement
    output = self.get_correct_output_for_input(sample)

    input = [TRACR_BOS] + sample + pad
    output = [TRACR_BOS] + output + pad

    return input, output

  def gen_all_data(self, min_seq_len, max_seq_len) -> (
  HookedTracrTransformerBatchInput, HookedTracrTransformerBatchInput):
    """Generates all possible sequences for the vocab on this case."""
    vals = sorted(list(self.get_vocab()))

    input_data: HookedTracrTransformerBatchInput = []
    output_data: HookedTracrTransformerBatchInput = []

    for seq_len in range(min_seq_len, max_seq_len + 1):
      pad_len = max_seq_len - seq_len
      pad = [TRACR_PAD] * pad_len

      count = len(vals) ** (seq_len - 1)
      for index in range(count):
        # we want to produce all possible sequences for these lengths, so we convert the index to base len(vals) and
        # then convert each digit to the corresponding value in vals
        sample = []
        base = len(vals)
        num = index
        while num:
          sample.append(vals[num % base])
          num //= base

        if len(sample) < seq_len - 1:
          # extend with the first value in vocab to fill the sequence
          sample.extend([vals[0]] * (seq_len - 1 - len(sample)))

        # reverse the list to produce the sequence in the correct order
        sample = sample[::-1]

        output = self.get_correct_output_for_input(sample)

        input_data.append([TRACR_BOS] + sample + pad)
        output_data.append([TRACR_BOS] + output + pad)

    return input_data, output_data

  def get_correct_output_for_input(self, input: Sequence) -> Sequence:
    """Returns the correct output for the given input.
    By default, we run the program and use its output as ground truth.
    """
    return self.get_program()(input)

  def get_validation_metric(
      self,
      ll_model: HookedTransformer,
      data: t.Tensor,
      metric_name: Optional[str] = None,
      *args, **kwargs
  ) -> Callable[[Tensor], Float[Tensor, ""]]:
    """Returns the validation metric for the benchmark case.
    By default, only the l2 and kl metrics are available. Other metrics should override this method.
    """
    if metric_name is None:
      hl_model = self.get_hl_model()
      metric_name = "l2" if not hl_model.is_categorical() else "kl"

    is_categorical = self.get_hl_model().is_categorical()
    with t.no_grad():
      baseline_output = ll_model(data)
    if metric_name == "l2":
      return partial(l2_metric, baseline_output=baseline_output, is_categorical=is_categorical)
    elif metric_name == "kl":
      return partial(kl_metric, baseline_output=baseline_output, is_categorical=is_categorical)
    else:
      raise ValueError(f"Metric {metric_name} not available for this case.")

  def get_max_seq_len(self) -> int:
    """Returns the maximum sequence length for the benchmark case (including BOS).
    Default implementation: 10."""
    return 10

  def get_min_seq_len(self) -> int:
    """Returns the minimum sequence length for the benchmark case (including BOS).
    Default implementation: 4."""
    return 4

  def get_relative_path_from_root(self) -> str:
    return f"circuits_benchmark/benchmark/cases/case_{self.get_name()}.py"

  def get_tracr_output(self) -> TracrOutput:
    """Compiles a single case to a tracr model."""
    if self.tracr_output is not None:
      return self.tracr_output

    # Reset the RASPExpr ids to ensure reproducibility of Tracr labels
    RASPExpr._ids = itertools.count(1)

    program = self.get_program()
    max_seq_len_without_BOS = self.get_max_seq_len() - 1
    vocab = self.get_vocab()

    # Tracr assumes that max_seq_len in the following call means the maximum sequence length without BOS
    tracr_output = compiling.compile_rasp_to_model(
      program,
      vocab=vocab,
      max_seq_len=max_seq_len_without_BOS,
      compiler_bos=TRACR_BOS,
      compiler_pad=TRACR_PAD,
      causal=self.supports_causal_masking(),
    )
    self.tracr_output = tracr_output

    return tracr_output

  def get_ll_gt_circuit(self, granularity: CircuitGranularity = "acdc_hooks", *args, **kwargs) -> Circuit:
    """Returns the ground truth circuit for the LL model."""
    # This is the identity for now
    return self.get_hl_gt_circuit(granularity=granularity, *args, **kwargs)

  def get_hl_gt_circuit(self, granularity: CircuitGranularity = "acdc_hooks", *args, **kwargs) -> Circuit:
    """Returns the ground truth circuit for the HL model. I.e., the Tracr-generated model."""
    tacr_output = self.get_tracr_output()
    tracr_circuits = build_tracr_circuits(tacr_output.graph, tacr_output.craft_model, granularity=granularity)
    return tracr_circuits.tracr_transformer_circuit
