import importlib
import os.path
import random
from functools import partial
from typing import Set, Optional, Sequence

import numpy as np
import torch as t
from networkx import DiGraph
from torch import Tensor

from circuits_benchmark.benchmark.case_dataset import CaseDataset
from circuits_benchmark.benchmark.vocabs import TRACR_BOS, TRACR_PAD
from circuits_benchmark.metrics.validation_metrics import l2_metric, kl_metric
from circuits_benchmark.transformers.alignment import Alignment
from circuits_benchmark.transformers.circuit import Circuit
from circuits_benchmark.transformers.circuit_granularity import CircuitGranularity
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer, \
  HookedTracrTransformerBatchInput
from circuits_benchmark.transformers.tracr_circuits_builder import build_tracr_circuits
from circuits_benchmark.utils.cloudpickle import load_from_pickle, dump_to_pickle
from circuits_benchmark.utils.compare_tracr_output import compare_valid_positions
from circuits_benchmark.utils.project_paths import detect_project_root
from tracr.compiler import compiling
from tracr.compiler.assemble import AssembledTransformerModel
from tracr.compiler.compiling import TracrOutput
from tracr.craft.transformers import SeriesWithResiduals
from tracr.rasp import rasp
from tracr.transformer.encoder import CategoricalEncoder


class BenchmarkCase(object):

  def __init__(self):
    self.case_file_absolute_path = os.path.join(detect_project_root(), self.get_relative_path_from_root())
    self.data_size_for_tests = 100
    self.tracr_output: TracrOutput | None = None

  @staticmethod
  def get_instance_for_file_path(file_path_from_root: str):
    """Returns an instance of the benchmark case for the corresponding given file path."""
    # load module for the file
    # The module name is built by replacing "/" with "." and removing the ".py" extension from the file path
    module_name = file_path_from_root.replace("/", ".")[:-3]
    module = importlib.import_module(module_name)

    # Create class for case dinamically
    index_str = file_path_from_root.split("/")[1].split("-")[1]
    class_name = f"Case{index_str}"
    case_instance = getattr(module, class_name)(file_path_from_root)

    return case_instance

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

  def get_task_description(self) -> str:
    """Returns the task description for the benchmark case."""
    return ""

  def get_clean_data(self,
                     count: Optional[int] = 10,
                     seed: Optional[int] = 42,
                     variable_length_seqs: Optional[bool] = False,
                     remove_duplicates: Optional[bool] = False) -> CaseDataset:
    """Returns the clean data for the benchmark case."""
    max_seq_len = self.get_max_seq_len()

    if variable_length_seqs:
      min_seq_len = self.get_min_seq_len()
    else:
      min_seq_len = max_seq_len

    # assert min_seq_len is at least 2 elementsto account for BOS
    assert min_seq_len >= 2, "min_seq_len must be at least 2 to account for BOS"

    # set numpy seed and sort vocab to ensure reproducibility
    if seed is not None:
      t.random.manual_seed(seed)
      np.random.seed(seed)
      random.seed(seed)

    if count is None:
      # produce all possible sequences for this vocab
      input_data, output_data = self.gen_all_data(min_seq_len, max_seq_len)
    else:
      input_data, output_data = self.sample_data(count, min_seq_len, max_seq_len)

    assert len(set([tuple(o) for o in output_data])) > 1, "All outputs are the same for this case"

    unique_inputs = set()
    if remove_duplicates:
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

    return CaseDataset(input_data, output_data)

  def sample_data(self, count, min_seq_len, max_seq_len):
    """Samples random data for the benchmark case."""
    vals = sorted(list(self.get_vocab()))

    input_data: HookedTracrTransformerBatchInput = []
    output_data: HookedTracrTransformerBatchInput = []

    for _ in range(count):
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

  def gen_all_data(self, min_seq_len, max_seq_len) -> (HookedTracrTransformerBatchInput, HookedTracrTransformerBatchInput):
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

  def get_validation_metric(self,
                            metric_name: str,
                            tl_model: HookedTracrTransformer,
                            data_size: Optional[int] = 100) -> Tensor:
    """Returns the validation metric for the benchmark case.
    By default, only the l2 and kl metrics are available. Other metrics should override this method.
    """
    inputs = self.get_clean_data(count=data_size).get_inputs()
    is_categorical = self.get_tl_model().is_categorical()
    with t.no_grad():
      baseline_output = tl_model(inputs)
    if metric_name == "l2":
      return partial(l2_metric, baseline_output=baseline_output, is_categorical=is_categorical)
    elif metric_name == "kl":
      return partial(kl_metric, baseline_output=baseline_output, is_categorical=is_categorical)
    else:
      raise ValueError(f"Metric {metric_name} not available for this case.")

  def get_corrupted_data(self, count: Optional[int] = 10, seed: Optional[int] = 43) -> CaseDataset:
    """Returns the corrupted data for the benchmark case.
    Default implementation: re-generate clean data with a different seed."""
    dataset = self.get_clean_data(count=count, seed=seed)
    return dataset

  def get_max_seq_len(self) -> int:
    """Returns the maximum sequence length for the benchmark case (including BOS).
    Default implementation: 10."""
    return 10

  def get_min_seq_len(self) -> int:
    """Returns the minimum sequence length for the benchmark case (including BOS).
    Default implementation: 4."""
    return self.get_max_seq_len()

  def get_index(self) -> str:
    class_name = self.__class__.__name__  # Looks like "CaseN"
    return class_name[4:]

  def __str__(self):
    return self.case_file_absolute_path

  def get_tracr_output(self) -> TracrOutput:
    """Returns the tracr output for the benchmark case."""
    if self.tracr_output is None:
      return self.build_tracr_model()
    return self.tracr_output

  def get_tracr_model_pickle_path(self) -> str:
    return self.case_file_absolute_path.replace(".py", "_tracr_model.pkl")

  def get_tracr_graph_pickle_path(self) -> str:
    return self.case_file_absolute_path.replace(".py", "_tracr_graph.pkl")

  def get_craft_model_pickle_path(self) -> str:
    return self.case_file_absolute_path.replace(".py", "_craft_model.pkl")

  def get_tl_model_pickle_path(self) -> str:
    return self.case_file_absolute_path.replace(".py", "_tl_model.pkl")

  def get_tracr_model(self) -> AssembledTransformerModel | None:
    """Loads the tracr model from disk, if it exists."""
    tracr_model: AssembledTransformerModel | None = load_from_pickle(self.get_tracr_model_pickle_path())

    if tracr_model is None:
      tracr_output = self.build_tracr_model()
      return tracr_output.model
    else:
      return tracr_model

  def get_tracr_graph(self) -> DiGraph | None:
    """Loads the tracr graph from disk, if it exists, otherwise build."""
    tracr_graph: DiGraph | None = load_from_pickle(self.get_tracr_graph_pickle_path())

    if tracr_graph is None:
      tracr_output = self.build_tracr_model()
      return tracr_output.graph
    else:
      return tracr_graph

  def get_craft_model(self) -> SeriesWithResiduals | None:
    """Loads the craft model from disk, if it exists, otherwise build."""
    craft_model: SeriesWithResiduals | None = load_from_pickle(self.get_craft_model_pickle_path())

    if craft_model is None:
      tracr_output = self.build_tracr_model()
      return tracr_output.craft_model
    else:
      return craft_model

  def get_tl_model(self,
                   device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
                   *args, **kwargs
                   ) -> HookedTracrTransformer | None:
    """Loads the transformer_lens model from disk, if it exists, otherwise build."""
    tl_model: HookedTracrTransformer | None = load_from_pickle(self.get_tl_model_pickle_path())

    if tl_model is None:
      tracr_model = self.get_tracr_model()
      tl_model = self.build_transformer_lens_model(tracr_model=tracr_model, device=device, *args, **kwargs)
    else:
      # move the model to the correct device
      tl_model.to(device)

    return tl_model

  def dump_tracr_model(self, tracr_model: AssembledTransformerModel) -> None:
    """Dumps the tracr model to disk."""
    dump_to_pickle(self.get_tracr_model_pickle_path(), tracr_model)

  def dump_tracr_graph(self, tracr_graph: DiGraph) -> None:
    """Dumps the tracr graph to disk."""
    dump_to_pickle(self.get_tracr_graph_pickle_path(), tracr_graph)

  def dump_craft_model(self, craft_model: SeriesWithResiduals) -> None:
    """Dumps the craft model to disk."""
    dump_to_pickle(self.get_craft_model_pickle_path(), craft_model)

  def dump_tl_model(self, tl_model: HookedTracrTransformer) -> None:
    """Dumps the transformer_lens model to disk."""
    dump_to_pickle(self.get_tl_model_pickle_path(), tl_model)

  def get_relative_path_from_root(self) -> str:
    return f"circuits_benchmark/benchmark/cases/case_{self.get_index()}.py"

  def build_tracr_model(self) -> TracrOutput:
    """Compiles a single case to a tracr model."""
    if self.tracr_output is not None:
      return self.tracr_output
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

    # write tracr model and graph to disk
    self.dump_tracr_model(tracr_output.model)
    self.dump_tracr_graph(tracr_output.graph)
    self.dump_craft_model(tracr_output.craft_model)

    return tracr_output

  def get_tracr_circuit(self, granularity: CircuitGranularity = "component") -> tuple[Circuit, Circuit, Alignment]:
    """Returns the tracr circuit for the benchmark case."""
    tracr_graph = self.get_tracr_graph()
    craft_model = self.get_craft_model()
    return build_tracr_circuits(tracr_graph, craft_model, granularity)

  def build_transformer_lens_model(self,
                                   tracr_model: AssembledTransformerModel = None,
                                   device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
                                   *args, **kwargs
                                   ) -> HookedTracrTransformer:
    """Compiles a tracr model to transformer lens."""
    if tracr_model is None:
      tracr_model = self.get_tracr_model()

    tl_model = HookedTracrTransformer.from_tracr_model(tracr_model, device=device, *args, **kwargs)
    self.dump_tl_model(tl_model)

    return tl_model

  def run_case_tests_on_tracr_model(self,
                                    tracr_model: AssembledTransformerModel = None,
                                    atol: float = 1.e-2,
                                    fail_on_error: bool = True) -> float:
    if tracr_model is None:
      tracr_model = self.get_tracr_model()

    dataset = self.get_clean_data(count=self.data_size_for_tests)
    inputs = dataset.get_inputs()
    expected_outputs = dataset.get_correct_outputs()

    is_categorical = isinstance(tracr_model.output_encoder, CategoricalEncoder)

    correct_count = 0
    for i in range(len(inputs)):
      input = inputs[i]
      expected_output = expected_outputs[i]
      decoded_output = tracr_model.apply(input).decoded
      correct = all(compare_valid_positions(expected_output, decoded_output, is_categorical, atol))

      if not correct and fail_on_error:
        raise ValueError(f"Failed test for {self} on tracr model."
                         f"\n >>> Input: {input}"
                         f"\n >>> Expected: {expected_output}"
                         f"\n >>> Got: {decoded_output}")
      elif correct:
        correct_count += 1

    return correct_count / len(inputs)

  def run_case_tests_on_tl_model(self,
                                 tl_model: HookedTracrTransformer = None,
                                 atol: float = 1.e-2,
                                 fail_on_error: bool = True) -> float:
    if tl_model is None:
      tl_model = self.get_tl_model()

    dataset = self.get_clean_data(count=self.data_size_for_tests)
    inputs = dataset.get_inputs()
    expected_outputs = dataset.get_correct_outputs()
    decoded_outputs = tl_model(inputs, return_type="decoded")

    correct_count = 0
    for i in range(len(expected_outputs)):
      input = inputs[i]
      expected_output = expected_outputs[i]
      decoded_output = decoded_outputs[i]
      correct = all(compare_valid_positions(expected_output, decoded_output, tl_model.is_categorical(), atol))

      if not correct and fail_on_error:
        raise ValueError(f"Failed test for {self} on tl model."
                         f"\n >>> Input: {input}"
                         f"\n >>> Expected: {expected_output}"
                         f"\n >>> Got: {decoded_output}")
      elif correct:
        correct_count += 1

    return correct_count / len(expected_outputs)