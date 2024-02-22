import random
import time
import unittest

import numpy as np
import plotly.express as px
import torch as t
from torch.nn import init

from benchmark.case_dataset import CaseDataset
from benchmark.cases.case_3 import Case3
from utils.hooked_tracr_transformer import HookedTracrTransformer
from utils.project_paths import get_default_output_dir
from utils.resampling_ablation_loss.resample_ablation_loss import get_resample_ablation_loss


class ResampleAblationLossTest(unittest.TestCase):
  def get_randomly_initialized_transformer(self, original_tracr_model):
    random_model = HookedTracrTransformer.from_hooked_tracr_transformer(
      original_tracr_model,
      init_params_fn=lambda x: init.kaiming_uniform_(x) if len(x.shape) > 1 else init.normal_(x, std=0.02),
    )
    return random_model

  def get_new_transformer_with_used_matrix_re_initialized(self, original_tracr_model, used_modules):
    new_model = HookedTracrTransformer.from_hooked_tracr_transformer(
      original_tracr_model,
    )

    # get the first relevant MLP
    mlp_block_name = [module_name for module_name in used_modules if "mlp" in module_name][0]
    layer = int(mlp_block_name.split(".")[1])

    # Re-initialize randomly the W_in and W_out of that MLP
    init.normal_(new_model.blocks[layer].mlp.W_in, std=0.02)
    init.normal_(new_model.blocks[layer].mlp.W_out, std=0.02)

    return new_model

  def get_new_transformer_with_no_op_matrix_re_initialized(self, original_tracr_model, used_modules):
    new_model = HookedTracrTransformer.from_hooked_tracr_transformer(
      original_tracr_model,
    )

    # get all MLPs
    n_layers = new_model.cfg.n_layers
    mlp_block_names = [module_name for module_name in used_modules if "mlp" in module_name]
    mlp_layers = [int(mlp_block_name.split(".")[1]) for mlp_block_name in mlp_block_names]

    # find the first layer that has a no-op MLP
    no_op_mlp_layer = None
    for i in range(n_layers):
      if i not in mlp_layers:
        no_op_mlp_layer = i
        break

    # Re-initialize randomly the W_in and W_out of that MLP
    init.normal_(new_model.blocks[no_op_mlp_layer].mlp.W_in, std=0.02)
    init.normal_(new_model.blocks[no_op_mlp_layer].mlp.W_out, std=0.02)

    return new_model

  def set_fixed_seed(self, seed):
    np.random.seed(seed)
    t.manual_seed(seed)
    random.seed(seed)

  def set_random_seed(self):
    self.set_fixed_seed(int(time.time()))

  def get_fixed_clean_and_corrupted_data(self, case, data_sizes):
    full_clean_data = case.get_clean_data(count=None, seed=42*6)
    full_corrupted_data = case.get_corrupted_data(count=None, seed=42*7)
    clean_datas = {}
    corrupted_datas = {}
    for data_size in data_sizes:
      clean_datas[data_size] = CaseDataset(full_clean_data[:data_size][CaseDataset.INPUT_FIELD],
                                           full_clean_data[:data_size][CaseDataset.CORRECT_OUTPUT_FIELD])
      corrupted_datas[data_size] = CaseDataset(full_corrupted_data[:data_size][CaseDataset.INPUT_FIELD],
                                               full_corrupted_data[:data_size][CaseDataset.CORRECT_OUTPUT_FIELD])
    return clean_datas, corrupted_datas

  def test_variance_to_random_weights_data_and_interventions(self):
    self.skipTest("This test takes a long time to run, so it is skipped by default.")
    output_dir = get_default_output_dir()
    case = Case3()
    original_tracr_model: HookedTracrTransformer = case.get_tl_model()

    random_model = self.get_randomly_initialized_transformer(original_tracr_model)

    # Build a 2D heatmap varying data_size from 1 to 256 and max_interventions from 1 to 64.
    # At the end, plot using plotly.
    max_data_size = 256
    data_sizes_step = int(max_data_size * 0.1)
    data_sizes = list(range(1, max_data_size + 1, data_sizes_step)) + [max_data_size]

    max_interventions = 64
    max_interventions_step = int(max_interventions * 0.1)
    intervention_numbers = list(range(1, max_interventions + 1, max_interventions_step)) + [max_interventions]

    heatmap_data = []

    for data_size in data_sizes:
      row = []
      for n_interventions in intervention_numbers:
        self.set_random_seed()
        loss = get_resample_ablation_loss(
          case.get_clean_data(count=data_size, seed=None),
          case.get_corrupted_data(count=data_size, seed=None),
          original_tracr_model,
          random_model,
          max_interventions=n_interventions
        )
        row.append(loss)

      # add row to the beginning of the list
      heatmap_data.insert(0, row)

    # Plot heatmap using plotly
    fig = px.imshow(heatmap_data,
                    color_continuous_scale="RdBu_r",  # reverse RdBu (red is positive)
                    x=[str(i) for i in intervention_numbers],
                    y=[str(i) for i in reversed(data_sizes)],
                    width=800,
                    height=800)
    fig.update_layout(
      xaxis={"title": "Interventions"},
      yaxis={"title": "Data Size"},
    )
    fig.write_image(f"{output_dir}/resample-ablation-loss-variance-to-random-weights-data-and-interventions.png")

  def test_variance_to_random_weights_fixed_data_and_interventions(self):
    self.skipTest("This test takes a long time to run, so it is skipped by default.")
    output_dir = get_default_output_dir()
    case = Case3()
    original_tracr_model: HookedTracrTransformer = case.get_tl_model()

    random_model = self.get_randomly_initialized_transformer(original_tracr_model)

    # Build a 2D heatmap varying data_size from 1 to 256 and max_interventions from 1 to 64.
    # At the end, plot using plotly.
    max_data_size = 256
    data_sizes_step = int(max_data_size * 0.1)
    data_sizes = list(range(1, max_data_size + 1, data_sizes_step)) + [max_data_size]

    # prepare the data in a fixed way
    clean_datas, corrupted_datas = self.get_fixed_clean_and_corrupted_data(case, data_sizes)

    max_interventions = 64
    max_interventions_step = int(max_interventions * 0.1)
    intervention_numbers = list(range(1, max_interventions + 1, max_interventions_step)) + [max_interventions]

    heatmap_data = []

    for data_size in data_sizes:
      row = []
      for n_interventions in intervention_numbers:
        # set the seed to have fixed interventions
        self.set_fixed_seed(42 * 4)

        loss = get_resample_ablation_loss(
          clean_datas[data_size],
          corrupted_datas[data_size],
          original_tracr_model,
          random_model,
          max_interventions=n_interventions
        )

        row.append(loss)

      # add row to the beginning of the list
      heatmap_data.insert(0, row)

    # Plot heatmap using plotly
    fig = px.imshow(heatmap_data,
                    color_continuous_scale="RdBu_r",  # reverse RdBu (red is positive)
                    x=[str(i) for i in intervention_numbers],
                    y=[str(i) for i in reversed(data_sizes)],
                    width=800,
                    height=800)
    fig.update_layout(
      xaxis={"title": "Interventions"},
      yaxis={"title": "Data Size"},
    )
    fig.write_image(f"{output_dir}/resample-ablation-loss-variance-to-random-weights-fixed-data-and-interventions.png")

  def test_variance_to_data_size(self):
    self.skipTest("This test takes a long time to run, so it is skipped by default.")
    output_dir = get_default_output_dir()
    case = Case3()
    original_tracr_model: HookedTracrTransformer = case.get_tl_model()

    # set seed to have fixed weights
    self.set_fixed_seed(42 * 2)

    random_model = self.get_randomly_initialized_transformer(original_tracr_model)

    # Build a 2D heatmap varying data_size from 1 to 256 and max_interventions from 1 to 64.
    # At the end, plot using plotly.
    max_data_size = 256
    data_sizes_step = int(max_data_size * 0.1)
    data_sizes = list(range(1, max_data_size + 1, data_sizes_step)) + [max_data_size]

    # prepare the data in a fixed way
    clean_datas, corrupted_datas = self.get_fixed_clean_and_corrupted_data(case, data_sizes)

    max_interventions = 64
    intervention_numbers = [max_interventions]

    heatmap_data = []

    for data_size in data_sizes:
      row = []
      for n_interventions in intervention_numbers:
        # set the seed to have fixed interventions
        self.set_fixed_seed(42 * 5)

        loss = get_resample_ablation_loss(
          clean_datas[data_size],
          corrupted_datas[data_size],
          original_tracr_model,
          random_model,
          max_interventions=n_interventions
        )
        row.append(loss)

      # add row to the beginning of the list
      heatmap_data.insert(0, row)

    # Plot heatmap using plotly
    fig = px.imshow(heatmap_data,
                    color_continuous_scale="RdBu_r",  # reverse RdBu (red is positive)
                    x=[str(i) for i in intervention_numbers],
                    y=[str(i) for i in reversed(data_sizes)],
                    width=800,
                    height=800)
    fig.update_layout(
      xaxis={"title": "Interventions"},
      yaxis={"title": "Data Size"},
    )
    fig.write_image(f"{output_dir}/resample-ablation-loss-variance-to-data-size.png")

  def test_variance_to_interventions_size(self):
    self.skipTest("This test takes a long time to run, so it is skipped by default.")
    output_dir = get_default_output_dir()
    case = Case3()
    original_tracr_model: HookedTracrTransformer = case.get_tl_model()

    # set seed to have fixed weights
    self.set_fixed_seed(42 * 2)

    random_model = self.get_randomly_initialized_transformer(original_tracr_model)

    # Build a 2D heatmap varying data_size from 1 to 256 and max_interventions from 1 to 64.
    # At the end, plot using plotly.
    max_data_size = 256
    data_sizes = [max_data_size]
    clean_datas, corrupted_datas = self.get_fixed_clean_and_corrupted_data(case, data_sizes)

    max_interventions = 64
    max_interventions_step = int(max_interventions * 0.1)
    intervention_numbers = list(range(1, max_interventions + 1, max_interventions_step)) + [max_interventions]

    heatmap_data = []

    for data_size in data_sizes:
      row = []
      for n_interventions in intervention_numbers:
        # set the seed to have fixed interventions
        self.set_fixed_seed(42 * 5)

        loss = get_resample_ablation_loss(
          clean_datas[data_size],
          corrupted_datas[data_size],
          original_tracr_model,
          random_model,
          max_interventions=n_interventions
        )
        row.append(loss)

      # add row to the beginning of the list
      heatmap_data.insert(0, row)

    # Plot heatmap using plotly
    fig = px.imshow(heatmap_data,
                    color_continuous_scale="RdBu_r",  # reverse RdBu (red is positive)
                    x=[str(i) for i in intervention_numbers],
                    y=[str(i) for i in reversed(data_sizes)],
                    width=800,
                    height=800)
    fig.update_layout(
      xaxis={"title": "Interventions"},
      yaxis={"title": "Data Size"},
    )
    fig.write_image(f"{output_dir}/resample-ablation-loss-variance-to-interventions-size.png")

  def test_variance_re_init_used_matrix_random_data_and_interventions(self):
    self.skipTest("This test takes a long time to run, so it is skipped by default.")
    output_dir = get_default_output_dir()
    case = Case3()
    original_tracr_model: HookedTracrTransformer = case.get_tl_model()

    n_plots = 3

    for idx_plot in range(n_plots):
      random_model = self.get_new_transformer_with_used_matrix_re_initialized(original_tracr_model,
                                                                              case.get_relevant_module_names())

      # Build a 2D heatmap varying data_size from 1 to 256 and max_interventions from 1 to 64.
      # At the end, plot using plotly.
      max_data_size = 256
      data_sizes_step = int(max_data_size * 0.1)
      data_sizes = list(range(1, max_data_size + 1, data_sizes_step)) + [max_data_size]

      max_interventions = 64
      max_interventions_step = int(max_interventions * 0.1)
      intervention_numbers = list(range(1, max_interventions + 1, max_interventions_step)) + [max_interventions]

      heatmap_data = []

      for data_size in data_sizes:
        row = []
        for n_interventions in intervention_numbers:
          self.set_random_seed()
          loss = get_resample_ablation_loss(
            case.get_clean_data(count=data_size, seed=None),
            case.get_corrupted_data(count=data_size, seed=None),
            original_tracr_model,
            random_model,
            max_interventions=n_interventions
          )
          row.append(loss)

        # add row to the beginning of the list
        heatmap_data.insert(0, row)

      # Plot heatmap using plotly
      fig = px.imshow(heatmap_data,
                      color_continuous_scale="RdBu_r",  # reverse RdBu (red is positive)
                      x=[str(i) for i in intervention_numbers],
                      y=[str(i) for i in reversed(data_sizes)],
                      width=800,
                      height=800)
      fig.update_layout(
        xaxis={"title": "Interventions"},
        yaxis={"title": "Data Size"},
      )
      fig.write_image(f"{output_dir}/resample-ablation-loss-variance-re-init-used-matrix-random-data-and-interventions-{idx_plot + 1}.png")

  def test_variance_re_init_no_op_matrix_random_data_and_interventions(self):
    self.skipTest("This test takes a long time to run, so it is skipped by default.")
    output_dir = get_default_output_dir()
    case = Case3()
    original_tracr_model: HookedTracrTransformer = case.get_tl_model()

    n_plots = 3

    for idx_plot in range(n_plots):
      random_model = self.get_new_transformer_with_no_op_matrix_re_initialized(original_tracr_model,
                                                                              case.get_relevant_module_names())

      # Build a 2D heatmap varying data_size from 1 to 256 and max_interventions from 1 to 64.
      # At the end, plot using plotly.
      max_data_size = 256
      data_sizes_step = int(max_data_size * 0.1)
      data_sizes = list(range(1, max_data_size + 1, data_sizes_step)) + [max_data_size]

      max_interventions = 64
      max_interventions_step = int(max_interventions * 0.1)
      intervention_numbers = list(range(1, max_interventions + 1, max_interventions_step)) + [max_interventions]

      heatmap_data = []

      for data_size in data_sizes:
        row = []
        for n_interventions in intervention_numbers:
          self.set_random_seed()
          loss = get_resample_ablation_loss(
            case.get_clean_data(count=data_size, seed=None),
            case.get_corrupted_data(count=data_size, seed=None),
            original_tracr_model,
            random_model,
            max_interventions=n_interventions
          )
          row.append(loss)

        # add row to the beginning of the list
        heatmap_data.insert(0, row)

      # Plot heatmap using plotly
      fig = px.imshow(heatmap_data,
                      color_continuous_scale="RdBu_r",  # reverse RdBu (red is positive)
                      x=[str(i) for i in intervention_numbers],
                      y=[str(i) for i in reversed(data_sizes)],
                      width=800,
                      height=800)
      fig.update_layout(
        xaxis={"title": "Interventions"},
        yaxis={"title": "Data Size"},
      )
      fig.write_image(
        f"{output_dir}/resample-ablation-loss-variance-re-init-no-op-matrix-random-data-and-interventions-{idx_plot + 1}.png")
