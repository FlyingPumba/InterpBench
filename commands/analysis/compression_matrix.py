import os

import numpy as np
import plotly
import plotly.express as px
from transformer_lens.utils import to_numpy

from tracr.compiler.assemble import AssembledTransformerModel
from utils.cloudpickle import load_from_pickle


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("compression-matrix")
  parser.add_argument("--matrix", type=str, required=True,
                      help="The path to the matrix to use (.npy file).")
  parser.add_argument("--tracr-model", type=str, required=True,
                      help="The path to the tracr model to use (.pkl file).")
  parser.add_argument("-o", "--output-dir", type=str, default="results",
                      help="The directory to save the results to.")


def run(args):
  output_dir = args.output_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  print(f"Loading matrix from {args.matrix}")
  matrix: np.ndarray = np.load(args.matrix)
  if matrix is None:
    raise ValueError(f"Failed to load matrix from {args.matrix}")

  print(f"Loading tracr model from {args.tracr_model}")
  tracr_model: AssembledTransformerModel = load_from_pickle(args.tracr_model)
  if tracr_model is None:
    raise ValueError(f"Failed to load tracr model from {args.tracr_model}")

  fig = px.imshow(to_numpy(matrix.T @ matrix),
                  color_continuous_midpoint=0.0,
                  color_continuous_scale="RdBu_r",  # reverse RdBu (red is positive)
                  x=tracr_model.residual_labels,
                  y=tracr_model.residual_labels,
                  width=800,
                  height=800)

  fig.update_layout(
    xaxis={"mirror": "allticks", 'side': 'top'},
  )
  fig.update_xaxes(tickangle=-90)
  fig.update_layout(
    title=dict(text=f"SGD Compression Matrix",
               font=dict(size=25),
               y=0.1,
               x=0.5,
               xanchor='center',
               yanchor='top')
  )

  fig.write_image(f"{output_dir}/sgd-compression.png")
  plotly.offline.plot(fig, filename=f"{output_dir}/sgd-compression.html")

  # U, S, Vh = torch.svd(W_E)
  #
  # px.line(utils.to_numpy(S), title="Singular values").show()
  # px.imshow(utils.to_numpy(U), title="Principal Components on the Input").show()
  #
  # px.imshow(utils.to_numpy(torch.corrcoef(W_E))).show()
  # px.imshow(utils.to_numpy(torch.corrcoef(W_E.T))).show()
  #
  # px.imshow(utils.to_numpy(fourier_basis @ neuron_acts[:, 5].reshape(p, p) @ fourier_basis.T),
  # title="2D Fourier Transformer of neuron 5",
  # color_continuous_midpoint=0.0,
  # color_continuous_scale="RdBu",
  # labels={"x": "a", "y": "b"},
  # x=fourier_basis_names, y=fourier_basis_names)
