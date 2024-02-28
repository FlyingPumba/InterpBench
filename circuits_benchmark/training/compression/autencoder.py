import os

import torch as t
import wandb
from torch import nn
from wandb.sdk.wandb_run import Run


class AutoEncoder(nn.Module):
  def __init__(self,
               encoder_input_size: int,
               encoder_output_size: int,
               n_layers: int,
               first_hidden_layer_shape: str = "wide",
               device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")):
    """An autoencoder that compresses the input to `encoder_output_size` and then decompresses it back to
    `encoder_input_size`. The encoder and decoder are fully connected networks with `n_layers` layers each.
    """
    super().__init__()
    self.device = device
    self.encoder = nn.Sequential()
    self.decoder = nn.Sequential()

    self.input_size = encoder_input_size
    self.compression_size = encoder_output_size
    self.use_bias = True

    self.setup_encoder(encoder_input_size, encoder_output_size, n_layers, first_hidden_layer_shape)
    self.setup_decoder(encoder_input_size, encoder_output_size, n_layers, first_hidden_layer_shape)

    self.print_architecture()

  def setup_encoder(self, encoder_input_size, encoder_output_size, n_layers, first_hidden_layer_shape):
    """Set up the encoder layers. The encoder is a fully connected network with `n_layers` layers. The last layer has
    size `encoder_output_size`.
    """
    assert n_layers > 0, "The number of layers must be greater than 0"
    assert encoder_input_size > encoder_output_size, "The input size must be greater than the output size"

    input_size = encoder_input_size
    current_layer = 0

    if first_hidden_layer_shape == "wide":
      # The first hidden layer has size equal to the first power of 2 larger than double the input size
      output_size = 2 ** (int(encoder_input_size.bit_length()) + 1)
      self.encoder.add_module("encoder_0", nn.Linear(input_size, output_size,
                                                     bias=self.use_bias,
                                                     device=self.device))
      self.encoder.add_module("encoder_0_relu", nn.ReLU())
      input_size = output_size
      current_layer += 1

    # The size decrease for each layer is the difference between the input and output size divided by the number of
    # remaining layers
    size_decrease = (input_size - encoder_output_size) // (n_layers - current_layer)
    output_size = input_size - size_decrease

    for i in range(current_layer, n_layers - 1):
      self.encoder.add_module(f"encoder_{i}", nn.Linear(input_size, output_size,
                                                        bias=self.use_bias,
                                                        device=self.device))
      self.encoder.add_module(f"encoder_{i}_relu", nn.ReLU())
      input_size = output_size
      output_size = input_size - size_decrease

    self.encoder.add_module(f"encoder_{n_layers - 1}", nn.Linear(input_size, encoder_output_size,
                                                                 bias=self.use_bias, device=self.device))

  def setup_decoder(self, encoder_input_size, encoder_output_size, n_layers, first_hidden_layer_shape):
    """Set up the decoder layers. The decoder is a fully connected network with `n_layers` layers. The last layer has
    size `encoder_input_size`.
    """
    # Mimic exactly the same layers as the encoder but in reverse order
    for i in range(n_layers - 1, -1, -1):
      linear_layer_index_in_encoder = i * 2
      encoder_linear_layer = self.encoder[linear_layer_index_in_encoder]
      decoder_layer_input_size = encoder_linear_layer.out_features
      decoder_layer_output_size = encoder_linear_layer.in_features
      self.decoder.add_module(f"decoder_{i}", nn.Linear(decoder_layer_input_size, decoder_layer_output_size,
                                                        bias=self.use_bias, device=self.device))
      if i > 0:
        self.decoder.add_module(f"decoder_{i}_relu", nn.ReLU())

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def freeze_all_weights(self):
    """Freezes all weights in the autoencoder."""
    for param in self.parameters():
      param.requires_grad = False

  def unfreeze_all_weights(self):
    """Unfreezes all weights in the autoencoder."""
    for param in self.parameters():
      param.requires_grad = True

  def print_architecture(self):
    print("AutoEncoder architecture:")
    print(self)

  def load_weights_from_file(self, path: str):
    """Loads the autoencoder weights from file."""
    self.load_state_dict(t.load(path))

  def save(self, output_dir: str, prefix: str, wandb_run: Run | None = None):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    # save the weights of the model using state_dict
    weights_path = os.path.join(output_dir, f"{prefix}-autoencoder-weights.pt")
    t.save(self.state_dict(), weights_path)

    # save the entire model
    model_path = os.path.join(output_dir, f"{prefix}-autoencoder.pt")
    t.save(self, model_path)

    if wandb_run is not None:
      # save the files as artifacts to wandb
      artifact = wandb.Artifact(f"{prefix}-autoencoder", type="model")
      artifact.add_file(weights_path)
      artifact.add_file(model_path)
      wandb_run.log_artifact(artifact)