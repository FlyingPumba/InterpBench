import torch as t
from torch import nn


class AutoEncoder(nn.Module):
  def __init__(self,
               encoder_input_size: int,
               encoder_output_size: int,
               n_layers: int,
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

    self.setup_encoder(encoder_input_size, encoder_output_size, n_layers)
    self.setup_decoder(encoder_input_size, encoder_output_size, n_layers)

  def setup_encoder(self, encoder_input_size, encoder_output_size, n_layers):
    """Set up the encoder layers. The encoder is a fully connected network with `n_layers` layers. The last layer has
    size `encoder_output_size`.
    """
    input_size = encoder_input_size
    size_decrease = (encoder_input_size - encoder_output_size) // n_layers
    output_size = input_size - size_decrease
    for i in range(n_layers - 1):
      self.encoder.add_module(f"encoder_{i}", nn.Linear(input_size, output_size, bias=False, device=self.device))
      self.encoder.add_module(f"encoder_{i}_relu", nn.ReLU())
      input_size = output_size
      output_size = input_size - size_decrease
    self.encoder.add_module(f"encoder_{n_layers - 1}", nn.Linear(input_size, encoder_output_size,
                                                                 bias=False, device=self.device))

  def setup_decoder(self, encoder_input_size, encoder_output_size, n_layers):
    """Set up the decoder layers. The decoder is a fully connected network with `n_layers` layers. The last layer has
    size `encoder_input_size`.
    """
    input_size = encoder_output_size
    size_increase = (encoder_input_size - encoder_output_size) // n_layers
    output_size = input_size + size_increase
    for i in range(n_layers - 1):
      self.decoder.add_module(f"decoder_{i}", nn.Linear(input_size, output_size, bias=False, device=self.device))
      self.decoder.add_module(f"decoder_{i}_relu", nn.ReLU())
      input_size = output_size
      output_size = input_size + size_increase
    self.decoder.add_module(f"decoder_{n_layers - 1}", nn.Linear(input_size, encoder_input_size,
                                                                 bias=False, device=self.device))

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

  def load_weights_from_file(self, path: str):
    """Loads the autoencoder weights from file."""
    self.load_state_dict(t.load(path))