import unittest

from torch import nn

from training.compression.autencoder import AutoEncoder


class AutoEncoderTest(unittest.TestCase):
  def test_correct_layers_for_narrow(self):
    autoencoder = AutoEncoder(14, 5, 2, "narrow")

    expected_layers = 2

    # We have one ReLU layer for each layer except the last one
    expected_total_components = (expected_layers * 2) - 1

    assert len(autoencoder.encoder) == expected_total_components
    assert len(autoencoder.decoder) == expected_total_components

    assert isinstance(autoencoder.encoder[0], nn.Linear)
    assert autoencoder.encoder[0].in_features == 14
    assert autoencoder.encoder[0].out_features == 10

    assert isinstance(autoencoder.encoder[1], nn.ReLU)

    assert isinstance(autoencoder.encoder[2], nn.Linear)
    assert autoencoder.encoder[2].in_features == 10
    assert autoencoder.encoder[2].out_features == 5

    assert isinstance(autoencoder.decoder[0], nn.Linear)
    assert autoencoder.decoder[0].in_features == 5
    assert autoencoder.decoder[0].out_features == 10

    assert isinstance(autoencoder.decoder[1], nn.ReLU)

    assert isinstance(autoencoder.decoder[2], nn.Linear)
    assert autoencoder.decoder[2].in_features == 10
    assert autoencoder.decoder[2].out_features == 14


  def test_correct_layers_for_wide(self):
    autoencoder = AutoEncoder(14, 5, 2, "wide")

    expected_layers = 2

    # We have one ReLU layer for each layer except the last one
    expected_total_components = (expected_layers * 2) - 1

    assert len(autoencoder.encoder) == expected_total_components
    assert len(autoencoder.decoder) == expected_total_components

    assert isinstance(autoencoder.encoder[0], nn.Linear)
    assert autoencoder.encoder[0].in_features == 14
    assert autoencoder.encoder[0].out_features == 32

    assert isinstance(autoencoder.encoder[1], nn.ReLU)

    assert isinstance(autoencoder.encoder[2], nn.Linear)
    assert autoencoder.encoder[2].in_features == 32
    assert autoencoder.encoder[2].out_features == 5

    assert isinstance(autoencoder.decoder[0], nn.Linear)
    assert autoencoder.decoder[0].in_features == 5
    assert autoencoder.decoder[0].out_features == 32

    assert isinstance(autoencoder.decoder[1], nn.ReLU)

    assert isinstance(autoencoder.decoder[2], nn.Linear)
    assert autoencoder.decoder[2].in_features == 32
    assert autoencoder.decoder[2].out_features == 14


  def test_correct_layers_for_wide_bigger(self):
    autoencoder = AutoEncoder(91, 45, 2, "wide")

    expected_layers = 2

    # We have one ReLU layer for each layer except the last one
    expected_total_components = (expected_layers * 2) - 1

    assert len(autoencoder.encoder) == expected_total_components
    assert len(autoencoder.decoder) == expected_total_components

    assert isinstance(autoencoder.encoder[0], nn.Linear)
    assert autoencoder.encoder[0].in_features == 91
    assert autoencoder.encoder[0].out_features == 256

    assert isinstance(autoencoder.encoder[1], nn.ReLU)

    assert isinstance(autoencoder.encoder[2], nn.Linear)
    assert autoencoder.encoder[2].in_features == 256
    assert autoencoder.encoder[2].out_features == 45

    assert isinstance(autoencoder.decoder[0], nn.Linear)
    assert autoencoder.decoder[0].in_features == 45
    assert autoencoder.decoder[0].out_features == 256

    assert isinstance(autoencoder.decoder[1], nn.ReLU)

    assert isinstance(autoencoder.decoder[2], nn.Linear)
    assert autoencoder.decoder[2].in_features == 256
    assert autoencoder.decoder[2].out_features == 91


  def test_correct_layers_for_wide_bigger_more_layers(self):
    autoencoder = AutoEncoder(91, 45, 3, "wide")

    expected_layers = 3

    # We have one ReLU layer for each layer except the last one
    expected_total_components = (expected_layers * 2) - 1

    assert len(autoencoder.encoder) == expected_total_components
    assert len(autoencoder.decoder) == expected_total_components

    assert isinstance(autoencoder.encoder[0], nn.Linear)
    assert autoencoder.encoder[0].in_features == 91
    assert autoencoder.encoder[0].out_features == 256

    assert isinstance(autoencoder.encoder[1], nn.ReLU)

    assert isinstance(autoencoder.encoder[2], nn.Linear)
    assert autoencoder.encoder[2].in_features == 256
    assert autoencoder.encoder[2].out_features == 151

    assert isinstance(autoencoder.encoder[3], nn.ReLU)

    assert isinstance(autoencoder.encoder[4], nn.Linear)
    assert autoencoder.encoder[4].in_features == 151
    assert autoencoder.encoder[4].out_features == 45

    assert isinstance(autoencoder.decoder[0], nn.Linear)
    assert autoencoder.decoder[0].in_features == 45
    assert autoencoder.decoder[0].out_features == 151

    assert isinstance(autoencoder.decoder[1], nn.ReLU)

    assert isinstance(autoencoder.decoder[2], nn.Linear)
    assert autoencoder.decoder[2].in_features == 151
    assert autoencoder.decoder[2].out_features == 256

    assert isinstance(autoencoder.decoder[3], nn.ReLU)

    assert isinstance(autoencoder.decoder[4], nn.Linear)
    assert autoencoder.decoder[4].in_features == 256
    assert autoencoder.decoder[4].out_features == 91