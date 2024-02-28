def get_zero_weights_pct(model, atol=1e-8):
  """Returns the percentage of weights that are zero (or very close to zero given the atol)."""
  weights= []
  for param in model.parameters():
    weights.extend(param.flatten().tolist())

  non_zero_weights = len([w for w in weights if abs(w) > atol])
  non_zero_weights_pct = non_zero_weights / len(weights)

  return 1 - non_zero_weights_pct
