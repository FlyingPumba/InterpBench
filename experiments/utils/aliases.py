import numpy as np

def create_random_str(length: int = 7):
  alphabet = list('abcdefghijklmnopqrstuvwxyz0123456789')
  np_alphabet = np.array(alphabet)
  np_codes = np.random.choice(np_alphabet, (length))
  return ''.join(np_codes)

def replace_bad_things_in_arg(arg: str):
  return arg.replace(".", "-").replace("+", "--").replace(" ", "-").replace("_", "-")  

class AliasDict(dict):
    def __getitem__(self, key):
        if key not in self:
            return ""
        value = super().__getitem__(key)
        if callable(value):
            value = value()
        # replace ' ' and '_' with '-'
        return replace_bad_things_in_arg(value)

important_args_aliases = AliasDict({
      "main.py": "",
      "-t": "",
      "-i": "case",
      "--wandb-suffix": "",
      "random_suffix": create_random_str,
      "categorical-metric": "metric",
  })
