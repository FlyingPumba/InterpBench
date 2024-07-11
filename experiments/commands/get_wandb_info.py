from commands.templates import COMMANDS, CommandType, get_model_suffix


def get_wandb_info(command_type: str, subcommand: str,
                   case: str,
                   model_type: str,
                   threshold = None,
                   same_size: bool = False,
                   ) -> dict | None:
  model_suffix = get_model_suffix(model_type, case)
  wandb_info = COMMANDS[command_type].get("wandb_info", None)
  if wandb_info is None:
    wandb_info = {}
  try:
    wandb_info.update(COMMANDS[command_type][subcommand].get("wandb_info", {}))
  except KeyError:
    pass

  if wandb_info == {}:
    return None

  project = wandb_info["project"].format(same_size="_same_size" if same_size else "")
  if command_type == CommandType.CIRCUIT_DISCOVERY.value:
    group = wandb_info["group"].format(algorithm=subcommand, case=case, model_suffix=model_suffix)
    name = wandb_info["name"].format(threshold=threshold)
    return {
      "project": project,
      "group": group,
      "name": name,
    }
  elif command_type == CommandType.EVALUATION.value:
    raise NotImplementedError("Evaluation wandb info not implemented yet")
  else:
    raise ValueError(f"Unknown command type {command_type}")