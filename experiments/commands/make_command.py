from commands.arguments import OptionalArgument, OptionalStoreTrueArgument, OptionalVariableArgument, RequiredArgument, StoreTrueArgument, VariableArgument
from commands.templates import COMMANDS


from string import Formatter


def make_command(command_type: str, subcommand: str, **kwargs):
  command = COMMANDS[command_type][subcommand]["command"]
  variable_args = COMMANDS[command_type][subcommand].get("variable_args", [])
  common_args = COMMANDS[command_type].get("common_args", [])
  optional_args = COMMANDS[command_type][subcommand].get("optional_args", [])
  all_args = common_args + variable_args + optional_args

  # populate final args and optional args with:
  # 1. default values for non-optional arguments, values from kwargs for required arguments
  # 2. kwargs values for optional arguments if they exist, else do not include them
  final_args = {}
  optional_args = {}
  for arg in all_args:
    if isinstance(arg, OptionalArgument):
      if arg.name in kwargs and isinstance(arg, OptionalVariableArgument):
        value = kwargs.get(arg.name, None)
        optional_args[arg.name] = arg.make(value)
      elif isinstance(arg, OptionalStoreTrueArgument) and arg.name in kwargs and kwargs[arg.name]:
        optional_args[arg.name] = arg.make()
    else:
      if isinstance(arg, RequiredArgument):
        if arg.name not in kwargs:
          raise ValueError(f"Required argument {arg.name} not found in kwargs")
        final_args[arg.name] = arg.make(kwargs[arg.name])
      elif isinstance(arg, StoreTrueArgument):
        optional_args[arg.arg] = arg.make()
      else:
        correct_instance = isinstance(arg, VariableArgument) and not isinstance(arg, OptionalVariableArgument)
        assert correct_instance, RuntimeError(f"Unknown argument type {arg}")
        value = kwargs.get(arg.name, None)
        final_args[arg.name] = arg.make(value)

  # populate the command with the final args
  formatter = Formatter()
  command = formatter.format(command, **final_args)
  for _, value in optional_args.items():
      command += f" {value}"

  return command