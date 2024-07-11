from abc import ABC, abstractmethod


class Argument(ABC):
    name: str
    arg: str
    value: str

    def __init__(self, name, arg, value):
        self.name = name
        self.arg = arg
        self.value = value

    @abstractmethod
    def make(self):
        raise NotImplementedError


class RequiredArgument(Argument):
    def __init__(self, name):
        super().__init__(name, None, None)

    def make(self, value):
        return value


class StoreTrueArgument(Argument):
    def __init__(self, arg, name=None):
        super().__init__(name, arg, None)

    def make(self):
        return self.arg


class VariableArgument(Argument):
    def make(self, value=None):
        if value is not None:
            return f"{self.arg} {value}"
        return f"{self.arg} {self.value}"


class OptionalArgument(Argument):
    pass


class OptionalStoreTrueArgument(OptionalArgument, StoreTrueArgument):
    def __init__(self, name, arg):
        super().__init__(name=name, arg=arg)

    def make(self):
        return self.arg


class OptionalVariableArgument(OptionalArgument, VariableArgument):
    def __init__(self, name, arg, value):
        super().__init__(name, arg, value)

    def make(self, value=None):
        # use VariableArgument's make method
        return super().make(value)
