import importlib
from importlib.metadata import PackageNotFoundError, version

import tensorbridge.dtypes  # noqa


try:
    __version__ = version("tensorbridge-kernels")
except PackageNotFoundError:
    __version__ = "0.2.0+source"
RUNTIME_API_VERSION = 1


def __getattr__(name: str):
    if name == "ops":
        module = importlib.import_module("tensorbridge.ops")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
