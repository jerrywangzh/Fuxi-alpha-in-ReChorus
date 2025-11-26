from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')
]

# Export camel-case alias so `--model_name FuXi` resolves to the same module as `fuxi`.
try:
    from . import fuxi as _fuxi_module  # noqa: F401
    FuXi = _fuxi_module  # type: ignore
    if "FuXi" not in __all__:
        __all__.append("FuXi")
except Exception:
    pass
