"""Shared config loading and JAX environment bootstrap helpers."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any


def parse_bootstrap_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-c", "--config", default=None)
    return parser.parse_known_args(argv)[0]


def default_config_module_name(package: str | None) -> str:
    if package:
        return f"{package}.init"
    return "init"


def looks_like_file_ref(config_ref: str) -> bool:
    return config_ref.endswith(".py") or "/" in config_ref or "\\" in config_ref


def load_config_module_from_path(config_ref: str, prefix: str):
    path = Path(config_ref).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"config file does not exist: {path}")
    if path.suffix != ".py":
        raise ValueError(f"config file must be a .py file: {path}")

    module_name = f"_{prefix}_config_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load config file: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    added_paths: list[str] = []
    for import_path in (str(path.parent), str(Path.cwd())):
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
            added_paths.append(import_path)
    try:
        spec.loader.exec_module(module)
    finally:
        for import_path in added_paths:
            try:
                sys.path.remove(import_path)
            except ValueError:
                pass
    return module


def load_config_module(config_ref: str | None, package: str | None, prefix: str):
    if config_ref is None:
        source = default_config_module_name(package)
        return importlib.import_module(source), source
    if looks_like_file_ref(config_ref):
        return load_config_module_from_path(config_ref, prefix), str(Path(config_ref))
    return importlib.import_module(config_ref), config_ref


def load_cfg(config_ref: str | None, package: str | None, prefix: str = "text_lm") -> tuple[Any, type, str]:
    module, source = load_config_module(config_ref, package, prefix)
    if not hasattr(module, "CFG"):
        raise AttributeError(f"config source {source!r} must define CFG")
    cfg = module.CFG
    return cfg, getattr(module, "TextLMConfig", type(cfg)), source


def append_xla_flag(flag: str) -> None:
    flags = os.environ.get("XLA_FLAGS", "")
    if flag not in flags.split():
        os.environ["XLA_FLAGS"] = (flags + " " + flag).strip()


def setup_jax_environment(jax_platforms: str | None) -> None:
    if jax_platforms is not None and "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = jax_platforms
    append_xla_flag("--xla_gpu_autotune_level=0")
    os.environ.setdefault("ROCM_PATH", "/opt/rocm")
    os.environ.setdefault("LLVM_PATH", "/opt/rocm/llvm")
    if os.environ.get("LD_LIBRARY_PATH") is None:
        os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/lib"
    elif "/opt/rocm/lib" not in os.environ["LD_LIBRARY_PATH"].split(":"):
        os.environ["LD_LIBRARY_PATH"] += ":/opt/rocm/lib"
