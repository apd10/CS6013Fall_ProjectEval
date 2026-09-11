#!/usr/bin/env python3
"""Zero every state-dict tensor whose name contains ``visual``.

Examples:
    python ensure_visual_zero.py /path/to/hf_checkpoint
    python ensure_visual_zero.py model.safetensors --output /tmp/zeroed/
    python ensure_visual_zero.py pytorch_model.bin
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


def is_visual_name(name: str) -> bool:
    return "visual" in name.replace("\\", "/").lower()


def _unwrap_state_dict(obj: object) -> tuple[object, dict | None]:
    """Return ``(container, tensor_dict)``. tensor_dict is None if unsupported."""
    if not isinstance(obj, dict):
        return obj, None
    for wrap in ("state_dict", "model", "module"):
        inner = obj.get(wrap)
        if isinstance(inner, dict) and inner and all(
            torch.is_tensor(v) for v in list(inner.values())[:3]
        ):
            return obj, inner
    if obj and all(torch.is_tensor(v) for v in list(obj.values())[:3] if v is not None):
        return obj, obj
    return obj, None


def zero_visual_in_dict(tensors: dict) -> int:
    n = 0
    for key, value in list(tensors.items()):
        if not torch.is_tensor(value):
            continue
        if is_visual_name(str(key)):
            tensors[key] = torch.zeros_like(value)
            n += 1
    return n


def process_safetensors(src: Path, dst: Path) -> int:
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    metadata = None
    n = 0
    with safe_open(str(src), framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            tensor = f.get_tensor(key)
            if is_visual_name(key):
                tensor = torch.zeros_like(tensor)
                n += 1
            tensors[key] = tensor.contiguous()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if metadata:
        save_file(tensors, str(dst), metadata=metadata)
    else:
        save_file(tensors, str(dst))
    return n


def process_torch_file(src: Path, dst: Path) -> int:
    obj = torch.load(str(src), map_location="cpu", weights_only=True)
    container, tensors = _unwrap_state_dict(obj)
    if tensors is None:
        raise TypeError(f"Unsupported checkpoint object in {src}: {type(obj)}")
    n = zero_visual_in_dict(tensors)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(container, str(dst))
    return n


def process_file(src: Path, dst: Path) -> int:
    suffix = src.suffix.lower()
    if suffix == ".safetensors":
        return process_safetensors(src, dst)
    if suffix in {".bin", ".pt", ".pth"}:
        return process_torch_file(src, dst)
    if suffix == ".czip":
        raise TypeError("Cannot rewrite .czip containers; decompress first, then zero visual weights.")
    raise TypeError(f"Unsupported weight file: {src}")


def weight_files_in_dir(path: Path) -> list[Path]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        meta = json.loads(index.read_text(encoding="utf-8"))
        shards = sorted(set(meta.get("weight_map", {}).values()))
        return [path / name for name in shards]
    st = sorted(p for p in path.glob("*.safetensors") if p.is_file())
    if st:
        return st
    return (
        sorted(path.glob("pytorch_model*.bin"))
        + sorted(path.glob("*.pt"))
        + sorted(path.glob("*.pth"))
    )


def copy_sidecars(src_dir: Path, dst_dir: Path, weight_names: set[str]) -> None:
    if src_dir.resolve() == dst_dir.resolve():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.name in weight_names or item.name.endswith(".safetensors"):
            continue
        if item.suffix.lower() in {".bin", ".pt", ".pth"}:
            continue
        target = dst_dir / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir() and item.name not in {".git"}:
            shutil.copytree(item, target, dirs_exist_ok=True)


def ensure_visual_zero(src: Path, dst: Path) -> tuple[int, int]:
    """Zero visual tensors. Returns ``(n_files, n_tensors)``."""
    if src.is_file():
        n = process_file(src, dst)
        return 1, n

    if not src.is_dir():
        raise FileNotFoundError(src)

    files = weight_files_in_dir(src)
    if not files:
        czip = list(src.glob("*.czip"))
        if czip:
            print(f"No .safetensors / .bin / .pt shards under {src} (found {czip[0].name}); skipping visual zero")
            return 0, 0
        raise FileNotFoundError(f"No .safetensors / .bin / .pt weights found under {src}")

    copy_sidecars(src, dst, {p.name for p in files})
    n_tensors = 0
    for f in files:
        rel = f.relative_to(src)
        n_tensors += process_file(f, dst / rel)
    return len(files), n_tensors


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Set state-dict tensors to 0 when the tensor name contains "visual".'
    )
    parser.add_argument("checkpoint", type=Path, help="HF checkpoint dir or a weight file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write here (default: overwrite the input checkpoint)",
    )
    args = parser.parse_args()
    src = args.checkpoint
    if not src.exists():
        print(f"ERROR: path not found: {src}", file=sys.stderr)
        return 1
    dst = args.output if args.output is not None else src
    n_files, n_tensors = ensure_visual_zero(src, dst)
    print(f"Zeroed {n_tensors} visual tensor(s) across {n_files} weight file(s) → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
