#!/usr/bin/env python3
"""Measure checkpoint size in bits from the state dict.

For each tensor in a normal HF / torch checkpoint:
    bits = numel * dtype_bit_width

For CS6013 ``.czip`` containers (magic ``CS6013CZ``):
    bits = sum(on-disk DEFLATE block sizes) * 8
so size_frac reflects the actual compressed payload (not the restored dtype).

Visual parameters (keys matching visual/vision modules) are tallied separately
from all other parameters.

Examples:
    python measure_checkpoint_bits.py /path/to/hf_or_pt_checkpoint
    python measure_checkpoint_bits.py /path/to/compressed_model/   # may contain model.czip
    python measure_checkpoint_bits.py model.safetensors --json
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch

CZIP_MAGIC = b"CS6013CZ"


def dtype_bit_width(dtype: torch.dtype) -> int:
    """Bit width of one element for ``dtype`` (storage width)."""
    # itemsize is authoritative for torch storage (bool=8, float16=16, ...).
    return int(torch.empty((), dtype=dtype).element_size() * 8)


def is_visual_param(name: str) -> bool:
    """Heuristic: Qwen-VL / Qwen3.5 vision tower and related modules."""
    n = name.replace("\\", "/").lower()
    markers = (
        "visual.",
    )
    if any(m in n for m in markers):
        return True
    # Exact path segment match (avoids matching unrelated names containing "vision").
    parts = n.split(".")
    return any(p in {"visual", "vision", "vision_tower", "vision_model"} for p in parts)


def _czip_block_compressed_bytes(blocks: Any) -> int:
    """Sum on-disk compressed byte sizes from a czip tensor ``blocks`` entry."""
    if not blocks:
        return 0
    if isinstance(blocks, dict):
        if "size" in blocks and "offset" in blocks:
            return int(blocks["size"])
        total = 0
        for value in blocks.values():
            if isinstance(value, dict) and "size" in value:
                total += int(value["size"])
        return total
    if isinstance(blocks, list):
        return sum(int(b["size"]) for b in blocks if isinstance(b, dict) and "size" in b)
    return 0


def _read_czip_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        if fh.read(8) != CZIP_MAGIC:
            raise ValueError(f"{path} is not a CS6013 .czip container")
        _version = struct.unpack("<I", fh.read(4))[0]
        fh.seek(-24, io.SEEK_END)
        header_offset, header_len = struct.unpack("<QQ", fh.read(16))
        if fh.read(8) != CZIP_MAGIC:
            raise ValueError(f"{path} is truncated or corrupt")
        fh.seek(header_offset)
        return json.loads(fh.read(header_len).decode("utf-8"))


def iter_czip_compressed_bits(path: Path) -> Iterator[tuple[str, int, int]]:
    """Yield ``(name, compressed_bits, num_elements_estimate)`` from a ``.czip`` file."""
    header = _read_czip_header(path)
    for entry in header.get("tensors", []):
        name = str(entry.get("name", ""))
        compressed_bits = _czip_block_compressed_bytes(entry.get("blocks")) * 8
        shape = entry.get("shape") or []
        n_elem = 1
        for dim in shape:
            n_elem *= int(dim)
        if not shape:
            # Fall back to packed code count when present.
            n_elem = int(entry.get("num_codes") or 0)
        yield name, compressed_bits, n_elem


def find_czip_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".czip":
        return [path]
    if not path.is_dir():
        return []
    named = path / "model.czip"
    if named.is_file():
        return [named]
    return sorted(path.glob("*.czip"))


def iter_state_dict(path: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` from a file or Hugging Face-style directory."""
    if path.is_file():
        if path.suffix.lower() == ".czip":
            raise TypeError("use measure_bits() for .czip containers")
        yield from _iter_checkpoint_file(path)
        return

    if not path.is_dir():
        raise FileNotFoundError(path)

    st_files = sorted(path.glob("*.safetensors"))
    # Prefer sharded safetensors; skip non-model sidecars if index exists.
    index = path / "model.safetensors.index.json"
    if index.exists():
        meta = json.loads(index.read_text(encoding="utf-8"))
        weight_map = meta.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        for shard in shard_names:
            yield from _iter_checkpoint_file(path / shard)
        return

    if st_files:
        for f in st_files:
            # Skip files that are clearly not weight shards if both exist.
            if f.name.endswith(".json"):
                continue
            yield from _iter_checkpoint_file(f)
        return

    bin_files = sorted(path.glob("pytorch_model*.bin")) + sorted(
        path.glob("*.pt")
    ) + sorted(path.glob("*.pth"))
    if bin_files:
        for f in bin_files:
            yield from _iter_checkpoint_file(f)
        return

    if find_czip_files(path):
        raise TypeError("use measure_bits() for .czip containers")

    raise FileNotFoundError(
        f"No .safetensors / .bin / .pt / .czip weights found under {path}"
    )


def _iter_checkpoint_file(path: Path) -> Iterator[tuple[str, torch.Tensor]]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)
        return

    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        # Unwrap common containers.
        for wrap in ("state_dict", "model", "module"):
            inner = obj.get(wrap)
            if isinstance(inner, dict) and inner and all(
                torch.is_tensor(v) for v in list(inner.values())[:3]
            ):
                obj = inner
                break
        for k, v in obj.items():
            if torch.is_tensor(v):
                yield str(k), v
        return

    raise TypeError(f"Unsupported checkpoint object in {path}: {type(obj)}")


def _empty_by_dtype() -> dict[str, dict[str, int]]:
    return defaultdict(
        lambda: {
            "visual_bits": 0,
            "other_bits": 0,
            "visual_tensors": 0,
            "other_tensors": 0,
        }
    )


def _accumulate(
    *,
    name: str,
    bits: int,
    n_elem: int,
    dtype_s: str,
    visual_bits: int,
    other_bits: int,
    visual_params: int,
    other_params: int,
    visual_tensors: int,
    other_tensors: int,
    by_dtype: dict[str, dict[str, int]],
) -> tuple[int, int, int, int, int, int]:
    bucket = by_dtype[dtype_s]
    if is_visual_param(name):
        visual_bits += bits
        visual_params += n_elem
        visual_tensors += 1
        bucket["visual_bits"] += bits
        bucket["visual_tensors"] += 1
    else:
        other_bits += bits
        other_params += n_elem
        other_tensors += 1
        bucket["other_bits"] += bits
        bucket["other_tensors"] += 1
    return (
        visual_bits,
        other_bits,
        visual_params,
        other_params,
        visual_tensors,
        other_tensors,
    )


def measure_bits(path: Path) -> dict:
    visual_bits = 0
    other_bits = 0
    visual_params = 0
    other_params = 0
    visual_tensors = 0
    other_tensors = 0
    by_dtype: dict[str, dict[str, int]] = _empty_by_dtype()
    source = "state_dict"

    czip_files = find_czip_files(path)
    # Prefer standard weight shards when both exist; otherwise measure .czip.
    has_standard = False
    if path.is_dir():
        has_standard = bool(
            list(path.glob("*.safetensors"))
            or (path / "model.safetensors.index.json").exists()
            or list(path.glob("pytorch_model*.bin"))
            or list(path.glob("*.pt"))
            or list(path.glob("*.pth"))
        )
    elif path.is_file() and path.suffix.lower() != ".czip":
        has_standard = True

    if czip_files and not has_standard:
        source = "czip_compressed"
        for czip in czip_files:
            for name, bits, n_elem in iter_czip_compressed_bits(czip):
                (
                    visual_bits,
                    other_bits,
                    visual_params,
                    other_params,
                    visual_tensors,
                    other_tensors,
                ) = _accumulate(
                    name=name,
                    bits=bits,
                    n_elem=n_elem,
                    dtype_s="czip_compressed",
                    visual_bits=visual_bits,
                    other_bits=other_bits,
                    visual_params=visual_params,
                    other_params=other_params,
                    visual_tensors=visual_tensors,
                    other_tensors=other_tensors,
                    by_dtype=by_dtype,
                )
    else:
        for name, tensor in iter_state_dict(path):
            if not torch.is_tensor(tensor):
                continue
            bits = int(tensor.numel()) * dtype_bit_width(tensor.dtype)
            n_elem = int(tensor.numel())
            dtype_s = str(tensor.dtype).replace("torch.", "")
            (
                visual_bits,
                other_bits,
                visual_params,
                other_params,
                visual_tensors,
                other_tensors,
            ) = _accumulate(
                name=name,
                bits=bits,
                n_elem=n_elem,
                dtype_s=dtype_s,
                visual_bits=visual_bits,
                other_bits=other_bits,
                visual_params=visual_params,
                other_params=other_params,
                visual_tensors=visual_tensors,
                other_tensors=other_tensors,
                by_dtype=by_dtype,
            )

    total_bits = visual_bits + other_bits
    return {
        "checkpoint": str(path.resolve()),
        "source": source,
        "total_bits": total_bits,
        "total_bytes": total_bits / 8.0,
        "visual_bytes": visual_bits / 8.0,
        "text_bytes": other_bits / 8.0,
        "visual": {
            "bits": visual_bits,
            "bytes": visual_bits / 8.0,
            "num_parameters": visual_params,
            "num_tensors": visual_tensors,
        },
        "other": {
            "bits": other_bits,
            "bytes": other_bits / 8.0,
            "num_parameters": other_params,
            "num_tensors": other_tensors,
        },
        "by_dtype": dict(by_dtype),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure checkpoint bit size from state_dict (visual vs text)."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="HF checkpoint dir, .czip container, or a .safetensors / .bin / .pt file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a human summary",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: path not found: {args.checkpoint}", file=sys.stderr)
        return 1

    result = measure_bits(args.checkpoint)
    total_gb = result["total_bits"] / 8.0 / (1024**3)
    visual_gb = result["visual"]["bits"] / 8.0 / (1024**3)
    text_gb = result["other"]["bits"] / 8.0 / (1024**3)

    if args.json:
        print(
            json.dumps(
                {
                    "total_gb": total_gb,
                    "visual_gb": visual_gb,
                    "text_gb": text_gb,
                },
                indent=2,
            )
        )
        return 0

    print(f"Total: {total_gb:.4f} GB, visual: {visual_gb:.4f} GB, text: {text_gb:.4f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
