"""Build a generated CUDA extension for the currently selected GPU."""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess


FA2_COMMIT = "49774b31c8d1172a5b28f72e9aed0f322bf4ccba"
CUTLASS_COMMIT = "dc4817921edda44a549197ff3a9dcf5df0636e7b"
CUDA_FEATURES = (
    "BACKWARD", "DROPOUT", "ALIBI", "SOFTCAP", "UNEVEN_K", "LOCAL",
)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def checked_headers(flash_attention_root):
    """Resolve the pinned source checkout or an existing hashed vendor snapshot."""
    root = Path(flash_attention_root).resolve()
    flash = root / "csrc" / "flash_attn"
    cutlass = root / "csrc" / "cutlass"
    if not (flash / "src" / "utils.h").is_file() or not (
        cutlass / "include" / "cute" / "tensor.hpp"
    ).is_file():
        raise ValueError(f"Expected FlashAttention and its CUTLASS submodule in {root}")

    if (root / ".git").exists():
        def git(path, *arguments):
            return subprocess.check_output(
                ["git", "-C", str(path), *arguments], text=True
            ).strip()

        if git(root, "rev-parse", "HEAD") != FA2_COMMIT:
            raise ValueError(f"Check out FlashAttention {FA2_COMMIT}")
        if git(cutlass, "rev-parse", "HEAD") != CUTLASS_COMMIT:
            raise ValueError(f"Initialize pinned CUTLASS submodule {CUTLASS_COMMIT}")
        if git(root, "status", "--porcelain", "--", "csrc/flash_attn", "csrc/cutlass"):
            raise ValueError("Build with an unmodified FlashAttention/CUTLASS checkout")
    else:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["commit"] != FA2_COMMIT
                or manifest["cutlass_submodule"].split()[0] != CUTLASS_COMMIT):
            raise ValueError("Unexpected upstream revisions in vendor manifest")
        files = manifest["files"]
        for required in ("csrc/flash_attn/src/utils.h", "csrc/cutlass/include/cute/tensor.hpp"):
            if required not in files:
                raise ValueError(f"Missing pinned vendor entry {required}")
        for name, metadata in files.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or file_sha256(path) != metadata["sha256"]:
                raise ValueError(f"Vendor source hash mismatch for {name}")
    return flash, cutlass


def cuda_device(device="cuda:0"):
    """Validate the NVIDIA device selected by the caller or CUDA visibility."""
    import torch

    selected = torch.device(device)
    if selected.type != "cuda" or not torch.version.cuda or not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA device and CUDA-enabled PyTorch are required")
    index = torch.cuda.current_device() if selected.index is None else selected.index
    selected = torch.device("cuda", index)
    capability = torch.cuda.get_device_capability(selected)
    if capability < (8, 0):
        raise RuntimeError("The kernels require CUDA compute capability 8.0 or newer")
    return selected, capability


def build_cuda(source_dir, flash_attention_root, device="cuda:0"):
    """Return the dispatcher namespace and its reproducible build metadata."""
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    device, (major, minor) = cuda_device(device)
    flash, cutlass = checked_headers(flash_attention_root)
    source_dir = Path(source_dir).resolve()
    files = sorted(
        path for path in source_dir.iterdir()
        if path.is_file() and path.suffix in (".cpp", ".cu", ".cuh", ".h", ".hpp")
    )
    units = [str(path) for path in files if path.suffix in (".cpp", ".cu")]
    if not any(path.endswith(".cu") for path in units):
        raise ValueError(f"No generated CUDA translation unit in {source_dir}")
    nvcc = Path(CUDA_HOME) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc") if CUDA_HOME else None
    if nvcc is None or not nvcc.is_file():
        raise RuntimeError("Set CUDA_HOME to an installed CUDA Toolkit with nvcc")
    target = f"{major}{minor}"
    compiler = subprocess.check_output([str(nvcc), "--version"], text=True).strip()
    known_targets = subprocess.run(
        [str(nvcc), "--list-gpu-code"], capture_output=True, text=True, check=False
    )
    if known_targets.returncode == 0 and f"sm_{target}" not in known_targets.stdout.split():
        raise RuntimeError(f"This CUDA Toolkit cannot compile sm_{target}")

    shared_flags = [f"-DFLASHATTENTION_DISABLE_{feature}" for feature in CUDA_FEATURES]
    shared_flags.append("-DFLASH_NAMESPACE=TORCH_EXTENSION_NAME")
    host_flags = ["/O2", "/std:c++20"] if os.name == "nt" else ["-O3", "-std=c++20"]
    cuda_flags = [
        "-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "--use_fast_math", "-lineinfo", "-Xptxas=-v",
        f"-gencode=arch=compute_{target},code=sm_{target}",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
    identity = {
        "source_sha256": {path.name: file_sha256(path) for path in files},
        "flash_attention_commit": FA2_COMMIT,
        "cutlass_commit": CUTLASS_COMMIT,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_compiler": compiler,
        "target": f"sm_{target}",
        "host_flags": host_flags + shared_flags,
        "cuda_flags": cuda_flags + shared_flags,
    }
    key = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    name = "tilecadence_" + key
    with torch.cuda.device(device):
        library = load(
            name=name,
            sources=units,
            extra_include_paths=[str(source_dir), str(flash), str(flash / "src"),
                                 str(cutlass / "include")],
            extra_cflags=identity["host_flags"],
            extra_cuda_cflags=identity["cuda_flags"],
            with_cuda=True,
            is_python_module=False,
        )
    receipt = {
        **identity,
        "device_name": torch.cuda.get_device_name(device),
        "device_index": device.index,
        "module": name,
        "module_path": str(library),
        "module_sha256": file_sha256(library),
    }
    return getattr(torch.ops, name), receipt
