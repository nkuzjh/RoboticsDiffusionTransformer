#!/usr/bin/env python3
"""Fetch and verify the official RDT/SigLIP/T5 assets into the project cache.

The downloader deliberately uses project-local Hugging Face cache variables and
``local_dir`` paths.  It downloads the official PyTorch T5 checkpoint and
tokenizer/configuration files needed by ``T5EncoderModel``; TensorFlow and
unrelated media files are excluded.  ``T5EncoderModel`` loads the encoder
portion of that checkpoint at runtime.
Repeated invocations resume through ``huggingface_hub`` and verify every file
against the public Hub API before writing ``asset_manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / ".cache" / "csgo_seen10" / "models"
DEFAULT_HF_CACHE = PROJECT_ROOT / ".cache" / "huggingface"
HF_ENDPOINT = "https://huggingface.co"


@dataclass(frozen=True)
class AssetSpec:
    name: str
    repo_id: str
    files: tuple[str, ...]


ASSETS = (
    AssetSpec(
        "rdt-1b",
        "robotics-diffusion-transformer/rdt-1b",
        ("config.json", "pytorch_model.bin"),
    ),
    AssetSpec(
        "siglip-so400m-patch14-384",
        "google/siglip-so400m-patch14-384",
        (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    ),
    AssetSpec(
        "t5-v1_1-xxl",
        "google/t5-v1_1-xxl",
        (
            "config.json",
            "generation_config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "spiece.model",
            "tokenizer_config.json",
        ),
    ),
)


def _configure_project_cache(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    # Force all Hub resolution/cache paths into this project.  This prevents a
    # missing local file from silently being written to ~/.cache/huggingface.
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_ASSETS_CACHE"] = str(cache_root / "assets")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
    # The configured proxy currently leaves hf-xet large-file transfers at
    # zero bytes.  The standard resolve/CDN path supports resumable downloads
    # and keeps the whole transfer inside this project's cache.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _request_json(url: str, attempts: int = 4) -> Any:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    headers = {"User-Agent": "rdt-csgo-seen10-asset-preparer/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"Hub API request failed after {attempts} attempts: {url}: {last_error}")


def _inventory(spec: AssetSpec) -> dict[str, Any]:
    repo_metadata = _request_json(f"{HF_ENDPOINT}/api/models/{spec.repo_id}")
    if not isinstance(repo_metadata, dict) or not repo_metadata.get("sha"):
        raise RuntimeError(f"Hub API has no immutable revision for {spec.repo_id}")
    revision = str(repo_metadata["sha"])
    url = f"{HF_ENDPOINT}/api/models/{spec.repo_id}/tree/{revision}?recursive=true&expand=true"
    payload = _request_json(url)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Hub API response for {spec.repo_id}: {type(payload).__name__}")
    by_path = {entry.get("path"): entry for entry in payload if isinstance(entry, dict)}
    missing = [name for name in spec.files if name not in by_path]
    if missing:
        raise RuntimeError(f"Hub repo {spec.repo_id} is missing required files: {missing}")

    entries: list[dict[str, Any]] = []
    for filename in spec.files:
        entry = by_path[filename]
        size = entry.get("size")
        if not isinstance(size, int) or size < 0:
            raise RuntimeError(f"Hub API has no usable size for {spec.repo_id}/{filename}")
        lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
        entries.append(
            {
                "path": filename,
                "size": size,
                "hub_oid": entry.get("oid"),
                "sha256": lfs.get("oid"),
                "last_commit": (entry.get("lastCommit") or {}).get("id"),
            }
        )
    return {
        "name": spec.name,
        "repo_id": spec.repo_id,
        "revision": revision,
        "files": entries,
        "total_bytes": sum(item["size"] for item in entries),
    }


def _human_bytes(value: int | float) -> str:
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, entry: dict[str, Any]) -> str:
    if not path.is_file():
        raise RuntimeError(f"Downloaded file is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != entry["size"]:
        raise RuntimeError(
            f"Size mismatch for {path}: expected {entry['size']}, got {actual_size}"
        )
    if entry.get("sha256"):
        actual = _sha256(path)
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: expected {entry['sha256']}, got {actual}"
            )
        return actual
    expected_oid = entry.get("hub_oid")
    if expected_oid:
        actual_oid = _git_blob_sha1(path)
        if actual_oid != expected_oid:
            raise RuntimeError(
                f"Git blob SHA-1 mismatch for {path}: expected {expected_oid}, got {actual_oid}"
            )
        return actual_oid
    return ""


def _download_with_curl(repo_id: str, revision: str, filename: str, target: Path) -> Path:
    """Resume one Hub file through the public resolve/CDN endpoint.

    ``hf_hub_download`` normally delegates large LFS files to hf-xet.  The
    current proxy leaves that path at zero bytes, while aria2c/curl ordinary
    redirected HTTP streams support byte-range continuation reliably.
    """

    aria2c = shutil.which("aria2c")
    curl = shutil.which("curl")
    if aria2c is None and curl is None:
        raise RuntimeError("aria2c or curl is required for official asset downloads")
    partial = target.with_name(target.name + ".partial")
    if target.is_file():
        return target
    if partial.exists() and not partial.is_file():
        raise RuntimeError(f"Download partial path is not a regular file: {partial}")
    url = (
        f"{HF_ENDPOINT}/{repo_id}/resolve/{revision}/{quote(filename, safe='/')}"
        f"?download=true&rdt_cache_bust={int(time.time())}"
    )
    if aria2c is not None:
        command = [
            aria2c,
            "--continue=true",
            "--allow-overwrite=true",
            "--file-allocation=none",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=4M",
            "--max-tries=10",
            "--retry-wait=2",
            "--timeout=60",
            "--connect-timeout=30",
            "--summary-interval=20",
            "--console-log-level=warn",
            "--out",
            partial.name,
            "--dir",
            str(partial.parent),
            url,
        ]
    else:
        command = [
            curl,
            "--location",
            "--fail",
            "--retry",
            "10",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "30",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            "--progress-bar",
            url,
        ]
    subprocess.run(command, check=True)
    if not partial.is_file():
        raise RuntimeError(f"curl completed without writing {partial}")
    partial.replace(target)
    return target


def _print_inventory(inventories: Iterable[dict[str, Any]]) -> None:
    total = 0
    print("Official CSGO Seen-10 asset inventory:", flush=True)
    for item in inventories:
        total += item["total_bytes"]
        print(
            f"  {item['repo_id']} revision={item['revision']} "
            f"total={_human_bytes(item['total_bytes'])}",
            flush=True,
        )
        for entry in item["files"]:
            checksum = entry.get("sha256") or entry.get("hub_oid") or "n/a"
            print(
                f"    {entry['path']}: {_human_bytes(entry['size'])} "
                f"checksum={checksum}",
                flush=True,
            )
    print(f"  selected total: {_human_bytes(total)}", flush=True)


def _select_assets(names: list[str] | None) -> tuple[AssetSpec, ...]:
    if not names:
        return ASSETS
    available = {item.name: item for item in ASSETS}
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(f"unknown asset name(s): {unknown}; choose from {sorted(available)}")
    return tuple(available[name] for name in names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory containing one local directory per selected Hub repo",
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=DEFAULT_HF_CACHE,
        help="Project-local Hugging Face download cache",
    )
    parser.add_argument(
        "--asset",
        action="append",
        dest="assets",
        choices=[item.name for item in ASSETS],
        help="Select one asset (repeatable); defaults to all three",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Query and print Hub files/sizes without downloading",
    )
    args = parser.parse_args(argv)
    selected = _select_assets(args.assets)
    output_root = args.output_root.expanduser().resolve()
    hf_cache = args.hf_cache.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _configure_project_cache(hf_cache)

    inventories = [_inventory(spec) for spec in selected]
    _print_inventory(inventories)
    if args.list_only:
        return 0

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_unix": time.time(),
        "project_root": str(PROJECT_ROOT),
        "output_root": str(output_root),
        "hf_cache": str(hf_cache),
        "assets": [],
    }
    for item in inventories:
        repo_dir = output_root / item["name"]
        repo_dir.mkdir(parents=True, exist_ok=True)
        recorded_files: list[dict[str, Any]] = []
        print(f"Preparing {item['repo_id']} -> {repo_dir}", flush=True)
        for entry in item["files"]:
            filename = entry["path"]
            target = repo_dir / filename
            started = time.monotonic()
            if target.is_file() and target.stat().st_size == entry["size"]:
                print(f"  verifying cached {filename} ({_human_bytes(entry['size'])})", flush=True)
                actual_path = target
                status = "cached"
            else:
                print(
                    f"  downloading {filename} ({_human_bytes(entry['size'])}); "
                    "hf_hub_download progress follows",
                    flush=True,
                )
                actual_path = _download_with_curl(
                    item["repo_id"], item["revision"], filename, target
                )
                status = "downloaded"
            checksum = _verify(actual_path, entry)
            elapsed = time.monotonic() - started
            mib_s = entry["size"] / (1024 * 1024) / elapsed if elapsed > 0 else 0.0
            print(
                f"  {status} {filename}: {_human_bytes(entry['size'])}, "
                f"elapsed={elapsed:.1f}s avg={mib_s:.2f} MiB/s verified={checksum}",
                flush=True,
            )
            recorded_files.append(
                {
                    **entry,
                    "local_path": str(actual_path),
                    "verified_checksum": checksum,
                    "status": status,
                    "elapsed_seconds": elapsed,
                }
            )
        manifest["assets"].append(
            {
                **item,
                "local_dir": str(repo_dir),
                "files": recorded_files,
            }
        )

    manifest_path = output_root.parent / "asset_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"All selected official assets verified; manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("prepare_csgo_assets: interrupted; rerun to resume", file=sys.stderr)
        raise SystemExit(130)
