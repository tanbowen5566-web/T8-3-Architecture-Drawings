"""
download_models.py

Purpose: Download the trained model files (title block detector + LayoutLMv3
         field classifier) from this repo's GitHub Release into models/, so
         that run_pipeline.py works right after setup. The model files are
         too large for a normal git repo (GitHub's limit is 100 MB per file),
         so they are published as Release assets instead.

Usage:
    python download_models.py            # download anything that is missing
    python download_models.py --force    # re-download everything

The repository is private, so a GitHub login is needed. The token is taken from,
in order:
    1. the GITHUB_TOKEN environment variable
           export GITHUB_TOKEN=<token>   # Mac/Linux
           set GITHUB_TOKEN=<token>      # Windows (cmd)
    2. the GitHub CLI, if installed and logged in (`gh auth login`)

Only the Python standard library is used, so this works before/without
installing requirements.txt. Internet access is needed once; after that the
whole pipeline runs offline.

To publish a new model version: upload the new files to a new Release, then
update RELEASE_TAG, MODELS (file names + sha256) and the paths in
run_pipeline.py. Get the checksum with:  shasum -a 256 <file>
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile

REPO = "tanbowen5566-web/T8-3-Architecture-Drawings"
RELEASE_TAG = "model-v2"
OUTPUT_DIR = "models"

# asset name -> {sha256, path that must exist afterwards, unzip?}
MODELS = {
    "title_block_detector_v2.pth": {
        "sha256": "c6e15db9ff8fc29520f8661be37aa0d35cf2f5239dfa7387cd21a022430ae597",
        "target": os.path.join(OUTPUT_DIR, "title_block_detector_v2.pth"),
        "unzip": False,
    },
    "field_classifier_v2.zip": {
        "sha256": "17df5bca0151bce6b4dc4e4468a0dff3df70c73cca47619aaf892346016d3614",
        "target": os.path.join(OUTPUT_DIR, "field_classifier_v2", "model.safetensors"),
        "unzip": True,  # zip contains a top-level field_classifier_v2/ folder
    },
}


def get_token():
    """GITHUB_TOKEN if set, else the GitHub CLI's login (`gh auth token`), else None."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    if shutil.which("gh"):
        try:
            result = subprocess.run(
                ["gh", "auth", "token"], capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and result.stdout.strip():
                print("Using the GitHub CLI login (gh auth token)")
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def open_url(url, token, accept=None):
    headers = {"User-Agent": "titleblock-extraction"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if accept:
        headers["Accept"] = accept
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))


def resolve_url(name, token):
    """Public repos: direct release URL. With a token (private repos): look up
    the asset via the API and download it through the API endpoint."""
    if not token:
        return f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{name}", None
    api = f"https://api.github.com/repos/{REPO}/releases/tags/{RELEASE_TAG}"
    with open_url(api, token, "application/vnd.github+json") as resp:
        release = json.load(resp)
    for asset in release["assets"]:
        if asset["name"] == name:
            return asset["url"], "application/octet-stream"
    raise FileNotFoundError(f"asset '{name}' not found in release '{RELEASE_TAG}'")


def download(name, dest, token):
    url, accept = resolve_url(name, token)
    with open_url(url, token, accept) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description="Download trained models from the GitHub Release")
    parser.add_argument("--force", action="store_true", help="re-download even if the files already exist")
    args = parser.parse_args()

    token = get_token()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, info in MODELS.items():
        if os.path.isfile(info["target"]) and not args.force:
            print(f"[skip] {info['target']} already exists")
            continue

        print(f"[download] {name}")
        tmp = os.path.join(OUTPUT_DIR, name + ".part")
        try:
            download(name, tmp, token)
        except urllib.error.HTTPError as e:
            hint = ""
            if e.code in (401, 403, 404) and not token:
                hint = " The repository is private: run `gh auth login` (GitHub CLI) or set the GITHUB_TOKEN environment variable, then retry."
            print(f"  Failed: HTTP {e.code} {e.reason}.{hint}")
            sys.exit(1)
        except (urllib.error.URLError, FileNotFoundError) as e:
            print(f"  Failed: {e}")
            sys.exit(1)

        if sha256_of(tmp) != info["sha256"]:
            os.remove(tmp)
            print("  Failed: checksum mismatch (corrupted download or wrong file). Please retry.")
            sys.exit(1)

        if info["unzip"]:
            with zipfile.ZipFile(tmp) as z:
                z.extractall(OUTPUT_DIR)
            os.remove(tmp)
        else:
            os.replace(tmp, info["target"])
        print(f"  OK -> {info['target']}")

    print("\nAll models are in place.")


if __name__ == "__main__":
    main()
