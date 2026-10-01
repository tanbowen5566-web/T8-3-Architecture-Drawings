"""
Project environment setup script (cross-platform: Windows / Mac / Linux)

Usage:
    gh repo clone tanbowen5566-web/T8-3-Architecture-Drawings
    cd T8-3-Architecture-Drawings
    python3 setup.py       # Mac/Linux
    py setup.py            # Windows

On Linux, PyTorch is installed as the CPU-only build (much smaller than the
default CUDA build). Pass --cuda to install the default build instead.

This script will automatically:
    1. Check that the installed Python version meets the requirement
    2. Create a virtual environment (venv)
    3. Install dependencies from requirements.txt
    4. Download the trained models from the GitHub Release (download_models.py)

No need to distinguish between Mac / Windows — the script detects
the current OS automatically.
"""

import argparse
import re
import shutil
import subprocess
import sys
import os
import platform

# Classic Windows consoles only render the ANSI colour codes below once
# virtual-terminal processing is switched on; an empty os.system call does that.
if platform.system() == "Windows":
    os.system("")

# ---- Config ----
VENV_DIR = "venv"
REQUIREMENTS_FILE = "requirements.txt"
MIN_PYTHON_VERSION = (3, 11)  # matches README; docTR 1.x, torch and transformers all support 3.11+
# On Linux, PyPI's default torch wheels bundle CUDA (several GB). The pipeline runs
# on CPU, so Linux installs torch/torchvision from PyTorch's CPU-only index instead.
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_PACKAGES = ("torch", "torchvision")


def print_step(msg):
    print(f"\n\033[1;34m==> {msg}\033[0m")


def print_success(msg):
    print(f"\033[1;32m✓ {msg}\033[0m")


def print_error(msg):
    print(f"\033[1;31m✗ {msg}\033[0m")


def check_python_version():
    """Check that the current Python version meets the minimum requirement."""
    print_step("Checking Python version...")
    current = sys.version_info[:2]

    if current < MIN_PYTHON_VERSION:
        print_error(
            f"Current Python version is {current[0]}.{current[1]}, "
            f"but this project requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or higher."
        )
        print(
            "\nPlease install a compatible Python version first:\n"
            "  Mac:     brew install python@3.11   then: python3.11 setup.py\n"
            "  Windows: winget install --id Python.Python.3.11   then: py -3.11 setup.py\n"
            "           (or download from https://www.python.org/downloads/)\n"
            "  Or with conda: conda create -n titleblock python=3.11\n"
        )
        sys.exit(1)

    print_success(f"Python version {current[0]}.{current[1]} meets the requirement")


def check_requirements_file():
    """Check that requirements.txt exists."""
    if not os.path.isfile(REQUIREMENTS_FILE):
        print_error(f"Could not find {REQUIREMENTS_FILE}. Make sure you're running this script from the project root.")
        sys.exit(1)


def create_venv():
    """Create a virtual environment, skipping if one already exists."""
    print_step("Creating virtual environment...")

    if os.path.isdir(VENV_DIR):
        print_success(f"Found existing virtual environment '{VENV_DIR}', skipping creation")
        return

    try:
        subprocess.run(
            [sys.executable, "-m", "venv", VENV_DIR],
            check=True,
        )
        print_success(f"Virtual environment created at ./{VENV_DIR}")
    except subprocess.CalledProcessError:
        # A failed attempt leaves a half-built venv/ behind, which the next run
        # would otherwise "reuse"; remove it so a re-run starts clean.
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        print_error("Failed to create virtual environment. Make sure Python's venv module is installed.")
        if platform.system() == "Linux":
            major, minor = sys.version_info[:2]
            print(
                "On Debian/Ubuntu the venv module is a separate package:\n"
                f"    sudo apt install python{major}.{minor}-venv\n"
                "Then re-run this script."
            )
        sys.exit(1)
    except FileNotFoundError:
        print_error("Could not find the Python executable. Make sure Python is installed and added to PATH.")
        sys.exit(1)


def get_venv_executable(name):
    """Return the path to an executable (pip / python) inside the virtual environment, based on OS."""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", f"{name}.exe")
    else:
        return os.path.join(VENV_DIR, "bin", name)


def torch_requirements():
    """Return the torch/torchvision lines from requirements.txt (e.g. 'torch>=2.2'),
    so version constraints stay defined in one place."""
    specs = []
    with open(REQUIREMENTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower()
            if name in TORCH_PACKAGES:
                specs.append(line)
    return specs


def install_requirements(use_cuda=False):
    """Install dependencies inside the virtual environment."""
    print_step("Installing dependencies (this requires internet access and may take a few minutes)...")

    venv_python = get_venv_executable("python")

    if not os.path.isfile(venv_python):
        print_error(f"Could not find Python inside the virtual environment: {venv_python}")
        print("The virtual environment may not have been created correctly. Try deleting the venv folder and re-running this script.")
        sys.exit(1)

    # Always call pip as `python -m pip`: on Windows, pip.exe refuses to upgrade
    # itself ("To modify pip, please run ... -m pip"), which would abort setup.
    pip_cmd = [venv_python, "-m", "pip"]

    try:
        # Upgrade pip first to reduce compatibility issues during install
        subprocess.run(
            pip_cmd + ["install", "--upgrade", "pip"],
            check=True,
        )
        if platform.system() == "Linux" and not use_cuda:
            specs = torch_requirements()
            if specs:
                print_step("Linux: installing CPU-only PyTorch (use --cuda for the GPU build)...")
                subprocess.run(
                    pip_cmd + ["install", "--index-url", TORCH_CPU_INDEX] + specs,
                    check=True,
                )
        # torch/torchvision are already satisfied on Linux at this point, so pip
        # keeps the CPU build and installs everything else from PyPI.
        subprocess.run(
            pip_cmd + ["install", "-r", REQUIREMENTS_FILE],
            check=True,
        )
        print_success("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print_error("Failed to install dependencies. Common causes:")
        print(
            "  - Unstable network connection — check your connection and re-run this script\n"
            "  - A package version in requirements.txt is incompatible with your system\n"
            f"\nError details: {e}"
        )
        sys.exit(1)
    except FileNotFoundError:
        print_error("Could not find pip. The virtual environment may not have been created correctly.")
        sys.exit(1)


def download_models():
    """Fetch the trained model files. A failure here is not fatal: the
    environment is still usable and download_models.py can be re-run."""
    print_step("Downloading trained models (large files, needs internet access)...")
    result = subprocess.run([get_venv_executable("python"), "download_models.py"])
    if result.returncode == 0:
        print_success("Models downloaded")
    else:
        print_error(
            "Model download failed. The environment is otherwise ready.\n"
            "The repository is private: run `gh auth login` (GitHub CLI) or set the\n"
            "GITHUB_TOKEN environment variable, then activate the environment and run:\n"
            "    python download_models.py"
        )


def print_activation_instructions():
    print_step("Setup complete!")

    if platform.system() == "Windows":
        activate_cmd = (
            rf"    cmd:        {VENV_DIR}\Scripts\activate" "\n"
            rf"    PowerShell: {VENV_DIR}\Scripts\Activate.ps1"
        )
    else:
        activate_cmd = f"    source {VENV_DIR}/bin/activate"

    print(
        f"\nActivate the virtual environment with:\n"
        f"{activate_cmd}\n"
        f"\nThen run the pipeline, e.g.:\n"
        f"    python run_pipeline.py --dataset_name YFA_cabana --run_name demo --limit 10\n"
        f"    python run_pipeline.py --input_dir <folder_of_drawings> --run_name demo\n"
        f"\nResults: results/extracted_metadata_<run_name>.csv\n"
        f"Note: docTR downloads its OCR weights on first run (internet needed once).\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Set up the project environment.")
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Linux only: install the default (CUDA) PyTorch build instead of the CPU-only one",
    )
    args = parser.parse_args()

    print("=" * 50)
    print(" Project Environment Setup")
    print("=" * 50)

    check_python_version()
    check_requirements_file()
    create_venv()
    install_requirements(use_cuda=args.cuda)
    download_models()
    print_activation_instructions()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\nSetup was interrupted by the user.")
        sys.exit(1)