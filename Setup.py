"""
Project environment setup script (cross-platform: Windows / Mac / Linux)

Usage:
    git clone <repo>
    cd <project-folder>
    python setup.py        # or python3 setup.py

This script will automatically:
    1. Check that the installed Python version meets the requirement
    2. Create a virtual environment (venv)
    3. Install dependencies from requirements.txt

No need to distinguish between Mac / Windows — the script detects
the current OS automatically.
"""

import subprocess
import sys
import os
import platform

# ---- Config ----
VENV_DIR = "venv"
REQUIREMENTS_FILE = "requirements.txt"
MIN_PYTHON_VERSION = (3, 11)  # docTR requires Python 3.11+


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
            "  Mac:     brew install python@3.11\n"
            "  Windows: download from https://www.python.org/downloads/\n"
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
        print_error("Failed to create virtual environment. Make sure Python's venv module is installed.")
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


def install_requirements():
    """Install dependencies inside the virtual environment."""
    print_step("Installing dependencies (this requires internet access and may take a few minutes)...")

    pip_path = get_venv_executable("pip")

    if not os.path.isfile(pip_path):
        print_error(f"Could not find pip inside the virtual environment: {pip_path}")
        print("The virtual environment may not have been created correctly. Try deleting the venv folder and re-running this script.")
        sys.exit(1)

    try:
        # Upgrade pip first to reduce compatibility issues during install
        subprocess.run(
            [pip_path, "install", "--upgrade", "pip"],
            check=True,
        )
        subprocess.run(
            [pip_path, "install", "-r", REQUIREMENTS_FILE],
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


def print_activation_instructions():
    print_step("Setup complete!")

    if platform.system() == "Windows":
        activate_cmd = rf"{VENV_DIR}\Scripts\activate"
    else:
        activate_cmd = f"source {VENV_DIR}/bin/activate"

    print(
        f"\nActivate the virtual environment with:\n"
        f"    {activate_cmd}\n"
        f"\nThen you can run the project scripts, e.g.:\n"
        f"    python your_script.py\n"
    )


def main():
    print("=" * 50)
    print(" Project Environment Setup")
    print("=" * 50)

    check_python_version()
    check_requirements_file()
    create_venv()
    install_requirements()
    print_activation_instructions()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\nSetup was interrupted by the user.")
        sys.exit(1)