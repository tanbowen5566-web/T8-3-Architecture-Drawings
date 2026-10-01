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
    2. Check that requirements.txt is present
    3. Create a virtual environment (venv)
    4. Install dependencies from requirements.txt
    5. Download the trained models from the GitHub Release (download_models.py)

At the end it prints a summary of every step. If a step failed, the summary
says which one, what went wrong and how to fix it; re-running the script then
reuses everything that already succeeded.

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
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = "venv"
REQUIREMENTS_FILE = "requirements.txt"
MIN_PYTHON_VERSION = (3, 11)  # matches README; docTR 1.x, torch and transformers all support 3.11+
# On Linux, PyPI's default torch wheels bundle CUDA (several GB). The pipeline runs
# on CPU, so Linux installs torch/torchvision from PyTorch's CPU-only index instead.
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_PACKAGES = ("torch", "torchvision")
README_HINT = "README → Quick start → Step 2: Check that setup succeeded"
# Where the trained models are published. The release tag is read from
# download_models.py (RELEASE_TAG); MODEL_RELEASE_TAG is only the fallback.
# Keep MODEL_FILES in sync with the asset names in download_models.py.
RELEASES_URL = "https://github.com/tanbowen5566-web/T8-3-Architecture-Drawings/releases"
MODEL_RELEASE_TAG = "models-v2"
MODEL_FILES = ("title_block_detector_v2.pth", "field_classifier_v2.zip")

# (key, label) in the order the steps run
STEPS = [
    ("python", "Check Python version"),
    ("requirements", "Find requirements.txt"),
    ("venv", "Create virtual environment"),
    ("deps", "Install dependencies"),
    ("models", "Download trained models"),
]
# A failure in these steps does not stop the remaining steps
NON_FATAL_STEPS = {"models"}

SYSTEM = platform.system()  # "Windows", "Darwin" (Mac) or "Linux"


# ---- Output helpers ----

def print_step(msg):
    print(f"\n\033[1;34m==> {msg}\033[0m")


def print_success(msg):
    print(f"\033[1;32m✓ {msg}\033[0m")


def print_error(msg):
    print(f"\033[1;31m✗ {msg}\033[0m")


class StepFailed(Exception):
    """Raised by a step when it fails. `problem` says what went wrong,
    `fix` tells the user what to do; both are shown in the final summary."""

    def __init__(self, problem, fix):
        super().__init__(problem)
        self.problem = problem
        self.fix = fix


def setup_command():
    """The command the user should type to re-run setup on this OS."""
    if SYSTEM == "Windows":
        return "py setup.py"
    return "python3 setup.py"


def install_python_fix():
    """OS-specific instructions for installing a suitable Python."""
    want = f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"
    if SYSTEM == "Darwin":
        return (
            f"Install Python {want}: download the macOS installer from https://www.python.org/downloads/macos/\n"
            f"(then double-click 'Install Certificates.command' in Applications → Python {want}),\n"
            f"or with Homebrew: brew install python@{want}\n"
            f"Then run: python{want} setup.py"
        )
    if SYSTEM == "Windows":
        return (
            f"Install Python {want}: winget install --id Python.Python.{want}\n"
            f"(or the 64-bit installer from https://www.python.org/downloads/windows/ with 'Add python.exe to PATH' ticked).\n"
            f"Close and reopen the terminal, then run: py -{want} setup.py"
        )
    return (
        f"Install Python {want}, e.g. Ubuntu: sudo apt install python{want} python{want}-venv\n"
        f"RHEL/Rocky/Alma: sudo dnf install python{want}\n"
        f"Then run: python{want} setup.py"
    )


def get_venv_executable(name):
    """Return the path to an executable (pip / python) inside the virtual environment, based on OS."""
    if SYSTEM == "Windows":
        return os.path.join(VENV_DIR, "Scripts", f"{name}.exe")
    return os.path.join(VENV_DIR, "bin", name)


def run_and_capture(cmd):
    """Run a command, show its output live, and also return the output text
    so failures can be diagnosed afterwards."""
    sys.stdout.flush()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    output = bytearray()
    while True:
        chunk = proc.stdout.read1(4096)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        output += chunk
    proc.wait()
    return proc.returncode, output.decode("utf-8", errors="replace")


# ---- Steps (each returns a short detail string on success, raises StepFailed on failure) ----

def check_python_version(args):
    """Check that the current Python version meets the minimum requirement."""
    print_step("Checking Python version...")
    current = sys.version_info[:2]
    version = f"{current[0]}.{current[1]}"
    want = f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"

    if current < MIN_PYTHON_VERSION:
        print_error(f"Current Python version is {version}, but this project requires Python {want} or higher.")
        raise StepFailed(
            f"Setup was started with Python {version} ({sys.executable}); Python {want} or higher is required.",
            install_python_fix(),
        )

    print_success(f"Python version {version} meets the requirement")
    return f"Python {version}"


def check_requirements_file(args):
    """Check that requirements.txt exists next to this script."""
    print_step("Looking for requirements.txt...")
    if not os.path.isfile(REQUIREMENTS_FILE):
        print_error(f"Could not find {REQUIREMENTS_FILE} in {PROJECT_ROOT}")
        raise StepFailed(
            f"{REQUIREMENTS_FILE} is missing from the project folder ({PROJECT_ROOT}).",
            "The project folder is incomplete. Ask the team for a complete copy of the project,\n"
            "then run setup from the new folder.",
        )
    print_success(f"Found {REQUIREMENTS_FILE}")
    return "found"


def venv_python_version(venv_python):
    """Return (major, minor) of the venv's Python, or None if it cannot be run."""
    try:
        out = subprocess.run(
            [venv_python, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        return int(out[0]), int(out[1])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def create_venv(args):
    """Create a virtual environment. An existing, working one is reused; a broken
    or outdated one (e.g. left by an interrupted run) is rebuilt."""
    print_step("Creating virtual environment...")

    if os.path.isdir(VENV_DIR):
        existing = venv_python_version(get_venv_executable("python"))
        if existing and existing >= MIN_PYTHON_VERSION:
            print_success(f"Found existing virtual environment '{VENV_DIR}', reusing it")
            return "reused existing venv/"
        reason = "it is incomplete" if existing is None else f"it uses Python {existing[0]}.{existing[1]}"
        print(f"Existing '{VENV_DIR}' folder cannot be used ({reason}); rebuilding it...")
        try:
            shutil.rmtree(VENV_DIR)
        except OSError as e:
            print_error(f"Could not delete the old '{VENV_DIR}' folder: {e}")
            raise StepFailed(
                f"The existing '{VENV_DIR}' folder is broken and could not be deleted ({e}).",
                f"Close any terminal or editor that is using the project, delete the '{VENV_DIR}' folder\n"
                f"inside the project folder by hand, then run: {setup_command()}",
            )

    try:
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
    except subprocess.CalledProcessError:
        # A failed attempt leaves a half-built venv/ behind; remove it so a re-run starts clean.
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        print_error("Failed to create virtual environment.")
        if SYSTEM == "Linux":
            major, minor = sys.version_info[:2]
            fix = (
                "The Python 'venv' module is missing. On Debian/Ubuntu install it with:\n"
                f"    sudo apt install python{major}.{minor}-venv\n"
                f"Then run: python{major}.{minor} setup.py"
            )
        else:
            fix = (
                "This Python installation is missing its 'venv' module.\n"
                "Reinstall Python from https://www.python.org/downloads/ and run setup again."
            )
        raise StepFailed(f"'{sys.executable} -m venv {VENV_DIR}' failed.", fix)
    except FileNotFoundError:
        print_error("Could not find the Python executable.")
        raise StepFailed(
            f"The Python executable ({sys.executable}) could not be started.",
            install_python_fix(),
        )

    print_success(f"Virtual environment created at ./{VENV_DIR}")
    return f"created ./{VENV_DIR}"


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


def install_requirements(args):
    """Install dependencies inside the virtual environment."""
    print_step("Installing dependencies (this requires internet access and may take a few minutes)...")

    venv_python = get_venv_executable("python")
    if not os.path.isfile(venv_python):
        print_error(f"Could not find Python inside the virtual environment: {venv_python}")
        raise StepFailed(
            f"The virtual environment has no Python at {venv_python}.",
            f"Delete the '{VENV_DIR}' folder inside the project folder, then run: {setup_command()}",
        )

    # Always call pip as `python -m pip`: on Windows, pip.exe refuses to upgrade
    # itself ("To modify pip, please run ... -m pip"), which would abort setup.
    pip_cmd = [venv_python, "-m", "pip"]

    stages = [("upgrading pip", ["install", "--upgrade", "pip"])]
    if SYSTEM == "Linux" and not args.cuda:
        specs = torch_requirements()
        if specs:
            stages.append(("installing CPU-only PyTorch",
                           ["install", "--index-url", TORCH_CPU_INDEX] + specs))
    # On Linux torch/torchvision are already satisfied at this point, so pip
    # keeps the CPU build and installs everything else from PyPI.
    stages.append((f"installing the packages in {REQUIREMENTS_FILE}", ["install", "-r", REQUIREMENTS_FILE]))

    for stage, pip_args in stages:
        print(f"\n--- {stage[0].upper() + stage[1:]} ---")
        sys.stdout.flush()
        result = subprocess.run(pip_cmd + pip_args)
        if result.returncode != 0:
            print_error(f"pip failed while {stage}")
            raise StepFailed(
                f"pip failed while {stage} (exit code {result.returncode}).",
                "Scroll up to the first line starting with 'ERROR:' just above this summary; it names the cause.\n"
                "  - Connection errors / timeouts: check your internet connection, then run setup again.\n"
                "  - 'No space left on device': free up about 3 GB of disk space, then run setup again.\n"
                "  - 'Could not find a version that satisfies the requirement torch': this Python is 32-bit\n"
                "    or newer than PyTorch supports. Install 64-bit Python 3.11 and run setup with it.\n"
                "Already installed packages are kept, so running setup again continues from here.",
            )

    print_success("Dependencies installed successfully")
    return "installed"


def model_release_tag():
    """Release tag that holds the models. Read from download_models.py so the two
    files cannot disagree; falls back to MODEL_RELEASE_TAG if it cannot be read."""
    try:
        with open("download_models.py", encoding="utf-8") as f:
            match = re.search(r"""^RELEASE_TAG\s*=\s*["']([^"']+)["']""", f.read(), re.M)
        if match:
            return match.group(1)
    except OSError:
        pass
    return MODEL_RELEASE_TAG


def manual_download_fix():
    """Step-by-step instructions for downloading the models from the GitHub Release by hand."""
    detector, classifier_zip = MODEL_FILES
    classifier_dir = classifier_zip[:-len(".zip")]
    return (
        "Download the models by hand from GitHub instead:\n"
        f"1. Open {RELEASES_URL} and find the release '{model_release_tag()}'.\n"
        f"2. Under 'Assets', download {detector} and {classifier_zip}.\n"
        f"3. In the project folder, open the 'models' folder (create a folder named 'models' if it is missing)\n"
        f"   and move {detector} into it.\n"
        f"4. Unzip {classifier_zip} and move the resulting '{classifier_dir}' folder into 'models'.\n"
        f"   Check: the file models/{classifier_dir}/config.json must exist (on Windows, 'Extract All'\n"
        f"   can create {classifier_dir}/{classifier_dir}/; move the inner folder up if so).\n"
        "5. Replace any file with the same name that is already there.\n"
        "Setup is otherwise complete: continue with README → Step 3 (no need to run setup again)."
    )


def diagnose_download_failure(output):
    """Turn download_models.py output into a short description of what went wrong."""
    text = output.lower()
    if "certificate_verify_failed" in text:
        return "Python could not verify GitHub's SSL certificate, so the download was refused."
    if "404" in text or "not found" in text:
        return ("GitHub answered 'not found' (HTTP 404): this computer is not logged in to GitHub,\n"
                "or the account has no access to the repository.")
    if "401" in text or "403" in text or "bad credentials" in text:
        return "GitHub rejected the login (HTTP 401/403): the saved login or token is invalid or expired."
    if any(s in text for s in ("checksum", "sha256", "hash mismatch")):
        return "A downloaded model file is corrupted (checksum mismatch), usually from an interrupted download."
    if any(s in text for s in ("timed out", "urlopen error", "name or service not known", "getaddrinfo",
                               "temporary failure in name resolution", "connection", "network")):
        return "Could not reach GitHub (network problem or firewall)."
    return "download_models.py stopped with an error (see its output just above this summary)."


def download_models(args):
    """Fetch the trained model files (only the ones that are missing)."""
    print_step("Downloading trained models (large files, needs internet access)...")
    if not os.path.isfile("download_models.py"):
        print_error("download_models.py not found")
        raise StepFailed("download_models.py is missing from the project folder.",
                         "The project folder is incomplete. Ask the team for a complete copy of the project.")

    returncode, output = run_and_capture([get_venv_executable("python"), "download_models.py"])
    if returncode != 0:
        print_error("Model download failed")
        raise StepFailed(diagnose_download_failure(output), manual_download_fix())

    print_success("Models ready")
    return "models/ ready"


RUNNERS = {
    "python": check_python_version,
    "requirements": check_requirements_file,
    "venv": create_venv,
    "deps": install_requirements,
    "models": download_models,
}


# ---- Final summary ----

def activation_instructions():
    if SYSTEM == "Windows":
        activate_cmd = (
            rf"    PowerShell: {VENV_DIR}\Scripts\Activate.ps1" "\n"
            rf"    cmd:        {VENV_DIR}\Scripts\activate"
        )
    else:
        activate_cmd = f"    source {VENV_DIR}/bin/activate"
    return (
        f"Activate the virtual environment with:\n"
        f"{activate_cmd}\n"
        f"\nThen run the pipeline, e.g.:\n"
        f"    python run_pipeline.py --dataset_name YFA_cabana --run_name demo --limit 10\n"
        f"    python run_pipeline.py --input_dir \"<folder_of_drawings>\" --run_name demo\n"
        f"\nResults: results/extracted_metadata_<run_name>.csv\n"
        f"Note: docTR downloads its OCR weights on the first run (internet needed once)."
    )


def indent(text, prefix="           "):
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def print_summary(results):
    print("\n" + "=" * 60)
    print(" Setup summary")
    print("=" * 60)

    width = max(len(label) for _, label in STEPS)
    for i, (key, label) in enumerate(STEPS, start=1):
        status, detail = results[key]
        if status == "ok":
            mark, colour, text = "✓", "32", detail
        elif status == "failed":
            mark, colour, text = "✗", "31", "FAILED"
        elif status == "interrupted":
            mark, colour, text = "!", "33", "INTERRUPTED"
        else:
            mark, colour, text = "–", "90", "not run"
        print(f" \033[{colour}m{mark}\033[0m {i}. {label:<{width}}   \033[{colour}m{text}\033[0m")

    numbered = {key: (i, label) for i, (key, label) in enumerate(STEPS, start=1)}
    failed = [k for k, _ in STEPS if results[k][0] == "failed"]
    interrupted = [k for k, _ in STEPS if results[k][0] == "interrupted"]

    for key in failed:
        i, label = numbered[key]
        err = results[key][1]
        print()
        print_error(f"Step {i} failed: {label}")
        print(f"  Problem: {err.problem.splitlines()[0]}")
        for line in err.problem.splitlines()[1:]:
            print(f"           {line}")
        print(f"  Fix:     {err.fix.splitlines()[0]}")
        rest = "\n".join(err.fix.splitlines()[1:])
        if rest:
            print(indent(rest))

    if interrupted:
        i, label = numbered[interrupted[0]]
        print()
        print(f"\033[33m! Setup was interrupted during step {i}: {label}.\033[0m")
        print(f"  Run it again: {setup_command()}  (finished steps are reused)")

    if not failed and not interrupted:
        print_step("Setup complete!")
        print(activation_instructions())
        return True

    print()
    if failed and all(k in NON_FATAL_STEPS for k in failed) and not interrupted:
        print("\033[1;33m==> The environment is ready, but the step above must be fixed before running the pipeline.\033[0m")
        print(f"More help: {README_HINT}\n")
        print(activation_instructions())
        return False

    if any(k in ("python", "venv") for k in failed):
        # The Fix text already gives the exact (version-specific) command to re-run
        print("\033[1;31m==> Setup did not finish.\033[0m Fix the problem above, then run setup again as shown under Fix.")
    else:
        print(f"\033[1;31m==> Setup did not finish.\033[0m Fix the problem above, then run setup again: {setup_command()}")
    print(f"More help: {README_HINT}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Set up the project environment.")
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Linux only: install the default (CUDA) PyTorch build instead of the CPU-only one",
    )
    args = parser.parse_args()

    # Work from the project folder even if setup.py was started from somewhere else
    os.chdir(PROJECT_ROOT)

    print("=" * 60)
    print(" Project Environment Setup")
    print("=" * 60)

    results = {key: ("not run", "") for key, _ in STEPS}
    current = None
    try:
        for key, _ in STEPS:
            current = key
            try:
                results[key] = ("ok", RUNNERS[key](args))
            except StepFailed as err:
                results[key] = ("failed", err)
                if key not in NON_FATAL_STEPS:
                    break
    except KeyboardInterrupt:
        results[current] = ("interrupted", None)
        print()

    ok = print_summary(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()