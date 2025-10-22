#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# ScriptBots cross-compilation bootstrap script for fresh Ubuntu machines.
# This provisions toolchains for:
#   - Windows (x86_64-pc-windows-gnullvm) via llvm-mingw + Wine-based FXC
#   - macOS (x86_64/aarch64-apple-darwin) scaffolding via osxcross (SDK needed)
#
# Usage:
#   chmod +x script_for_setting_up_ubuntu_for_cross_compilation.sh
#   ./script_for_setting_up_ubuntu_for_cross_compilation.sh
# Environment overrides:
#   TOOLCHAIN_ROOT   : directory for downloaded toolchains (default: ./toolchains)
#   LLVM_MINGW_VER   : llvm-mingw release tag          (default: 20251021)
#   WINSDK_ISO_URL   : Windows SDK ISO download URL    (default: 26100.4948...)
#   RUST_TOOLCHAIN   : rustup toolchain to target      (default: stable)
###############################################################################

###############################
# Logging helpers
###############################
log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

###############################
# Defaults & directories
###############################
ROOT_DIR="$(pwd)"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-${ROOT_DIR}/toolchains}"
WINDOWS_DIR="${TOOLCHAIN_ROOT}/windows"
LLVM_MINGW_VER="${LLVM_MINGW_VER:-20251021}"
LLVM_MINGW_ARCHIVE="llvm-mingw-${LLVM_MINGW_VER}-ucrt-ubuntu-22.04-x86_64"
LLVM_MINGW_URL="https://github.com/mstorsjo/llvm-mingw/releases/download/${LLVM_MINGW_VER}/${LLVM_MINGW_ARCHIVE}.tar.xz"
LLVM_MINGW_DEST="${TOOLCHAIN_ROOT}/${LLVM_MINGW_ARCHIVE}"

WINSDK_ISO_URL="${WINSDK_ISO_URL:-https://download.microsoft.com/download/3a857edb-459d-4fbb-88dc-5153f6183142/26100.4948.250812-1640.ge_release_svc_im_WindowsSDK.iso}"
WINSDK_ISO_PATH="${WINDOWS_DIR}/WindowsSDK.iso"
WINSDK_EXTRACT_DIR="${WINDOWS_DIR}/extracted"
FXC_INSTALLER_NAME="Windows SDK for Windows Store Apps Tools-x86_en-us.msi"

OSXCROSS_DIR="${TOOLCHAIN_ROOT}/osxcross"
OSXCROSS_TARBALL_DIR="${OSXCROSS_DIR}/tarballs"

RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-stable}"

###############################
# Command availability helpers
###############################
require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        err "Missing required command '$1'. Install it first."
    fi
}

require_root_or_sudo() {
    if [[ $EUID -ne 0 ]]; then
        if ! command -v sudo >/dev/null 2>&1; then
            err "Script needs sudo privileges for apt operations. Install sudo or run as root."
        fi
    fi
}

run_apt() {
    if [[ $EUID -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

###############################
# System preparation
###############################
ensure_packages() {
    local packages=(
        build-essential clang mold cmake ninja-build pkg-config curl git unzip xz-utils
        p7zip-full python3 python3-venv python3-pip
        msitools wine wine64 wine32
    )
    log "Updating apt repositories…"
    run_apt apt-get update -y
    log "Installing prerequisite packages…"
    run_apt DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
}

enable_multiarch() {
    if ! dpkg --print-foreign-architectures | grep -q '^i386$'; then
        log "Enabling i386 multiarch support (required for Wine)…"
        run_apt dpkg --add-architecture i386
        run_apt apt-get update -y
    fi
}

###############################
# Rust target configuration
###############################
ensure_rust_targets() {
    if ! command -v rustup >/dev/null 2>&1; then
        warn "rustup not found. Install Rust toolchain before rerunning this script."
        return
    fi
    log "Ensuring rustup toolchain '${RUST_TOOLCHAIN}' is installed…"
    rustup toolchain install "${RUST_TOOLCHAIN}" >/dev/null

    local targets=(
        x86_64-pc-windows-gnullvm
        x86_64-pc-windows-gnu
        x86_64-apple-darwin
        aarch64-apple-darwin
    )
    for target in "${targets[@]}"; do
        log "Adding Rust target ${target}…"
        rustup target add "${target}" --toolchain "${RUST_TOOLCHAIN}" >/dev/null
    done
}

###############################
# llvm-mingw toolchain
###############################
install_llvm_mingw() {
    if [[ -d "${LLVM_MINGW_DEST}" ]]; then
        log "llvm-mingw already present at ${LLVM_MINGW_DEST}"
        return
    fi
    mkdir -p "${TOOLCHAIN_ROOT}"
    log "Downloading llvm-mingw release ${LLVM_MINGW_VER}…"
    curl -L "${LLVM_MINGW_URL}" -o "${TOOLCHAIN_ROOT}/${LLVM_MINGW_ARCHIVE}.tar.xz"
    log "Extracting llvm-mingw archive…"
    tar -xf "${TOOLCHAIN_ROOT}/${LLVM_MINGW_ARCHIVE}.tar.xz" -C "${TOOLCHAIN_ROOT}"
}

###############################
# Windows SDK & FXC extraction
###############################
download_windows_sdk_iso() {
    mkdir -p "${WINDOWS_DIR}"
    if [[ ! -f "${WINSDK_ISO_PATH}" ]]; then
        log "Downloading Windows 11 SDK ISO…"
        curl -L "${WINSDK_ISO_URL}" -o "${WINSDK_ISO_PATH}"
    else
        log "Windows SDK ISO already present at ${WINSDK_ISO_PATH}"
    fi
}

extract_windows_sdk() {
    if [[ -d "${WINSDK_EXTRACT_DIR}" ]]; then
        log "Windows SDK ISO already extracted."
        return
    fi
    log "Extracting Windows SDK ISO with 7z…"
    mkdir -p "${WINSDK_EXTRACT_DIR}"
    7z x "${WINSDK_ISO_PATH}" -o"${WINSDK_EXTRACT_DIR}" >/dev/null
}

extract_fxc() {
    local installer_path="${WINSDK_EXTRACT_DIR}/Installers/${FXC_INSTALLER_NAME}"
    local target_bin="${WINDOWS_DIR}/bin"
    local out_dir="${WINDOWS_DIR}/fxc_extract"

    if [[ ! -f "${installer_path}" ]]; then
        err "Cannot find ${FXC_INSTALLER_NAME} inside extracted ISO. ISO format may have changed."
    fi

    if [[ -f "${target_bin}/fxc.exe" ]]; then
        log "fxc.exe already extracted."
        return
    fi

    log "Extracting FXC and Direct3D redistributables from ${FXC_INSTALLER_NAME}…"
    mkdir -p "${out_dir}"
    msiextract --directory "${out_dir}" "${installer_path}" >/dev/null

    mkdir -p "${target_bin}"
    cp "${out_dir}/Program Files/Windows Kits/10/bin/"*/x64/fxc.exe "${target_bin}/fxc-x64.exe"
    cp "${out_dir}/Program Files/Windows Kits/10/bin/"*/x86/fxc.exe "${target_bin}/fxc-x86.exe"
    cp "${target_bin}/fxc-x64.exe" "${target_bin}/fxc.exe"
    log "FXC binaries copied into ${target_bin}"
}

create_fxc_wrapper() {
    local wrapper="${WINDOWS_DIR}/bin/fxc-wrapper"
    cat > "${wrapper}" <<'PY'
#!/usr/bin/env python3
import os
import subprocess
import sys

FXC_EXE = os.path.join(os.path.dirname(__file__), "fxc.exe")
PATH_OPTIONS = {
    "fi", "fo", "fh", "fl", "fd", "fe", "fx", "p",
    "setrootsignature", "extractrootsignature", "verifyrootsignature"
}

def to_windows_path(path: str) -> str:
    result = subprocess.run(
        ["winepath", "-w", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        text=True,
    )
    return result.stdout.strip()

def needs_conversion(arg: str) -> bool:
    if not arg:
        return False
    if arg.startswith('-'):
        return False
    if arg.startswith('/'):
        # POSIX absolute path
        return True
    if os.path.isabs(arg) or os.path.sep in arg:
        return True
    return False

def is_flag(token: str) -> bool:
    if not token:
        return False
    if token.startswith(('--', '-')):
        return True
    if token.startswith('/') and '/' not in token[1:] and '\\' not in token:
        return True
    return False

def main() -> int:
    args = sys.argv[1:]
    converted = []
    i = 0
    while i < len(args):
        current = args[i]
        option_key = ""
        if is_flag(current):
            option_key = current.lstrip('/-').lower()
            converted.append(current)
            if option_key in PATH_OPTIONS and i + 1 < len(args):
                i += 1
                path_arg = args[i]
                abs_path = os.path.abspath(path_arg)
                converted.append(to_windows_path(abs_path))
            i += 1
            continue

        value = current
        if needs_conversion(value):
            value = to_windows_path(os.path.abspath(value))
        converted.append(value)
        i += 1

    env = os.environ.copy()
    env.setdefault('WINEDEBUG', '-all')
    process = subprocess.run(["wine", FXC_EXE, *converted], env=env)
    return process.returncode

if __name__ == "__main__":
    sys.exit(main())
PY
    chmod +x "${wrapper}"
    log "Created FXC wrapper at ${wrapper}"
}

###############################
# .cargo configuration tweaks
###############################
update_cargo_config() {
    local cargo_dir="${ROOT_DIR}/.cargo"
    local cargo_cfg="${cargo_dir}/config.toml"
    mkdir -p "${cargo_dir}"

    local llvm_prefix="${LLVM_MINGW_DEST}"
    local win_bin="${WINDOWS_DIR}/bin"

    local snippet
    read -r -d '' snippet <<EOF || true
[env]
CC_x86_64_pc_windows_gnullvm = "${llvm_prefix}/bin/x86_64-w64-mingw32-clang"
CXX_x86_64_pc_windows_gnullvm = "${llvm_prefix}/bin/x86_64-w64-mingw32-clang++"
AR_x86_64_pc_windows_gnullvm = "${llvm_prefix}/bin/llvm-ar"
CXXFLAGS_x86_64_pc_windows_gnullvm = "-isystem ${llvm_prefix}/x86_64-w64-mingw32/include/c++/v1"
GPUI_FXC_PATH = "${win_bin}/fxc-wrapper"

[target.x86_64-pc-windows-gnullvm]
linker = "${llvm_prefix}/bin/x86_64-w64-mingw32-clang"
ar = "${llvm_prefix}/bin/llvm-ar"

[target.x86_64-apple-darwin]
linker = "${OSXCROSS_DIR}/target/bin/o64-clang"
ar = "${OSXCROSS_DIR}/target/bin/llvm-ar"

[target.aarch64-apple-darwin]
linker = "${OSXCROSS_DIR}/target/bin/oa64-clang"
ar = "${OSXCROSS_DIR}/target/bin/llvm-ar"
EOF

    if [[ -f "${cargo_cfg}" ]] && grep -q "GPUI_FXC_PATH" "${cargo_cfg}"; then
        warn "${cargo_cfg} already contains cross-compilation settings; skipping automatic update."
        return
    fi

    log "Writing cross-compilation settings to ${cargo_cfg}"
    printf '%s\n' "${snippet}" >> "${cargo_cfg}"
}

###############################
# osxcross scaffolding
###############################
prepare_osxcross() {
    if [[ -d "${OSXCROSS_DIR}" ]]; then
        log "osxcross repository already exists."
    else
        log "Cloning osxcross into ${OSXCROSS_DIR}…"
        git clone --depth 1 https://github.com/tpoechtrager/osxcross.git "${OSXCROSS_DIR}"
    fi
    mkdir -p "${OSXCROSS_TARBALL_DIR}"
    cat <<'INFO'
------------------------------------------------------------------------------
NEXT STEP (manual): place your macOS SDK tarball inside:
    ${OSXCROSS_DIR}/tarballs/
Recommended naming: MacOSX<version>.sdk.tar.xz (or .tar.gz)

Once the SDK is available, run:
    cd ${OSXCROSS_DIR}
    SDK_VERSION=<version> ./build.sh

Finally, export:
    export OSXCROSS_ROOT=${OSXCROSS_DIR}
    export PATH="$OSXCROSS_ROOT/target/bin:$PATH"
------------------------------------------------------------------------------
INFO
}

###############################
# Main routine
###############################
main() {
    require_cmd curl
    require_cmd git
    require_cmd tar
    require_root_or_sudo

    enable_multiarch
    ensure_packages
    ensure_rust_targets
    install_llvm_mingw
    download_windows_sdk_iso
    extract_windows_sdk
    extract_fxc
    create_fxc_wrapper
    update_cargo_config
    prepare_osxcross

    log "Bootstrap complete!"
    cat <<'SUMMARY'
==============================================================================
Cross-compilation toolchains staged. Remaining manual tasks:
  1. Add macOS SDK tarball to osxcross/tarballs and run its build.sh.
  2. Consider adding toolchains/windows/ and toolchains/osxcross/ to .gitignore.
  3. Open a new shell or source ~/.bashrc to use the updated PATH if modified.
==============================================================================
SUMMARY
}

main "$@"
