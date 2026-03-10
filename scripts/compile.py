import os
import subprocess
from pathlib import Path
import sys

# Paths configuration
CUDA_PATH = "/usr/local/cuda"  # Path to CUDA installation
COMBINED_SUFFIX = "_combined.bc"  # Suffix for the combined .bc files

# Manually define PyTorch include paths
try:
    import torch
    TORCH_INCLUDE = os.path.join(os.path.dirname(torch.__file__), "include")
    print(TORCH_INCLUDE)
    if not os.path.exists(TORCH_INCLUDE):
        raise FileNotFoundError(f"PyTorch include directory not found at {TORCH_INCLUDE}")
except ImportError:
    print("PyTorch is not installed. Please install it and try again.")
    sys.exit(1)


# Utility function to run shell commands
def run_command(command, cwd=None):
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        return False
    else:
        print(result.stdout)
        return True


def compile_cu_file(cu_file, depPath):
    print(f"Compiling {cu_file}...")

    clang_command = [
        "clang++-13",
        # "-g", # cause broken module
        # "-O0",
        # "-disable-O0-optnone",
        "-x", "cuda", 
        "--cuda-gpu-arch=sm_80",  
        "-std=c++17",
        # "-DENABLE_BF16",
        "-Xclang",
        "-fcuda-allow-variadic-functions",
        "-D__CUDA_ARCH__=800",
        "-I", f"{CUDA_PATH}/include",
        "-I", TORCH_INCLUDE,
        "-I", TORCH_INCLUDE + "/torch/csrc/api/include",
        "-I", depPath,
        "-I", "/usr/include/python3.10",
        "-I", "/opt/rocm/include",
        "-I", "./include",
        "-I", "./include/cutlass",
        "-I", "./include/cutlass/examples",
    ]

    if "nvfp4" in cu_file:
        clang_command.append("-DENABLE_NVFP4=1")
        clang_command.append("-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1")
    
    if "cutlass_mla_entry" in cu_file:
        clang_command.append("-DENABLE_CUTLASS_MLA=1")
    
    if "sparse_scaled_mm_entry" in cu_file or "sparse_scaled_mm_c3x" in cu_file:
        clang_command.append("-DENABLE_SPARSE_SCALED_MM_C3X=1")
        clang_command.append("-DCUDA_VERSION=12020")
    
    # if "cutlass_w8a8" in cu_file:
    #     clang_command.append("--cuda-gpu-arch=sm_80")
    #     clang_command.append("-DENABLE_CUTLASS_MOE_SM90=1")
    #     clang_command.append("-DENABLE_SCALED_MM_SM90=1")
    # else:
    #     clang_command.append("--cuda-gpu-arch=sm_80")
    
    clang_command.extend(["-emit-llvm",
        "-c",
        cu_file])
    
    if "rocm" in cu_file:
        clang_command = [
            "clang++-13",
            "-x", "hip",
            "--offload-arch=gfx90a",  
            "-std=c++17",
            "-D__HIPCC__",
            "-D__gfx942__",
            "-I", f"{CUDA_PATH}/include",
            "-I", TORCH_INCLUDE,
            "-I", TORCH_INCLUDE + "/torch/csrc/api/include",
            "-I", depPath,
            "-I", "/usr/include/python3.10",
            "-I", "/opt/rocm/include",
            "-I", "./include",
            "-I", "./include/cutlass",
            "-I", "./include/cutlass/examples",
            "-emit-llvm",
            "-c",
            cu_file
        ]
    
    return run_command(clang_command)


def link_combine(dir):
    for file in os.listdir(dir):
        if file.endswith("cuda-nvptx64-nvidia-cuda-sm_80.bc") or file.endswith("combined.bc"):
            continue
        if not file.endswith(".bc"):
            continue
        
        output_prefix = dir+"/"+file[:file.find(".bc")]
        host_bc_file = f"{output_prefix}.bc"
        cuda_bc_file = f"{output_prefix}-cuda-nvptx64-nvidia-cuda-sm_80.bc"
        combined_bc_file = f"{output_prefix}{COMBINED_SUFFIX}"

        if os.path.exists(combined_bc_file):
            continue

        if os.path.exists(host_bc_file) and os.path.exists(cuda_bc_file):
            print(f"Combining {host_bc_file} and {cuda_bc_file} into {combined_bc_file}...")
            run_command([
                "llvm-link-13",
                "-o", combined_bc_file,
                host_bc_file,
                cuda_bc_file
            ])
            run_command([
                "llvm-dis-13",
                combined_bc_file
            ])
        else:
            print(f"Error: Missing {host_bc_file} or {cuda_bc_file} file. Skipping combination.")

def compileDir(outputDir, inputDir):
    print("Compiling CUDA files...")
    cu_files = list(Path(inputDir).resolve().rglob("*.cu"))
    if not cu_files:
        print("No .cu files found in the source directory.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(outputDir, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(outputDir)
    
    failed_files = []

    for cu_file in cu_files:
        filename = cu_file.stem
        output_prefix = filename

        host_bc_file = f"{output_prefix}.bc"
        cuda_bc_file = f"{output_prefix}-cuda-nvptx64-nvidia-cuda-sm_80.bc"
        combined_bc_file = f"{output_prefix}{COMBINED_SUFFIX}"

        if not os.path.exists(host_bc_file):
            if not compile_cu_file(str(cu_file), inputDir):
                failed_files.append(str(cu_file))
                continue

        if os.path.exists(host_bc_file) and os.path.exists(cuda_bc_file):
            print(f"Combining {host_bc_file} and {cuda_bc_file} into {combined_bc_file}...")
            run_command([
                "llvm-link-13",
                "-o", combined_bc_file,
                host_bc_file,
                cuda_bc_file
            ])
            run_command([
                "llvm-dis-13",
                combined_bc_file
            ])
        else:
            print(f"Error: Missing .bc or -cuda.bc file for {cu_file}. Skipping combination.")

    if failed_files:
        print(failed_files)
    print("# of .cu:", len(cu_files), "# of failed:", len(failed_files))
    os.chdir(original_dir)
