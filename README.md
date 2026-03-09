# Model2Kernel: Model-Aware Symbolic Execution For Safe CUDA Kernels
Model2Kernel combines model-aware dynamic analysis and CUDA-specialized symbolic execution to identify memory bugs in kernels.

## Overview

## Architecture

## Project Structure
```text
Model2Kernel/
├── data/                               # cuda files and compiled llvm files
├── exp/                                # input files in the experiments
├── gklee/                              # docker files to run GKLEE
├── honeycomb/                          # docker files to run Honeycomb
├── klee/                               # implementation of cuKLEE
├── HFProbe/                            # implementation of HFProbe
├── scripts/                            # scripts to compile, run
├── src/                                # entry of cuKLEE
└── README.md
```

## Quick Start
### Prerequistes
- OS: Linux
- Python >= 3.9
- Docker
- Z3
- Git

#### Prerequisites Setup
```bash
$ sudo apt update

# install Git
$ sudo apt install -y git 

# install z3
$ sudo apt install -y z3

# install docker
$ sudo apt install -y ca-certificates curl
$ sudo install -m 0755 -d /etc/apt/keyrings
$ sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
$  sudo chmod a+r /etc/apt/keyrings/docker.asc
$ sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF
$ sudo apt update
$ sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# install python
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt update
$ sudo apt install -y python3.12
$ sudo apt install -y python3-pip
```

### Project Setup
```bash
$ git clone https://github.com/Model2Kernel/Model2Kernel.git
$ mkdir build 
$ cd build 
$ cmake ..
$ make -j8 
$ export PATH="<Model2KernelPath>/build/bin:$PATH"
```

#### Run cuKLEE Only
```bash
# run on a compiled llvm bc file
$ python3 scripts/run_cuKLEE.py --bc <file.bc> --out-dir <output_path> --threads <number_of_threads>
# run on a json file containing bc filepath and parameter constraints
$ python3 scripts/run_cuKLEE.py --json <file.json> --out-dir <output_path> --threads <number_of_threads>
# run on a directory of bc files
$ python3 scripts/run_cuKLEE.py --bc-dir <directory_path> --out-dir <output_path> --threads <number_of_threads>
# run on a directory of json files
$ python3 scripts/run_cuKLEE.py --json-dir <directory_path> --out-dir <output_path> --threads <number_of_threads>
# --out-dir and --threads are optional. Default outdir is scripts/out. Default number of threads is 5.
```
example:
```bash
$ python3 scripts/run_cuKLEE.py --json exp/vllm/input/fused_add_rms_norm.json
```

