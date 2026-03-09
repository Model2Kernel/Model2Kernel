FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set timezone environment variable
ENV TZ=America/New_York

# Set DEBIAN_FRONTEND to noninteractive to avoid user prompts during apt-get install
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

# Configure timezone and clean up
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean

RUN apt-get update && apt-get -y install g++ \
	gcc \
	cmake \
	git \
	ninja-build \
	wget \
	curl \
	lld \
	python3 \
	python3-pip \
	software-properties-common \
	lldb-13 \
	gdb

RUN add-apt-repository universe

RUN apt-get update && apt-get install -y file g++-multilib gcc-multilib libcap-dev libgoogle-perftools-dev libncurses5-dev libsqlite3-dev libtcmalloc-minimal4 unzip graphviz doxygen
RUN pip3 install lit wllvm
RUN apt-get install -y python3-tabulate 
# RUN apt-get install pipx
# RUN pipx install lit wllvm

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add -
RUN apt-get install -y clang-13 llvm-13 llvm-13-dev llvm-13-tools

ENV LLVM_DIR=/usr/lib/llvm-13
ENV PATH="$LLVM_DIR/bin:$PATH"

WORKDIR '/home'
RUN [ ! -d /home/z3 ] && git clone https://github.com/Z3Prover/z3.git
WORKDIR '/home/z3'
RUN python3 scripts/mk_make.py
WORKDIR '/home/z3/build'
RUN make
RUN make install

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN apt install -y wget gnupg2 curl lsb-release
RUN apt install -y nano
# RUN nano /etc/apt/sources.list.d/rocm.list
# deb [arch=amd64 trusted=yes] https://repo.radeon.com/rocm/apt/debian jammy main
# RUN curl -sSL https://repo.radeon.com/rocm/rocm.asc | tee /etc/apt/trusted.gpg.d/rocm.asc
# RUN apt update
# RUN apt install -y hip-base

WORKDIR '/home/klee_cuda'
# RUN [ ! -d /home/klee_cuda/klee ] && git clone https://github.com/klee/klee.git && cd /home/klee_cuda/klee && git checkout 3.1.x
# WORKDIR '/home/klee_cuda/klee'
# RUN LLVM_VERSION=13 BASE=/usr/local/lib/libc++ ENABLE_OPTIMIZED=1 DISABLE_ASSERTIONS=1 ENABLE_DEBUG=0 REQUIRES_RTTI=1 scripts/build/build.sh libcxx

# RUN test -d /home/klee_cuda/klee/build && rm -r /home/klee_cuda/klee/build || echo "klee/build does not exist"
# RUN mkdir build
# WORKDIR '/home/klee_cuda/klee/build'
# RUN cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_KLEE_LIBCXX=true -DENABLE_SOLVER_Z3=ON -D CMAKE_SYSTEM_PROCESSOR=x86_64
# RUN make 

ENV KLEE_DIR=/home/klee_cuda/klee/build
ENV PATH="$KLEE_DIR/bin:$PATH"

# ENV LD_LIBRARY_PATH=/usr/lib/llvm-13/lib:$LD_LIBRARY_PATH

# RUN git clone https://github.com/llvm/llvm-project.git
# WORKDIR '/home/llvm-project'
# RUN git checkout release/19.x
# RUN cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang"
# RUN ninja -C build install