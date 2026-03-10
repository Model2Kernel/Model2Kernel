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
RUN apt install -y z3

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add -
RUN apt-get install -y clang-13 llvm-13 llvm-13-dev llvm-13-tools

ENV LLVM_DIR=/usr/lib/llvm-13
ENV PATH="$LLVM_DIR/bin:$PATH"

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN apt install -y wget gnupg2 curl lsb-release
RUN apt install -y nano
