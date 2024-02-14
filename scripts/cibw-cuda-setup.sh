#!/bin/bash

# Set CUDA version and architecture
CU_VER=${1//./-}
ARCH="x86_64"

# Install CUDA compiler and libraries
yum install -y yum-utils
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/${ARCH}/cuda-rhel7.repo
yum -y install cuda-toolkit-${CU_VER}.${ARCH} \
              nvidia-driver-latest-dkms

# Clean up YUM caches
yum clean all
rm -rf /var/cache/yum/*

# Configure dynamic linker run-time bindings
echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/999_nvidia_cuda.conf

# Set environment variables
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CUDADIR=/usr/local/cuda