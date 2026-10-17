# Build every frictiontune pipeline image from the one cascading Containerfile.
#
#   make -C container all                     # os → base → data → train → serve + dev + llm
#   make -C container train                   # one pipeline
#   make -C container all IMAGE_PREFIX=registry.example/frictiontune
#
# The cascade is vendor-neutral: it builds for AMD (ROCm) by default and for
# NVIDIA (CUDA) by switching one variable. `VARIANT=cuda` supplies the whole
# NVIDIA argument set (OS image, torch index, llama.cpp backend), so the two
# vendors are one knob apart:
#
#   make -C container all VARIANT=cuda
#   make -C container all VARIANT=cuda CUDA_ARCH=89
#
# Every other build ARG in the Containerfile can be overridden on the command
# line too:
#
#   make -C container train TORCH_VERSION=2.10.0+rocm7.0
#   make -C container llm   GGML_EXTRA_ARGS="-DAMDGPU_TARGETS=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm"

ENGINE        ?= podman
FILE          ?= Containerfile
CONTEXT       ?= ..
IMAGE_PREFIX  ?= frictiontune

# Vendor switch: rocm (default) or cuda.
VARIANT       ?= rocm
# CUDA compute capability for the llama.cpp build (e.g. 89 = Ada, 90 = Hopper,
# 120 = Blackwell). A `;`-separated list covers a mixed box.
CUDA_ARCH     ?= 89

# NVIDIA argument set. ROCm is the Containerfile default, so only cuda needs a set.
CUDA_ARGS = \
	--build-arg VARIANT=cuda \
	--build-arg OS_IMAGE=nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04 \
	--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 \
	--build-arg TORCH_VERSION=2.10.0 \
	--build-arg TORCH_EXTRA_PACKAGES= \
	--build-arg GGML_BACKEND=CUDA \
	--build-arg GGML_EXTRA_ARGS="-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)"
VENDOR_ARGS = $(if $(filter cuda,$(VARIANT)),$(CUDA_ARGS),)

# Build args forwarded verbatim to the engine. Append NAME=value to extend.
ARGS ?=
BUILD_ARGS = $(VENDOR_ARGS) $(ARGS)

.PHONY: all os rocm base data train serve dev llm pod pod-down clean

all: os base data train serve dev llm

# `rocm` is an alias for the shared OS layer kept for muscle memory.
os rocm:
	$(ENGINE) build -f $(FILE) --target os -t $(IMAGE_PREFIX):os $(BUILD_ARGS) $(CONTEXT)

base: os
	$(ENGINE) build -f $(FILE) --target base -t $(IMAGE_PREFIX):base $(BUILD_ARGS) $(CONTEXT)

data: base
	$(ENGINE) build -f $(FILE) --target data -t $(IMAGE_PREFIX):data $(BUILD_ARGS) $(CONTEXT)

train: data
	$(ENGINE) build -f $(FILE) --target train -t $(IMAGE_PREFIX):train $(BUILD_ARGS) $(CONTEXT)

serve: train
	$(ENGINE) build -f $(FILE) --target serve -t $(IMAGE_PREFIX):serve $(BUILD_ARGS) $(CONTEXT)

dev: train
	$(ENGINE) build -f $(FILE) --target dev -t $(IMAGE_PREFIX):dev $(BUILD_ARGS) $(CONTEXT)

llm: os
	$(ENGINE) build -f $(FILE) --target llm -t $(IMAGE_PREFIX):llm $(BUILD_ARGS) $(CONTEXT)

# Render the declarative pod (its only machine-specific value is the repo path)
# and hand it to kube play.
pod: data
	FT_REPO=$(realpath $(CONTEXT)) envsubst < pod.yaml | $(ENGINE) kube play -

pod-down:
	FT_REPO=$(realpath $(CONTEXT)) envsubst < pod.yaml | $(ENGINE) kube down -

clean:
	-$(ENGINE) rmi $(IMAGE_PREFIX):os $(IMAGE_PREFIX):base $(IMAGE_PREFIX):data \
	    $(IMAGE_PREFIX):train $(IMAGE_PREFIX):serve $(IMAGE_PREFIX):dev $(IMAGE_PREFIX):llm