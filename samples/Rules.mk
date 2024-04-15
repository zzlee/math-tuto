# # Clear the flags from env
# CXXFLAGS :=
# LDFLAGS :=

# Verbose flag
ifeq ($(V), 1)
AT =
else
AT = @
endif

# TARGET_ROOTFS ?= /usr/targets/aarch64-unknown/
# CROSS_COMPILE ?= aarch64-unknown-linux-gnu-

AS             = $(AT) $(CROSS_COMPILE)as
LD             = $(AT) $(CROSS_COMPILE)ld
CC             = $(AT) $(CROSS_COMPILE)gcc
CXX            = $(AT) $(CROSS_COMPILE)g++
AR             = $(AT) $(CROSS_COMPILE)ar
NM             = $(AT) $(CROSS_COMPILE)nm
STRIP          = $(AT) $(CROSS_COMPILE)strip
OBJCOPY        = $(AT) $(CROSS_COMPILE)objcopy
OBJDUMP        = $(AT) $(CROSS_COMPILE)objdump
NVCC           = $(AT) /usr/local/cuda/bin/nvcc

# Specify the logical root directory for headers and libraries.
ifneq ($(TARGET_ROOTFS),)
CXXFLAGS += --sysroot=$(TARGET_ROOTFS)
CFLAGS += --sysroot=$(TARGET_ROOTFS)
LDFLAGS +=
endif

# CUDA code generation flags
GENCODE_SM53=-gencode arch=compute_53,code=sm_53
GENCODE_SM62=-gencode arch=compute_62,code=sm_62
GENCODE_SM72=-gencode arch=compute_72,code=sm_72
# GENCODE_SM87=-gencode arch=compute_87,code=sm_87
GENCODE_SM_PTX=-gencode arch=compute_72,code=compute_72
GENCODE_FLAGS=$(GENCODE_SM53) $(GENCODE_SM62) $(GENCODE_SM72) $(GENCODE_SM87) $(GENCODE_SM_PTX)

CXXFLAGS += \
-Ofast
CFLAGS += \
-Ofast

# All common header files
CXXFLAGS += -std=c++11 \
-I../common
CFLAGS += \
-I../common

# All common dependent libraries
LDFLAGS += \
-pthread

COMMON_DIR := \
../common
