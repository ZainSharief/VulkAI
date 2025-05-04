CC = clang -fsanitize=address -g0
CFLAGS = -Iinclude/VulkML/tensor -Isrc -Iexternal/vulkan/include -Wall -Wextra -std=c11
SRC = $(shell find src -name '*.c')
TESTS = tests/tensor_init_test.c tests/lib_vulkan_test.c
OUT = main

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
	LDFLAGS = -Lexternal/vulkan/lib/macOS -lMoltenVK -Wl,-rpath,@loader_path/external/vulkan/lib/macOS 
endif
ifeq ($(UNAME_S), Linux)
	LDFLAGS = -Lexternal/vulkan/lib/linux -lvulkan -Wl,-rpath,\$$ORIGIN/external/vulkan/lib/linux
endif
ifeq ($(OS), Windows_NT)
	LDFLAGS = -Lexternal/vulkan/lib/windows -lvulkan-1
endif

all:
	$(CC) $(CFLAGS) $(SRC) $(TESTS) -o $(OUT) $(LDFLAGS)
