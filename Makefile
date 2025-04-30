CC = clang -fsanitize=address -g0
CFLAGS = -Iinclude -Isrc -I${VULKAN_SDK}/include -Wall -Wextra -std=c11
SRC = src/tensor_utils.c src/tensor_init.c
TESTS = tests/tensor_init_test.c
OUT = main

all:
	$(CC) $(CFLAGS) $(SRC) $(TESTS) -o $(OUT)
