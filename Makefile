CC = clang -fsanitize=address -g0
CFLAGS = -Iinclude -Wall -Wextra -std=c11
SRC = src/main.c src/tensor_utils.c src/tensor_init.c
OUT = main

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)
