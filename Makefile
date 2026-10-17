# Trivikrama - PC build and tests
CC      ?= gcc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror
INC      = -Icore/include
SRC      = core/src/trv_kernels.c core/src/trv_model.c core/src/trv_source_file.c
HDR      = core/include/trivikrama.h core/include/trivikrama_file.h

.PHONY: all test clean

all: build/test_kernels

build/test_kernels: $(SRC) $(HDR) tests/test_kernels.c
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) tests/test_kernels.c -lm

test: build/test_kernels
	PYTHONPATH=python python3 tests/check_kernels.py build/test_kernels

clean:
	rm -rf build
