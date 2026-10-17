# Trivikrama - PC build and tests
CC      ?= gcc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror
INC      = -Icore/include
SRC      = core/src/trv_kernels.c core/src/trv_linear.c core/src/trv_sources.c \
           core/src/trv_model.c core/src/trv_source_file.c \
           core/src/trv_modernbert.c
HDR      = $(wildcard core/include/*.h)

.PHONY: all test test-kernels test-features test-modernbert virtual-mcu clean

all: build/test_kernels build/trv_run

build/test_kernels: $(SRC) $(HDR) tests/test_kernels.c
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) tests/test_kernels.c -lm

build/trv_run: $(SRC) $(HDR) core/src/trv_posix.c tools/trv_run.c
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) core/src/trv_posix.c tools/trv_run.c -lm -lpthread

test-kernels: build/test_kernels
	PYTHONPATH=python python3 tests/check_kernels.py build/test_kernels

test-features: build/trv_run
	PYTHONPATH=python python3 tests/check_features.py build/trv_run

# needs: pip install torch transformers
test-modernbert: build/trv_run
	PYTHONPATH=python python3 tests/check_modernbert.py build/trv_run

test: test-kernels test-features test-modernbert

# usage: make virtual-mcu MODEL=model.trv
virtual-mcu: build/trv_run
	PYTHONPATH=python python3 tools/virtual_mcu.py build/trv_run $(MODEL)

clean:
	rm -rf build
