# Trivikrama - PC build and tests
CC      ?= gcc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror
INC      = -Icore/include
SRC      = core/src/trv_kernels.c core/src/trv_linear.c core/src/trv_sources.c \
           core/src/trv_model.c core/src/trv_source_file.c \
           core/src/trv_modernbert.c core/src/trv_laya.c \
           core/src/trv_laya_text.c
HDR      = $(wildcard core/include/*.h)

.PHONY: all test test-kernels test-features test-export test-modernbert test-laya test-tokenizer virtual-mcu clean

all: build/test_kernels build/trv_run build/trv_laya build/laya_device build/trv_tokenize \
     build/laya_cli

build/test_kernels: $(SRC) $(HDR) tests/test_kernels.c
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) tests/test_kernels.c -lm

build/trv_run: $(SRC) $(HDR) core/src/trv_posix.c tools/trv_run.c
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) core/src/trv_posix.c tools/trv_run.c -lm -lpthread

build/trv_laya: $(SRC) $(HDR) core/src/trv_posix.c tools/trv_laya.c
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) core/src/trv_posix.c tools/trv_laya.c -lm -lpthread

build/laya_device: $(SRC) $(HDR) examples/laya_device/laya_device.c
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) examples/laya_device/laya_device.c -lm

build/trv_tokenize: $(SRC) $(HDR) tools/trv_tokenize.c
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) tools/trv_tokenize.c -lm

build/laya_cli: $(SRC) $(HDR) examples/laya_cli/laya_cli.c
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) examples/laya_cli/laya_cli.c -lm

test-kernels: build/test_kernels
	PYTHONPATH=python python3 tests/check_kernels.py build/test_kernels

test-features: build/trv_run
	PYTHONPATH=python python3 tests/check_features.py build/trv_run

# needs: pip install safetensors
test-export: build/trv_run
	PYTHONPATH=python python3 tests/check_export.py build/trv_run

# needs: pip install torch transformers
test-modernbert: build/trv_run
	PYTHONPATH=python python3 tests/check_modernbert.py build/trv_run

test-laya: build/trv_run build/trv_laya build/laya_device build/laya_cli
	PYTHONPATH=python python3 tests/check_laya.py build/trv_laya build/trv_run

test-tokenizer: build/trv_tokenize
	PYTHONPATH=python python3 tests/check_tokenizer.py build/trv_tokenize

test: test-tokenizer test-kernels test-features test-export test-laya test-modernbert

# usage: make virtual-mcu MODEL=model.trv
virtual-mcu: build/trv_run
	PYTHONPATH=python python3 tools/virtual_mcu.py build/trv_run $(MODEL)

clean:
	rm -rf build
