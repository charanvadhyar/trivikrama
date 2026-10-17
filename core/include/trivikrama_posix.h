/*
 * POSIX helpers for PC builds: a thread pool for the multi-core hook, and a
 * file source that can read asynchronously and simulate SD-card speed.
 * Not needed on a microcontroller; there you plug in your own versions.
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TRIVIKRAMA_POSIX_H
#define TRIVIKRAMA_POSIX_H

#include "trivikrama.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- thread pool -> trv_runtime.parallel ---------------------------------- */

typedef struct trv_pool trv_pool;

trv_pool *trv_pool_create(uint32_t n_workers);
void trv_pool_destroy(trv_pool *pool);
/* fill rt->parallel / parallel_ctx / n_workers from the pool */
void trv_pool_runtime(trv_pool *pool, trv_runtime *rt);

/* ---- file source with optional async reads and SD-speed simulation --------- */

typedef struct trv_filex trv_filex;

/*
 *   async          1: provide read_async/wait (one I/O thread, like an SD card
 *                  with DMA working next to the CPU)
 *   bytes_per_s    0 for full speed, else sleep to mimic a slower card
 *   overhead_s     fixed cost added to every read (command + seek)
 */
trv_filex *trv_filex_open(trv_source *src, const char *path, int async,
                          double bytes_per_s, double overhead_s);
void trv_filex_close(trv_filex *fx);
uint64_t trv_filex_bytes_read(const trv_filex *fx);
uint32_t trv_filex_reads(const trv_filex *fx);
uint64_t trv_filex_size(const trv_filex *fx);

#ifdef __cplusplus
}
#endif

#endif /* TRIVIKRAMA_POSIX_H */
