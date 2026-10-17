/*
 * POSIX helpers for PC builds (threads, async + throttled file source).
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#define _POSIX_C_SOURCE 200809L

#include "trivikrama_posix.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- thread pool ------------------------------------------------------------ */

struct trv_pool {
    uint32_t n;
    pthread_t *th;
    pthread_mutex_t mu;
    pthread_cond_t cv_start, cv_done;
    uint64_t gen;
    uint32_t pending;
    trv_task_fn fn;
    void *arg;
    int quit;
};

typedef struct {
    trv_pool *pool;
    uint32_t id;
} pool_worker_arg;

static void *pool_worker(void *varg)
{
    pool_worker_arg *wa = (pool_worker_arg *)varg;
    trv_pool *p = wa->pool;
    uint32_t id = wa->id;
    uint64_t seen = 0;
    free(wa);
    for (;;) {
        trv_task_fn fn;
        void *arg;
        pthread_mutex_lock(&p->mu);
        while (p->gen == seen && !p->quit) pthread_cond_wait(&p->cv_start, &p->mu);
        if (p->quit) {
            pthread_mutex_unlock(&p->mu);
            return NULL;
        }
        seen = p->gen;
        fn = p->fn;
        arg = p->arg;
        pthread_mutex_unlock(&p->mu);

        fn(arg, id, p->n);

        pthread_mutex_lock(&p->mu);
        if (--p->pending == 0) pthread_cond_signal(&p->cv_done);
        pthread_mutex_unlock(&p->mu);
    }
}

static void pool_parallel(void *ctx, trv_task_fn fn, void *arg, uint32_t n_workers)
{
    trv_pool *p = (trv_pool *)ctx;
    if (n_workers != p->n || p->n < 2) {
        uint32_t w;
        for (w = 0; w < n_workers; ++w) fn(arg, w, n_workers);
        return;
    }
    pthread_mutex_lock(&p->mu);
    p->fn = fn;
    p->arg = arg;
    p->pending = p->n - 1;
    p->gen++;
    pthread_cond_broadcast(&p->cv_start);
    pthread_mutex_unlock(&p->mu);

    fn(arg, 0, p->n); /* the calling thread is worker 0 */

    pthread_mutex_lock(&p->mu);
    while (p->pending) pthread_cond_wait(&p->cv_done, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

trv_pool *trv_pool_create(uint32_t n_workers)
{
    trv_pool *p;
    uint32_t i;
    if (n_workers < 1) n_workers = 1;
    p = (trv_pool *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->n = n_workers;
    p->th = (pthread_t *)calloc(n_workers, sizeof(pthread_t));
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_start, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    for (i = 1; i < n_workers; ++i) {
        pool_worker_arg *wa = (pool_worker_arg *)malloc(sizeof *wa);
        wa->pool = p;
        wa->id = i;
        pthread_create(&p->th[i], NULL, pool_worker, wa);
    }
    return p;
}

void trv_pool_destroy(trv_pool *p)
{
    uint32_t i;
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    p->quit = 1;
    pthread_cond_broadcast(&p->cv_start);
    pthread_mutex_unlock(&p->mu);
    for (i = 1; i < p->n; ++i) pthread_join(p->th[i], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_start);
    pthread_cond_destroy(&p->cv_done);
    free(p->th);
    free(p);
}

void trv_pool_runtime(trv_pool *p, trv_runtime *rt)
{
    rt->parallel = pool_parallel;
    rt->parallel_ctx = p;
    rt->n_workers = p ? p->n : 1;
}

/* ---- async / throttled file source ----------------------------------------------- */

#define QCAP 64

typedef struct {
    uint32_t off, n;
    void *dst;
} req_t;

struct trv_filex {
    FILE *f;
    uint64_t size;
    double bytes_per_s, overhead_s;
    pthread_mutex_t file_mu; /* one reader at a time, like a real card */
    pthread_mutex_t mu;
    pthread_cond_t cv_req, cv_done;
    req_t q[QCAP];
    uint32_t head, tail, count, pending;
    int err, quit, async;
    pthread_t io;
    uint64_t bytes_read;
    uint32_t n_reads;
};

static void sleep_s(double s)
{
    struct timespec ts;
    if (s <= 0) return;
    ts.tv_sec = (time_t)s;
    ts.tv_nsec = (long)((s - (double)ts.tv_sec) * 1e9);
    nanosleep(&ts, NULL);
}

static int do_read(trv_filex *fx, uint32_t off, void *dst, uint32_t n)
{
    int rc = 0;
    pthread_mutex_lock(&fx->file_mu);
    if (fseek(fx->f, (long)off, SEEK_SET) != 0 || fread(dst, 1, n, fx->f) != n) rc = -1;
    if (fx->bytes_per_s > 0) sleep_s(fx->overhead_s + (double)n / fx->bytes_per_s);
    fx->bytes_read += n;
    fx->n_reads += 1;
    pthread_mutex_unlock(&fx->file_mu);
    return rc;
}

static int fx_read(void *ctx, uint32_t off, void *dst, uint32_t n)
{
    return do_read((trv_filex *)ctx, off, dst, n);
}

static int fx_read_async(void *ctx, uint32_t off, void *dst, uint32_t n)
{
    trv_filex *fx = (trv_filex *)ctx;
    pthread_mutex_lock(&fx->mu);
    while (fx->count == QCAP) pthread_cond_wait(&fx->cv_done, &fx->mu);
    fx->q[fx->tail].off = off;
    fx->q[fx->tail].n = n;
    fx->q[fx->tail].dst = dst;
    fx->tail = (fx->tail + 1) % QCAP;
    fx->count++;
    fx->pending++;
    pthread_cond_signal(&fx->cv_req);
    pthread_mutex_unlock(&fx->mu);
    return 0;
}

static int fx_wait(void *ctx)
{
    trv_filex *fx = (trv_filex *)ctx;
    int err;
    pthread_mutex_lock(&fx->mu);
    while (fx->pending) pthread_cond_wait(&fx->cv_done, &fx->mu);
    err = fx->err;
    fx->err = 0;
    pthread_mutex_unlock(&fx->mu);
    return err ? -1 : 0;
}

static void *io_thread(void *ctx)
{
    trv_filex *fx = (trv_filex *)ctx;
    for (;;) {
        req_t r;
        int rc;
        pthread_mutex_lock(&fx->mu);
        while (fx->count == 0 && !fx->quit) pthread_cond_wait(&fx->cv_req, &fx->mu);
        if (fx->count == 0 && fx->quit) {
            pthread_mutex_unlock(&fx->mu);
            return NULL;
        }
        r = fx->q[fx->head];
        fx->head = (fx->head + 1) % QCAP;
        fx->count--;
        pthread_mutex_unlock(&fx->mu);

        rc = do_read(fx, r.off, r.dst, r.n);

        pthread_mutex_lock(&fx->mu);
        if (rc) fx->err = 1;
        fx->pending--;
        pthread_cond_broadcast(&fx->cv_done);
        pthread_mutex_unlock(&fx->mu);
    }
}

trv_filex *trv_filex_open(trv_source *src, const char *path, int async,
                          double bytes_per_s, double overhead_s)
{
    trv_filex *fx = (trv_filex *)calloc(1, sizeof *fx);
    if (!fx) return NULL;
    fx->f = fopen(path, "rb");
    if (!fx->f) {
        free(fx);
        return NULL;
    }
    fseek(fx->f, 0, SEEK_END);
    fx->size = (uint64_t)ftell(fx->f);
    fx->bytes_per_s = bytes_per_s;
    fx->overhead_s = overhead_s;
    fx->async = async;
    pthread_mutex_init(&fx->file_mu, NULL);
    pthread_mutex_init(&fx->mu, NULL);
    pthread_cond_init(&fx->cv_req, NULL);
    pthread_cond_init(&fx->cv_done, NULL);
    memset(src, 0, sizeof *src);
    src->read = fx_read;
    src->ctx = fx;
    if (async) {
        pthread_create(&fx->io, NULL, io_thread, fx);
        src->read_async = fx_read_async;
        src->wait = fx_wait;
    }
    return fx;
}

void trv_filex_close(trv_filex *fx)
{
    if (!fx) return;
    if (fx->async) {
        pthread_mutex_lock(&fx->mu);
        fx->quit = 1;
        pthread_cond_signal(&fx->cv_req);
        pthread_mutex_unlock(&fx->mu);
        pthread_join(fx->io, NULL);
    }
    fclose(fx->f);
    pthread_mutex_destroy(&fx->file_mu);
    pthread_mutex_destroy(&fx->mu);
    pthread_cond_destroy(&fx->cv_req);
    pthread_cond_destroy(&fx->cv_done);
    free(fx);
}

uint64_t trv_filex_bytes_read(const trv_filex *fx) { return fx->bytes_read; }
uint32_t trv_filex_reads(const trv_filex *fx) { return fx->n_reads; }
uint64_t trv_filex_size(const trv_filex *fx) { return fx->size; }
