#pragma once

// Ubuntu's newer glibc exposes C23 rsqrt() under _GNU_SOURCE, which
// conflicts with CUDA 13.1 math_functions.h. The CUDA translation units
// therefore disable _GNU_SOURCE; declare the two GNU pthread functions
// required by libstdc++ without changing any system headers.
#include <pthread.h>
#include <time.h>

#ifndef __USE_GNU
extern "C" int pthread_cond_clockwait(
    pthread_cond_t*, pthread_mutex_t*, clockid_t, const struct timespec*);
extern "C" int pthread_mutex_clocklock(
    pthread_mutex_t*, clockid_t, const struct timespec*) noexcept;
#endif
