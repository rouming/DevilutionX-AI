#ifndef FUTEX_H
#define FUTEX_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>

static inline int futex_wait(void* addr, int expected, int timeout_ms)
{
	struct timespec ts, *timeout = NULL;
	if (timeout_ms > 0) {
		ts.tv_sec  = timeout_ms / 1000;
		ts.tv_nsec = timeout_ms % 1000 * 1000 * 1000;
		timeout = &ts;
	}
	return syscall(SYS_futex, addr, FUTEX_WAIT, expected, timeout, NULL, 0);
}

static inline int futex_wake(void* addr, int n)
{
	return syscall(SYS_futex, addr, FUTEX_WAKE, n, NULL, NULL, 0);
}

#endif //FUTEX_H
