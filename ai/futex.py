"""
futex.py - Simple Futex Wrapper

Wraps futex syscalls

Author: Roman Penyaev <r.peniaev@gmail.com>
"""
import ctypes

libc = ctypes.CDLL("libc.so.6", use_errno=True)
SYS_futex = 202   # x86_64, check `unistd.h` for your arch
FUTEX_WAIT = 0
FUTEX_WAKE = 1
INT_MAX = (1<<31)-1

def wait(addr, expected):
    return libc.syscall(SYS_futex,
                        ctypes.c_void_p(addr),
                        FUTEX_WAIT,
                        expected,
                        0, 0, 0)

def wake(addr, n=INT_MAX):
    return libc.syscall(SYS_futex,
                        ctypes.c_void_p(addr),
                        FUTEX_WAKE,
                        n, 0, 0, 0)
