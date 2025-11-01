#!/usr/bin/env python3
"""
ring.py - Lock-Free Ring Queue for Shared Memory Input Events

Implements a fixed-size lock-free ring buffer using DevilutionX's
`ring_queue` layout. Supports submitting and retrieving entries
without locks.

WARNING:
    This code assumes x86-64 memory ordering. On architectures
    with weaker models (e.g., ARM, RISC-V), it is unsafe without
    proper memory barriers:

    - Writers must use RELEASE semantics when updating `write_idx`.
    - Readers must use ACQUIRE semantics when loading `write_idx`.

    These barriers must be added via C extensions or atomic
    intrinsics.

Author: Roman Penyaev <r.peniaev@gmail.com>
"""

import numpy as np
import futex
import devilutionx as dx


def offsetof(cls, field):
    return cls.fields[field][1]

def addr(obj):
    # https://numpy.org/devdocs/reference/generated/numpy.ndarray.ctypes.html
    return obj.__array_interface__['data'][0]


# Define constants
RING_QUEUE_CAPACITY = dx.ring_queue.fields['array'][0].shape[0]
RING_QUEUE_MASK = RING_QUEUE_CAPACITY - 1

# Define the ring_entry_type enum
class RingEntryType:
    RING_ENTRY_KEY_LEFT  = 1<<0
    RING_ENTRY_KEY_RIGHT = 1<<1
    RING_ENTRY_KEY_UP    = 1<<2
    RING_ENTRY_KEY_DOWN  = 1<<3
    RING_ENTRY_KEY_X     = 1<<4
    RING_ENTRY_KEY_Y     = 1<<5
    RING_ENTRY_KEY_A     = 1<<6
    RING_ENTRY_KEY_B     = 1<<7
    RING_ENTRY_KEY_NEW   = 1<<8
    RING_ENTRY_KEY_SAVE  = 1<<9
    RING_ENTRY_KEY_LOAD  = 1<<10
    RING_ENTRY_KEY_PAUSE = 1<<11
    RING_ENTRY_KEY_NOOP  = 1<<12

    # Events
    RING_ENTRY_EVENT_STEP_FINISHED = 1<<30

    # Flags
    RING_ENTRY_F_SINGLE_TICK_PRESS = 1<<31

    # Common flags
    RING_ENTRY_FLAGS	 = (RING_ENTRY_F_SINGLE_TICK_PRESS)


def init(ring):
    ring.write_idx = 0
    ring.read_idx = 0

def nr_submitted_entries(ring):
    """Get number of already submitted entries in the queue."""
    return ring.write_idx - ring.read_idx

def has_capacity_to_submit(ring):
    """Returns true if enough capacity to submit a new entry."""
    return ring.write_idx - ring.read_idx < RING_QUEUE_CAPACITY

def get_entry_to_submit(ring):
    """Get the next entry to submit."""
    return ring.array[ring.write_idx & RING_QUEUE_MASK]

def submit(ring):
    """Submit the current entry."""
    # TODO: for architectures other than x86-64 we need
    # __atomic_store_n(&write_idx, write_idx + 1, __ATOMIC_RELEASE)
    ring.write_idx += 1

    addr_write_idx = addr(ring) + offsetof(dx.ring_queue, 'write_idx')
    futex.wake(addr_write_idx)

def wait_all_retrieved(ring, read_idx=None):
    """Wait for all retrieved. Called on the producer side"""
    if read_idx is None:
        read_idx = ring.read_idx
    base_addr = addr(ring)
    while ring.write_idx != read_idx:
        addr_read_idx = base_addr + offsetof(dx.ring_queue, 'read_idx')
        futex.wait(addr_read_idx, read_idx)
        read_idx = ring.read_idx

def get_entry_to_retrieve(ring, read_idx=None):
    """Get the next entry to retrieve, or None if the queue is empty."""
    # TODO: for architectures other than x86-64 we need
    # write_idx = __atomic_load_n(&write_idx, __ATOMIC_ACQUIRE);
    write_idx = ring.write_idx
    if read_idx is None:
        read_idx = ring.read_idx
    if write_idx == read_idx:
        return None
    return ring.array[read_idx & RING_QUEUE_MASK]

def retrieve(ring):
    """Mark the current entry as retrieved."""
    ring.read_idx += 1
    base_addr = addr(ring)
    addr_read_idx = base_addr + offsetof(dx.ring_queue, 'read_idx')
    futex.wake(addr_read_idx)

def wait_any_submitted(ring, read_idx=None):
    """Wait for anything submitted. Called on the consimer side"""
    if read_idx is None:
        read_idx = ring.read_idx
    base_addr = addr(ring)
    while ring.write_idx == read_idx:
        # Pass `read_idx` as expected value to avoid a race, since
        # `write_idx` can be increased by the producer
        addr_write_idx = base_addr + offsetof(dx.ring_queue, 'write_idx')
        futex.wait(addr_write_idx, read_idx)

# Example Usage
if __name__ == "__main__":
    import mmap
    import os
    import time

    # Create shared memory for the ring queue
    mm = mmap.mmap(-1, dx.ring_queue.itemsize)

    # Create a 1-element view and return the scalar object. The
    # `view(np.recarray)` is needed to allow access using dot
    # notation.
    rings = np.frombuffer(mm, dtype=dx.ring_queue, count=1).view(np.recarray)
    ring = rings[0]

    # Initialize the ring queue
    init(ring)

    entry = get_entry_to_retrieve(ring)
    assert entry is None

    res = has_capacity_to_submit(ring)
    assert res

    # Consume the whole capacity
    i = 0
    while has_capacity_to_submit(ring):
        entry = get_entry_to_submit(ring)
        entry.en_type = i
        submit(ring)
        i += 1
    assert i == RING_QUEUE_CAPACITY

    # Retrieve everything
    i = 0;
    while True:
        entry = get_entry_to_retrieve(ring)
        if entry == None:
            break
        assert entry.en_type == i
        retrieve(ring)
        i += 1
    assert i == RING_QUEUE_CAPACITY

    # Get an entry to submit and submit it
    entry = get_entry_to_submit(ring)
    assert entry is not None
    entry.en_type = 0x666
    entry.en_data = 0x555
    submit(ring)

    # Retrieve the entry after submitting
    entry = get_entry_to_retrieve(ring)
    assert entry is not None
    assert entry.en_type == 0x666
    assert entry.en_data == 0x555

    # Retrieve the next entry, it should be the same as the previous one
    entry2 = get_entry_to_retrieve(ring)
    base_addr1 = addr(entry)
    base_addr2 = addr(entry2)
    assert base_addr1 == base_addr2

    # Now retrieve the entry from the ring queue
    retrieve(ring)

    # The ring queue should be empty now
    entry = get_entry_to_retrieve(ring)
    assert entry is None

    print("-- fork test: parent producer, child receiver --")

    # Do some futex test
    pid = os.fork()
    if pid == 0:
        # Child
        print("child:  wait submission")
        wait_any_submitted(ring)

        # Retrieve the submitted entry
        entry = get_entry_to_retrieve(ring)
        assert entry is not None
        assert entry.en_type == 0x111
        assert entry.en_data == 0x111
        print("child:  retrieved from parent")
        retrieve(ring)

        # Sleep a bit before changing roles
        time.sleep(0.5)

        print("-- fork test: child producer, parent receiver --")

        # Get an entry to submit and submit it
        entry = get_entry_to_submit(ring)
        assert entry is not None
        entry.en_type = 0x222
        entry.en_data = 0x222
        print("child:  submitted to parent")
        submit(ring)
        wait_all_retrieved(ring);
        print("child:  all retrieved by parent")
    else:
        # Parent
        time.sleep(1)

        # Get an entry to submit and submit it
        entry = get_entry_to_submit(ring)
        assert entry is not None
        entry.en_type = 0x111
        entry.en_data = 0x111
        print("parent: submitted to child")
        submit(ring)

        wait_all_retrieved(ring)

        print("parent: all retrieved by child")

        time.sleep(0.5)

        print("parent: wait submission")
        wait_any_submitted(ring)

        # Retrieve the submitted entry
        entry = get_entry_to_retrieve(ring)
        assert entry is not None
        assert entry.en_type == 0x222
        assert entry.en_data == 0x222
        print("parent: retrieved from child")
        retrieve(ring)

        os.waitpid(pid, 0)
