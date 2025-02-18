#!/usr/bin/env python3
"""
ring.py - Lock-Free Ring Queue for Shared Memory Input Events

Implements a fixed-size lock-free ring buffer using ctypes and
DevilutionX's `ring_queue` layout. Supports submitting and retrieving
entries without locks.

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

import ctypes
import devilutionx as dx

# Define constants
RING_QUEUE_CAPACITY = 128
RING_QUEUE_MASK = RING_QUEUE_CAPACITY - 1

# Define the ring_entry_type enum
class RingEntryType(ctypes.c_int):
    RING_ENTRY_KEY_LEFT  = 1<<0
    RING_ENTRY_KEY_RIGHT = 1<<1
    RING_ENTRY_KEY_UP    = 1<<2
    RING_ENTRY_KEY_DOWN  = 1<<3
    RING_ENTRY_KEY_X     = 1<<4
    RING_ENTRY_KEY_Y     = 1<<5
    RING_ENTRY_KEY_A     = 1<<6
    RING_ENTRY_KEY_B     = 1<<7
    RING_ENTRY_KEY_SAVE  = 1<<8
    RING_ENTRY_KEY_LOAD  = 1<<9
    RING_ENTRY_KEY_PAUSE = 1<<10

    RING_ENTRY_F_SINGLE_TICK_PRESS = 1<<31

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

# Example Usage
if __name__ == "__main__":
    import mmap

    # Create shared memory for the ring queue
    shm = mmap.mmap(-1, ctypes.sizeof(dx.ring_queue))

    # Initialize RingQueue with the shared memory address
    ring = dx.ring_queue(ctypes.addressof(ctypes.c_char.from_buffer(shm)))

    # Initialize the ring queue
    init(ring)

    entry = get_entry_to_retrieve(ring)
    assert entry is None

    res = has_capacity_to_submit(ring);
    assert res

    # Consume the whole capacity
    i = 0
    while has_capacity_to_submit(ring):
        entry = get_entry_to_submit(ring)
        entry.type = i
        submit(ring)
        i += 1
    assert i == RING_QUEUE_CAPACITY

    # Retrieve everything
    i = 0;
    while True:
        entry = get_entry_to_retrieve(ring)
        if entry == None:
            break
        assert entry.type == i
        retrieve(ring)
        i += 1
    assert i == RING_QUEUE_CAPACITY

    # Get an entry to submit and submit it
    entry = get_entry_to_submit(ring)
    assert entry is not None
    entry.type = 0x666
    entry.data = 0x555
    submit(ring)

    # Retrieve the entry after submitting
    entry = get_entry_to_retrieve(ring)
    assert entry is not None
    assert entry.type == 0x666
    assert entry.data == 0x555

    # Retrieve the next entry, it should be the same as the previous one
    entry2 = get_entry_to_retrieve(ring)
    assert ctypes.addressof(entry) == ctypes.addressof(entry2)

    # Now retrieve the entry from the ring queue
    retrieve(ring)

    # The ring queue should be empty now
    entry = get_entry_to_retrieve(ring)
    assert entry is None
