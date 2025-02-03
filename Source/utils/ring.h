#ifndef RING_H
#define RING_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

#include "futex.h"

#define RING_QUEUE_CAPACITY 128
#define RING_QUEUE_MASK		(RING_QUEUE_CAPACITY - 1)

enum ring_entry_type {
	RING_ENTRY_KEY_LEFT	 = 1<<0,
	RING_ENTRY_KEY_RIGHT = 1<<1,
	RING_ENTRY_KEY_UP	 = 1<<2,
	RING_ENTRY_KEY_DOWN	 = 1<<3,
	RING_ENTRY_KEY_X	 = 1<<4,
	RING_ENTRY_KEY_Y	 = 1<<5,
	RING_ENTRY_KEY_A	 = 1<<6,
	RING_ENTRY_KEY_B	 = 1<<7,
	RING_ENTRY_KEY_NEW	 = 1<<8,
	RING_ENTRY_KEY_SAVE	 = 1<<9,
	RING_ENTRY_KEY_LOAD	 = 1<<10,
	RING_ENTRY_KEY_PAUSE = 1<<11,
	RING_ENTRY_KEY_NOOP	 = 1<<12,

	/* Events */
	RING_ENTRY_EVENT_STEP_FINISHED = 1<<30,

	/* Flags */
	RING_ENTRY_F_SINGLE_TICK_PRESS = 1<<31,

	/* Common flags */
	RING_ENTRY_FLAGS	 = (RING_ENTRY_F_SINGLE_TICK_PRESS)
};

struct ring_entry {
	uint32_t type;
	uint32_t data;
};

struct ring_queue {
	uint32_t		  write_idx;
	uint32_t		  read_idx;
	struct ring_entry array[RING_QUEUE_CAPACITY];
};

static inline void ring_queue_init(struct ring_queue *ring)
{
	*ring = (struct ring_queue){};
}

static inline bool
ring_queue_has_capacity_to_submit(struct ring_queue *ring)
{
	return (ring->write_idx - ring->read_idx < RING_QUEUE_CAPACITY);
}

static inline struct ring_entry *
ring_queue_get_entry_to_submit(struct ring_queue *ring)
{
	return &ring->array[ring->write_idx & RING_QUEUE_MASK];
}

static inline void ring_queue_submit(struct ring_queue *ring)
{
	__atomic_store_n(&ring->write_idx, ring->write_idx + 1,
					 __ATOMIC_RELEASE);
	futex_wake(&ring->write_idx, INT_MAX);
}

static inline bool ring_queue_wait_all_retrieved(struct ring_queue *ring,
												 int timeout_ms)
{
	unsigned read_idx;
	int rv;

	read_idx = __atomic_load_n(&ring->read_idx, __ATOMIC_ACQUIRE);
	while (ring->write_idx != read_idx) {
		rv = futex_wait(&ring->read_idx, read_idx, timeout_ms);
		if (rv == -1)
			/* Timeout or fault */
			return false;

		read_idx = __atomic_load_n(&ring->read_idx, __ATOMIC_ACQUIRE);
	}

	return true;
}

static inline struct ring_entry *
ring_queue_get_entry_to_retrieve(struct ring_queue *ring)
{
	unsigned write_idx;

	write_idx = __atomic_load_n(&ring->write_idx, __ATOMIC_ACQUIRE);
	if (write_idx == ring->read_idx)
		return NULL;

	return &ring->array[ring->read_idx & RING_QUEUE_MASK];
}

static inline void ring_queue_retrieve(struct ring_queue *ring)
{
	__atomic_store_n(&ring->read_idx, ring->read_idx + 1,
					 __ATOMIC_RELEASE);
	futex_wake(&ring->read_idx, INT_MAX);
}

static inline bool ring_queue_wait_any_submitted(struct ring_queue *ring,
												 int timeout_ms)
{
	unsigned write_idx;
	int rv;

	write_idx = __atomic_load_n(&ring->write_idx, __ATOMIC_ACQUIRE);
	while (write_idx == ring->read_idx) {
		rv = futex_wait(&ring->write_idx, write_idx, timeout_ms);
		if (rv == -1)
			/* Timeout or fault */
			return false;
		write_idx = __atomic_load_n(&ring->write_idx, __ATOMIC_ACQUIRE);
	}

	return true;
}

/*
 * To build a test from the header:
 * gcc -std=c99 -Wall -x c -o ring ring.h -DTEST
 */
#ifdef	TEST

#include <stdio.h>
#include <assert.h>
#include <sys/wait.h>
#include <sys/mman.h>

int main()
{
	struct ring_entry *entry, *entry2;
	void *ring;
	pid_t pid;
	bool res;
	int i;

	/* Create shared memory for the ring queue */
	ring = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
				MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	assert(ring != MAP_FAILED);

	ring_queue_init(ring);

	entry = ring_queue_get_entry_to_retrieve(ring);
	assert(entry == NULL);

	res = ring_queue_has_capacity_to_submit(ring);
	assert(res);

	/* Consume the whole queue capacity */
	i = 0;
	while (ring_queue_has_capacity_to_submit(ring)) {
		entry = ring_queue_get_entry_to_submit(ring);
		entry->type = i;
		ring_queue_submit(ring);
		i++;
	}
	assert(i == RING_QUEUE_CAPACITY);

	/* Retrieve everything */
	i = 0;
	while (1) {
		entry = ring_queue_get_entry_to_retrieve(ring);
		if (!entry)
			break;
		assert(entry->type == i);
		ring_queue_retrieve(ring);
		i++;
	}
	assert(i == RING_QUEUE_CAPACITY);

	entry = ring_queue_get_entry_to_submit(ring);
	assert(entry != NULL);
	entry->type = 666;

	ring_queue_submit(ring);
	entry = ring_queue_get_entry_to_retrieve(ring);
	assert(entry != NULL);
	assert(entry->type == 666);

	entry2 = ring_queue_get_entry_to_retrieve(ring);
	assert(entry == entry2);

	ring_queue_retrieve(ring);
	entry = ring_queue_get_entry_to_retrieve(ring);
	assert(entry == NULL);

	printf("-- fork test: parent producer, child receiver --\n");

	/* Do some futex test */
	pid = fork();
	if (pid == 0) {
		/* Child */
		printf("child:	wait submission\n");
		ring_queue_wait_any_submitted(ring, -1);

		/* Retrieve the submitted entry */
		entry = ring_queue_get_entry_to_retrieve(ring);
		assert(entry != NULL);
		assert(entry->type == 0x111);
		assert(entry->data == 0x111);
		printf("child:	retrieved from parent\n");
		ring_queue_retrieve(ring);

		/* Sleep a bit before changing roles */
		usleep(500 * 1000);

		printf("-- fork test: child producer, parent receiver --\n");

		/* Get an entry to submit and submit it */
		entry = ring_queue_get_entry_to_submit(ring);
		assert(entry != NULL);
		entry->type = 0x222;
		entry->data = 0x222;
		printf("child:	submitted to parent\n");
		ring_queue_submit(ring);
		ring_queue_wait_all_retrieved(ring, -1);
		printf("child:	all retrieved by parent\n");
	} else {
		/* Parent */
		sleep(1);

		/* Get an entry to submit and submit it */
		entry = ring_queue_get_entry_to_submit(ring);
		assert(entry != NULL);
		entry->type = 0x111;
		entry->data = 0x111;
		printf("parent: submitted to child\n");
		ring_queue_submit(ring);

		ring_queue_wait_all_retrieved(ring, -1);

		printf("parent: all retrieved by child\n");

		usleep(500 * 1000);

		printf("parent: wait submission\n");
		ring_queue_wait_any_submitted(ring, -1);

		/* Retrieve the submitted entry */
		entry = ring_queue_get_entry_to_retrieve(ring);
		assert(entry != NULL);
		assert(entry->type == 0x222);
		assert(entry->data == 0x222);
		printf("parent: retrieved from child\n");
		ring_queue_retrieve(ring);

		/* Wait for child */
		wait(NULL);
	}
}

#endif /* TEST */

#endif /* RING_H */
