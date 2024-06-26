#ifndef _MNIST_DB_H_
#define _MNIST_DB_H_

#include <stddef.h>
#include <stdio.h>

#define CHANGE_ENDIANNESS(a)			\
	( 									\
		(((a) >>  0) & 0xFF) << 24 |	\
        (((a) >>  8) & 0xFF) << 16 |	\
        (((a) >> 16) & 0xFF) <<  8 |	\
        (((a) >> 24) & 0xFF) <<  0 		\
	)

typedef struct MnistEntry MnistEntry;
struct MnistEntry {
    int class;
    double *pixels;
};

typedef struct MnistDataset MnistDataset;
struct MnistDataset {
    unsigned int width;
	unsigned int height;
	int transpose;

	size_t entries_count;
	size_t entries_read;

	FILE *fimage;
	FILE *flabel;

	size_t batch_size;
    MnistEntry *batch_entries;
};

#define CLASS_COUNT 10

/* Read a dataset from .
   Returns -1 in case of a failure, and 0 otherwise. */
int mnist_init(MnistDataset *output, const char *images_file, const char *labels_file, int transpose, size_t batch_size);

size_t mnist_load_batch(MnistDataset *dt);

/* Libère la la base de donnée de la mémoire. */
void mnist_free(MnistDataset *dt);

#endif /* _MNIST_DB_H_ */
