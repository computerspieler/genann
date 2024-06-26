#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "genann.h"
#include "mnist_db.h"

#define CLASS_COUNT 10

int main(int argc, char* argv[]) 
{
    size_t i;
	int j;
	double output[CLASS_COUNT];
    MnistDataset training, tests;

	if(argc != 5) {
		printf("./mnist [NUMBER OF HIDDEN LAYERS] [NEURON PER HIDDEN LAYERS] [TRAINING ITERATION COUNT] [OUTPUT FILE]");
		return 1;
	}

	if(mnist_init(&training,
		"mnist_data/train-images-idx3-ubyte",
		"mnist_data/train-labels-idx1-ubyte",
		0, 0
	))
		return 1;

	if(mnist_load_batch(&training) != training.batch_size) {
		mnist_free(&training);
		return 1;
	}
    
	if(mnist_init(&tests,
		"mnist_data/t10k-images-idx3-ubyte",
		"mnist_data/t10k-labels-idx1-ubyte",
		0, 0
	)) {
		mnist_free(&training);
		return 1;
	}

	if(mnist_load_batch(&tests) != tests.batch_size) {
		mnist_free(&tests);
		mnist_free(&training);
		return 1;
	}

	assert(training.width == tests.width);
	assert(training.height == tests.height);
	assert(training.width != 0);
	assert(training.height != 0);

    genann *ann = genann_init(training.width * training.height,
		atoi(argv[1]),
		atoi(argv[2]),
		CLASS_COUNT
	);
	assert(ann != NULL);

	memset(output, 0, CLASS_COUNT * sizeof(double));

	for(j = 0; j < atoi(argv[3]); j ++) {
		for (i = 0; i < training.batch_size; ++i) {
			printf("[Training number %d]: %zd%%\r",
				j+1,
				(100 * (i+1)) / training.batch_size
            );
			
			output[training.batch_entries[i].class] = 1;
			genann_train(ann, training.batch_entries[i].pixels, output, 0.25);
			output[training.batch_entries[i].class] = 0;
		}
		printf("\n");
	}

	FILE *output_file = fopen(argv[4], "w");
	if(output_file) {
		genann_write(ann, output_file);
		fclose(output_file);
	} else
		perror("fopen");

	genann_free(ann);

	mnist_free(&training);
	mnist_free(&tests);

	return 0;
}

