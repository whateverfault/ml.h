#include <stdio.h>

#define ML_IMPLEMENTATION
#define ML_STRIP_PREFIXES
#include "ml.h"

MatItem train_data[] = {
    0, 0, 0, 0,
    1, 0, 0, 1,
    0, 1, 0, 1,
    1, 1, 1, 0,
};

int main() {
    size_t arch[] = {2, 2, 2, 2};
    size_t arch_len = ML_ARRAY_LEN(arch);
    
    Neural_Network nn = nn_alloc(arch, arch_len);
    Neural_Network gradient = nn_alloc(arch, arch_len);

    nn_randomize(nn, -1, 1);

    size_t stride = 4;
    size_t rows = sizeof(train_data)/sizeof(train_data[0])/stride;
    TrainingSet training_data = {
        .in = (Mat){
            .rows = rows,
            .cols = 2,
            .stride = stride,
            .items = &train_data[0],
        },
    .out = {
            .rows = rows,
            .cols = 2,
            .stride = stride,
            .items = &train_data[2],
        },
    };
    
    
    nn_train_finite_difference(100*1000, nn, gradient, training_data);
    
    printf("cost = %f\n", nn_cost(nn, training_data));
    for (size_t i = 0; i < training_data.in.rows; ++i) {
        Mat x = mat_row(training_data.in, i);

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        printf("%f + %f = (%f; %f)\n", ML_MAT_AT(x, 0, 0), ML_MAT_AT(x, 0, 1), MAT_AT(NN_OUTPUT(nn), 0, 0), MAT_AT(NN_OUTPUT(nn), 0, 1));
    }
}