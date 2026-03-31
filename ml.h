#ifndef ML_H
#define ML_H

#define ML_ARRAY_LEN(xs) sizeof(xs)/sizeof((xs)[0])

#define ML_MAT_AT(mat, y, x) (mat).items[(x) + (mat).stride * (y)]
#define ML_MAT_PRINT(m) ml_mat_print(m, #m)

#define ML_NN_INPUT(nn) (nn).as[0]
#define ML_NN_OUTPUT(nn) (nn).as[(nn).layers]
#define ML_NN_PRINT(nn) ml_nn_print(nn, #nn)

#ifndef ML_EPS
#define ML_EPS 1e-1
#endif //ML_EPS

#ifndef ML_RATE
#define ML_RATE 1e-1
#endif //ML_RATE

typedef double MatItem;
typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    MatItem *items;
} Mat;

typedef struct {
    size_t *arch;
    size_t layers;
    Mat *ws;
    Mat *bs;
    Mat *as; // includes input
} Neural_Network;

typedef struct {
    Mat in;
    Mat out;
} TrainingSet;

typedef void (*train_func_t)(Neural_Network nn, Neural_Network gradient, TrainingSet training_data);

Neural_Network ml_nn_alloc(size_t *arch, size_t arch_count);
void ml_nn_forward(Neural_Network nn);
MatItem ml_nn_cost(Neural_Network nn, TrainingSet training_data);
void ml_nn_finite_diff(Neural_Network nn, Neural_Network gradient, TrainingSet training_data);
void ml_nn_backprop(Neural_Network nn, Neural_Network gradient, TrainingSet training_data);
void ml_nn_train(train_func_t train_func, size_t iterations, Neural_Network nn, TrainingSet training_data);
void ml_nn_randomize(Neural_Network nn, MatItem min, MatItem max);
void ml_nn_zero(Neural_Network nn);
void ml_nn_print(Neural_Network nn, const char *name);

Mat ml_mat_alloc(size_t rows, size_t cols);
Mat ml_mat_row(Mat m, size_t row);
Mat ml_mat_col(Mat m, size_t col);
void ml_mat_copy(Mat dest, Mat m);
MatItem ml_mat_extract(Mat m);
void ml_mat_dot(Mat dest, Mat a, Mat b);
void ml_mat_sum(Mat a, Mat b);
void ml_mat_map(Mat m, MatItem (*func)(MatItem));
void ml_mat_randomize(Mat m, MatItem min, MatItem max);
void ml_mat_fill(Mat m, MatItem val);
void ml_mat_print(Mat m, const char *name);
void ml_mat_print_indent(Mat m, const char *name);

void ml_rand_seed(int seed);
MatItem ml_rand_double(MatItem min, MatItem max);
MatItem ml_sigmoid(MatItem x);

#ifdef ML_IMPLEMENTATION

#ifndef ML_MALLOC
#include <stdlib.h>
#define ML_MALLOC malloc
#endif //ML_MALLOC

#ifndef ML_ASSERT
#include <assert.h>
#define ML_ASSERT assert
#endif //ML_ASSERT

#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

Neural_Network ml_nn_alloc(size_t *arch, size_t arch_count) {
    ML_ASSERT(arch_count > 1);

    size_t count = arch_count - 1;
    Neural_Network nn = {
        .arch = arch,
        .layers = count,
        .ws = (Mat*)ML_MALLOC(count * sizeof(Mat)),
        .bs = (Mat*)ML_MALLOC(count * sizeof(Mat)),
        .as = (Mat*)ML_MALLOC(arch_count * sizeof(Mat)),
    };

    ML_ASSERT(nn.ws != NULL);
    ML_ASSERT(nn.bs != NULL);
    ML_ASSERT(nn.as != NULL);

    ML_ASSERT(arch[0] > 0);
    nn.as[0] = ml_mat_alloc(1, arch[0]);
    for (size_t i = 1; i < arch_count; ++i) {
        ML_ASSERT(arch[i] > 0);
        nn.ws[i - 1] = ml_mat_alloc(nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = ml_mat_alloc(1, arch[i]);
        nn.as[i] = ml_mat_alloc(1, arch[i]);
    }

    return nn;
}

void ml_nn_forward(Neural_Network nn) {
    for (size_t i = 0; i < nn.layers; ++i) {
        ml_mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        ml_mat_sum(nn.as[i + 1], nn.bs[i]);
        ml_mat_map(nn.as[i + 1], ml_sigmoid);
    }
}

MatItem ml_nn_cost(Neural_Network nn, TrainingSet training_data) {
    ML_ASSERT(training_data.in.rows == training_data.out.rows);
    ML_ASSERT(training_data.out.cols == ML_NN_OUTPUT(nn).cols);
    size_t n = training_data.in.rows;

    MatItem c = 0;
    for (size_t i = 0; i < n; ++i) {
        Mat x = ml_mat_row(training_data.in, i);
        Mat y = ml_mat_row(training_data.out, i);

        ml_mat_copy(ML_NN_INPUT(nn), x);
        ml_nn_forward(nn);

        for (size_t j = 0; j < y.cols; ++j) {
            MatItem d = ML_MAT_AT(ML_NN_OUTPUT(nn), 0, j) - ML_MAT_AT(y, 0, j);
            c += d*d;
        }
    }

    return c/n;
}

void ml_nn_finite_diff(Neural_Network nn, Neural_Network gradient, TrainingSet training_data) {
    ML_ASSERT(training_data.in.rows == training_data.out.rows);
    
    MatItem c = ml_nn_cost(nn, training_data);
    MatItem saved;

    for (size_t k = 0; k < nn.layers; ++k) {
        for (size_t i = 0; i < nn.ws[k].rows; ++i) {
            for (size_t j = 0; j < nn.ws[k].cols; ++j) {
                saved = ML_MAT_AT(nn.ws[k], i, j);
                ML_MAT_AT(nn.ws[k], i, j) += ML_EPS;
                ML_MAT_AT(gradient.ws[k], i, j) = (ml_nn_cost(nn, training_data) - c) / ML_EPS;
                ML_MAT_AT(nn.ws[k], i, j) = saved;
            }
        }
    }

    for (size_t k = 0; k < nn.layers; ++k) {
        for (size_t i = 0; i < nn.bs[k].rows; ++i) {
            for (size_t j = 0; j < nn.bs[k].cols; ++j) {
                saved = ML_MAT_AT(nn.bs[k], i, j);
                ML_MAT_AT(nn.bs[k], i, j) += ML_EPS;
                ML_MAT_AT(gradient.bs[k], i, j) = (ml_nn_cost(nn, training_data) - c) / ML_EPS;
                ML_MAT_AT(nn.bs[k], i, j) = saved;
            }
        }
    }
}

void ml_nn_backprop(Neural_Network nn, Neural_Network gradient, TrainingSet training_data) {
    ML_ASSERT(training_data.in.rows == training_data.out.rows);
    ML_ASSERT(ML_NN_OUTPUT(nn).cols == training_data.out.cols);

    ml_nn_zero(gradient);
    
    for (size_t i = 0; i < training_data.in.rows; ++i) {
        ml_mat_copy(ML_NN_INPUT(nn), ml_mat_row(training_data.in, i));
        ml_nn_forward(nn);

        for (size_t j = 0;  j <= gradient.layers; ++j) {
            ml_mat_fill(gradient.as[j], 0);
        }
        
        for (size_t j = 0; j < training_data.out.cols; ++j) {
            ML_MAT_AT(ML_NN_OUTPUT(gradient), 0, j) = 2 * (ML_MAT_AT(ML_NN_OUTPUT(nn), 0, j) - ML_MAT_AT(training_data.out, i, j));
        }

        for (size_t l = nn.layers; l > 0; --l) {
            for (size_t j = 0; j < nn.as[l].cols; ++j) {
                float a = ML_MAT_AT(nn.as[l], 0, j);
                float da = ML_MAT_AT(gradient.as[l], 0, j);
                float delta = da * a * (1 - a);

                ML_MAT_AT(gradient.bs[l-1], 0, j) += delta;

                for (size_t k = 0; k < nn.as[l-1].cols; ++k) {
                    float pa = ML_MAT_AT(nn.as[l-1], 0, k);
                    float w = ML_MAT_AT(nn.ws[l-1], k, j);
                    
                    ML_MAT_AT(gradient.ws[l-1], k, j) += delta * pa;
                    ML_MAT_AT(gradient.as[l-1], 0, k) += delta * w;
                }
            }
        }
    }
    
    for (size_t i = 0; i < gradient.layers; ++i) {
        for (size_t j = 0; j < gradient.ws[i].rows; ++j) {
            for (size_t k = 0; k < gradient.ws[i].cols; ++k) {
                ML_MAT_AT(gradient.ws[i], j, k) /= training_data.in.rows;
            }
        }

        for (size_t j = 0; j < gradient.bs[i].rows; ++j) {
            for (size_t k = 0; k < gradient.bs[i].cols; ++k) {
                ML_MAT_AT(gradient.bs[i], j, k) /= training_data.in.rows;
            }
        }
    }
}

void apply_gradient(Neural_Network nn, Neural_Network gradient) {
    for (size_t k = 0; k < nn.layers; ++k) {
        for (size_t n = 0; n < nn.ws[k].rows; ++n) {
            for (size_t j = 0; j < nn.ws[k].cols; ++j) {
                ML_MAT_AT(nn.ws[k], n, j) -= ML_MAT_AT(gradient.ws[k], n, j) * ML_RATE;
            }
        }
    }

    for (size_t k = 0; k < nn.layers; ++k) {
        for (size_t n = 0; n < nn.bs[k].rows; ++n) {
            for (size_t j = 0; j < nn.bs[k].cols; ++j) {
                ML_MAT_AT(nn.bs[k], n, j) -= ML_MAT_AT(gradient.bs[k], n, j) * ML_RATE;
            }
        }
    }
}

void ml_nn_train(train_func_t train_func, size_t iterations, Neural_Network nn, TrainingSet training_data) {
    Neural_Network gradient = ml_nn_alloc(nn.arch, nn.layers + 1);
    
    for (size_t i = 0; i < iterations; ++i) {
        train_func(nn, gradient, training_data);
        apply_gradient(nn, gradient);
    }
}

void ml_nn_randomize(Neural_Network nn, MatItem min, MatItem max) {
    for (size_t i = 0; i < nn.layers; ++i) {
        ml_mat_randomize(nn.ws[i], min, max);
        ml_mat_randomize(nn.bs[i], min, max);
    }
}

void ml_nn_zero(Neural_Network nn) {
    for (size_t i = 0; i < nn.layers; ++i) {
        ml_mat_fill(nn.ws[i], 0);
        ml_mat_fill(nn.bs[i], 0);
        ml_mat_fill(nn.as[i], 0);
    }

    ml_mat_fill(nn.as[nn.layers], 0);
}

void ml_nn_print(Neural_Network nn, const char *name) {
    char buf[256] = {0};

    printf("%s = [\n", name);
    
    for (size_t i = 0; i < nn.layers; ++i) {
        snprintf(buf, sizeof(buf), "ws[%zu]", i);
        ml_mat_print_indent(nn.ws[i], buf);
        snprintf(buf, sizeof(buf), "bs[%zu]", i);
        ml_mat_print_indent(nn.bs[i], buf);
    }
    
    printf("]\n");
}

Mat ml_mat_alloc(size_t rows, size_t cols) {
    Mat m = {
        .rows = rows,
        .cols = cols,
        .stride = cols,
        .items = (MatItem*)ML_MALLOC(rows*cols * sizeof(MatItem)),
    };
    
    ML_ASSERT(m.items != NULL);
    return m;
}

Mat ml_mat_row(Mat m, size_t row) {
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .items = &ML_MAT_AT(m, row, 0),
    };
}

Mat ml_mat_col(Mat m, size_t col) {
    return (Mat){
        .rows = m.rows,
        .cols = 1,
        .stride = m.cols,
        .items = &ML_MAT_AT(m, 0, col),
    };
}

void ml_mat_copy(Mat dest, Mat m) {
    ML_ASSERT(dest.rows == m.rows && dest.cols == m.cols);

    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            ML_MAT_AT(dest, i, j) = ML_MAT_AT(m, i, j);
        }
    }
}

MatItem ml_mat_extract(Mat m) {
    ML_ASSERT(m.rows == 1 && m.cols == 1);
    return m.items[0];
}

void ml_mat_dot(Mat dest, Mat a, Mat b) {
    ML_ASSERT(a.cols == b.rows);
    ML_ASSERT(dest.rows == a.rows && dest.cols == b.cols);

    ml_mat_fill(dest, 0);
    
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < b.cols; ++j){
            for (size_t n = 0; n < a.cols; ++n) {
                ML_MAT_AT(dest, i, j) += ML_MAT_AT(a, i, n) * ML_MAT_AT(b, n, j);
            }
        }
    }
}

void ml_mat_sum(Mat a, Mat b) {
    ML_ASSERT(a.rows == b.rows && a.cols == b.cols);

    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            ML_MAT_AT(a, i, j) += ML_MAT_AT(b, i, j);
        }
    }
}

void ml_mat_map(Mat m, MatItem (*func)(MatItem)) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            ML_MAT_AT(m, i, j) = func(ML_MAT_AT(m, i, j));
        }
    }
}

void ml_mat_randomize(Mat m, MatItem min, MatItem max) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            ML_MAT_AT(m, i, j) = ml_rand_double(min, max);
        }
    }
}

void ml_mat_fill(Mat m, MatItem val) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            ML_MAT_AT(m, i, j) = val;
        }
    }
}

void ml_mat_print(Mat m, const char *name) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("    %f", ML_MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

void ml_mat_print_indent(Mat m, const char *name) {
    printf("    %s = [\n", name);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("        %f", ML_MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("    ]\n");
}

static int seeded = 0;
void ml_rand_seed(int seed) {
    srand(seed);
    seeded = 1;
}

MatItem ml_rand_double(MatItem min, MatItem max) {
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
    
    unsigned long long r = rand();
    r ^= rand() << 15;
    return (MatItem)(r % RAND_MAX) / (MatItem)RAND_MAX * (max-min) + min;
}

MatItem ml_sigmoid(MatItem x) {
    return 1./ (1. + exp(-x));
}

#endif //ML_IMPLEMENTATION

#ifdef ML_STRIP_PREFIXES
#define ARRAY_LEN ML_ARRAY_LEN
#define MAT_AT ML_MAT_AT
#define MAT_PRINT ML_MAT_PRINT

#define NN_INPUT(nn) ML_NN_INPUT(nn)
#define NN_OUTPUT(nn) ML_NN_OUTPUT(nn)
#define NN_PRINT(nn) ML_NN_PRINT(nn)

#define EPS ML_EPS
#define RATE ML_RATE

#define nn_alloc ml_nn_alloc
#define nn_forward ml_nn_forward
#define nn_cost ml_nn_cost
#define nn_finite_diff ml_nn_finite_diff
#define nn_backprop ml_nn_backprop
#define nn_train ml_nn_train
#define nn_randomize ml_nn_randomize
#define nn_zero ml_nn_zero
#define nn_print ml_nn_print

#define mat_alloc ml_mat_alloc
#define mat_row ml_mat_row
#define mat_col ml_mat_col
#define mat_copy ml_mat_copy
#define mat_dot ml_mat_dot
#define mat_extract ml_mat_extract
#define mat_sum ml_mat_sum
#define mat_map ml_mat_map
#define mat_randomize ml_mat_randomize
#define mat_fill ml_mat_fill
#define mat_print ml_mat_print
#define mat_print_indent ml_mat_print_indent

#define rand_seed ml_rand_seed
#define rand_double ml_rand_double
#define sigmoid ml_sigmoid
#endif //ML_STRIP_PREFIXES

#endif //ML_H
