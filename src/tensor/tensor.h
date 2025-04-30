#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>

#include "dtype.h"
#include "device.h"

typedef struct Tensor {
    void* data;

    uint8_t dims;
    uint8_t* shape;

    AI_TensorDType dtype;
    size_t dtype_size;

    AI_TensorDevice device;

    bool requires_grad;
    void* grad;

    AI_TensorDType grad_dtype;
    size_t grad_dtype_size;

} Tensor;

#endif