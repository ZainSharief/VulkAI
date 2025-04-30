#ifndef DTYPE_H
#define DTYPE_H

#include <stdlib.h>
#include <stdbool.h>

typedef enum
{
    TENSOR_f32,
    TENSOR_i32,
    TENSOR_bool

} AI_TensorDType;

size_t tensorDTypeSize(AI_TensorDType dtype);

#endif