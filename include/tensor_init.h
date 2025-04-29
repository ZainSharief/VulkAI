#ifndef TENSOR_INIT_H
#define TENSOR_INIT_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

typedef enum
{
    TENSOR_f32,
    TENSOR_i32,
    TENSOR_bool

} AI_TensorDType;

typedef enum
{
    TENSOR_CPU,
    TENSOR_GPU

} AI_TensorDevice;

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

Tensor* AI_InitTensor(void* data, uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, void* grad, AI_TensorDType grad_dtype);

Tensor* AI_InitFullTensor(uint8_t* shape, uint8_t dims, void* fill_value, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype);
Tensor* AI_InitEmptyTensor(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype);
Tensor* AI_InitZerosTensor(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype);
Tensor* AI_InitLikeTensor(Tensor* other, AI_TensorDevice device);
Tensor* AI_CopyTensor(Tensor* other, AI_TensorDevice device);

void AI_DestroyTensor(Tensor* tensor);

#endif