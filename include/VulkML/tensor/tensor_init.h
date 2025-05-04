#ifndef TENSOR_INIT_H
#define TENSOR_INIT_H

#include <stdlib.h>
#include <stdbool.h>

#include "tensor.h"

Tensor* VML_InitTensor(void* data, uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, void* grad, AI_TensorDType grad_dtype);

Tensor* VML_InitFullTensor(uint8_t* shape, uint8_t dims, void* fill_value, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype);
Tensor* VML_InitEmptyTensor(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype);
Tensor* VML_InitZerosTensor(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype);
Tensor* VML_InitLikeTensor(Tensor* other, AI_TensorDevice device);
Tensor* VML_CopyTensor(Tensor* other, AI_TensorDevice device);

void VML_DestroyTensor(Tensor* tensor, AI_TensorDevice device);

#endif