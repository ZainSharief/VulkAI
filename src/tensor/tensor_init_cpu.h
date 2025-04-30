#ifndef TENSOR_INIT_CPU_H
#define TENSOR_INIT_CPU_H

#include <stdlib.h>
#include <stdbool.h>

#include "tensor.h"

Tensor* initTensorCPU(void* data, uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, void* grad, AI_TensorDType grad_dtype);

Tensor* initFullTensorCPU(uint8_t* shape, uint8_t dims, void* fill_value, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype);
Tensor* initEmptyTensorCPU(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype);
Tensor* initZerosTensorCPU(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype);
Tensor* initLikeTensorCPU(Tensor* other, AI_TensorDevice device);
Tensor* copyTensorCPU(Tensor* other, AI_TensorDevice device);

void destroyTensorCPU(Tensor* tensor);

#endif