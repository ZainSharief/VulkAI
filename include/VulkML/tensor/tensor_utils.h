#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "tensor_init.h"

void VML_PrintTensor(Tensor* tensor);
Tensor* VML_IndexTensor(Tensor* tensor, uint8_t* indices);

#endif