#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "tensor_init.h"

void AI_PrintTensor(Tensor* tensor);
Tensor* AI_IndexTensor(Tensor* tensor, uint8_t* indices);

#endif