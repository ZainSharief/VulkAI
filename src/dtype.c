#include "dtype.h"

size_t tensorDTypeSize(AI_TensorDType dtype)
{
    switch (dtype) {
        case TENSOR_f32: return sizeof(float); 
        case TENSOR_i32: return sizeof(int32_t); 
        case TENSOR_bool: return sizeof(bool); 
        default: return 0;
    }
}