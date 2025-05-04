#include "tensor_init_vulkan.h"

Tensor* initTensorVULKAN(void* data, uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, void* grad, AI_TensorDType grad_dtype)
{
    return NULL;
}

Tensor* initFullTensorVULKAN(uint8_t* shape, uint8_t dims, void* fill_value, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype)
{
    return NULL;
}

Tensor* initEmptyTensorVULKAN(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype)
{
    return NULL;
}

Tensor* initZerosTensorVULKAN(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype)
{
    return NULL;
}

Tensor* initLikeTensorVULKAN(Tensor* other, AI_TensorDevice device)
{
    return NULL;
}

Tensor* copyTensorVULKAN(Tensor* other, AI_TensorDevice device)
{   
    return NULL;
}

void destroyTensorVULKAN(Tensor* tensor)
{
    return;
}