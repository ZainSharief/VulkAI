#include "tensor_init.h"

size_t AI_TensorDTypeSize(AI_TensorDType dtype);

Tensor* AI_InitTensor(void* data, uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, void* grad, AI_TensorDType grad_dtype)
{
    Tensor* tensor;
    tensor = malloc(sizeof(Tensor));

    tensor->data = data;
    tensor->dims = dims;

    // copies the given shape into the tensor
    tensor->shape = (uint8_t*)malloc(dims * sizeof(uint8_t));
    memcpy(tensor->shape, shape, dims * sizeof(uint8_t));

    tensor->dtype = dtype;
    tensor->dtype_size = AI_TensorDTypeSize(dtype);
    tensor->device = device;

    tensor->requires_grad = requires_grad;
    tensor->grad = grad;

    tensor->grad_dtype = grad_dtype;
    tensor->grad_dtype_size = AI_TensorDTypeSize(grad_dtype);

    return tensor;
}

Tensor* AI_InitFullTensor(uint8_t* shape, uint8_t dims, void* fill_value, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype)
{
    Tensor* tensor = AI_InitTensor(NULL, shape, dims, dtype, device, requires_grad, NULL, grad_dtype);

    int total = 1;
    for (int i = 0; i < dims; i++) total *= shape[i];

    // TODO: could be done with block memcpy
    void* data = malloc(total * tensor->dtype_size);
    for (int i = 0; i < total; i++) {
        memcpy(data + i * tensor->dtype_size, fill_value, tensor->dtype_size);
    }
    tensor->data = data;

    return tensor;
}

Tensor* AI_InitEmptyTensor(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype)
{
    Tensor* tensor = AI_InitTensor(NULL, shape, dims, dtype, device, requires_grad, NULL, grad_dtype);

    int total = 1;
    for (int i = 0; i < dims; i++) total *= shape[i];

    void* data = malloc(total * tensor->dtype_size);
    tensor->data = data;

    return tensor;
}

Tensor* AI_InitZerosTensor(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype)
{
    Tensor* tensor = AI_InitTensor(NULL, shape, dims, dtype, device, requires_grad, NULL, grad_dtype);

    int total = 1;
    for (int i = 0; i < dims; i++) total *= shape[i];

    void* data = calloc(total, tensor->dtype_size);
    tensor->data = data;

    return tensor;
}

Tensor* AI_CopyTensor(Tensor* other, AI_TensorDevice device)
{
    void* data = NULL;
    void* grad = NULL;
    uint8_t* shape;

    int total = 1;
    for (int i = 0; i < other->dims; i++) total *= other->shape[i];

    if (other->data != NULL) {
        data = malloc(total * other->dtype_size);
        memcpy(data, other->data, total * other->dtype_size);
    }
    if (other->grad != NULL) {
        grad = malloc(total * other->dtype_size);
        memcpy(grad, other->grad, total * other->grad_dtype_size);
    }
    
    shape = malloc(other->dims * sizeof(uint8_t));
    memcpy(shape, other->shape, other->dims * sizeof(uint8_t));

    Tensor* tensor = AI_InitTensor(data, shape, other->dims, other->dtype, device, other->requires_grad, grad, other->grad_dtype);

    return tensor;
}

size_t AI_TensorDTypeSize(AI_TensorDType dtype)
{
    switch (dtype) {
        case TENSOR_f32: return sizeof(float);
        case TENSOR_i32: return sizeof(int32_t);
        case TENSOR_bool: return sizeof(bool);
        default: return 0;
    }
}

void AI_DestroyTensor(Tensor* tensor)
{
    if (!tensor) return;

    free(tensor->data);
    free(tensor->grad);
    free(tensor->shape);
    free(tensor);
}