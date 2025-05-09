#include "tensor_init_cpu.h"

#include <string.h>

//helper gets data size given a shape, dims, dtype - 64bit allows for tensors >4.3GB
uint64_t getDataSize(uint8_t* shape, uint8_t dims, AI_TensorDType dtype)
{
	uint64_t size;
	for (int s = 0; s < dims; s++) size *= shape[s];
	return size * tensorDTypeSize(dtype);
}

Tensor* initTensor(void* data, uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, void* grad, AI_TensorDType grad_dtype)
{
    Tensor* tensor;
    tensor = malloc(sizeof(Tensor));

    tensor->dims = dims;

    if (shape != NULL) {
        tensor->shape = (uint8_t*)malloc(dims * sizeof(uint8_t));
        memcpy(tensor->shape, shape, dims * sizeof(uint8_t));
    }
    
    tensor->dtype = dtype;
    tensor->dtype_size = tensorDTypeSize(dtype);

    tensor->requires_grad = requires_grad;
    tensor->grad = grad;
        
    tensor->grad_dtype = grad_dtype;
    tensor->grad_dtype_size = tensorDTypeSize(grad_dtype);

	tensor->device = device;
	uint64_t size = getDataSize(shape, dims, dtype);
	switch (device)
	{
		case TENSOR_CPU:
			//heap allocation ensures tensor can outlive user generated data
			tensor->data = malloc(size);
			memcpy(tensor->data, data, size);
			tensor->gpuBuffer = NULL;
			break;
		case TENSOR_GPU_VULKAN:
			tensor->gpuBuffer = (GPUBuffer*)malloc(sizeof(GPUBuffer));
			*tensor->gpuBuffer = createGPUBuffer(size);
			uploadToGPUBuffer(tensor->gpuBuffer, data, size);
			tensor->data = NULL;
			break;
	}

    return tensor;
}

Tensor* initFullTensor(uint8_t* shape, uint8_t dims, void* fill_value, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype)
{
    Tensor* tensor = initTensor(NULL, shape, dims, dtype, device, requires_grad, NULL, grad_dtype);

    size_t total = 1;
    for (int i = 0; i < dims; i++) total *= shape[i];

    void* data = malloc(total * tensor->dtype_size);    

    // Fill the first element with the fill value
    memcpy(data, fill_value, tensor->dtype_size);

    // Exponentially fill the rest of the tensor -> O(logn) 
    size_t filled = 1;
    while (filled < total) {
        size_t copyNum = (filled < (total - filled)) ? filled : (total - filled);
        memcpy((char*)data + filled * tensor->dtype_size, data, copyNum * tensor->dtype_size);
        filled += copyNum;
    }
    tensor->data = data;

    return tensor;
}

Tensor* initEmptyTensor(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype)
{
    Tensor* tensor = initTensorCPU(NULL, shape, dims, dtype, device, requires_grad, NULL, grad_dtype);

    int total = 1;
    for (int i = 0; i < dims; i++) total *= shape[i];

    void* data = malloc(total * tensor->dtype_size);
    tensor->data = data;

    return tensor;
}

Tensor* initZerosTensor(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype)
{
    Tensor* tensor = initTensorCPU(NULL, shape, dims, dtype, device, requires_grad, NULL, grad_dtype);

    int total = 1;
    for (int i = 0; i < dims; i++) total *= shape[i];

    void* data = calloc(total, tensor->dtype_size);
    tensor->data = data;

    return tensor;
}

Tensor* initLikeTensor(Tensor* other, AI_TensorDevice device)
{
    void* data = NULL;
    void* grad = NULL;
    uint8_t* shape = NULL;

    Tensor* tensor = initTensorCPU(data, shape, other->dims, other->dtype, device, other->requires_grad, grad, other->grad_dtype);

    return tensor;
}

Tensor* copyTensorCPU(Tensor* other, AI_TensorDevice device)
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
        grad = malloc(total * other->grad_dtype_size);
        memcpy(grad, other->grad, total * other->grad_dtype_size);
    }
    
    shape = malloc(other->dims * sizeof(uint8_t));
    memcpy(shape, other->shape, other->dims * sizeof(uint8_t));

    Tensor* tensor = initTensorCPU(data, shape, other->dims, other->dtype, device, other->requires_grad, grad, other->grad_dtype);

    return tensor;
}

void destroyTensor(Tensor* tensor)
{
    if (tensor == NULL) {
        return;
    }

    if (tensor->data != NULL) {
        free(tensor->data);
    }
    if (tensor->grad != NULL) {
        free(tensor->grad);
    }
    if (tensor->shape != NULL) {
        free(tensor->shape);
    }
    
    free(tensor);
}
