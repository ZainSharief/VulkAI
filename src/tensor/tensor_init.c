#include "tensor_init.h"

#include "tensor_init_cpu.h"
#include "tensor_init_vulkan.h"

Tensor* VML_InitTensor(void* data, uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, void* grad, AI_TensorDType grad_dtype)
{
	switch (device)
	{
		case TENSOR_CPU:
			return initTensorCPU(data, shape, dims, dtype, device, requires_grad, grad, grad_dtype);
		case TENSOR_GPU_VULKAN:
			return initTensorVULKAN(data, shape, dims, dtype, device, requires_grad, grad, grad_dtype);
		default:
			return NULL;
	}
}

Tensor* VML_InitFullTensor(uint8_t* shape, uint8_t dims, void* fill_value, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype)
{
    switch (device)
	{
		case TENSOR_CPU:
			return initFullTensorCPU(shape, dims, fill_value, dtype, device, requires_grad, grad_dtype);
		case TENSOR_GPU_VULKAN:
			return initFullTensorVULKAN(shape, dims, fill_value, dtype, device, requires_grad, grad_dtype);
		default:
			return NULL;
	}
}

Tensor* VML_InitEmptyTensor(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype)
{
    switch (device)
	{
		case TENSOR_CPU:
			return initEmptyTensorCPU(shape, dims, dtype, device, requires_grad, grad_dtype);
		case TENSOR_GPU_VULKAN:
			return initEmptyTensorVULKAN(shape, dims, dtype, device, requires_grad, grad_dtype);
		default:
			return NULL;
	}
}

Tensor* VML_InitZerosTensor(uint8_t* shape, uint8_t dims, AI_TensorDType dtype, AI_TensorDevice device, bool requires_grad, AI_TensorDType grad_dtype)
{
    switch (device)
	{
		case TENSOR_CPU:
			return initZerosTensorCPU(shape, dims, dtype, device, requires_grad, grad_dtype);
		case TENSOR_GPU_VULKAN:
			return initZerosTensorVULKAN(shape, dims, dtype, device, requires_grad, grad_dtype);
		default:
			return NULL;
	}
}

Tensor* VML_InitLikeTensor(Tensor* other, AI_TensorDevice device)
{
    switch (device)
	{
		case TENSOR_CPU:
			return initLikeTensorCPU(other, device);
		case TENSOR_GPU_VULKAN:
			return initLikeTensorVULKAN(other, device);
		default:
			return NULL;
	}
}

Tensor* VML_CopyTensor(Tensor* other, AI_TensorDevice device)
{
    switch (device)
	{
		case TENSOR_CPU:
			return copyTensorCPU(other, device);
		case TENSOR_GPU_VULKAN:
			return copyTensorVULKAN(other, device);
		default:
			return NULL;
	}
}

void VML_DestroyTensor(Tensor* tensor, AI_TensorDevice device)
{
    switch (device)
	{
		case TENSOR_CPU:
			destroyTensorCPU(tensor);
            break;
		case TENSOR_GPU_VULKAN:
			destroyTensorVULKAN(tensor);
            break;
	}
}