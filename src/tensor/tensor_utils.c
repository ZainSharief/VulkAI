#include "tensor_utils.h"

#include <stdio.h>
#include <string.h>

void printType(void* data, AI_TensorDType dtype);
void recursivePrint(uint8_t* shape, uint8_t currentDim, uint8_t dims, void* data, int dataSize, AI_TensorDType dtype, size_t dtype_size);
void recursiveIndex(void* data, uint32_t* dataIndex, void* oldData, uint32_t oldDataIndex, uint8_t* indices, uint8_t* shape, uint32_t* strides, uint8_t dims, uint8_t currentDim, size_t dtype_size);

Tensor* VML_IndexTensor(Tensor* tensor, uint8_t* indices)
{
    Tensor* newTensor = VML_InitLikeTensor(tensor, tensor->device);

    void* data;
    uint8_t shape[tensor->dims];
    uint32_t strides[tensor->dims];
    
    for (int i = 0; i < tensor->dims; i++) {
        strides[i] = 1;
        for (int j = i + 1; j < tensor->dims; j++) {
            strides[i] *= tensor->shape[j];
        }
    }

    size_t total = 1;
    for (int i = 0; i < tensor->dims; i++) {
        uint8_t start = indices[i * 3];
        uint8_t stop = indices[i * 3 + 1];
        uint8_t step = indices[i * 3 + 2];

        shape[i] = ((stop - start) / step);
        total *= shape[i];
    }
    data = malloc(total * tensor->dtype_size);

    uint32_t dataIndex = 0;
    recursiveIndex(data, &dataIndex, tensor->data, 0, indices, shape, strides, tensor->dims, 0, tensor->dtype_size);

    newTensor->data = data;
    newTensor->shape = malloc(tensor->dims * sizeof(uint8_t));
    memcpy(newTensor->shape, shape, tensor->dims * sizeof(uint8_t));

    return newTensor;
}

void recursiveIndex(void* data, uint32_t* dataIndex, void* oldData, uint32_t oldDataIndex, uint8_t* indices, uint8_t* shape, uint32_t* strides, uint8_t dims, uint8_t currentDim, size_t dtype_size)
{
    if (currentDim == dims - 1) {
        memcpy((char*)data + (*dataIndex * dtype_size), (char*)oldData + (oldDataIndex * dtype_size), shape[currentDim] * dtype_size);
        (*dataIndex) += shape[currentDim];
        return;
    }

    uint8_t start = indices[currentDim * 3];
    uint8_t stop = indices[currentDim * 3 + 1];
    uint8_t step = indices[currentDim * 3 + 2];
    
    for  (int i = start; i < stop; i += step) {
        recursiveIndex(data, dataIndex, oldData, oldDataIndex + (strides[currentDim] * i), indices, shape, strides, dims, currentDim + 1, dtype_size);
    }
}

void VML_PrintTensor(Tensor* tensor)
{
    printf("Shape: (");
    for (int i = 0; i < tensor->dims; i++) {
        printf("%d", tensor->shape[i]);
        if (i != tensor->dims - 1) printf(", ");
    }
    printf("), ");

    size_t total = 1;
    for (int i = 0; i < tensor->dims; i++) total *= tensor->shape[i];

    recursivePrint(tensor->shape, 0, tensor->dims, tensor->data, total, tensor->dtype, tensor->dtype_size);
    printf("\n");
}

void recursivePrint(uint8_t* shape, uint8_t currentDim, uint8_t dims, void* data, int dataSize, AI_TensorDType dtype, size_t dtype_size)
{
    printf("[");
    if (currentDim == dims - 1) { 
        uint8_t currDimSize = shape[currentDim];

        if (currDimSize > 4) {
            for (int i = 0; i < 2; i++) {
                printType(((char*)data + i * dtype_size), dtype);
                printf(", ");
            }
            printf(" ... ");
            for (int i = currDimSize-2; i < currDimSize; i++) {
                printType(((char*)data + i * dtype_size), dtype);
                if (i != currDimSize-1) printf(", ");
            }
        }
        else {
            for (int i = 0; i < currDimSize; i++) {
                printType(((char*)data + i * dtype_size), dtype);
                if (i != currDimSize-1) printf(", ");
            }
        }
    }
    else {
        uint8_t currDimSize = shape[currentDim];
        dataSize /= currDimSize;

        if (currDimSize > 8) {
            for (int i = 0; i < 4; i++) {
                if (currentDim == 0 && i > 0) printf("\n");
                void* newData = (char*)data + i * dtype_size * dataSize;
                recursivePrint(shape, currentDim + 1, dims, newData, dataSize, dtype, dtype_size);
                printf(", \n");
            }
            printf("  ... \n");
            for (int i = currDimSize-4; i < currDimSize; i++) {
                if (currentDim == 0 && i > currDimSize-4) printf("\n");
                void* newData = (char*)data + i * dtype_size * dataSize;
                recursivePrint(shape, currentDim + 1, dims, newData, dataSize, dtype, dtype_size);
                if (i != currDimSize-1) printf(", \n");
            }
        }
        else {
            for (int i = 0; i < currDimSize; i++) {
                if (currentDim == 0 && i > 0) printf("\n");
                void* newData = (char*)data + i * dtype_size * dataSize;
                recursivePrint(shape, currentDim + 1, dims, newData, dataSize, dtype, dtype_size);
                if (i != currDimSize-1) printf(", \n");
            }
        }
    }
    printf("]");
}

void printType(void* data, AI_TensorDType dtype)
{
    switch (dtype) {
        case TENSOR_f32: printf("%f", *(float*)data); break;
        case TENSOR_i32: printf("%d", *(int32_t*)data); break;
        case TENSOR_bool: printf("%s", *(bool*)data ? "true" : "false"); break;
        default: printf("Unknown type"); break;
    }
}