#include "tensor_utils.h"

#include <stdio.h>

void printType(void* data, AI_TensorDType dtype);
void recursivePrint(uint8_t* shape, uint8_t currentDim, uint8_t dims, void* data, int dataSize, AI_TensorDType dtype, size_t dtype_size);

void AI_PrintTensor(Tensor* tensor)
{
    if (!tensor) return;

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
                printType((data + i * dtype_size), dtype);
                printf(", ");
            }
            printf(" ... ");
            for (int i = currDimSize-2; i < currDimSize; i++) {
                printType((data + i * dtype_size), dtype);
                if (i != currDimSize-1) printf(", ");
            }
        }
        else {
            for (int i = 0; i < currDimSize; i++) {
                printType((data + i * dtype_size), dtype);
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
                void* newData = data + i * dtype_size * dataSize;
                recursivePrint(shape, currentDim + 1, dims, newData, dataSize, dtype, dtype_size);
                printf(", \n");
            }
            printf("  ... \n");
            for (int i = currDimSize-4; i < currDimSize; i++) {
                if (currentDim == 0 && i > currDimSize-4) printf("\n");
                void* newData = data + i * dtype_size * dataSize;
                recursivePrint(shape, currentDim + 1, dims, newData, dataSize, dtype, dtype_size);
                if (i != currDimSize-1) printf(", \n");
            }
        }
        else {
            for (int i = 0; i < currDimSize; i++) {
                if (currentDim == 0 && i > 0) printf("\n");
                void* newData = data + i * dtype_size * dataSize;
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