#include "tensor/tensor_utils.h"
#include "tensor/tensor_init.h"

#include <stdio.h>

int main()
{
    uint8_t shape[3] = { 10, 28, 28 };
    float fill_value = 0.5;

    Tensor* newTensor = VML_InitFullTensor(shape, 3, &fill_value, TENSOR_f32, TENSOR_CPU, false, TENSOR_f32);

    uint8_t indices[9] = { 2, 5, 1, 4, 12, 1, 3, 8, 1 };  
    Tensor* indexTensor = VML_IndexTensor(newTensor, indices);

    printf("Original Tensor:\n");
    VML_PrintTensor(newTensor);

    printf("\n\nIndexed Tensor:\n");
    VML_PrintTensor(indexTensor);

    VML_DestroyTensor(newTensor, newTensor->device);
    VML_DestroyTensor(indexTensor, indexTensor->device);
    return 0;
}