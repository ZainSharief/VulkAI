#include "tensor_utils.h"
#include "tensor_init.h"

#include <stdio.h>

int main()
{
    uint8_t shape[3] = { 10, 28, 28 };
    float fill_value = 0.5;

    Tensor* newTensor = AI_InitFullTensor(shape, 3, &fill_value, TENSOR_f32, TENSOR_CPU, false, TENSOR_f32);

    uint8_t indices[9] = { 2, 5, 1, 4, 12, 1, 3, 8, 1 };  
    Tensor* indexTensor = AI_IndexTensor(newTensor, indices);

    printf("Original Tensor:\n");
    AI_PrintTensor(newTensor);

    printf("\n\nIndexed Tensor:\n");
    AI_PrintTensor(indexTensor);

    AI_DestroyTensor(newTensor);
    AI_DestroyTensor(indexTensor);
    return 0;
}