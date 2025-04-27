#include "tensor_utils.h"
#include "tensor_init.h"

int main()
{
    uint8_t shape[3] = { 10, 28, 28 };
    float fill_value = 0.5;

    Tensor* newTensor = AI_InitFullTensor(shape, 3, &fill_value, TENSOR_f32, TENSOR_CPU, false, TENSOR_f32);
    Tensor* deepcopy = AI_CopyTensor(newTensor, TENSOR_CPU);

    AI_DestroyTensor(newTensor);
    AI_DestroyTensor(deepcopy);
    return 0;
}