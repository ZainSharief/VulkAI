#include "lib_init.h"
#include "lib_vulkan.h"

#include <stdlib.h>

int main1()
{
    VulkanContext* context;
    context = malloc(sizeof(VulkanContext));

    free(context);

    return 0;
}