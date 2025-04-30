#include "lib_vulkan.h"

static VulkanContext innerVulkanContext;

VulkanContext* get_VkContext()
{
	return &innerVulkanContext;
}
