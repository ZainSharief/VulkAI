#ifndef LIB_VULKAN_H
#define LIB_VULKAN_H

#include <vulkan/vulkan.h>

struct VulkanContext
{	
	VkInstance instance;
	VkPhysicalDevice physicalDevice;
	VkDevice device;
	VkQueue queue;
	uint32_t queueFamilyIndex;
};
typedef struct VulkanContext VulkanContext;

VulkanContext* get_VkContext();

#endif
