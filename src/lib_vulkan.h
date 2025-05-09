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
	VkCommandPool commandPool;
};
typedef struct VulkanContext VulkanContext;

struct VulkanBuffer
{
	VkBuffer data;
	VkDeviceMemory address;
	size_t size;
};
typedef struct GPUBuffer GPUBuffer;

VulkanContext* get_VkContext();
GPUBuffer createGPUBuffer(size_t data_size);
void uploadToGPUBuffer(GPUBuffer* buffer, void* data, size_t size);

#endif
