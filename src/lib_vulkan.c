#include "lib_vulkan.h"

static VulkanContext innerVulkanContext;

VulkanContext* get_VkContext()
{
	return &innerVulkanContext;
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) && 
			(memProperties.memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}
}

GPUBuffer createGPUBuffer(size_t data_size)
{
	VulkanContext* vkctx = get_VkContext();
	GPUBuffer buffer;
	buffer.size = data_size;

	VkBufferCreateInfo bufferInfo = 
	{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = data_size;
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		.sharingMode = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
	};

	vkCreateBuffer(vkctx->device, &bufferInfo, NULL, &buffer.data);

	VkMemoryRequirements memreq;
	vkGetBufferMemoryRequirements(vkctx->device, buffer.data, &memreq);

	VkMemoryAllocateInfo allocInfo = 
	{
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = memreq.size,
		.memoryTypeIndex = findMemoryType
		(
			vkctx->physicalDevice, memreq.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		)
	};

	vkAllocateMemory(vkctx->device, &allocInfo, NULL, &buffer.address);
	vkBindBufferMemory(vkctx->device, buffer.data, buffer.address, 0);

	return buffer;
}

void uploadToGPUBuffer(GPUBuffer* buffer, void* data, size_t size)
{
	VulkanContext* vkctx = getVkContext();

	VkBuffer stagingBuffer;
	VkBufferCreateInfo stagingBufferInfo = 
	{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE
	};
	vkCreateBuffer(vkctx->device, &stagingBufferInfo, NULL, &stagingBuffer);

	VkMemoryRequirements stagingMemreq;
	vkGetBufferMemoryRequiremenets(vkctx->device, stagingBuffer, &stagingMemreq);

	VkMemoryAllocateInfo stagingAllocInfo = 
	{
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = stagingMemreq.size,
		.memoryTypeIndex = findMemoryType
		(
			vkctx->physicalDevice, stagingMemreq.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
		)
	}

	VkDeviceMemory stagingMemory;
	vkAllocateMemory(vkctx->device, &stagingAllocInfo, NULL, &stagingMemory);
	vkBindBufferMemory(vkctx->device, stagingBuffer, stagingMemory, 0);

	void* mapped;
	vkMapMemory(vkctx->device, stagingMemory, 0, size, 0, &mapped);
	memcpy(mapped, data, size);
	vkUnmapMemory(vkctx->device, stagingMemory);

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(vkctx->device, &cmdAllocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo = 
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	VkBufferCopy copyRegion = 
	{
		.srcOffset = 0,
		.dstOffset = 0,
		.size = size
	};
	vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer->data, 1, &copyRegion);

	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo = 
	{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &commandBuffer
	};

	vkQueueSubmit(vkctx->queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(vkctx->queue);

	vkFreeCommandBuffers(vkctx->device, vkctx->commandPool, 1, &commandBuffer);
	vkDestroyBuffer(vkctx->device, stagingBuffer, NULL);
	vkFreeMemory(vkctx->device, stagingMemory, NULL);
}
