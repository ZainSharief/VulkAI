#include "lib_init.h"
#include "lib_vulkan.h"

void init_Vulkan()
{
	VulkanContext* vkctx = get_VkContext();
	
	VkApplicationInfo appInfo = 
	{
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "Bare Vulkan",
		.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		.pEngineName = "None",
		.engineVersion = VK_MAKE_VERSION(1, 0, 0),
		.apiVersion = VK_API_VERSION_1_0
	};

	VkInstanceCreateInfo instanceInfo = 
	{
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &appInfo
	};

	check_vk(vkCreateInstance(&instanceInfo, NULL, &vkctx->instance));

	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(vkctx->instance, &deviceCount, NULL);
	if (deviceCount == 0)
	{
		//printf will be replaced soon with logError, option to disable logging.
		printf("<AI-LIB> CRUCIAL ERROR c001 : No Vulkan-Compatible GPU found\n");
	}

	VkPhysicalDevice devices[deviceCount];
	vkEnumeratePhysicalDevices(vkctx->instance, &deviceCount, devices);
	vkctx->physicalDevice = devices[0];

	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(vkctx->physicalDevice, &queueFamilyCount, NULL);

	VkQueueFamilyProperties qfp[queueFamilyCount];
	vkGetPhysicalDeviceQueueFamilyProperties(vkctx->physicalDevice, &queueFamilyCount, qfp);

	for (uint32_t i = 0; i < queueFamilyCount; i++)
	{
		if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
		{
			vkctx->queueFamilyIndex = i;
			break;
		}
	}

	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo queueInfo = 
	{
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = vkctx->queueFamilyIndex,
		.queueCount = 1,
		.pQueuePriorities = &queuePriority
	};

	VkDeviceCreateInfo deviceInfo = 
	{
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &queueInfo
	};

	check_vk(vkCreateDevice(vkctx->physicalDevice, &deviceInfo, NULL, &vkctx->device));
	vkGetDeviceQueue(vkctx->device, vkctx->queueFamilyIndex, 0, &vkctx->queue);
}

void AI_Init()
{
	init_Vulkan();
}

