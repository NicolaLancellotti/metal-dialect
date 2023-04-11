#import "MetalRuntime.h"
#include <iostream>

static const uint32_t count = 5;
static bool isStorageModeManaged = false;

void generateData(float *arg) {
  for (unsigned long index = 0; index < count; index++) {
    arg[index] = index;
  }
}

void addMetal() {
  const int8_t * libPath = (const int8_t *)"./default.metallib";
  uint64_t typeSize = sizeof(float);
  
  intptr_t device = _MetalDeviceMakeDefault();
  intptr_t commandQueue = _MetalDeviceMakeCommandQueue(device);
  intptr_t buffer0 = _MetalDeviceMakeBuffer(device, isStorageModeManaged, count, typeSize);
  intptr_t buffer1 = _MetalDeviceMakeBuffer(device, isStorageModeManaged, count, typeSize);
  intptr_t buffer2 = _MetalDeviceMakeBuffer(device, isStorageModeManaged, count, typeSize);
  
  float *arg0 = (float *)_MetalBufferGetContents2(buffer0);
  generateData(arg0);
  float *arg1 = (float *)_MetalBufferGetContents2(buffer1);
  generateData(arg1);
  float *result = (float *)_MetalBufferGetContents2(buffer2);
  
  
  intptr_t commandBuffer =
  _MetalCommandQueueMakeCommandBuffer(commandQueue,
                                      libPath,
                                      (const int8_t *)"addArrays",
                                      count, 1, 1);
  _MetalCommandBufferAddBuffer(commandBuffer, buffer0, 0);
  _MetalCommandBufferAddBuffer(commandBuffer, buffer1, 1);
  _MetalCommandBufferAddBuffer(commandBuffer, buffer2, 2);
  
  _MetalCommandBufferCommit(commandBuffer);
  _MetalCommandBufferWaitUntilCompleted(commandBuffer);
  
  for (uint32_t index = 0; index < count; index++) {
    printf("%f\n", result[index]);
    assert(result[index] == (arg0[index] + arg1[index]));
  }
  printf("Compute results as expected\n");
  
  _MetalRelease(commandBuffer);
  _MetalRelease(device);
  _MetalRelease(buffer0);
  _MetalRelease(buffer1);
  _MetalRelease(buffer2);
  _MetalRelease(commandQueue);
}
