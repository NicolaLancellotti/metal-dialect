#import "MetalRuntime.h"
#include <iostream>

using BufferType = int32_t;

static int32_t ITER = 1;
static int32_t rows = 100000000;
//static int32_t rows = 5;
static int32_t columns = 5;
static bool print = false;
static bool isStorageModeManaged = false;

static void printMatrix(BufferType *buffer, int32_t rows, int32_t columns) {
  for (int32_t i = 0; i < rows; ++i) {
    for (int32_t j = 0; j < columns; ++j) {
      std::cout << buffer[i * columns + j];
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

static void genValues(BufferType *buffer, int32_t columns) {
  buffer[0 * columns + 0] =
  buffer[0 * columns + 1] =
  buffer[0 * columns + 2] =
  buffer[0 * columns + 3] = 1;
}

void lifeMetal() {
  uint64_t typeSize = sizeof(BufferType);
  uint64_t count = (uint64_t)rows * columns;
  const int8_t * libPath = (const int8_t *)"./default.metallib";
  
  intptr_t device = _MetalDeviceMakeDefault();
  intptr_t commandQueue = _MetalDeviceMakeCommandQueue(device);
  
  intptr_t buffer0 = _MetalDeviceMakeBuffer(device, isStorageModeManaged, count, typeSize);
  intptr_t buffer1 = _MetalDeviceMakeBuffer(device, isStorageModeManaged, 1, typeSize);
  intptr_t buffer2 = _MetalDeviceMakeBuffer(device, isStorageModeManaged, count, typeSize);
  
  BufferType *arg0 = (BufferType *)_MetalBufferGetContents2(buffer0);
  genValues(arg0, columns);
  
  BufferType *arg1 = (BufferType *)_MetalBufferGetContents2(buffer1);
  arg1[0] = rows;
  arg1[1] = columns;
  
  for (int32_t i = 0; i < ITER; ++i) {
    intptr_t commandBuffer =
    _MetalCommandQueueMakeCommandBuffer(commandQueue,
                                        libPath,
                                        (const int8_t *)"life",
                                        rows, columns, 1);
    auto even = i % 2 == 0;
    _MetalCommandBufferAddBuffer(commandBuffer, even ? buffer0 : buffer2, 0);
    _MetalCommandBufferAddBuffer(commandBuffer, buffer1, 1);
    _MetalCommandBufferAddBuffer(commandBuffer, even ? buffer2 : buffer0, 2);
    
    _MetalCommandBufferCommit(commandBuffer);
    _MetalCommandBufferWaitUntilCompleted(commandBuffer);
    if (print) {
      auto result = (BufferType *)_MetalBufferGetContents2(even ? buffer2 : buffer0);
      printMatrix(result, rows, columns);
    }
    _MetalRelease(commandBuffer);
  }
  
  _MetalRelease(device);
  _MetalRelease(buffer0);
  _MetalRelease(buffer1);
  _MetalRelease(buffer2);
  _MetalRelease(commandQueue);
}

void lifeCPUCompute(const BufferType *matrix, const int32_t *sides,
                    BufferType *result, int32_t x, int32_t y) {
  int32_t rows = sides[0];
  int32_t columns = sides[1];
  int32_t liveNeighbors = 0;
  
  for (int32_t i = x - 1; i <= x + 1; ++i) {
    for (int32_t j = y - 1; j <= y + 1; ++j) {
      if (i >= 0 && i < rows && j >= 0 && j < columns &&
          (i != x || j != y) && matrix[i * columns + j]) {
        liveNeighbors++;
      }
    }
  }
  
  int32_t index = x * columns + y;
  if (matrix[index]) {
    result[index] = (liveNeighbors == 2 || liveNeighbors == 3) ? 1 : 0;
  } else {
    result[index] = liveNeighbors == 3 ? 1 : 0;
  }
}

void lifeCPU() {
  int32_t size = rows * columns;
  BufferType *buff0 = (BufferType *)calloc(size, sizeof(int32_t));
  BufferType *buff1 = (BufferType *)malloc(size * sizeof(int32_t));
  BufferType *arg0 = buff0;
  BufferType *result = buff1;
  
  for (BufferType i = 0; i < rows * columns; ++i)
    arg0[i] = 0;
  
  genValues(arg0, columns);
  
  int32_t sides[2] = {rows, columns};
  for (int32_t int32_ter = 0; int32_ter < ITER; ++int32_ter) {
    for (int32_t i = 0; i < rows; ++i) {
      for (int32_t j = 0; j < columns; ++j) {
        lifeCPUCompute(arg0, sides, result, i, j);
      }
    }
    
    if (print)
      printMatrix(result, rows, columns);
    
    if (arg0 == buff1) {
      arg0 = buff0;
      result = buff1;
    } else {
      arg0 = buff1;
      result = buff0;
    }
  }
  
  free(buff0);
  free(buff1);
}
