#include "MetalRuntime.h"
@import Foundation;
@import MetalRuntimeSwift;

// _____________________________________________________________________________
// Release

void _MetalRelease(intptr_t ref) {
  @autoreleasepool {
    [Wrappable release:ref];
  }
}

// _____________________________________________________________________________
// Device

intptr_t _MetalDeviceMakeDefault(void) {
  intptr_t value = 0;
  @autoreleasepool {
    value = [[Device makeDefault] wrap];
  }
  return value;
}

intptr_t _MetalDeviceMakeCommandQueue(intptr_t ref) {
  intptr_t value = 0;
  @autoreleasepool {
    Device *instance = [Device unwrap:ref];
    value = [[instance makeCommandQueue] wrap];
  }
  return value;
}

intptr_t _MetalDeviceMakeBuffer(intptr_t ref,
                                bool isStorageModeManaged,
                                int64_t count,
                                int64_t sizeType) {
  intptr_t value = 0;
  @autoreleasepool {
    Device *instance = [Device unwrap:ref];
    value = [[instance makeBufferWithIsStorageModeManaged:isStorageModeManaged
                                               bufferSize:count * sizeType
                                                    count:count] wrap];
  }
  return value;
}


// _____________________________________________________________________________
// Buffer

void _MetalBufferGetContents(intptr_t ref, void *memRef) {
  typedef struct {
    void *allocated;
    void *aligned;
    int64_t offset;
    int64_t sizes;
    int64_t strides;
  } MemRef;
  
  @autoreleasepool {
    Buffer *instance = [Buffer unwrap:ref];
    void *contents = [instance contents];
    MemRef *mem = memRef;
    mem->allocated = contents;
    mem->aligned = contents;
    mem->offset = 0;
    mem->sizes = [instance getCount];
    mem->strides = 1;
  }
}

void* _MetalBufferGetContents2(intptr_t ref) {
  void *value;
  @autoreleasepool {
    Buffer *instance = [Buffer unwrap:ref];
    void *contents = [instance contents];
    value = contents;
  }
  return value;
}

// _____________________________________________________________________________
// CommandQueue

intptr_t _MetalCommandQueueMakeCommandBuffer(intptr_t ref,
                                             const int8_t *libPath,
                                             const int8_t *functionName,
                                             int64_t width,
                                             int64_t height,
                                             int64_t depth) {
  intptr_t value = 0;
  @autoreleasepool {
    CommandQueue *instance = [CommandQueue unwrap:ref];
    NSString *nsLibPath = [NSString stringWithCString:(const char *)libPath
                                             encoding:NSUTF8StringEncoding];
    NSString *nsFunctionName = [NSString stringWithCString:(const char *)functionName
                                                  encoding:NSUTF8StringEncoding];
    value = [[instance makeCommandBufferWithLibPath:nsLibPath
                                       functionName:nsFunctionName
                                              width:width
                                             height:height
                                              depth:depth] wrap];
  }
  return value;
}

// _____________________________________________________________________________
// CommandBuffer

void _MetalCommandBufferAddBuffer(intptr_t ref,
                                  intptr_t bufferRef,
                                  int64_t index) {
  @autoreleasepool {
    CommandBuffer *instance = [CommandBuffer unwrap:ref];
    Buffer *buffer = [Buffer unwrap:bufferRef];
    [instance addBufferWithBuffer:buffer index:index];
  }
}

void _MetalCommandBufferCommit(intptr_t ref) {
  @autoreleasepool {
    CommandBuffer *instance = [CommandBuffer unwrap:ref];
    [instance commit];
  }
}

void _MetalCommandBufferWaitUntilCompleted(intptr_t ref) {
  @autoreleasepool {
    CommandBuffer *instance = [CommandBuffer unwrap:ref];
    [instance waitUntilCompleted];
  }
}
