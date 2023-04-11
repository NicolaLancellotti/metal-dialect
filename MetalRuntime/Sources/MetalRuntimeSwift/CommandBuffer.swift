import Metal

@objc
public class CommandBuffer: Wrappable {
  private let device: MTLDevice
  private let commandBuffer: MTLCommandBuffer
  private var computeEncoder: MTLComputeCommandEncoder
  private var buffers: [MTLBuffer]
  private let gridSize: MTLSize
  private let threads: MTLSize
  
  init(device: MTLDevice,
       commandBuffer: MTLCommandBuffer,
       computeEncoder: MTLComputeCommandEncoder,
       gridSize: MTLSize,
       threads: MTLSize) {
    self.buffers = [MTLBuffer]()
    self.device = device
    self.commandBuffer = commandBuffer
    self.computeEncoder = computeEncoder
    self.gridSize = gridSize
    self.threads = threads
  }
  
  @objc
  public func addBuffer(buffer: Buffer, index: Int) {
    self.buffers.append(buffer.buffer)
    self.computeEncoder.setBuffer(buffer.buffer, offset: 0, index: index)
  }
  
  @objc
  public func commit() {
    self.computeEncoder.dispatchThreads(gridSize,
                                        threadsPerThreadgroup: threads)
    self.computeEncoder.endEncoding()
    
    
    let isStorageModeManaged = self.buffers.first { $0.storageMode == .managed } != nil
    if isStorageModeManaged {
      let blitCommandEncoder = commandBuffer.makeBlitCommandEncoder()!
      for buffer in buffers {
        buffer.didModifyRange(0..<buffer.length)
        blitCommandEncoder.synchronize(resource: buffer)
      }
      blitCommandEncoder.endEncoding()
    }
    
    self.commandBuffer.commit();
  }
  
  @objc
  public func waitUntilCompleted() {
    self.commandBuffer.waitUntilCompleted()
  }
  
}
