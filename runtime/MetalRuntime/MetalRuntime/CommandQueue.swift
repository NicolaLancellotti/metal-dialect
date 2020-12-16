import Metal

@objc
class CommandQueue: Wrappable {
  private let device: MTLDevice
  private let commandQueue: MTLCommandQueue
  
  @objc
  init(device: MTLDevice, commandQueue: MTLCommandQueue) {
    self.device = device
    self.commandQueue = commandQueue
  }
  
  @objc
  func makeCommandBuffer(libPath: String,
                         functionName: String,
                         width: Int,
                         height: Int,
                         depth: Int) -> CommandBuffer? {
    do {
      let library = try device.makeLibrary(filepath: libPath)
      guard let function = library.makeFunction(name: functionName),
            let commandBuffer = commandQueue.makeCommandBuffer(),
            let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
        return nil
      }
      let pipelineState = try device.makeComputePipelineState(function: function)
      computeEncoder.setComputePipelineState(pipelineState)
      
      let gridSize = MTLSize(width: width, height: height, depth: depth)
      var w = pipelineState.threadExecutionWidth
      w = w > gridSize.width ? gridSize.width : w
      var h = pipelineState.maxTotalThreadsPerThreadgroup / w
      h = h > gridSize.height ? gridSize.height : h
      let threads = MTLSize(width: w, height: h, depth: 1);
      
      return CommandBuffer(device: device,
                           commandBuffer: commandBuffer,
                           computeEncoder: computeEncoder,
                           gridSize: gridSize,
                           threads: threads)
    } catch {
      print(error)
      return nil
    }
  }
  
}
