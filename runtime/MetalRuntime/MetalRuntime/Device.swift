import Metal

@objc
class Device: Wrappable {
  private let device: MTLDevice
  
  @objc
  static func makeDefault() -> Device? {
    guard let device = MTLCreateSystemDefaultDevice() else {
      return nil
    }
    return Device(device: device)
  }
  
  private init(device: MTLDevice) {
    self.device = device
  }
  
  @objc
  func makeCommandQueue() -> CommandQueue? {
    guard let commandQueue = device.makeCommandQueue() else {
      return nil
    }
    return CommandQueue(device: device, commandQueue: commandQueue)
  }
  
  @objc
  func makeBuffer(isStorageModeManaged: Bool, bufferSize: Int, count: Int) -> Buffer? {
    let option: MTLResourceOptions = isStorageModeManaged
      ? .storageModeManaged : .storageModeShared
    if let buffer = device.makeBuffer(length: bufferSize, options: option) {
      return Buffer(buffer: buffer, count: count)
    }
    return nil
  }
}
