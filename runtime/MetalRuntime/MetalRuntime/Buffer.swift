import Metal

@objc
class Buffer: Wrappable {
  let buffer: MTLBuffer
  public let count: Int
  
  init(buffer: MTLBuffer, count: Int) {
    self.buffer = buffer;
    self.count = count
  }
  
  @objc
  func contents() -> UnsafeMutableRawPointer {
    return self.buffer.contents()
  }
  
  @objc
  func getCount() -> Int {
    return count
  }
}
