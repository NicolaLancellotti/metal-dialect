import Metal

@objc
public class Buffer: Wrappable {
  let buffer: MTLBuffer
  public let count: Int
  
  init(buffer: MTLBuffer, count: Int) {
    self.buffer = buffer;
    self.count = count
  }
  
  @objc
  public func contents() -> UnsafeMutableRawPointer {
    return self.buffer.contents()
  }
  
  @objc
  public func getCount() -> Int {
    return count
  }
}
