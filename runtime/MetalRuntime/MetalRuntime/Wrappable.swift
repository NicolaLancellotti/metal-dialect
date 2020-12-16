import Foundation

@objc
class Wrappable: NSObject {
  @objc
  func wrap() -> Int {
    let opaquePointer = Unmanaged.passRetained(self).toOpaque()
    return Int(bitPattern: opaquePointer)
  }
  
  @objc
  static func unwrap(_ bitPattern: Int) -> Self {
    if let opaquePointer = UnsafeMutableRawPointer(bitPattern: bitPattern) {
      return Unmanaged.fromOpaque(opaquePointer).takeUnretainedValue()
    } else {
      fatalError(Self.className())
    }
  }
  
  @objc
  static func release(_ bitPattern: Int) {
    let opaquePointer = UnsafeMutableRawPointer(bitPattern: bitPattern)!
    Unmanaged<Self>.fromOpaque(opaquePointer).release()
  }
}
