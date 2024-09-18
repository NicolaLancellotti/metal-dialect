// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "MetalRuntime",
  products: [
    .library(
      name: "MetalRuntime",
      type: .static,
      targets: ["MetalRuntime"]),
  ],
  dependencies: [
  ],
  targets: [
    .target(
      name: "MetalRuntimeSwift"),
    .target(
      name: "MetalRuntime",
      dependencies: ["MetalRuntimeSwift"]),
  ]
)
