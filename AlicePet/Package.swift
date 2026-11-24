// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "AlicePet",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .executable(
            name: "AlicePet",
            targets: ["AlicePet"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/arkavo-org/VRMMetalKit", from: "1.3.0")
    ],
    targets: [
        .executableTarget(
            name: "AlicePet",
            dependencies: [
                .product(name: "VRMMetalKit", package: "VRMMetalKit"),
                .product(name: "GLTFMetalKit", package: "VRMMetalKit")
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Info.plist"
                ])
            ]
        )
    ]
)
