#!/usr/bin/env swift
import AppKit
import CoreGraphics
import Foundation

// Usage: swift make_contact_sheet.swift <output.png> <label1=path1> <label2=path2> ...
let args = Array(CommandLine.arguments.dropFirst())
guard args.count >= 2 else {
    fputs("Usage: make_contact_sheet.swift <output.png> <label=path>...\n", stderr)
    exit(1)
}

let outputPath = args[0]
var tiles: [(label: String, image: NSImage)] = []

for spec in args.dropFirst() {
    guard let eqIdx = spec.firstIndex(of: "=") else { continue }
    let label = String(spec[..<eqIdx])
    let path = String(spec[spec.index(after: eqIdx)...])
    if let img = NSImage(contentsOfFile: path) {
        tiles.append((label: label, image: img))
    } else {
        fputs("Warning: failed to load \(path)\n", stderr)
    }
}

guard !tiles.isEmpty else {
    fputs("Error: no valid images to stitch\n", stderr)
    exit(1)
}

let cols = min(3, tiles.count)
let rows = (tiles.count + cols - 1) / cols
let tileW: CGFloat = 420
let tileH: CGFloat = 580
let headerH: CGFloat = 38
let pad: CGFloat = 18
let footerH: CGFloat = 34

let totalW = CGFloat(cols) * tileW + CGFloat(cols + 1) * pad
let totalH = CGFloat(rows) * (tileH + headerH) + CGFloat(rows + 1) * pad + footerH

guard let rep = NSBitmapImageRep(
    bitmapDataPlanes: nil,
    pixelsWide: Int(totalW),
    pixelsHigh: Int(totalH),
    bitsPerSample: 8,
    samplesPerPixel: 4,
    hasAlpha: true,
    isPlanar: false,
    colorSpaceName: .deviceRGB,
    bytesPerRow: 0,
    bitsPerPixel: 0
) else {
    fputs("Error: failed to create bitmap rep\n", stderr)
    exit(1)
}

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

// Canvas background
NSColor(calibratedRed: 0.10, green: 0.11, blue: 0.13, alpha: 1.0).setFill()
NSRect(x: 0, y: 0, width: totalW, height: totalH).fill()

func drawCheckerboard(in rect: NSRect, cellSize: CGFloat = 16) {
    let dark = NSColor(calibratedWhite: 0.17, alpha: 1.0)
    let light = NSColor(calibratedWhite: 0.22, alpha: 1.0)
    dark.setFill()
    rect.fill()
    light.setFill()
    let colsCount = Int(ceil(rect.width / cellSize))
    let rowsCount = Int(ceil(rect.height / cellSize))
    for r in 0..<rowsCount {
        for c in 0..<colsCount where (r + c) % 2 == 0 {
            let x = rect.minX + CGFloat(c) * cellSize
            let y = rect.minY + CGFloat(r) * cellSize
            let w = min(cellSize, rect.maxX - x)
            let h = min(cellSize, rect.maxY - y)
            NSRect(x: x, y: y, width: w, height: h).fill()
        }
    }
}

let titleAttrs: [NSAttributedString.Key: Any] = [
    .font: NSFont.systemFont(ofSize: 15, weight: .semibold),
    .foregroundColor: NSColor(calibratedWhite: 0.94, alpha: 1.0),
]

for (idx, tile) in tiles.enumerated() {
    let row = idx / cols
    let col = idx % cols
    let x = pad + CGFloat(col) * (tileW + pad)
    // Flip row index so row 0 is at the top (NSGraphicsContext origin is bottom-left)
    let cellTop = totalH - pad - CGFloat(row) * (tileH + headerH + pad)
    let headerRect = NSRect(x: x, y: cellTop - headerH, width: tileW, height: headerH)
    let imageRect = NSRect(x: x, y: headerRect.minY - tileH, width: tileW, height: tileH)

    // Header bar
    NSColor(calibratedRed: 0.16, green: 0.18, blue: 0.22, alpha: 1.0).setFill()
    headerRect.fill()

    let labelStr = NSAttributedString(string: tile.label, attributes: titleAttrs)
    let labelSize = labelStr.size()
    labelStr.draw(at: NSPoint(x: headerRect.minX + 12, y: headerRect.minY + (headerH - labelSize.height) * 0.5))

    // Checkerboard backdrop for alpha edge inspection
    drawCheckerboard(in: imageRect)

    // Aspect-fit image into imageRect
    let srcSize = tile.image.size
    if srcSize.width > 0 && srcSize.height > 0 {
        let scale = min(imageRect.width / srcSize.width, imageRect.height / srcSize.height)
        let drawW = srcSize.width * scale
        let drawH = srcSize.height * scale
        let dest = NSRect(
            x: imageRect.minX + (imageRect.width - drawW) * 0.5,
            y: imageRect.minY + (imageRect.height - drawH) * 0.5,
            width: drawW,
            height: drawH
        )
        tile.image.draw(in: dest, from: .zero, operation: .sourceOver, fraction: 1.0)
    }

    // Subtle tile border
    NSColor(calibratedWhite: 0.30, alpha: 1.0).setStroke()
    let border = NSBezierPath(rect: NSRect(x: x, y: imageRect.minY, width: tileW, height: tileH + headerH))
    border.lineWidth = 1.0
    border.stroke()
}

// Footer timestamp
let df = DateFormatter()
df.dateFormat = "yyyy-MM-dd HH:mm:ss"
let footerText = "AlicePet Visual Regression Sheet · Generated \(df.string(from: Date())) · \(tiles.count) scenes"
let footerAttrs: [NSAttributedString.Key: Any] = [
    .font: NSFont.monospacedSystemFont(ofSize: 12, weight: .regular),
    .foregroundColor: NSColor(calibratedWhite: 0.65, alpha: 1.0),
]
NSAttributedString(string: footerText, attributes: footerAttrs).draw(at: NSPoint(x: pad, y: 10))

NSGraphicsContext.restoreGraphicsState()

guard let pngData = rep.representation(using: .png, properties: [:]) else {
    fputs("Error: failed to encode PNG\n", stderr)
    exit(1)
}

do {
    try pngData.write(to: URL(fileURLWithPath: outputPath))
    print("Contact sheet saved to \(outputPath)")
} catch {
    fputs("Error writing contact sheet: \(error)\n", stderr)
    exit(1)
}
