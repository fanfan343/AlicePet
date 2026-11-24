import AppKit
import MetalKit

@MainActor
final class PetMetalView: MTKView {
    var onOrbitDrag: ((Float, Float) -> Void)?
    var onPanDrag: ((Float, Float) -> Void)?
    var onZoom: ((Float) -> Void)?
    var onDoubleClick: (() -> Void)?
    var contextMenuProvider: (() -> NSMenu)?

    private enum DragMode {
        case none
        case windowMove
        case orbit
        case pan
    }

    private var dragMode: DragMode = .none
    private var dragStartMouseScreen: NSPoint = .zero
    private var dragStartWindowOrigin: NSPoint = .zero
    private var rightDragAccumulated: CGFloat = 0

    override var acceptsFirstResponder: Bool { true }

    override func mouseDown(with event: NSEvent) {
        if event.clickCount == 2 {
            onDoubleClick?()
            return
        }

        if event.modifierFlags.contains(.option) {
            dragMode = .orbit
            return
        }

        if event.modifierFlags.contains(.shift) {
            dragMode = .pan
            return
        }

        guard let window else { return }
        dragMode = .windowMove
        dragStartMouseScreen = NSEvent.mouseLocation
        dragStartWindowOrigin = window.frame.origin
    }

    override func mouseDragged(with event: NSEvent) {
        switch dragMode {
        case .orbit:
            onOrbitDrag?(Float(event.deltaX), Float(event.deltaY))
        case .pan:
            onPanDrag?(Float(event.deltaX), Float(event.deltaY))
        case .windowMove:
            guard let window else { return }
            let currentMouse = NSEvent.mouseLocation
            let dx = currentMouse.x - dragStartMouseScreen.x
            let dy = currentMouse.y - dragStartMouseScreen.y
            window.setFrameOrigin(NSPoint(
                x: dragStartWindowOrigin.x + dx,
                y: dragStartWindowOrigin.y + dy
            ))
        case .none:
            break
        }
    }

    override func mouseUp(with event: NSEvent) {
        dragMode = .none
    }

    // Middle mouse button (buttonNumber == 2) pans the canvas without moving the window
    override func otherMouseDown(with event: NSEvent) {
        if event.buttonNumber == 2 {
            dragMode = .pan
        }
    }

    override func otherMouseDragged(with event: NSEvent) {
        if event.buttonNumber == 2 || dragMode == .pan {
            onPanDrag?(Float(event.deltaX), Float(event.deltaY))
        }
    }

    override func otherMouseUp(with event: NSEvent) {
        if event.buttonNumber == 2 {
            dragMode = .none
        }
    }

    override func rightMouseDown(with event: NSEvent) {
        rightDragAccumulated = 0
    }

    override func rightMouseDragged(with event: NSEvent) {
        rightDragAccumulated += abs(event.deltaX) + abs(event.deltaY)
        onOrbitDrag?(Float(event.deltaX), Float(event.deltaY))
    }

    override func rightMouseUp(with event: NSEvent) {
        if rightDragAccumulated < 4.0, let menu = contextMenuProvider?() {
            NSMenu.popUpContextMenu(menu, with: event, for: self)
        }
    }

    override func scrollWheel(with event: NSEvent) {
        let delta = event.hasPreciseScrollingDeltas ? event.scrollingDeltaY * 0.15 : event.scrollingDeltaY
        onZoom?(Float(delta))
    }
}
