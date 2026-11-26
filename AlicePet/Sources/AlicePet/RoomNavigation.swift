import Foundation
import simd

/// Top-down walkable floor grid produced by `services/scene/estimate_scene.py`.
///
/// Every cell is `cellSize` metres; `rows[j][i] == "1"` means Alice may stand there.
/// The grid already has furniture footprints and hip clearance baked in, and it was
/// flood-filled from the spawn, so anything marked walkable is actually reachable.
struct WalkableGrid: Codable, Equatable, Sendable {
    let origin: [Float]
    let cellSize: Float
    let width: Int
    let height: Int
    let rows: [String]
    /// Quadratic ground surface `[c0, cx, cz, cxx, cxz, czz]`, i.e.
    /// `y = c0 + cx·x + cz·z + cxx·x² + cxz·x·z + czz·z²`.
    ///
    /// Depth from a drawn room drifts 15-30 cm across the floor, so a flat Y=0 plane would
    /// leave Alice floating at one end and sunk at the other. The analyser fits this surface
    /// to the visible floor, and everything the avatar stands on is lifted onto it.
    let groundSurface: [Float]?

    func groundHeight(x: Float, z: Float) -> Float {
        guard let g = groundSurface, g.count == 6 else { return 0 }
        return g[0] + g[1] * x + g[2] * z + g[3] * x * x + g[4] * x * z + g[5] * z * z
    }

    private var cells: [Bool] {
        var out = [Bool](repeating: false, count: max(0, width * height))
        guard origin.count == 2, cellSize > 0.001 else { return out }
        for j in 0..<min(height, rows.count) {
            let row = Array(rows[j].utf8)
            for i in 0..<min(width, row.count) where row[i] == UInt8(ascii: "1") {
                out[j * width + i] = true
            }
        }
        return out
    }

    var parsed: ParsedGrid { ParsedGrid(grid: self) }

    /// Pre-parsed form, built once per scene instead of per query.
    struct ParsedGrid: Sendable {
        let originX: Float
        let originZ: Float
        let cell: Float
        let width: Int
        let height: Int
        private let open: [Bool]
        private let ground: [Float]?

        init(grid: WalkableGrid) {
            originX = grid.origin.count == 2 ? grid.origin[0] : 0
            originZ = grid.origin.count == 2 ? grid.origin[1] : 0
            cell = max(grid.cellSize, 0.01)
            width = grid.width
            height = grid.height
            open = grid.cells
            ground = grid.groundSurface
        }

        /// World Y of the floor under a point.
        func groundHeight(x: Float, z: Float) -> Float {
            guard let g = ground, g.count == 6 else { return 0 }
            return g[0] + g[1] * x + g[2] * z + g[3] * x * x + g[4] * x * z + g[5] * z * z
        }

        var isEmpty: Bool { width <= 0 || height <= 0 || open.isEmpty }

        func isWalkable(x: Float, z: Float) -> Bool {
            let i = Int(floor((x - originX) / cell))
            let j = Int(floor((z - originZ) / cell))
            guard i >= 0, j >= 0, i < width, j < height else { return false }
            return open[j * width + i]
        }

        func center(i: Int, j: Int) -> SIMD2<Float> {
            SIMD2<Float>(originX + (Float(i) + 0.5) * cell,
                         originZ + (Float(j) + 0.5) * cell)
        }

        /// Closest walkable cell centre, searched in growing rings.
        func nearestWalkable(x: Float, z: Float, maxRadius: Int = 10) -> SIMD2<Float>? {
            let i0 = Int(floor((x - originX) / cell))
            let j0 = Int(floor((z - originZ) / cell))
            if i0 >= 0, j0 >= 0, i0 < width, j0 < height, open[j0 * width + i0] {
                return center(i: i0, j: j0)
            }
            var best: SIMD2<Float>?
            var bestDist = Float.greatestFiniteMagnitude
            for r in 1...max(1, maxRadius) {
                for dj in -r...r {
                    for di in -r...r where max(abs(di), abs(dj)) == r {
                        let i = i0 + di
                        let j = j0 + dj
                        guard i >= 0, j >= 0, i < width, j < height, open[j * width + i] else { continue }
                        let c = center(i: i, j: j)
                        let d = simd_length_squared(SIMD2<Float>(x - c.x, z - c.y))
                        if d < bestDist {
                            bestDist = d
                            best = c
                        }
                    }
                }
                if best != nil { return best }
            }
            return best
        }

        /// A* over the 8-connected grid, then string-pulled into a handful of straight legs.
        /// Returns world XZ waypoints (excluding the start, ending at the goal) or nil.
        func findPath(from start: SIMD2<Float>, to goal: SIMD2<Float>) -> [SIMD2<Float>]? {
            guard !isEmpty else { return nil }
            guard let startCell = nearestWalkable(x: start.x, z: start.y),
                  let goalCell = nearestWalkable(x: goal.x, z: goal.y) else { return nil }

            let si = Int(floor((startCell.x - originX) / cell))
            let sj = Int(floor((startCell.y - originZ) / cell))
            let gi = Int(floor((goalCell.x - originX) / cell))
            let gj = Int(floor((goalCell.y - originZ) / cell))
            if si == gi && sj == gj { return [goal] }

            let n = width * height
            var gScore = [Float](repeating: .greatestFiniteMagnitude, count: n)
            var cameFrom = [Int32](repeating: -1, count: n)
            var closed = [Bool](repeating: false, count: n)
            let goalIdx = gj * width + gi
            let startIdx = sj * width + si

            func heuristic(_ i: Int, _ j: Int) -> Float {
                let di = Float(abs(i - gi)), dj = Float(abs(j - gj))
                return (di + dj) + (1.41421356 - 2.0) * min(di, dj)
            }

            // Small binary heap; a 0.06 m grid over a room stays tiny.
            var heap: [(f: Float, idx: Int)] = []
            func push(_ item: (f: Float, idx: Int)) {
                heap.append(item)
                var k = heap.count - 1
                while k > 0 {
                    let parent = (k - 1) / 2
                    if heap[parent].f <= heap[k].f { break }
                    heap.swapAt(parent, k)
                    k = parent
                }
            }
            func pop() -> (f: Float, idx: Int)? {
                guard !heap.isEmpty else { return nil }
                let top = heap[0]
                heap[0] = heap[heap.count - 1]
                heap.removeLast()
                var k = 0
                while true {
                    let l = 2 * k + 1, r = 2 * k + 2
                    var m = k
                    if l < heap.count && heap[l].f < heap[m].f { m = l }
                    if r < heap.count && heap[r].f < heap[m].f { m = r }
                    if m == k { break }
                    heap.swapAt(m, k)
                    k = m
                }
                return top
            }

            gScore[startIdx] = 0
            push((heuristic(si, sj), startIdx))

            let neighbours: [(Int, Int, Float)] = [
                (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                (1, 1, 1.41421356), (1, -1, 1.41421356),
                (-1, 1, 1.41421356), (-1, -1, 1.41421356),
            ]

            var found = false
            var guardCounter = 0
            while let cur = pop() {
                if cur.idx == goalIdx { found = true; break }
                if closed[cur.idx] { continue }
                closed[cur.idx] = true
                guardCounter += 1
                if guardCounter > n { break }

                let ci = cur.idx % width
                let cj = cur.idx / width
                for (di, dj, cost) in neighbours {
                    let ni = ci + di, nj = cj + dj
                    guard ni >= 0, nj >= 0, ni < width, nj < height else { continue }
                    let nIdx = nj * width + ni
                    guard open[nIdx], !closed[nIdx] else { continue }
                    // No corner cutting: a diagonal step needs both orthogonal cells open.
                    if di != 0 && dj != 0 {
                        guard open[cj * width + ni], open[nj * width + ci] else { continue }
                    }
                    let tentative = gScore[cur.idx] + cost
                    if tentative < gScore[nIdx] {
                        gScore[nIdx] = tentative
                        cameFrom[nIdx] = Int32(cur.idx)
                        push((tentative + heuristic(ni, nj), nIdx))
                    }
                }
            }

            guard found else { return nil }
            var cellsPath: [Int] = []
            var walk: Int32 = Int32(goalIdx)
            while walk >= 0 {
                cellsPath.append(Int(walk))
                walk = cameFrom[Int(walk)]
            }
            cellsPath.reverse()

            var points: [SIMD2<Float>] = cellsPath.map {
                center(i: $0 % width, j: $0 / width)
            }
            points[0] = startCell
            points[points.count - 1] = goalCell
            let simplified = simplify(points)
            return simplified.isEmpty ? [goal] : simplified
        }

        /// String pulling: drop every intermediate point that is still visible in a straight line.
        private func simplify(_ points: [SIMD2<Float>]) -> [SIMD2<Float>] {
            guard points.count > 2 else {
                return points.dropFirst().map { $0 }
            }
            var result: [SIMD2<Float>] = []
            var anchor = 0
            var probe = 1
            while probe + 1 < points.count {
                if lineIsClear(points[anchor], points[probe + 1]) {
                    probe += 1
                } else {
                    result.append(points[probe])
                    anchor = probe
                    probe += 1
                }
            }
            result.append(points[points.count - 1])
            return result
        }

        private func lineIsClear(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Bool {
            let delta = b - a
            let dist = simd_length(delta)
            guard dist > 0.001 else { return true }
            let dir = delta / dist
            var travelled: Float = 0
            while travelled <= dist {
                let p = a + dir * travelled
                if !isWalkable(x: p.x, z: p.y) { return false }
                travelled += cell * 0.5
            }
            return true
        }
    }
}
