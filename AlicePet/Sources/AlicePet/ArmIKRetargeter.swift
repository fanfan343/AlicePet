import Foundation
import VRMMetalKit
import simd

/// Universal proportional 2-Bone Arm IK & torso/head anti-clipping solver for VMC bone streams.
///
/// Adapts adult-proportion motion capture / generative skeletons (e.g. ARDY Core27) to narrower
/// anime VRM proportions without arm crossing, chest clipping, or face penetration.
/// Activates automatically only when the incoming VMC frame carries valid FK joint positions
/// (so rotation-only VMC senders like MediaPipe / Waidayo remain untouched).
@MainActor
enum ArmIKRetargeter {
    struct ArmDiagnostics {
        let side: String
        let targetShiftCm: Float
        let clippedAdjusted: Bool
    }

    /// Applies proportional 2-Bone Arm IK and anti-clipping onto `model` after `VMCDriver.apply`.
    /// Returns diagnostics for the left/right arms when IK ran, or empty if the frame has no FK positions.
    @discardableResult
    static func apply(
        to model: VRMModel,
        frame: VMCFrame,
        convention: VMCCoordinateConvention
    ) -> [ArmDiagnostics] {
        guard let humanoid = model.humanoid else { return [] }

        // Require source FK positions for both arms (non-zero arm length guards against rotation-only streams)
        guard let srcLS = srcPos(.leftUpperArm, in: frame, convention: convention),
              let srcLE = srcPos(.leftLowerArm, in: frame, convention: convention),
              let srcLH = srcPos(.leftHand, in: frame, convention: convention),
              let srcRS = srcPos(.rightUpperArm, in: frame, convention: convention),
              let srcRE = srcPos(.rightLowerArm, in: frame, convention: convention),
              let srcRH = srcPos(.rightHand, in: frame, convention: convention) else {
            return []
        }

        let srcLeftLen = simd_length(srcLE - srcLS) + simd_length(srcLH - srcLE)
        let srcRightLen = simd_length(srcRE - srcRS) + simd_length(srcRH - srcRE)
        guard srcLeftLen > 0.10, srcRightLen > 0.10 else { return [] }

        // Resolve target VRM humanoid arm chains
        guard let lUpper = node(.leftUpperArm, humanoid: humanoid, model: model),
              let lLower = node(.leftLowerArm, humanoid: humanoid, model: model),
              let lHand = node(.leftHand, humanoid: humanoid, model: model),
              let rUpper = node(.rightUpperArm, humanoid: humanoid, model: model),
              let rLower = node(.rightLowerArm, humanoid: humanoid, model: model),
              let rHand = node(.rightHand, humanoid: humanoid, model: model) else {
            return []
        }

        // Ensure world matrices reflect the FK pose just written by VMCDriver
        updateRootWorldTransforms(of: model)

        let pLS = lUpper.worldPosition
        let pLE = lLower.worldPosition
        let pLH = lHand.worldPosition
        let pRS = rUpper.worldPosition
        let pRE = rLower.worldPosition
        let pRH = rHand.worldPosition

        let tgtL1 = simd_length(pLE - pLS)
        let tgtL2 = simd_length(pLH - pLE)
        let tgtR1 = simd_length(pRE - pRS)
        let tgtR2 = simd_length(pRH - pRE)
        guard tgtL1 > 0.02, tgtL2 > 0.02, tgtR1 > 0.02, tgtR2 > 0.02 else { return [] }

        let srcMid = 0.5 * (srcLS + srcRS)
        let tgtMid = 0.5 * (pLS + pRS)
        let srcShoulderSpan = simd_length(srcLS - srcRS)
        let tgtShoulderSpan = simd_length(pLS - pRS)
        guard srcShoulderSpan > 0.05, tgtShoulderSpan > 0.04 else { return [] }

        let srcHalfShoulder = 0.5 * srcShoulderSpan
        let tgtHalfShoulder = 0.5 * tgtShoulderSpan
        let srcLeftAxis = safeNormalize(srcLS - srcRS, fallback: SIMD3<Float>(1, 0, 0))
        let tgtLeftAxis = safeNormalize(pLS - pRS, fallback: SIMD3<Float>(1, 0, 0))

        // Estimate torso up & forward axes in target model space for anatomical pole fallbacks and capsules
        let hipsPos = node(.hips, humanoid: humanoid, model: model)?.worldPosition
            ?? (tgtMid - SIMD3<Float>(0, tgtShoulderSpan * 1.8, 0))
        let neckPos = node(.neck, humanoid: humanoid, model: model)?.worldPosition
            ?? (tgtMid + SIMD3<Float>(0, tgtShoulderSpan * 0.25, 0))
        let headPos = node(.head, humanoid: humanoid, model: model)?.worldPosition
            ?? (neckPos + SIMD3<Float>(0, tgtShoulderSpan * 0.35, 0))

        let torsoUp = safeNormalize(neckPos - hipsPos, fallback: SIMD3<Float>(0, 1, 0))
        let torsoFwd = safeNormalize(simd_cross(tgtLeftAxis, torsoUp), fallback: SIMD3<Float>(0, 0, 1))

        // Dynamic body & flared-skirt collision primitives scaled proportionally to the loaded VRM
        let torsoCapsuleStart = hipsPos + torsoUp * (tgtHalfShoulder * 0.35)
        let torsoCapsuleEnd = neckPos - torsoUp * (tgtHalfShoulder * 0.15)
        let torsoRadius = max(0.075, tgtHalfShoulder * 0.88)

        // Tapered hip/skirt clearance cone below waist so resting & swinging hands glide cleanly
        // outside flared anime dresses instead of burying into or snagging the skirt
        let skirtConeTop = hipsPos + torsoUp * 0.02
        let skirtConeBottom = hipsPos - torsoUp * max(0.32, (tgtL1 + tgtL2) * 0.85)
        let skirtTopRadius = torsoRadius * 1.12
        let skirtBottomRadius = max(0.18, torsoRadius * 2.15)

        let headRadius = max(0.105, tgtHalfShoulder * 1.15)
        let headCenter = headPos + torsoUp * (headRadius * 0.72)

        // Compute proportional world hand targets (lateral midline-preserving inside shoulder width,
        // arm-length scaled outside shoulder width and along vertical/depth axes)
        let leftArmScale = (tgtL1 + tgtL2) / srcLeftLen
        let rightArmScale = (tgtR1 + tgtR2) / srcRightLen

        var targetLH = mapHandTarget(
            srcHand: srcLH,
            srcShoulder: srcLS,
            srcMid: srcMid,
            srcHalfShoulder: srcHalfShoulder,
            srcLeftAxis: srcLeftAxis,
            tgtShoulder: pLS,
            tgtMid: tgtMid,
            tgtHalfShoulder: tgtHalfShoulder,
            tgtLeftAxis: tgtLeftAxis,
            armScale: leftArmScale
        )
        var targetRH = mapHandTarget(
            srcHand: srcRH,
            srcShoulder: srcRS,
            srcMid: srcMid,
            srcHalfShoulder: srcHalfShoulder,
            srcLeftAxis: srcLeftAxis,
            tgtShoulder: pRS,
            tgtMid: tgtMid,
            tgtHalfShoulder: tgtHalfShoulder,
            tgtLeftAxis: tgtLeftAxis,
            armScale: rightArmScale
        )

        // Push hand targets outside torso capsule, flared skirt cone, & head sphere
        let handClearance: Float = 0.032
        let (lh1, lhClip1) = pushOutOfCapsule(
            point: targetLH,
            segA: torsoCapsuleStart,
            segB: torsoCapsuleEnd,
            radius: torsoRadius + handClearance,
            preferredOutward: safeNormalize(tgtLeftAxis * 0.35 + torsoFwd * 0.94, fallback: torsoFwd)
        )
        let (lh1b, lhClipSkirt) = pushOutOfTaperedCone(
            point: lh1,
            top: skirtConeTop,
            bottom: skirtConeBottom,
            topRadius: skirtTopRadius + handClearance,
            bottomRadius: skirtBottomRadius + handClearance,
            preferredOutward: safeNormalize(tgtLeftAxis * 0.85 + torsoFwd * 0.52, fallback: tgtLeftAxis)
        )
        let (lh2, lhClip2) = pushOutOfSphere(
            point: lh1b,
            center: headCenter,
            radius: headRadius + handClearance,
            preferredOutward: safeNormalize(tgtLeftAxis * 0.5 + torsoFwd * 0.86, fallback: torsoFwd)
        )
        targetLH = lh2

        let (rh1, rhClip1) = pushOutOfCapsule(
            point: targetRH,
            segA: torsoCapsuleStart,
            segB: torsoCapsuleEnd,
            radius: torsoRadius + handClearance,
            preferredOutward: safeNormalize(-tgtLeftAxis * 0.35 + torsoFwd * 0.94, fallback: torsoFwd)
        )
        let (rh1b, rhClipSkirt) = pushOutOfTaperedCone(
            point: rh1,
            top: skirtConeTop,
            bottom: skirtConeBottom,
            topRadius: skirtTopRadius + handClearance,
            bottomRadius: skirtBottomRadius + handClearance,
            preferredOutward: safeNormalize(-tgtLeftAxis * 0.85 + torsoFwd * 0.52, fallback: -tgtLeftAxis)
        )
        let (rh2, rhClip2) = pushOutOfSphere(
            point: rh1b,
            center: headCenter,
            radius: headRadius + handClearance,
            preferredOutward: safeNormalize(-tgtLeftAxis * 0.5 + torsoFwd * 0.86, fallback: torsoFwd)
        )
        targetRH = rh2

        // Prevent left/right hands from crossing or interpenetrating during claps & close gestures.
        // Note: `targetLH/RH` are WRIST joints; each palm+fingers extends ~7-8cm inward when angled
        // toward midline, so wrists need ~9.5cm signed lateral separation to let palms kiss cleanly
        // without X-crossing through each other.
        let srcLatSep = simd_dot(srcLH - srcRH, srcLeftAxis)
        var handsSeparated = false
        let handVec = targetLH - targetRH
        let curLatSep = simd_dot(handVec, tgtLeftAxis)
        let handDY = abs(simd_dot(handVec, torsoUp))
        let handDZ = abs(simd_dot(handVec, torsoFwd))

        if srcLatSep >= -0.025 && handDY < 0.16 && handDZ < 0.16 {
            // Non-crossing palm-aware clap separation along the canonical shoulder axis (+tgtLeftAxis)
            let minClapWristSep: Float = max(0.112, tgtHalfShoulder * 1.68)
            if curLatSep < minClapWristSep {
                let push = 0.5 * (minClapWristSep - curLatSep)
                targetLH += tgtLeftAxis * push
                targetRH -= tgtLeftAxis * push
                handsSeparated = true
            }
        } else {
            let minHandSeparation: Float = 0.060
            let handDist = simd_length(handVec)
            if handDist < minHandSeparation {
                let sepDir = (handDist > 1e-4) ? (handVec / handDist) : tgtLeftAxis
                let push = 0.5 * (minHandSeparation - handDist)
                targetLH += sepDir * push
                targetRH -= sepDir * push
                handsSeparated = true
            }
        }

        // Solve Left & Right 2-Bone IK chains while preserving FK axial twist and wrist orientation
        let leftDiag = solveArmChain(
            side: "Left",
            isLeft: true,
            upperNode: lUpper,
            lowerNode: lLower,
            handNode: lHand,
            l1: tgtL1,
            l2: tgtL2,
            targetHandPos: targetLH,
            srcElbowOffset: srcLE - srcLS,
            torsoCapsuleA: torsoCapsuleStart,
            torsoCapsuleB: torsoCapsuleEnd,
            torsoRadius: torsoRadius,
            tgtLeftAxis: tgtLeftAxis,
            torsoUp: torsoUp,
            torsoFwd: torsoFwd,
            hadTargetClip: lhClip1 || lhClipSkirt || lhClip2 || handsSeparated
        )
        let rightDiag = solveArmChain(
            side: "Right",
            isLeft: false,
            upperNode: rUpper,
            lowerNode: rLower,
            handNode: rHand,
            l1: tgtR1,
            l2: tgtR2,
            targetHandPos: targetRH,
            srcElbowOffset: srcRE - srcRS,
            torsoCapsuleA: torsoCapsuleStart,
            torsoCapsuleB: torsoCapsuleEnd,
            torsoRadius: torsoRadius,
            tgtLeftAxis: tgtLeftAxis,
            torsoUp: torsoUp,
            torsoFwd: torsoFwd,
            hadTargetClip: rhClip1 || rhClipSkirt || rhClip2 || handsSeparated
        )

        updateRootWorldTransforms(of: model)
        return [leftDiag, rightDiag]
    }

    // MARK: - Proportional Mapping

    private static func mapHandTarget(
        srcHand: SIMD3<Float>,
        srcShoulder: SIMD3<Float>,
        srcMid: SIMD3<Float>,
        srcHalfShoulder: Float,
        srcLeftAxis: SIMD3<Float>,
        tgtShoulder: SIMD3<Float>,
        tgtMid: SIMD3<Float>,
        tgtHalfShoulder: Float,
        tgtLeftAxis: SIMD3<Float>,
        armScale: Float
    ) -> SIMD3<Float> {
        let srcRelMid = srcHand - srcMid
        let srcLat = simd_dot(srcRelMid, srcLeftAxis)

        // Inside shoulder width: scale by shoulder width ratio so midline gestures (claps, chest touches)
        // never overshoot/cross on narrow-shoulder anime rigs. Outside shoulder width: scale by arm length.
        let absLat = abs(srcLat)
        let signLat: Float = (srcLat >= 0) ? 1.0 : -1.0
        let tgtLat: Float
        if absLat <= srcHalfShoulder {
            tgtLat = srcLat * (tgtHalfShoulder / srcHalfShoulder)
        } else {
            tgtLat = signLat * (tgtHalfShoulder + (absLat - srcHalfShoulder) * armScale)
        }

        // Preserve non-lateral (vertical + front/back) shoulder-relative vector aligned to target shoulder axis
        let srcRelShoulder = srcHand - srcShoulder
        let srcPerp = srcRelShoulder - simd_dot(srcRelShoulder, srcLeftAxis) * srcLeftAxis
        let alignQ = quatFromTo(srcLeftAxis, tgtLeftAxis)
        let alignedPerp = alignQ.act(srcPerp)
        let tgtShoulderPerp = (tgtShoulder - tgtMid) - simd_dot(tgtShoulder - tgtMid, tgtLeftAxis) * tgtLeftAxis

        return tgtMid + tgtLeftAxis * tgtLat + tgtShoulderPerp + alignedPerp * armScale
    }

    // MARK: - Analytic 2-Bone IK Chain Solver

    private static func solveArmChain(
        side: String,
        isLeft: Bool,
        upperNode: VRMNode,
        lowerNode: VRMNode,
        handNode: VRMNode,
        l1: Float,
        l2: Float,
        targetHandPos: SIMD3<Float>,
        srcElbowOffset: SIMD3<Float>,
        torsoCapsuleA: SIMD3<Float>,
        torsoCapsuleB: SIMD3<Float>,
        torsoRadius: Float,
        tgtLeftAxis: SIMD3<Float>,
        torsoUp: SIMD3<Float>,
        torsoFwd: SIMD3<Float>,
        hadTargetClip: Bool
    ) -> ArmDiagnostics {
        let pS = upperNode.worldPosition
        let pEFk = lowerNode.worldPosition
        let pHFk = handNode.worldPosition
        let fkHandWorldRot = worldRotation(of: handNode)

        let rawVec = targetHandPos - pS
        let rawDist = simd_length(rawVec)
        let minReach = max(abs(l1 - l2) + 0.015, 0.04)
        let maxReach = (l1 + l2) * 0.995
        let targetDir = safeNormalize(rawVec, fallback: safeNormalize(pHFk - pS, fallback: isLeft ? tgtLeftAxis : -tgtLeftAxis))
        let clampedDist = min(max(rawDist, minReach), maxReach)
        let pTarget = pS + targetDir * clampedDist

        // Derive pole direction from FK elbow first, falling back to source elbow / anatomical outward-back
        let sideSign: Float = isLeft ? 1.0 : -1.0
        let anatomicalPole = safeNormalize(
            tgtLeftAxis * (sideSign * 0.55) - torsoFwd * 0.72 - torsoUp * 0.42,
            fallback: SIMD3<Float>(sideSign, -0.5, -0.7)
        )
        var poleDir = perpendicularUnit(pEFk - pS, to: targetDir)
            ?? perpendicularUnit(srcElbowOffset, to: targetDir)
            ?? perpendicularUnit(anatomicalPole, to: targetDir)
            ?? anatomicalPole

        var pEIk = solveElbowPosition(
            shoulder: pS,
            targetDir: targetDir,
            dist: clampedDist,
            l1: l1,
            l2: l2,
            poleDir: poleDir
        )

        // Elbow torso anti-clipping: if solved elbow dips inside torso capsule, swing pole outward and re-solve
        let (pushedElbow, elbowClipped) = pushOutOfCapsule(
            point: pEIk,
            segA: torsoCapsuleA,
            segB: torsoCapsuleB,
            radius: torsoRadius + 0.022,
            preferredOutward: anatomicalPole
        )
        if elbowClipped, let adjustedPole = perpendicularUnit(pushedElbow - pS, to: targetDir) {
            poleDir = adjustedPole
            pEIk = solveElbowPosition(
                shoulder: pS,
                targetDir: targetDir,
                dist: clampedDist,
                l1: l1,
                l2: l2,
                poleDir: poleDir
            )
        }

        // 1. Apply minimal world-space swing onto UpperArm (preserves FK axial roll/twist)
        let uFk = safeNormalize(pEFk - pS, fallback: targetDir)
        let uIk = safeNormalize(pEIk - pS, fallback: targetDir)
        let qUpperSwing = quatFromTo(uFk, uIk)
        let upperParentRot = worldRotation(of: upperNode.parent)
        let upperWorldFk = simd_normalize(upperParentRot * upperNode.rotation)
        let upperWorldIk = simd_normalize(qUpperSwing * upperWorldFk)
        upperNode.rotation = simd_normalize(upperParentRot.inverse * upperWorldIk)
        upperNode.updateLocalMatrix()
        upperNode.updateWorldTransform()

        // 2. Apply minimal world-space swing onto LowerArm to land Hand exactly on pTarget
        let pECur = lowerNode.worldPosition
        let pHCur = handNode.worldPosition
        let fCur = safeNormalize(pHCur - pECur, fallback: targetDir)
        let fIk = safeNormalize(pTarget - pECur, fallback: targetDir)
        let qLowerSwing = quatFromTo(fCur, fIk)
        let lowerParentRot = worldRotation(of: lowerNode.parent)
        let lowerWorldCur = simd_normalize(lowerParentRot * lowerNode.rotation)
        let lowerWorldIk = simd_normalize(qLowerSwing * lowerWorldCur)
        lowerNode.rotation = simd_normalize(lowerParentRot.inverse * lowerWorldIk)
        lowerNode.updateLocalMatrix()
        lowerNode.updateWorldTransform()

        // 3. Anatomical Forearm-Twist Split + Wrist Joint Limits (eliminates candy-wrapper wrist pinching
        //    and extreme 60-80° side-bends from 1-DOF elbow mocap skeletons like ARDY Core27)
        let identityQ = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        let handParentRot = worldRotation(of: handNode.parent)
        var rawHandLocal = simd_normalize(handParentRot.inverse * fkHandWorldRot)
        if rawHandLocal.real < 0 { rawHandLocal = -rawHandLocal }

        // Self-calibrating longitudinal forearm axis in bone-local coordinates (works for VRM 0.x & 1.0)
        let localArmAxis = safeNormalize(handNode.translation, fallback: SIMD3<Float>(isLeft ? -1 : 1, 0, 0))
        let (rawSwing, rawTwistAngle) = decomposeSwingTwist(rawHandLocal, axis: localArmAxis)

        // Transfer 55% of pronation/supination roll onto the forearm bone, keeping 45% gently on the wrist
        let clampedTotalTwist = min(max(rawTwistAngle, -1.15), 1.15) // ±66° total forearm+wrist roll
        let forearmTwistAngle = clampedTotalTwist * 0.55
        let wristTwistAngle = min(max(clampedTotalTwist * 0.45, -0.40), 0.40) // ±23° max wrist roll

        if abs(forearmTwistAngle) > 0.005 {
            let qForearmTwist = simd_quatf(angle: forearmTwistAngle, axis: localArmAxis)
            lowerNode.rotation = simd_normalize(lowerNode.rotation * qForearmTwist)
            lowerNode.updateLocalMatrix()
            lowerNode.updateWorldTransform()
        }

        // Softly damp and clamp wrist flexion/deviation swing so wrists stay graceful (<= 24°)
        let dampedSwing = clampQuaternionAngle(
            simd_slerp(identityQ, rawSwing, 0.58),
            maxAngleRad: 24.0 * .pi / 180.0
        )
        let qWristTwist = simd_quatf(angle: wristTwistAngle, axis: localArmAxis)
        handNode.rotation = simd_normalize(dampedSwing * qWristTwist)
        handNode.updateLocalMatrix()
        handNode.updateWorldTransform()

        let shiftCm = simd_length(pTarget - pHFk) * 100.0
        return ArmDiagnostics(side: side, targetShiftCm: shiftCm, clippedAdjusted: hadTargetClip || elbowClipped)
    }

    /// Decomposes a unit quaternion `q` into `swing * twist` where `twist` is a pure rotation
    /// around `axis` (unit vector) and returns `(swing, signedTwistAngleRadians)`.
    private static func decomposeSwingTwist(_ qIn: simd_quatf, axis: SIMD3<Float>) -> (swing: simd_quatf, twistAngle: Float) {
        var q = simd_normalize(qIn)
        if q.real < 0 { q = -q }
        let proj = simd_dot(q.imag, axis) * axis
        let twistLenSq = simd_length_squared(proj) + q.real * q.real
        let twist: simd_quatf
        if twistLenSq > 1e-8 {
            let invLen = 1.0 / sqrt(twistLenSq)
            twist = simd_quatf(ix: proj.x * invLen, iy: proj.y * invLen, iz: proj.z * invLen, r: q.real * invLen)
        } else {
            twist = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        }
        let swing = simd_normalize(q * twist.inverse)
        let sinHalf = simd_dot(twist.imag, axis)
        let twistAngle = 2.0 * atan2(sinHalf, twist.real)
        return (swing, twistAngle)
    }

    private static func clampQuaternionAngle(_ qIn: simd_quatf, maxAngleRad: Float) -> simd_quatf {
        var q = simd_normalize(qIn)
        if q.real < 0 { q = -q }
        let halfAngle = acos(min(max(q.real, -1.0), 1.0))
        let angle = 2.0 * halfAngle
        guard angle > maxAngleRad, angle > 1e-5 else { return q }
        let sinHalf = sin(halfAngle)
        guard sinHalf > 1e-5 else { return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1) }
        let axis = q.imag / sinHalf
        return simd_quatf(angle: maxAngleRad, axis: axis)
    }

    private static func pushOutOfTaperedCone(
        point: SIMD3<Float>,
        top: SIMD3<Float>,
        bottom: SIMD3<Float>,
        topRadius: Float,
        bottomRadius: Float,
        preferredOutward: SIMD3<Float>
    ) -> (SIMD3<Float>, Bool) {
        let ab = bottom - top
        let abLenSq = simd_length_squared(ab)
        guard abLenSq > 1e-6 else { return (point, false) }
        let rawT = simd_dot(point - top, ab) / abLenSq
        // Only apply skirt clearance when the hand is actually in the vertical waist-to-hem band
        guard rawT >= -0.05 && rawT <= 1.08 else { return (point, false) }
        let t = min(max(rawT, 0.0), 1.0)
        let axisPt = top + ab * t
        let radiusAtT = topRadius + (bottomRadius - topRadius) * t
        let radial = point - axisPt
        let dist = simd_length(radial)
        guard dist < radiusAtT else { return (point, false) }
        let outDir = (dist > 1e-4) ? (radial / dist) : preferredOutward
        return (axisPt + outDir * radiusAtT, true)
    }

    private static func solveElbowPosition(
        shoulder: SIMD3<Float>,
        targetDir: SIMD3<Float>,
        dist: Float,
        l1: Float,
        l2: Float,
        poleDir: SIMD3<Float>
    ) -> SIMD3<Float> {
        let denom = max(2.0 * l1 * dist, 1e-5)
        let cosAlpha = min(max((l1 * l1 + dist * dist - l2 * l2) / denom, -1.0), 1.0)
        let sinAlpha = sqrt(max(0.0, 1.0 - cosAlpha * cosAlpha))
        return shoulder + targetDir * (l1 * cosAlpha) + poleDir * (l1 * sinAlpha)
    }

    // MARK: - Collision Helpers

    private static func pushOutOfCapsule(
        point: SIMD3<Float>,
        segA: SIMD3<Float>,
        segB: SIMD3<Float>,
        radius: Float,
        preferredOutward: SIMD3<Float>
    ) -> (SIMD3<Float>, Bool) {
        let ab = segB - segA
        let abLenSq = simd_length_squared(ab)
        let t: Float = abLenSq > 1e-6 ? min(max(simd_dot(point - segA, ab) / abLenSq, 0.0), 1.0) : 0.0
        let closest = segA + ab * t
        let diff = point - closest
        let dist = simd_length(diff)
        guard dist < radius else { return (point, false) }
        let outDir = (dist > 1e-4) ? (diff / dist) : preferredOutward
        return (closest + outDir * radius, true)
    }

    private static func pushOutOfSphere(
        point: SIMD3<Float>,
        center: SIMD3<Float>,
        radius: Float,
        preferredOutward: SIMD3<Float>
    ) -> (SIMD3<Float>, Bool) {
        let diff = point - center
        let dist = simd_length(diff)
        guard dist < radius else { return (point, false) }
        let outDir = (dist > 1e-4) ? (diff / dist) : preferredOutward
        return (center + outDir * radius, true)
    }

    // MARK: - Node & Quaternion Utilities

    private static func srcPos(
        _ bone: VRMHumanoidBone,
        in frame: VMCFrame,
        convention: VMCCoordinateConvention
    ) -> SIMD3<Float>? {
        guard let tr = frame.bones[bone] else { return nil }
        return convention.convert(position: tr.position)
    }

    private static func node(_ bone: VRMHumanoidBone, humanoid: VRMHumanoid, model: VRMModel) -> VRMNode? {
        guard let idx = humanoid.getBoneNode(bone), idx < model.nodes.count else { return nil }
        return model.nodes[idx]
    }

    private static func updateRootWorldTransforms(of model: VRMModel) {
        for node in model.nodes where node.parent == nil {
            node.updateWorldTransform()
        }
    }

    private static func worldRotation(of node: VRMNode?) -> simd_quatf {
        var rot = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        var cursor = node
        while let current = cursor {
            rot = current.rotation * rot
            cursor = current.parent
        }
        return simd_normalize(rot)
    }

    private static func safeNormalize(_ v: SIMD3<Float>, fallback: SIMD3<Float>) -> SIMD3<Float> {
        let len = simd_length(v)
        return len > 1e-5 ? (v / len) : simd_normalize(fallback)
    }

    private static func perpendicularUnit(_ v: SIMD3<Float>, to axis: SIMD3<Float>) -> SIMD3<Float>? {
        let perp = v - simd_dot(v, axis) * axis
        let len = simd_length(perp)
        return len > 1e-4 ? (perp / len) : nil
    }

    private static func quatFromTo(_ from: SIMD3<Float>, _ to: SIMD3<Float>) -> simd_quatf {
        let a = safeNormalize(from, fallback: SIMD3<Float>(1, 0, 0))
        let b = safeNormalize(to, fallback: a)
        let d = min(max(simd_dot(a, b), -1.0), 1.0)
        if d > 0.99999 {
            return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        }
        if d < -0.99999 {
            let ref = (abs(a.x) < 0.9) ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(0, 1, 0)
            let ortho = safeNormalize(simd_cross(a, ref), fallback: SIMD3<Float>(0, 0, 1))
            return simd_quatf(angle: .pi, axis: ortho)
        }
        let axis = simd_cross(a, b)
        return simd_normalize(simd_quatf(ix: axis.x, iy: axis.y, iz: axis.z, r: 1.0 + d))
    }
}
