//
//  ARDrawingView.swift
//  AirPaintAR
//
//  ARKit scene view for 3D stroke rendering
//

import SwiftUI
import ARKit
import SceneKit

struct ARDrawingView: UIViewRepresentable {
    @ObservedObject var drawingVM: DrawingViewModel

    func makeUIView(context: Context) -> ARSCNView {
        let arView = ARSCNView(frame: .zero)
        arView.delegate = context.coordinator
        arView.session.delegate = context.coordinator
        arView.autoenablesDefaultLighting = true
        arView.automaticallyUpdatesLighting = true

        // Debug stats (disable feature points - only show drawing)
        arView.showsStatistics = false
        // arView.debugOptions = [.showFeaturePoints]  // Disabled - too noisy

        // Configure AR session
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = []  // Don't detect planes

        // NOTE: We intentionally DO NOT use ARKit's image detection for the ArUco marker.
        // The Mac server handles ArUco detection and sends world-space coordinates.
        // Using ARKit's separate marker detection causes coordinate system mismatch
        // because Mac and iPhone detect the marker from different angles/algorithms.
        //
        // Original code disabled:
        // if let markerImage = createArUcoMarkerImage() { ... }
        print("ℹ️ ArUco detection handled by Mac server (world-space coordinates)")

        print("🚀 Starting AR session...")
        arView.session.run(configuration)

        // Store reference for updates
        context.coordinator.arView = arView
        context.coordinator.drawingVM = drawingVM

        // Connect coordinator to ViewModel
        drawingVM.setARCoordinator(context.coordinator)

        // Show origin axes to help user orient - will be 1m in front
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            context.coordinator.showOriginAxes()
        }

        return arView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        // Updates handled by coordinator
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    // ArUco marker image loading removed - Mac handles all ArUco detection

    class Coordinator: NSObject, ARSCNViewDelegate, ARSessionDelegate {
        weak var arView: ARSCNView?
        weak var drawingVM: DrawingViewModel?

        // Stroke rendering
        private var strokeNodes: [Int: SCNNode] = [:]  // stroke_id -> parent node
        private var lastPointPerStroke: [Int: SCNVector3] = [:]

        // MARK: - ARSessionDelegate

        func session(_ session: ARSession, didFailWithError error: Error) {
            print("❌ AR Session failed: \(error.localizedDescription)")
        }

        func sessionWasInterrupted(_ session: ARSession) {
            print("⚠️ AR Session interrupted")
        }

        func sessionInterruptionEnded(_ session: ARSession) {
            print("✅ AR Session interruption ended")
        }

        // MARK: - ARSCNViewDelegate
        // Note: Image anchor detection disabled - Mac handles ArUco detection

        // MARK: - Origin Visualization

        private var originNode: SCNNode?

        /// Add visual axes at the drawing origin (1m in front of start position)
        /// This helps users understand where drawing will appear and its orientation
        func showOriginAxes() {
            guard let arView = arView, originNode == nil else { return }

            let origin = SCNNode()
            origin.position = SCNVector3(0, 0, -1.0)  // 1m in front (Z=-1 in ARKit)

            let axisLength: Float = 0.15  // 15cm axes
            let axisRadius: CGFloat = 0.003

            // X axis - Red (right)
            let xAxis = SCNCylinder(radius: axisRadius, height: CGFloat(axisLength))
            xAxis.firstMaterial?.diffuse.contents = UIColor.red
            xAxis.firstMaterial?.emission.contents = UIColor.red.withAlphaComponent(0.3)
            let xNode = SCNNode(geometry: xAxis)
            xNode.position = SCNVector3(axisLength / 2, 0, 0)
            xNode.eulerAngles.z = -.pi / 2
            origin.addChildNode(xNode)

            // Y axis - Green (up in ARKit)
            let yAxis = SCNCylinder(radius: axisRadius, height: CGFloat(axisLength))
            yAxis.firstMaterial?.diffuse.contents = UIColor.green
            yAxis.firstMaterial?.emission.contents = UIColor.green.withAlphaComponent(0.3)
            let yNode = SCNNode(geometry: yAxis)
            yNode.position = SCNVector3(0, axisLength / 2, 0)
            origin.addChildNode(yNode)

            // Z axis - Blue (toward user in ARKit, which is "forward" from Mac's view)
            let zAxis = SCNCylinder(radius: axisRadius, height: CGFloat(axisLength))
            zAxis.firstMaterial?.diffuse.contents = UIColor.blue
            zAxis.firstMaterial?.emission.contents = UIColor.blue.withAlphaComponent(0.3)
            let zNode = SCNNode(geometry: zAxis)
            zNode.position = SCNVector3(0, 0, axisLength / 2)
            zNode.eulerAngles.x = .pi / 2
            origin.addChildNode(zNode)

            // Small sphere at origin
            let centerSphere = SCNSphere(radius: 0.01)
            centerSphere.firstMaterial?.diffuse.contents = UIColor.white
            let centerNode = SCNNode(geometry: centerSphere)
            origin.addChildNode(centerNode)

            // Label
            let text = SCNText(string: "ORIGIN", extrusionDepth: 0.5)
            text.font = UIFont.systemFont(ofSize: 10)
            text.firstMaterial?.diffuse.contents = UIColor.white
            let textNode = SCNNode(geometry: text)
            textNode.scale = SCNVector3(0.002, 0.002, 0.002)
            textNode.position = SCNVector3(-0.03, 0.05, 0)
            origin.addChildNode(textNode)

            arView.scene.rootNode.addChildNode(origin)
            originNode = origin

            print("📍 Origin axes added at (0, 0, -1) - point iPhone here to see drawing")
        }

        /// Hide origin axes (e.g., when drawing starts)
        func hideOriginAxes() {
            originNode?.removeFromParentNode()
            originNode = nil
        }

        // MARK: - Welding Animation Constants

        private let weldingDuration: TimeInterval = 0.8  // Slower fade
        private let weldingStartScale: Float = 3.0
        private let weldingGlowIntensity: CGFloat = 5.0  // Much brighter - hot white
        private let baseRadius: CGFloat = 0.002  // 2mm base radius

        // MARK: - Point Rendering

        func addPoint(_ point: SIMD3<Float>, strokeId: Int, color: UIColor = .red) {
            guard let arView = arView else {
                print("❌ addPoint: arView is nil")
                return
            }

            print("📍 Received point: (\(point.x), \(point.y), \(point.z)) stroke=\(strokeId)")

            // Transform point from Mac's coordinate system to ARKit
            let transformedPoint = transformPointToARKit(point)
            print("   Transformed to: (\(transformedPoint.x), \(transformedPoint.y), \(transformedPoint.z))")

            // Get or create stroke parent node
            let strokeNode: SCNNode
            if let existing = strokeNodes[strokeId] {
                strokeNode = existing
            } else {
                strokeNode = SCNNode()
                strokeNode.name = "stroke_\(strokeId)"
                arView.scene.rootNode.addChildNode(strokeNode)
                strokeNodes[strokeId] = strokeNode
            }

            // Create welding spark at the new point
            let sparkNode = createWeldingSpark(at: transformedPoint, color: color)
            strokeNode.addChildNode(sparkNode)

            // Create cylinder from last point to this point
            if let lastPoint = lastPointPerStroke[strokeId] {
                let cylinderNode = createCylinder(
                    from: lastPoint,
                    to: transformedPoint,
                    radius: baseRadius,
                    color: color
                )
                // Apply welding glow to cylinder too
                applyWeldingGlow(to: cylinderNode, color: color)
                strokeNode.addChildNode(cylinderNode)
            }

            lastPointPerStroke[strokeId] = transformedPoint
        }

        // MARK: - Welding Effect

        /// Create a welding spark effect - starts big and bright, shrinks and dims
        private func createWeldingSpark(at position: SCNVector3, color: UIColor) -> SCNNode {
            // Container node for spark + glow halo
            let containerNode = SCNNode()
            containerNode.position = position

            // Inner bright core - hot white
            let coreRadius = baseRadius * 1.5
            let core = SCNSphere(radius: coreRadius)
            let coreMaterial = SCNMaterial()
            coreMaterial.diffuse.contents = UIColor.white
            coreMaterial.emission.contents = UIColor.white
            coreMaterial.emission.intensity = weldingGlowIntensity
            coreMaterial.lightingModel = .constant  // Self-illuminated
            core.firstMaterial = coreMaterial

            let coreNode = SCNNode(geometry: core)

            // Outer glow halo - larger, more transparent
            let haloRadius = baseRadius * 4.0
            let halo = SCNSphere(radius: haloRadius)
            let haloMaterial = SCNMaterial()
            haloMaterial.diffuse.contents = UIColor.white.withAlphaComponent(0.6)
            haloMaterial.emission.contents = UIColor(red: 1.0, green: 0.9, blue: 0.7, alpha: 1.0)  // Warm white
            haloMaterial.emission.intensity = weldingGlowIntensity * 0.8
            haloMaterial.lightingModel = .constant
            haloMaterial.isDoubleSided = true
            haloMaterial.blendMode = .add  // Additive blending for glow
            halo.firstMaterial = haloMaterial

            let haloNode = SCNNode(geometry: halo)

            containerNode.addChildNode(haloNode)
            containerNode.addChildNode(coreNode)

            // Start scaled up (puffed)
            containerNode.scale = SCNVector3(weldingStartScale, weldingStartScale, weldingStartScale)

            // Animate scale down
            let scaleDown = SCNAction.scale(to: 1.0, duration: weldingDuration)
            scaleDown.timingMode = .easeOut

            // Animate core glow fade
            let fadeCoreGlow = SCNAction.customAction(duration: weldingDuration) { _, elapsedTime in
                let progress = elapsedTime / CGFloat(self.weldingDuration)
                let intensity = self.weldingGlowIntensity * (1.0 - progress)
                coreMaterial.emission.intensity = intensity
                // Transition from white to red
                coreMaterial.diffuse.contents = UIColor.white.interpolate(to: color, progress: progress)
                coreMaterial.emission.contents = UIColor.white.interpolate(to: color, progress: progress)
            }

            // Animate halo fade and shrink
            let fadeHaloGlow = SCNAction.customAction(duration: weldingDuration * 0.6) { _, elapsedTime in
                let progress = elapsedTime / CGFloat(self.weldingDuration * 0.6)
                let intensity = self.weldingGlowIntensity * 0.8 * (1.0 - progress)
                haloMaterial.emission.intensity = intensity
                haloMaterial.diffuse.contents = UIColor.white.withAlphaComponent(0.6 * (1.0 - progress))
            }

            // Halo disappears faster
            let removeHalo = SCNAction.sequence([
                SCNAction.wait(duration: weldingDuration * 0.7),
                SCNAction.run { _ in haloNode.removeFromParentNode() }
            ])

            // Run animations
            containerNode.runAction(SCNAction.group([scaleDown, fadeCoreGlow]))
            haloNode.runAction(fadeHaloGlow)
            containerNode.runAction(removeHalo)

            return containerNode
        }

        /// Apply welding glow effect to a node (for cylinders)
        private func applyWeldingGlow(to node: SCNNode, color: UIColor) {
            guard let material = node.geometry?.firstMaterial else { return }

            // Set initial hot white glow
            material.diffuse.contents = UIColor.white
            material.emission.contents = UIColor.white
            material.emission.intensity = weldingGlowIntensity * 0.7

            // Animate glow fade - white to color
            let fadeGlow = SCNAction.customAction(duration: weldingDuration) { node, elapsedTime in
                let progress = elapsedTime / CGFloat(self.weldingDuration)
                let intensity = self.weldingGlowIntensity * 0.7 * (1.0 - progress)
                if let geometry = node.geometry, let mat = geometry.firstMaterial {
                    mat.emission.intensity = intensity
                    mat.diffuse.contents = UIColor.white.interpolate(to: color, progress: progress)
                    mat.emission.contents = UIColor.white.interpolate(to: color, progress: progress)
                }
            }

            node.runAction(fadeGlow)
        }

        func startStroke(_ strokeId: Int) {
            // Clear last point for new stroke
            lastPointPerStroke[strokeId] = nil

            // Hide origin axes once drawing starts
            hideOriginAxes()
        }

        func endStroke(_ strokeId: Int) {
            // Stroke completed
        }

        func clearAllStrokes() {
            for (_, node) in strokeNodes {
                node.removeFromParentNode()
            }
            strokeNodes.removeAll()
            lastPointPerStroke.removeAll()

            // Show origin axes again after clearing
            showOriginAxes()
        }

        // MARK: - Coordinate Transform

        /// Transform point from Mac's world coordinates to ARKit coordinates
        ///
        /// Mac sends points in ArUco marker frame (meters) after world lock:
        ///   - Origin = center of ArUco marker
        ///   - X = right (along marker edge)
        ///   - Y = down (into marker plane)
        ///   - Z = forward (out of marker, toward camera)
        ///
        /// ARKit world space (at session start):
        ///   - Origin = device position
        ///   - X = right
        ///   - Y = up (gravity-aligned)
        ///   - Z = backward (toward user)
        private func transformPointToARKit(_ point: SIMD3<Float>) -> SCNVector3 {
            // Transform from Mac/ArUco coordinate system to ARKit
            // Flip Y (Mac: down, ARKit: up) and Z (Mac: forward, ARKit: backward)
            // Place drawing 1m in front of where the iPhone started
            return SCNVector3(
                point.x,         // X stays the same (right)
                -point.y,        // Y is flipped
                -point.z + 1.0   // Z is flipped, offset 1m in front
            )
        }

        // MARK: - Geometry Helpers

        private func createCylinder(
            from start: SCNVector3,
            to end: SCNVector3,
            radius: CGFloat,
            color: UIColor
        ) -> SCNNode {
            let vector = SCNVector3(
                end.x - start.x,
                end.y - start.y,
                end.z - start.z
            )

            let distance = CGFloat(sqrt(
                vector.x * vector.x +
                vector.y * vector.y +
                vector.z * vector.z
            ))

            let cylinder = SCNCylinder(radius: radius, height: distance)
            cylinder.firstMaterial?.diffuse.contents = color

            let node = SCNNode(geometry: cylinder)

            // Position at midpoint
            node.position = SCNVector3(
                (start.x + end.x) / 2,
                (start.y + end.y) / 2,
                (start.z + end.z) / 2
            )

            // Rotate to align with vector
            node.look(at: end, up: SCNVector3(0, 1, 0), localFront: SCNVector3(0, 1, 0))

            return node
        }
    }
}

// Extension for UIColor interpolation
extension UIColor {
    func interpolate(to color: UIColor, progress: CGFloat) -> UIColor {
        var r1: CGFloat = 0, g1: CGFloat = 0, b1: CGFloat = 0, a1: CGFloat = 0
        var r2: CGFloat = 0, g2: CGFloat = 0, b2: CGFloat = 0, a2: CGFloat = 0

        self.getRed(&r1, green: &g1, blue: &b1, alpha: &a1)
        color.getRed(&r2, green: &g2, blue: &b2, alpha: &a2)

        let p = max(0, min(1, progress))
        return UIColor(
            red: r1 + (r2 - r1) * p,
            green: g1 + (g2 - g1) * p,
            blue: b1 + (b2 - b1) * p,
            alpha: a1 + (a2 - a1) * p
        )
    }
}

// Extension for SCNNode lookAt
extension SCNNode {
    func look(at target: SCNVector3, up: SCNVector3, localFront: SCNVector3) {
        let direction = SCNVector3(
            target.x - position.x,
            target.y - position.y,
            target.z - position.z
        )

        let length = sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z)
        if length > 0.0001 {
            let normalizedDirection = SCNVector3(
                direction.x / length,
                direction.y / length,
                direction.z / length
            )

            // Calculate rotation
            let dotProduct = localFront.x * normalizedDirection.x +
                            localFront.y * normalizedDirection.y +
                            localFront.z * normalizedDirection.z

            let angle = acos(max(-1, min(1, dotProduct)))

            let crossProduct = SCNVector3(
                localFront.y * normalizedDirection.z - localFront.z * normalizedDirection.y,
                localFront.z * normalizedDirection.x - localFront.x * normalizedDirection.z,
                localFront.x * normalizedDirection.y - localFront.y * normalizedDirection.x
            )

            let crossLength = sqrt(crossProduct.x * crossProduct.x +
                                  crossProduct.y * crossProduct.y +
                                  crossProduct.z * crossProduct.z)

            if crossLength > 0.0001 {
                let axis = SCNVector3(
                    crossProduct.x / crossLength,
                    crossProduct.y / crossLength,
                    crossProduct.z / crossLength
                )
                rotation = SCNVector4(axis.x, axis.y, axis.z, angle)
            }
        }
    }
}
