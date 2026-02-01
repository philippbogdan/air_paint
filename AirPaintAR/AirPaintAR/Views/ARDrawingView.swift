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

        // Configure AR session
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = []  // Don't detect planes

        // Add ArUco marker as reference image for detection
        if let markerImage = createArUcoMarkerImage() {
            let referenceImage = ARReferenceImage(
                markerImage,
                orientation: .up,
                physicalWidth: 0.16  // 160mm marker
            )
            referenceImage.name = "ArUcoMarker0"
            configuration.detectionImages = [referenceImage]
            configuration.maximumNumberOfTrackedImages = 1
        }

        arView.session.run(configuration)

        // Store reference for updates
        context.coordinator.arView = arView
        context.coordinator.drawingVM = drawingVM

        // Connect coordinator to ViewModel
        drawingVM.setARCoordinator(context.coordinator)

        return arView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        // Updates handled by coordinator
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    /// Create ArUco marker image for detection
    private func createArUcoMarkerImage() -> CGImage? {
        // Try to load ArUco marker from bundle assets
        // Expected asset name: "aruco_marker_0" (without extension)
        if let image = UIImage(named: "aruco_marker_0") {
            print("Loaded ArUco marker image from assets")
            return image.cgImage
        }

        // Try loading from bundle directly
        if let path = Bundle.main.path(forResource: "aruco_marker_0", ofType: "png"),
           let image = UIImage(contentsOfFile: path) {
            print("Loaded ArUco marker image from bundle")
            return image.cgImage
        }

        print("Warning: ArUco marker image not found. Using world-space positioning.")
        return nil
    }

    class Coordinator: NSObject, ARSCNViewDelegate, ARSessionDelegate {
        weak var arView: ARSCNView?
        weak var drawingVM: DrawingViewModel?

        // Stroke rendering
        private var strokeNodes: [Int: SCNNode] = [:]  // stroke_id -> parent node
        private var lastPointPerStroke: [Int: SCNVector3] = [:]

        // Marker anchor
        private var markerAnchorNode: SCNNode?
        private var markerTransform: simd_float4x4?

        // MARK: - ARSCNViewDelegate

        func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
            if let imageAnchor = anchor as? ARImageAnchor {
                // ArUco marker detected by ARKit
                print("Marker detected: \(imageAnchor.referenceImage.name ?? "unknown")")
                markerAnchorNode = node
                markerTransform = imageAnchor.transform

                // Add visual indicator on marker
                let plane = SCNPlane(width: 0.16, height: 0.16)
                plane.firstMaterial?.diffuse.contents = UIColor.green.withAlphaComponent(0.3)
                let planeNode = SCNNode(geometry: plane)
                planeNode.eulerAngles.x = -.pi / 2
                node.addChildNode(planeNode)
            }
        }

        func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
            if let imageAnchor = anchor as? ARImageAnchor {
                markerTransform = imageAnchor.transform
            }
        }

        // MARK: - Point Rendering

        func addPoint(_ point: SIMD3<Float>, strokeId: Int, color: UIColor = .red) {
            guard let arView = arView else { return }

            // Transform point from Mac's coordinate system to ARKit
            let transformedPoint = transformPointToARKit(point)

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

            // Create cylinder from last point to this point
            if let lastPoint = lastPointPerStroke[strokeId] {
                let cylinderNode = createCylinder(
                    from: lastPoint,
                    to: transformedPoint,
                    radius: 0.002,  // 2mm radius
                    color: color
                )
                strokeNode.addChildNode(cylinderNode)
            } else {
                // First point - create small sphere
                let sphere = SCNSphere(radius: 0.003)
                sphere.firstMaterial?.diffuse.contents = color
                let sphereNode = SCNNode(geometry: sphere)
                sphereNode.position = transformedPoint
                strokeNode.addChildNode(sphereNode)
            }

            lastPointPerStroke[strokeId] = transformedPoint
        }

        func startStroke(_ strokeId: Int) {
            // Clear last point for new stroke
            lastPointPerStroke[strokeId] = nil
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
        }

        // MARK: - Coordinate Transform

        /// Transform point from Mac's world coordinates to ARKit coordinates
        private func transformPointToARKit(_ point: SIMD3<Float>) -> SCNVector3 {
            // Mac sends points in ArUco marker frame (meters)
            // If we have marker transform from ARKit, use it
            if let transform = markerTransform {
                // Point is already in marker-relative coordinates from Mac
                // Apply marker's ARKit transform
                let point4 = SIMD4<Float>(point.x, point.y, point.z, 1.0)
                let transformed = transform * point4
                return SCNVector3(transformed.x, transformed.y, transformed.z)
            } else {
                // No marker detected - use world-space positioning
                // Mac sends points in meters, place them directly in world space
                // The Mac's coordinate system has:
                //   X = right, Y = down, Z = forward (camera convention)
                // ARKit world space has:
                //   X = right, Y = up, Z = backward
                // So we need to flip Y and Z
                return SCNVector3(
                    point.x,        // X stays the same (right)
                    -point.y,       // Y is flipped (Mac: down, ARKit: up)
                    -point.z + 1.0  // Z is flipped, offset 1m in front of origin
                )
            }
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
