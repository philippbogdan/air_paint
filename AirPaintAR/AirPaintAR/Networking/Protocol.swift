//
//  Protocol.swift
//  AirPaintAR
//
//  WebSocket message protocol - mirrors Python server/protocol.py
//

import Foundation
import simd

// MARK: - Message Types

enum MessageType: String, Codable {
    // iPhone -> Mac
    case frame = "frame"
    case drawingToggle = "drawing_toggle"

    // Mac -> iPhone
    case point = "point"
    case strokeStart = "stroke_start"
    case strokeEnd = "stroke_end"
    case worldAnchor = "world_anchor"
    case status = "status"

    // Bidirectional
    case ping = "ping"
    case pong = "pong"
    case error = "error"
}

// MARK: - iPhone -> Mac Messages

struct FrameMessage: Codable {
    let type: String
    let data: String  // Base64-encoded JPEG
    let timestamp: Double
    let width: Int
    let height: Int

    init(jpegData: Data, width: Int, height: Int) {
        self.type = MessageType.frame.rawValue
        self.data = jpegData.base64EncodedString()
        self.timestamp = Date().timeIntervalSince1970
        self.width = width
        self.height = height
    }
}

struct DrawingToggleMessage: Codable {
    let type: String
    let timestamp: Double

    init() {
        self.type = MessageType.drawingToggle.rawValue
        self.timestamp = Date().timeIntervalSince1970
    }
}

// MARK: - Mac -> iPhone Messages

struct PointMessage: Codable {
    let type: String
    let x: Double  // meters
    let y: Double
    let z: Double
    let stroke_id: Int
    let timestamp: Double
    let confidence: Double

    var position: SIMD3<Float> {
        SIMD3<Float>(Float(x), Float(y), Float(z))
    }
}

struct StrokeStartMessage: Codable {
    let type: String
    let stroke_id: Int
    let timestamp: Double
    let color: [Double]  // RGB, 0-1
}

struct StrokeEndMessage: Codable {
    let type: String
    let stroke_id: Int
    let timestamp: Double
    let point_count: Int
}

struct WorldAnchorMessage: Codable {
    let type: String
    let marker_pose: [Double]  // 4x4 matrix, row-major (16 floats)
    let marker_id: Int
    let marker_size_m: Double
    let timestamp: Double
    let visible: Bool

    var transformMatrix: simd_float4x4 {
        guard marker_pose.count == 16 else {
            return simd_float4x4(1)  // Identity
        }

        return simd_float4x4(rows: [
            SIMD4<Float>(Float(marker_pose[0]), Float(marker_pose[1]), Float(marker_pose[2]), Float(marker_pose[3])),
            SIMD4<Float>(Float(marker_pose[4]), Float(marker_pose[5]), Float(marker_pose[6]), Float(marker_pose[7])),
            SIMD4<Float>(Float(marker_pose[8]), Float(marker_pose[9]), Float(marker_pose[10]), Float(marker_pose[11])),
            SIMD4<Float>(Float(marker_pose[12]), Float(marker_pose[13]), Float(marker_pose[14]), Float(marker_pose[15]))
        ])
    }
}

struct StatusMessage: Codable {
    let type: String
    let tracking: Bool
    let drawing: Bool
    let marker_visible: Bool
    let fps: Double
    let latency_ms: Double
    let stroke_id: Int
    let total_strokes: Int
    let timestamp: Double
}

// MARK: - Utility Messages

struct PingMessage: Codable {
    let type: String
    let timestamp: Double

    init() {
        self.type = MessageType.ping.rawValue
        self.timestamp = Date().timeIntervalSince1970
    }
}

struct PongMessage: Codable {
    let type: String
    let ping_timestamp: Double
    let timestamp: Double
}

struct ErrorMessage: Codable {
    let type: String
    let code: String
    let message: String
    let timestamp: Double
}

// MARK: - Message Parsing

enum ParsedMessage {
    case point(PointMessage)
    case strokeStart(StrokeStartMessage)
    case strokeEnd(StrokeEndMessage)
    case worldAnchor(WorldAnchorMessage)
    case status(StatusMessage)
    case pong(PongMessage)
    case error(ErrorMessage)
    case unknown
}

func parseMessage(_ data: Data) -> ParsedMessage {
    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let typeStr = json["type"] as? String else {
        return .unknown
    }

    let decoder = JSONDecoder()

    switch typeStr {
    case MessageType.point.rawValue:
        if let msg = try? decoder.decode(PointMessage.self, from: data) {
            return .point(msg)
        }
    case MessageType.strokeStart.rawValue:
        if let msg = try? decoder.decode(StrokeStartMessage.self, from: data) {
            return .strokeStart(msg)
        }
    case MessageType.strokeEnd.rawValue:
        if let msg = try? decoder.decode(StrokeEndMessage.self, from: data) {
            return .strokeEnd(msg)
        }
    case MessageType.worldAnchor.rawValue:
        if let msg = try? decoder.decode(WorldAnchorMessage.self, from: data) {
            return .worldAnchor(msg)
        }
    case MessageType.status.rawValue:
        if let msg = try? decoder.decode(StatusMessage.self, from: data) {
            return .status(msg)
        }
    case MessageType.pong.rawValue:
        if let msg = try? decoder.decode(PongMessage.self, from: data) {
            return .pong(msg)
        }
    case MessageType.error.rawValue:
        if let msg = try? decoder.decode(ErrorMessage.self, from: data) {
            return .error(msg)
        }
    default:
        break
    }

    return .unknown
}
