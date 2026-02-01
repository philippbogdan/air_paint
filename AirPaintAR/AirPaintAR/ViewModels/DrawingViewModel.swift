//
//  DrawingViewModel.swift
//  AirPaintAR
//
//  Manages WebSocket connection and stroke state
//

import Foundation
import Combine
import simd

class DrawingViewModel: ObservableObject, WebSocketClientDelegate {
    @Published var isConnected = false
    @Published var currentStrokePoints: [SIMD3<Float>] = []
    @Published var completedStrokes: [[SIMD3<Float>]] = []

    weak var appState: AppState?

    private let webSocketClient = WebSocketClient()
    private var arCoordinator: ARDrawingView.Coordinator?
    private var lastLatencyPing: Double = 0

    init() {
        webSocketClient.delegate = self
    }

    func setARCoordinator(_ coordinator: ARDrawingView.Coordinator) {
        self.arCoordinator = coordinator
    }

    // MARK: - Connection

    func connect(host: String, port: Int) {
        webSocketClient.connect(host: host, port: port)
    }

    func disconnect() {
        webSocketClient.disconnect()
    }

    // MARK: - Actions

    func toggleDrawing() {
        webSocketClient.sendDrawingToggle()
    }

    func clearStrokes() {
        currentStrokePoints.removeAll()
        completedStrokes.removeAll()
        arCoordinator?.clearAllStrokes()
    }

    // MARK: - Frame Sending (no-op in viewer mode)

    /// In viewer-only mode, the Mac handles all camera capture.
    /// This function is a no-op - iPhone doesn't send frames.
    func sendFrame(jpegData: Data, width: Int, height: Int) {
        // No-op in viewer mode - Mac captures from local dual cameras
    }

    // MARK: - WebSocketClientDelegate

    func webSocketDidConnect() {
        DispatchQueue.main.async {
            print("[DrawingVM] ✅ Server connected")
            self.isConnected = true
            self.appState?.isConnectedToServer = true
        }
    }

    func webSocketDidDisconnect(error: Error?) {
        DispatchQueue.main.async {
            print("[DrawingVM] ❌ Server disconnected: \(error?.localizedDescription ?? "no error")")
            self.isConnected = false
            self.appState?.isConnectedToServer = false
        }
    }

    func webSocketDidReceive(message: ParsedMessage) {
        DispatchQueue.main.async {
            self.handleMessage(message)
        }
    }

    // MARK: - Message Handling

    private func handleMessage(_ message: ParsedMessage) {
        switch message {
        case .point(let pointMsg):
            handlePoint(pointMsg)

        case .strokeStart(let startMsg):
            handleStrokeStart(startMsg)

        case .strokeEnd(let endMsg):
            handleStrokeEnd(endMsg)

        case .worldAnchor(let anchorMsg):
            handleWorldAnchor(anchorMsg)

        case .status(let statusMsg):
            handleStatus(statusMsg)

        case .pong(let pongMsg):
            handlePong(pongMsg)

        case .error(let errorMsg):
            print("Server error: \(errorMsg.code) - \(errorMsg.message)")

        case .unknown:
            break
        }
    }

    private func handlePoint(_ msg: PointMessage) {
        let point = msg.position

        // Add to current stroke
        currentStrokePoints.append(point)

        // Render in AR
        arCoordinator?.addPoint(point, strokeId: msg.stroke_id)
    }

    private func handleStrokeStart(_ msg: StrokeStartMessage) {
        // Clear current points for new stroke
        currentStrokePoints.removeAll()
        arCoordinator?.startStroke(msg.stroke_id)

        appState?.currentStrokeId = msg.stroke_id
        appState?.isDrawing = true
    }

    private func handleStrokeEnd(_ msg: StrokeEndMessage) {
        // Save completed stroke
        if !currentStrokePoints.isEmpty {
            completedStrokes.append(currentStrokePoints)
            currentStrokePoints.removeAll()
        }

        arCoordinator?.endStroke(msg.stroke_id)

        appState?.isDrawing = false
        appState?.totalStrokes = completedStrokes.count
    }

    private func handleWorldAnchor(_ msg: WorldAnchorMessage) {
        appState?.markerVisible = msg.visible

        // Update marker transform in AR view
        // This helps align coordinate systems
    }

    private func handleStatus(_ msg: StatusMessage) {
        appState?.fps = msg.fps
        appState?.handTracked = msg.tracking
        appState?.markerVisible = msg.marker_visible
        appState?.totalStrokes = msg.total_strokes
    }

    private func handlePong(_ msg: PongMessage) {
        let now = Date().timeIntervalSince1970
        let latency = (now - msg.ping_timestamp) * 1000  // ms
        appState?.latencyMs = latency
    }
}
