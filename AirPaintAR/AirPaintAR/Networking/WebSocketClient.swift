//
//  WebSocketClient.swift
//  AirPaintAR
//
//  WebSocket client for communication with Mac server
//

import Foundation

protocol WebSocketClientDelegate: AnyObject {
    func webSocketDidConnect()
    func webSocketDidDisconnect(error: Error?)
    func webSocketDidReceive(message: ParsedMessage)
}

class WebSocketClient: NSObject {
    weak var delegate: WebSocketClientDelegate?

    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession!
    private(set) var isConnected = false
    private var isConnecting = false
    private var pingTimer: Timer?

    override init() {
        super.init()
        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = false
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 300  // 5 minutes
        urlSession = URLSession(configuration: config, delegate: self, delegateQueue: .main)
    }

    func connect(host: String, port: Int) {
        guard !isConnected && !isConnecting else {
            print("Already connected or connecting")
            return
        }

        let urlString = "ws://\(host):\(port)"
        guard let url = URL(string: urlString) else {
            print("Invalid WebSocket URL: \(urlString)")
            return
        }

        print("Connecting to \(urlString)...")
        isConnecting = true

        webSocketTask = urlSession.webSocketTask(with: url)
        webSocketTask?.resume()

        // Start receiving immediately - it will queue until connected
        startReceiving()
    }

    func disconnect() {
        pingTimer?.invalidate()
        pingTimer = nil

        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        isConnected = false
        isConnecting = false

        delegate?.webSocketDidDisconnect(error: nil)
    }

    // MARK: - Sending Messages

    func send<T: Encodable>(_ message: T) {
        guard isConnected, let task = webSocketTask else {
            print("Cannot send - not connected")
            return
        }

        do {
            let data = try JSONEncoder().encode(message)
            let string = String(data: data, encoding: .utf8) ?? ""
            task.send(.string(string)) { error in
                if let error = error {
                    print("WebSocket send error: \(error)")
                }
            }
        } catch {
            print("Failed to encode message: \(error)")
        }
    }

    func sendFrame(jpegData: Data, width: Int, height: Int) {
        let message = FrameMessage(jpegData: jpegData, width: width, height: height)
        send(message)
    }

    func sendDrawingToggle() {
        let message = DrawingToggleMessage()
        send(message)
    }

    func sendPing() {
        let message = PingMessage()
        send(message)
    }

    // MARK: - Receiving

    private func startReceiving() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                print("[WS] Received message")
                self?.handleMessage(message)
                // Continue receiving
                self?.startReceiving()

            case .failure(let error):
                print("[WS] ❌ Receive error: \(error.localizedDescription)")
                self?.handleDisconnect(error: error)
            }
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .string(let text):
            if let data = text.data(using: .utf8) {
                let parsed = parseMessage(data)
                delegate?.webSocketDidReceive(message: parsed)
            }

        case .data(let data):
            let parsed = parseMessage(data)
            delegate?.webSocketDidReceive(message: parsed)

        @unknown default:
            break
        }
    }

    private func handleDisconnect(error: Error?) {
        pingTimer?.invalidate()
        pingTimer = nil
        isConnected = false
        isConnecting = false
        webSocketTask = nil
        delegate?.webSocketDidDisconnect(error: error)
    }

    // MARK: - Ping Timer

    private func startPingTimer() {
        pingTimer?.invalidate()
        // Send first ping immediately, then every 3 seconds to keep connection alive
        sendPing()
        pingTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            self?.sendPing()
        }
    }
}

// MARK: - URLSessionWebSocketDelegate

extension WebSocketClient: URLSessionWebSocketDelegate {
    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didOpenWithProtocol protocol: String?
    ) {
        print("[WS] ✅ Connected!")
        isConnected = true
        isConnecting = false
        startPingTimer()
        delegate?.webSocketDidConnect()
    }

    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
        reason: Data?
    ) {
        print("[WS] ❌ Closed with code: \(closeCode.rawValue)")
        handleDisconnect(error: nil)
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        if let error = error {
            print("[WS] ❌ Connection error: \(error.localizedDescription)")
            handleDisconnect(error: error)
        }
    }
}
