//
//  GlassesStreamViewModel.swift
//  AirPaintAR
//
//  Manages Meta Glasses camera streaming via MWDATCamera SDK
//

import AVFoundation
import CoreBluetooth
import Foundation
import MWDATCamera
import MWDATCore
import UIKit

@MainActor
class GlassesStreamViewModel: ObservableObject {
    @Published var isConnected = false
    @Published var isStreaming = false
    @Published var lastFrameTimestamp: Double = 0
    @Published var hasActiveDevice = false
    @Published var latestFrame: UIImage?

    weak var appState: AppState?
    weak var drawingVM: DrawingViewModel?

    // Wearables reference
    private let wearables: WearablesInterface
    private let deviceSelector: AutoDeviceSelector

    // Bluetooth manager - triggers permission dialog on init
    private var bluetoothManager: CBCentralManager?

    // Stream session for video frames
    private let streamSession: StreamSession
    private var videoFrameListenerToken: AnyListenerToken?
    private var stateListenerToken: AnyListenerToken?
    private var errorListenerToken: AnyListenerToken?

    // Device monitoring
    private var deviceMonitorTask: Task<Void, Never>?

    // For testing without glasses - use device camera
    @Published var useFallbackCamera = false
    private var fallbackSession: AVCaptureSession?
    private var fallbackOutput: AVCaptureVideoDataOutput?
    private let processingQueue = DispatchQueue(label: "glasses.processing")

    init(wearables: WearablesInterface) {
        self.wearables = wearables
        self.deviceSelector = AutoDeviceSelector(wearables: wearables)

        // Create StreamSession immediately for device discovery
        // Using 24 FPS for real-time drawing (matches plan)
        let config = StreamSessionConfig(
            videoCodec: .raw,
            resolution: .high,
            frameRate: 24
        )
        self.streamSession = StreamSession(streamSessionConfig: config, deviceSelector: deviceSelector)
        print("[GlassesStream] StreamSession created at init")

        // Trigger Bluetooth permission dialog
        self.bluetoothManager = CBCentralManager()

        // Monitor device availability
        deviceMonitorTask = Task { @MainActor [weak self] in
            guard let self else { return }
            for await deviceId in self.deviceSelector.activeDeviceStream() {
                let isActive = deviceId != nil
                self.hasActiveDevice = isActive
                self.appState?.isConnectedToGlasses = isActive
                if let deviceId, let device = wearables.deviceForIdentifier(deviceId) {
                    print("[GlassesStream] Device CONNECTED: \(device.nameOrId())")
                } else {
                    print("[GlassesStream] Device DISCONNECTED")
                }
            }
        }
    }

    deinit {
        deviceMonitorTask?.cancel()
    }

    func connect() {
        if useFallbackCamera {
            setupFallbackCamera()
        } else {
            Task {
                await startGlassesStream()
            }
        }
    }

    func disconnect() {
        if useFallbackCamera {
            fallbackSession?.stopRunning()
            fallbackSession = nil
        } else {
            Task {
                await stopGlassesStream()
            }
        }

        isConnected = false
        isStreaming = false
        appState?.isConnectedToGlasses = false
    }

    // MARK: - Glasses Stream

    private func startGlassesStream() async {
        // Check and request camera permission
        let permission = Permission.camera
        do {
            let status = try await wearables.checkPermissionStatus(permission)
            if status != .granted {
                print("[GlassesStream] Permission not granted, requesting...")
                let requestStatus = try await wearables.requestPermission(permission)
                if requestStatus != .granted {
                    print("[GlassesStream] Permission denied")
                    return
                }
            }
            print("[GlassesStream] Camera permission granted")
        } catch {
            print("[GlassesStream] Permission check failed: \(error)")
            return
        }

        // Subscribe to state changes
        stateListenerToken = streamSession.statePublisher.listen { [weak self] state in
            Task { @MainActor in
                self?.isStreaming = state == .streaming
                print("[GlassesStream] Session state: \(state)")
            }
        }

        // Subscribe to errors
        errorListenerToken = streamSession.errorPublisher.listen { [weak self] error in
            Task { @MainActor in
                print("[GlassesStream] Error: \(error)")
            }
        }

        // Subscribe to video frames
        videoFrameListenerToken = streamSession.videoFramePublisher.listen { [weak self] videoFrame in
            Task { @MainActor in
                guard let self else { return }
                if let image = videoFrame.makeUIImage() {
                    self.latestFrame = image
                    self.lastFrameTimestamp = CACurrentMediaTime()
                    self.processFrame(image)
                }
            }
        }

        // Start session
        await streamSession.start()
        print("[GlassesStream] Session started, waiting for streaming...")

        // Wait for streaming state
        do {
            try await waitForStreaming(timeout: 30.0)
            isConnected = true
            isStreaming = true
            appState?.isConnectedToGlasses = true
            print("[GlassesStream] Now streaming!")
        } catch {
            print("[GlassesStream] Failed to reach streaming state: \(error)")
        }
    }

    private func stopGlassesStream() async {
        await streamSession.stop()
        videoFrameListenerToken = nil
        stateListenerToken = nil
        errorListenerToken = nil
        latestFrame = nil
    }

    private func waitForStreaming(timeout: TimeInterval) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while streamSession.state != .streaming {
            if Date() > deadline {
                throw NSError(domain: "GlassesStream", code: 1, userInfo: [NSLocalizedDescriptionKey: "Timeout waiting for stream"])
            }
            try await Task.sleep(nanoseconds: 100_000_000)  // 0.1s
        }
    }

    // MARK: - Fallback Camera (for testing without glasses)

    private func setupFallbackCamera() {
        fallbackSession = AVCaptureSession()
        fallbackSession?.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                    for: .video,
                                                    position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            print("[GlassesStream] Failed to setup fallback camera")
            return
        }

        if fallbackSession?.canAddInput(input) == true {
            fallbackSession?.addInput(input)
        }

        fallbackOutput = AVCaptureVideoDataOutput()
        fallbackOutput?.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        fallbackOutput?.alwaysDiscardsLateVideoFrames = true

        let delegate = FallbackCameraDelegate { [weak self] image in
            Task { @MainActor in
                self?.latestFrame = image
                self?.lastFrameTimestamp = CACurrentMediaTime()
                self?.processFrame(image)
            }
        }
        fallbackOutput?.setSampleBufferDelegate(delegate, queue: processingQueue)
        // Keep delegate alive
        objc_setAssociatedObject(fallbackOutput as Any, "delegate", delegate, .OBJC_ASSOCIATION_RETAIN)

        if let output = fallbackOutput, fallbackSession?.canAddOutput(output) == true {
            fallbackSession?.addOutput(output)
        }

        processingQueue.async { [weak self] in
            self?.fallbackSession?.startRunning()
        }

        isConnected = true
        isStreaming = true
        appState?.isConnectedToGlasses = true
    }

    // MARK: - Frame Processing

    private func processFrame(_ image: UIImage) {
        // Convert to JPEG for WebSocket transmission
        guard let jpegData = image.jpegData(compressionQuality: 0.5) else { return }

        let width = Int(image.size.width)
        let height = Int(image.size.height)

        // Send to Mac server
        drawingVM?.sendFrame(jpegData: jpegData, width: width, height: height)
    }
}

// MARK: - Fallback Camera Delegate

private class FallbackCameraDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let handler: (UIImage) -> Void
    private let context = CIContext()

    init(handler: @escaping (UIImage) -> Void) {
        self.handler = handler
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }

        let image = UIImage(cgImage: cgImage)
        handler(image)
    }
}
