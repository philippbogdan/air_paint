//
//  AirPaintARApp.swift
//  AirPaintAR
//
//  Real-time AR Air Painting - Viewer Only Mode
//  iPhone receives 3D points from Mac and displays in AR
//

import SwiftUI

@main
struct AirPaintARApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        }
    }
}

/// Global app state
class AppState: ObservableObject {
    @Published var isConnectedToGlasses = false  // Not used in viewer mode
    @Published var isConnectedToServer = false
    @Published var isDrawing = false
    @Published var currentStrokeId = 0
    @Published var totalStrokes = 0
    @Published var fps: Double = 0.0
    @Published var latencyMs: Double = 0.0
    @Published var markerVisible = false
    @Published var handTracked = false

    // Server connection settings
    @Published var serverHost = "172.20.10.7"
    @Published var serverPort = 8765
}
