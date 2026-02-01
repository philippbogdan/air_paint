//
//  ContentView.swift
//  AirPaintAR
//
//  Main navigation and UI
//

import MWDATCore
import SwiftUI

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    let wearables: WearablesInterface
    @ObservedObject var wearablesVM: WearablesViewModel
    @StateObject private var glassesVM: GlassesStreamViewModel
    @StateObject private var drawingVM = DrawingViewModel()
    @State private var showSettings = false

    init(wearables: WearablesInterface, wearablesVM: WearablesViewModel) {
        self.wearables = wearables
        self.wearablesVM = wearablesVM
        self._glassesVM = StateObject(wrappedValue: GlassesStreamViewModel(wearables: wearables))
    }

    var body: some View {
        ZStack {
            // AR Drawing View (full screen)
            ARDrawingView(drawingVM: drawingVM)
                .ignoresSafeArea()

            // Overlay UI
            VStack {
                // Top status bar
                HStack {
                    // Connection status
                    StatusBadge(
                        label: "Glasses",
                        isConnected: glassesVM.hasActiveDevice || appState.isConnectedToGlasses,
                        color: .blue
                    )
                    StatusBadge(
                        label: "Server",
                        isConnected: appState.isConnectedToServer,
                        color: .green
                    )
                    StatusBadge(
                        label: "Marker",
                        isConnected: appState.markerVisible,
                        color: .orange
                    )

                    Spacer()

                    // Settings button
                    Button(action: { showSettings = true }) {
                        Image(systemName: "gear")
                            .font(.title2)
                            .foregroundColor(.white)
                            .padding(8)
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }
                }
                .padding()

                Spacer()

                // Bottom info bar
                HStack {
                    // Drawing state
                    if appState.isDrawing {
                        HStack {
                            Circle()
                                .fill(Color.red)
                                .frame(width: 12, height: 12)
                            Text("DRAWING")
                                .font(.headline)
                                .foregroundColor(.red)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(20)
                    }

                    Spacer()

                    // Stats
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("FPS: \(Int(appState.fps))")
                        Text("Latency: \(Int(appState.latencyMs))ms")
                        Text("Strokes: \(appState.totalStrokes)")
                    }
                    .font(.caption)
                    .foregroundColor(.white)
                    .padding(8)
                    .background(Color.black.opacity(0.5))
                    .cornerRadius(8)
                }
                .padding()
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(wearablesVM: wearablesVM, glassesVM: glassesVM, drawingVM: drawingVM)
        }
        .onAppear {
            // Initialize connections
            glassesVM.appState = appState
            glassesVM.drawingVM = drawingVM
            drawingVM.appState = appState
        }
    }
}

struct StatusBadge: View {
    let label: String
    let isConnected: Bool
    let color: Color

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(isConnected ? color : Color.gray)
                .frame(width: 8, height: 8)
            Text(label)
                .font(.caption)
                .foregroundColor(.white)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.black.opacity(0.5))
        .cornerRadius(12)
    }
}

struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @ObservedObject var wearablesVM: WearablesViewModel
    @ObservedObject var glassesVM: GlassesStreamViewModel
    @ObservedObject var drawingVM: DrawingViewModel
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            Form {
                Section("Server Connection") {
                    TextField("Host (e.g. 192.168.1.100)", text: $appState.serverHost)
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                    TextField("Port", value: $appState.serverPort, format: .number)
                        .keyboardType(.numberPad)

                    Button(appState.isConnectedToServer ? "Disconnect" : "Connect") {
                        if appState.isConnectedToServer {
                            drawingVM.disconnect()
                        } else {
                            drawingVM.connect(
                                host: appState.serverHost,
                                port: appState.serverPort
                            )
                        }
                    }
                    .foregroundColor(appState.isConnectedToServer ? .red : .blue)
                }

                Section("Meta Glasses") {
                    // Registration status
                    HStack {
                        Text("Registration")
                        Spacer()
                        switch wearablesVM.registrationState {
                        case .registered:
                            Text("Registered")
                                .foregroundColor(.green)
                        case .registering:
                            ProgressView()
                        default:
                            Text("Not Registered")
                                .foregroundColor(.secondary)
                        }
                    }

                    // Register/unregister button
                    if wearablesVM.registrationState == .registered {
                        Button("Disconnect & Re-pair", role: .destructive) {
                            wearablesVM.disconnectGlasses()
                        }
                    } else if wearablesVM.registrationState != .registering {
                        Button("Connect to Glasses") {
                            wearablesVM.connectGlasses()
                        }
                    }

                    // Stream control (only if registered)
                    if wearablesVM.registrationState == .registered {
                        Divider()

                        HStack {
                            Text("Device")
                            Spacer()
                            if glassesVM.hasActiveDevice {
                                Text("Connected")
                                    .foregroundColor(.green)
                            } else {
                                Text("Waiting...")
                                    .foregroundColor(.secondary)
                            }
                        }

                        Button(glassesVM.isStreaming ? "Stop Stream" : "Start Stream") {
                            if glassesVM.isStreaming {
                                glassesVM.disconnect()
                            } else {
                                glassesVM.connect()
                            }
                        }
                        .disabled(!glassesVM.hasActiveDevice)
                    }

                    // Fallback camera toggle
                    Toggle("Use iPhone Camera (Fallback)", isOn: $glassesVM.useFallbackCamera)
                }

                Section("Status") {
                    HStack {
                        Text("Hand Tracked")
                        Spacer()
                        Image(systemName: appState.handTracked ? "checkmark.circle.fill" : "xmark.circle")
                            .foregroundColor(appState.handTracked ? .green : .red)
                    }
                    HStack {
                        Text("Marker Visible")
                        Spacer()
                        Image(systemName: appState.markerVisible ? "checkmark.circle.fill" : "xmark.circle")
                            .foregroundColor(appState.markerVisible ? .green : .red)
                    }
                    HStack {
                        Text("Current Stroke")
                        Spacer()
                        Text("#\(appState.currentStrokeId)")
                            .foregroundColor(.secondary)
                    }
                }

                Section("Debug") {
                    Button("Clear All Strokes") {
                        drawingVM.clearStrokes()
                    }
                    .foregroundColor(.red)
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
