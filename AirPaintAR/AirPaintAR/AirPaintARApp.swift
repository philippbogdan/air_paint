//
//  AirPaintARApp.swift
//  AirPaintAR
//
//  Real-time AR Air Painting with Meta Glasses
//

import MWDATCore
import SwiftUI

@main
struct AirPaintARApp: App {
    private let wearables: WearablesInterface
    @StateObject private var appState = AppState()
    @StateObject private var wearablesViewModel: WearablesViewModel

    init() {
        // Initialize Meta Wearables SDK
        do {
            try Wearables.configure()
        } catch {
            #if DEBUG
            NSLog("[AirPaintAR] Failed to configure Wearables SDK: \(error)")
            #endif
        }

        let wearables = Wearables.shared
        self.wearables = wearables
        self._wearablesViewModel = StateObject(wrappedValue: WearablesViewModel(wearables: wearables))
    }

    var body: some Scene {
        WindowGroup {
            ContentView(wearables: wearables, wearablesVM: wearablesViewModel)
                .environmentObject(appState)
                .alert("Error", isPresented: $wearablesViewModel.showError) {
                    Button("OK") {
                        wearablesViewModel.dismissError()
                    }
                } message: {
                    Text(wearablesViewModel.errorMessage)
                }

            // Invisible view that handles OAuth callbacks from Meta app
            RegistrationView(viewModel: wearablesViewModel)
        }
    }
}

/// Global app state
class AppState: ObservableObject {
    @Published var isConnectedToGlasses = false
    @Published var isConnectedToServer = false
    @Published var isDrawing = false
    @Published var currentStrokeId = 0
    @Published var totalStrokes = 0
    @Published var fps: Double = 0.0
    @Published var latencyMs: Double = 0.0
    @Published var markerVisible = false
    @Published var handTracked = false

    // Server connection (iPhone hotspot default)
    @Published var serverHost = "172.20.10.7"  // Default, user can change
    @Published var serverPort = 8765
}

/// Handles OAuth callbacks from Meta mobile app
struct RegistrationView: View {
    @ObservedObject var viewModel: WearablesViewModel

    var body: some View {
        EmptyView()
            .onOpenURL { url in
                guard
                    let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
                    components.queryItems?.contains(where: { $0.name == "metaWearablesAction" }) == true
                else {
                    return  // URL is not related to DAT SDK
                }
                Task {
                    do {
                        _ = try await Wearables.shared.handleUrl(url)
                    } catch let error as RegistrationError {
                        viewModel.showError(error.description)
                    } catch {
                        viewModel.showError("Unknown error: \(error.localizedDescription)")
                    }
                }
            }
    }
}
