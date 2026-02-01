//
//  WearablesViewModel.swift
//  AirPaintAR
//
//  Manages Meta Glasses SDK registration and device discovery
//

import MWDATCore
import SwiftUI

@MainActor
class WearablesViewModel: ObservableObject {
    @Published var devices: [DeviceIdentifier]
    @Published var registrationState: RegistrationState
    @Published var showError: Bool = false
    @Published var errorMessage: String = ""

    private var registrationTask: Task<Void, Never>?
    private var deviceStreamTask: Task<Void, Never>?
    private let wearables: WearablesInterface
    private var compatibilityListenerTokens: [DeviceIdentifier: AnyListenerToken] = [:]

    init(wearables: WearablesInterface) {
        self.wearables = wearables
        self.devices = wearables.devices
        self.registrationState = wearables.registrationState

        registrationTask = Task {
            for await registrationState in wearables.registrationStateStream() {
                self.registrationState = registrationState
                if registrationState == .registered {
                    await setupDeviceStream()
                }
            }
        }
    }

    deinit {
        registrationTask?.cancel()
        deviceStreamTask?.cancel()
    }

    private func setupDeviceStream() async {
        if let task = deviceStreamTask, !task.isCancelled {
            task.cancel()
        }

        deviceStreamTask = Task {
            for await devices in wearables.devicesStream() {
                self.devices = devices
                monitorDeviceCompatibility(devices: devices)
            }
        }
    }

    private func monitorDeviceCompatibility(devices: [DeviceIdentifier]) {
        let deviceSet = Set(devices)
        compatibilityListenerTokens = compatibilityListenerTokens.filter { deviceSet.contains($0.key) }

        for deviceId in devices {
            guard compatibilityListenerTokens[deviceId] == nil else { continue }
            guard let device = wearables.deviceForIdentifier(deviceId) else { continue }

            let deviceName = device.nameOrId()
            let token = device.addCompatibilityListener { [weak self] compatibility in
                guard let self else { return }
                if compatibility == .deviceUpdateRequired {
                    Task { @MainActor in
                        self.showError("Device '\(deviceName)' requires an update to work with this app")
                    }
                }
            }
            compatibilityListenerTokens[deviceId] = token
        }
    }

    func connectGlasses() {
        guard registrationState != .registering else { return }
        do {
            try wearables.startRegistration()
        } catch {
            showError(error.localizedDescription)
        }
    }

    func disconnectGlasses() {
        do {
            try wearables.startUnregistration()
        } catch {
            showError(error.localizedDescription)
        }
    }

    func showError(_ error: String) {
        errorMessage = error
        showError = true
    }

    func dismissError() {
        showError = false
    }
}
