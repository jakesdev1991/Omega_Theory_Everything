// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

// T1 Wasmtime sandbox with REAL resource limits (§28).
//
// Fixes over the original:
//   * consume_fuel + epoch interruption enabled at engine level
//   * store limiter: memory / tables / instances bounded
//   * epoch deadline: hard wall-clock bound on any wasm execution
//   * actually EXECUTES the exported "run" function (was instantiate-only)
//   * trap surfaces as a typed SandboxResult, never kills the daemon

#include <wasm.h>
#include <wasmtime.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <atomic>
#include <functional>

namespace omni::bridge {

struct SandboxLimits {
    uint64_t fuel = 100'000'000;           // wasm instructions
    int64_t  memory_bytes = 64 * 1024 * 1024; // 64 MiB linear memory cap
    int64_t  table_elements = 10'000;
    int64_t  instances = 10;
    int64_t  tables = 100;
    int64_t  memories = 10;
    uint64_t epoch_ticks = 1;              // deadline delta; daemon bumps epochs
};

struct SandboxResult {
    bool        ok = false;
    bool        trapped = false;
    bool        out_of_fuel = false;
    bool        epoch_deadline_hit = false;
    std::string detail;
};

class WasmtimeSandbox {
private:
    wasm_engine_t*   engine_  = nullptr;
    wasm_config_t*   config_  = nullptr;
    wasmtime_store_t* store_  = nullptr;
    wasmtime_context_t* context_ = nullptr;
    SandboxLimits    limits_;

    static void raii_name(wasm_name_t* n) { /* helper marker */ }

public:
    explicit WasmtimeSandbox(SandboxLimits limits = {}) : limits_(limits) {
        // Engine with fuel + epoch interruption (§28 resource limits)
        config_ = wasm_config_new();
        if (!config_) throw std::runtime_error("wasm_config_new failed");
        wasmtime_config_consume_fuel_set(config_, true);
        wasmtime_config_epoch_interruption_set(config_, true);

        engine_ = wasm_engine_new_with_config(config_);
        if (!engine_) {
            wasm_config_delete(config_); config_ = nullptr;
            throw std::runtime_error("Failed to create Wasmtime engine with fuel+epoch config");
        }

        store_ = wasmtime_store_new(engine_, nullptr, nullptr);
        if (!store_) throw std::runtime_error("wasmtime_store_new failed");

        // Store limiter: memory/table/instance ceilings
        wasmtime_store_limiter(store_, limits_.memory_bytes, limits_.table_elements,
                               limits_.instances, limits_.tables, limits_.memories);

        context_ = wasmtime_store_context(store_);

        // Fuel for execution
        wasmtime_error_t* err = wasmtime_context_set_fuel(context_, limits_.fuel);
        if (err) {
            wasm_name_t msg; wasmtime_error_message(err, &msg);
            std::string m(msg.data, msg.size);
            wasm_byte_vec_delete(&msg);
            wasmtime_error_delete(err);
            throw std::runtime_error("set_fuel failed: " + m);
        }

        // Epoch deadline: any wasm that runs past this many engine-epoch
        // increments traps. Daemon calls bump_epoch() periodically.
        wasmtime_context_set_epoch_deadline(context_, limits_.epoch_ticks);
    }

    ~WasmtimeSandbox() {
        if (store_) wasmtime_store_delete(store_);
        if (engine_) wasm_engine_delete(engine_);
        // config_ is consumed by wasm_engine_new_with_config — do not double-free.
    }

    void bump_epoch() {
        if (engine_) wasmtime_engine_increment_epoch(engine_);
    }

    // Compile + instantiate + CALL the exported "run" (or given) function.
    SandboxResult execute_untrusted_module(const std::string& wat_code,
                                           const std::string& export_name = "run") {
        SandboxResult result;

        // Reset fuel for this execution — a prior run that exhausted fuel
        // must not poison the next (store-level fuel is monotonic down).
        wasmtime_error_t* ferr = wasmtime_context_set_fuel(context_, limits_.fuel);
        if (ferr) {
            wasm_name_t msg; wasmtime_error_message(ferr, &msg);
            result.detail = "fuel reset: " + std::string(msg.data, msg.size);
            wasm_byte_vec_delete(&msg);
            wasmtime_error_delete(ferr);
            return result;
        }
        // Re-arm the epoch deadline for this execution.
        wasmtime_context_set_epoch_deadline(context_, limits_.epoch_ticks);

        // 1. WAT -> wasm
        wasm_byte_vec_t wasm;
        wasmtime_error_t* error = wasmtime_wat2wasm(wat_code.data(), wat_code.size(), &wasm);
        if (error) {
            wasm_name_t msg; wasmtime_error_message(error, &msg);
            result.detail = "WAT parse: " + std::string(msg.data, msg.size);
            wasm_byte_vec_delete(&msg);
            wasmtime_error_delete(error);
            return result;
        }

        // 2. compile
        wasmtime_module_t* module = nullptr;
        error = wasmtime_module_new(engine_, (uint8_t*)wasm.data, wasm.size, &module);
        wasm_byte_vec_delete(&wasm);
        if (error) {
            wasm_name_t msg; wasmtime_error_message(error, &msg);
            result.detail = "compile: " + std::string(msg.data, msg.size);
            wasm_byte_vec_delete(&msg);
            wasmtime_error_delete(error);
            return result;
        }

        // 3. instantiate (imports omitted -> only import-free modules allowed)
        wasm_trap_t* trap = nullptr;
        wasmtime_instance_t instance;
        error = wasmtime_instance_new(context_, module, nullptr, 0, &instance, &trap);
        wasmtime_module_delete(module);
        if (error || trap) {
            if (error) {
                wasm_name_t msg; wasmtime_error_message(error, &msg);
                result.detail = "instantiate: " + std::string(msg.data, msg.size);
                wasm_byte_vec_delete(&msg);
                wasmtime_error_delete(error);
            } else {
                result.trapped = true;
                wasm_name_t msg; wasm_trap_message(trap, &msg);
                result.detail = "instantiate trap: " + std::string(msg.data, msg.size);
                wasm_byte_vec_delete(&msg);
                wasm_trap_delete(trap);
            }
            return result;
        }

        // 4. find exported function
        wasmtime_extern_t item;
        if (!wasmtime_instance_export_get(context_, &instance, export_name.c_str(),
                                         export_name.size(), &item)) {
            result.detail = "export not found: " + export_name;
            return result;
        }
        if (item.kind != WASMTIME_EXTERN_FUNC) {
            result.detail = "export is not a function: " + export_name;
            return result;
        }

        // 5. CALL with fuel + epoch deadline active
        wasm_trap_t* call_trap = nullptr;
        error = wasmtime_func_call(context_, &item.of.func, nullptr, 0,
                                   nullptr, 0, &call_trap);
        if (error) {
            wasm_name_t msg; wasmtime_error_message(error, &msg);
            result.detail = "call: " + std::string(msg.data, msg.size);
            wasm_byte_vec_delete(&msg);
            wasmtime_error_delete(error);
            // out-of-fuel and epoch-deadline surface as errors in the C API.
            if (result.detail.find("fuel") != std::string::npos) result.out_of_fuel = true;
            if (result.detail.find("epoch") != std::string::npos) result.epoch_deadline_hit = true;
            return result;
        }
        if (call_trap) {
            result.trapped = true;
            wasm_name_t msg; wasm_trap_message(call_trap, &msg);
            result.detail = "trap: " + std::string(msg.data, msg.size);
            wasm_byte_vec_delete(&msg);
            wasm_trap_delete(call_trap);
            return result;
        }

        result.ok = true;
        result.detail = "executed with fuel+epoch limits";
        return result;
    }
};

} // namespace omni::bridge
