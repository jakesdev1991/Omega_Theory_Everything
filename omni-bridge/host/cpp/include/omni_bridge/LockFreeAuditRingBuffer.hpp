// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <array>
#include <atomic>
#include <optional>
#include <cstdint>
#include <cstddef>

namespace omni::bridge {

struct CounterfactualRecord { 
    uint64_t original_route_id; 
    uint64_t simulated_route_id; 
    float budget_delta; 
    std::array<uint8_t, 32> blake3_root_hash; 
}; 

// Lock-free single-producer single-consumer ring buffer for decoupled telemetry ingestion
template <typename T, size_t Capacity>
class LockFreeAuditRingBuffer {
private: 
    static_assert((Capacity != 0) && ((Capacity & (Capacity - 1)) == 0), 
                  "Capacity must be a power of two for fast modulo");

    std::array<T, Capacity> storage_; 
    
    // Aligned to 64 bytes to prevent false sharing between producer and consumer threads
    alignas(64) std::atomic<size_t> head_{0}; 
    alignas(64) std::atomic<size_t> tail_{0}; 
    
    static constexpr size_t MASK = Capacity - 1;

public: 
    // SPSC push operation
    bool push(const T& item) noexcept { 
        const size_t current_tail = tail_.load(std::memory_order_relaxed); 
        const size_t current_head = head_.load(std::memory_order_acquire);

        if (current_tail - current_head == Capacity) {
            return false; // Buffer is full
        }

        storage_[current_tail & MASK] = item;
        tail_.store(current_tail + 1, std::memory_order_release);
        
        return true;
    }

    // SPSC pop operation
    std::optional<T> pop() noexcept {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail) {
            return std::nullopt; // Buffer is empty
        }

        T item = storage_[current_head & MASK];
        head_.store(current_head + 1, std::memory_order_release);

        return item;
    }
};

} // namespace omni::bridge
