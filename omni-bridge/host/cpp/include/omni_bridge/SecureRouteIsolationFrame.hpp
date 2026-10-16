// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <concepts>
#include <expected>
#include <span>
#include <string_view>
#include <cstdint>

namespace omni::bridge {

struct alignas(64) RouteHeader { 
    uint32_t magic_validation; 
    uint32_t payload_length; 
    uint32_t route_id; 
    uint32_t risk_tier; 
}; 

template <typename T>
concept ValidRoutePayload = std::is_trivial_v<T> && std::is_standard_layout_v<T>; 

enum class RouteError {
    InvalidSize,
    AlignmentFault,
    MagicMismatch,
    PayloadTruncated,
    TypeMismatch
};

class SecureRouteIsolationFrame {
private: 
    std::span<const uint8_t> raw_buffer_; 
    const RouteHeader* header_{nullptr}; 

    static constexpr uint32_t OMNI_MAGIC = 0x03170317;

public: 
    explicit SecureRouteIsolationFrame(std::span<const uint8_t> buffer) : raw_buffer_(buffer) {}

    std::expected<void, RouteError> validate() {
        if (raw_buffer_.size() < sizeof(RouteHeader)) {
            return std::unexpected(RouteError::InvalidSize);
        }

        // Enforce 64-byte cache line alignment
        if (reinterpret_cast<std::uintptr_t>(raw_buffer_.data()) % alignof(RouteHeader) != 0) {
            return std::unexpected(RouteError::AlignmentFault);
        }

        header_ = reinterpret_cast<const RouteHeader*>(raw_buffer_.data());

        if (header_->magic_validation != OMNI_MAGIC) {
            return std::unexpected(RouteError::MagicMismatch);
        }

        if (raw_buffer_.size() < sizeof(RouteHeader) + header_->payload_length) {
            return std::unexpected(RouteError::PayloadTruncated);
        }

        return {};
    }

    [[nodiscard]] const RouteHeader* header() const {
        return header_;
    }

    template <ValidRoutePayload T>
    std::expected<std::span<const T>, RouteError> extract_payload() const {
        if (!header_) {
            return std::unexpected(RouteError::MagicMismatch);
        }
        
        if (header_->payload_length % sizeof(T) != 0) {
            return std::unexpected(RouteError::TypeMismatch);
        }

        const auto* data = reinterpret_cast<const T*>(raw_buffer_.data() + sizeof(RouteHeader));
        const size_t count = header_->payload_length / sizeof(T);
        
        return std::span<const T>(data, count);
    }
};

} // namespace omni::bridge
