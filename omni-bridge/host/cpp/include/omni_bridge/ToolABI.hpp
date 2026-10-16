// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <stdexcept>

namespace omni::bridge {

struct OmniToolCall {
    std::string tool_name;
    std::string version{"1.0.0"};
    std::string json_arguments;
    int risk_tier{0};
    std::string idempotency_key;
};

struct OmniToolResponse {
    std::string status;
    std::string json_result;
    std::string error_message;
    int consumed_fuel{0};
};

class ToolABISerializer {
public:
    static std::string serialize_call(const OmniToolCall& call) {
        // Minimal JSON constructor (assuming sanitized inputs)
        return "{\"tool_name\":\"" + call.tool_name + 
               "\",\"version\":\"" + call.version + 
               "\",\"arguments\":" + (call.json_arguments.empty() ? "{}" : call.json_arguments) + 
               ",\"risk_tier\":" + std::to_string(call.risk_tier) + 
               ",\"idempotency_key\":\"" + call.idempotency_key + "\"}";
    }

    static OmniToolResponse deserialize_response(const std::string& json_data) {
        // Very basic stub parser (production would use simdjson)
        OmniToolResponse response;
        if (json_data.find("\"status\":\"success\"") != std::string::npos || json_data.find("\"status\": \"success\"") != std::string::npos) {
            response.status = "success";
        } else {
            response.status = "error";
            response.error_message = "Deserialization or explicit error.";
        }
        return response;
    }
};

} // namespace omni::bridge
