// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <blake3.h>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>

namespace omni::bridge {

struct ProvenanceRecord {
    std::string timestamp;
    std::string origin_agent;
    std::string action_type;
    std::string payload;
    std::string previous_hash;
    std::string current_hash;
};

class LedgerProvenanceEngine {
private:
    std::string latest_hash_{"0000000000000000000000000000000000000000000000000000000000000000"}; // 256-bit Genesis
    std::vector<ProvenanceRecord> chain_;

    std::string compute_blake3_hash(const std::string& input) const {
        blake3_hasher hasher;
        blake3_hasher_init(&hasher);
        blake3_hasher_update(&hasher, input.data(), input.size());

        uint8_t output[BLAKE3_OUT_LEN];
        blake3_hasher_finalize(&hasher, output, BLAKE3_OUT_LEN);

        std::stringstream ss;
        for (int i = 0; i < BLAKE3_OUT_LEN; ++i) {
            ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(output[i]);
        }
        return ss.str();
    }

public:
    LedgerProvenanceEngine() = default;

    ProvenanceRecord record_action(const std::string& agent, const std::string& action, const std::string& payload) {
        // Concatenate inputs with the previous block's hash to form a cryptographic chain
        std::string raw_data = latest_hash_ + agent + action + payload;
        std::string new_hash = compute_blake3_hash(raw_data);

        ProvenanceRecord record{
            "CURRENT_TIME_STUB", // Should be ISO-8601 in production
            agent,
            action,
            payload,
            latest_hash_,
            new_hash
        };

        chain_.push_back(record);
        latest_hash_ = new_hash;

        return record;
    }

    bool verify_chain_integrity() const {
        if (chain_.empty()) return true;

        std::string running_hash = "0000000000000000000000000000000000000000000000000000000000000000";
        for (const auto& block : chain_) {
            if (block.previous_hash != running_hash) {
                return false; // Broken chain link
            }
            std::string raw_data = running_hash + block.origin_agent + block.action_type + block.payload;
            std::string expected_hash = compute_blake3_hash(raw_data);
            
            if (expected_hash != block.current_hash) {
                return false; // Payload tampering
            }
            running_hash = expected_hash;
        }
        return true;
    }

    std::string get_latest_hash() const { return latest_hash_; }
};

} // namespace omni::bridge
