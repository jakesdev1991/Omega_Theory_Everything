// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace omni::bridge {

class RetrievalEngine {
private:
    std::vector<std::string> documents_;
    std::vector<std::vector<float>> normalized_vectors_;

    static void normalize(std::vector<float>& vec) {
        float sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f);
        if (sq_sum > 0.0f) {
            float norm = std::sqrt(sq_sum);
            for (auto& val : vec) val /= norm;
        }
    }

public:
    void add_document(std::string doc, std::vector<float> vec) {
        normalize(vec);
        documents_.push_back(std::move(doc));
        normalized_vectors_.push_back(std::move(vec));
    }

    struct SearchResult {
        std::string document;
        float score;
    };

    std::vector<SearchResult> search(std::vector<float> query, size_t k = 3) const {
        if (documents_.empty() || query.empty()) return {};

        normalize(query);
        std::vector<SearchResult> results;
        results.reserve(documents_.size());

        for (size_t i = 0; i < documents_.size(); ++i) {
            float sim = std::inner_product(
                normalized_vectors_[i].begin(), normalized_vectors_[i].end(),
                query.begin(), 0.0f
            );
            results.push_back({documents_[i], sim});
        }

        std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
            return a.score > b.score;
        });

        if (results.size() > k) {
            results.resize(k);
        }
        return results;
    }
};

} // namespace omni::bridge
