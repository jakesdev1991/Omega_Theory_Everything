// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

// ToE Grounding — the control boundary's "thinking instructions" (§20/§71).
//
// At startup this loads the kernel-checked Lean corpus index from the
// Omega Theory of Everything repo and attaches theorem-level citations to
// each pipeline stage. Grounding = honest citations to kernel-checked
// declarations, NOT prose claims. A missing corpus degrades to explicit
// "grounding: unavailable" — never fabricated citations.

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <optional>
#include <unordered_map>
#include <algorithm>

namespace omni::bridge {

using Json = nlohmann::json;

struct GroundingCitation {
    std::string file;
    int         line = 0;
    std::string kind;       // theorem | lemma | def ...
    std::string decl;
    std::string integrity;  // KERNEL_CHECKED | VOLUME_SOURCE
};

class ToeGrounding {
private:
    std::string lean_dir_;
    std::unordered_map<std::string, GroundingCitation> by_stage_;
    bool loaded_ = false;
    size_t kernel_checked_count_ = 0;

public:
    explicit ToeGrounding(std::string lean_dir)
        : lean_dir_(std::move(lean_dir)) {}

    // Map of pipeline stage -> declaration name in the Lean corpus.
    // Chosen from kernel_audit_targets.json — each must actually exist there.
    static const std::vector<std::pair<std::string, std::string>>& stage_theorems() {
        static const std::vector<std::pair<std::string, std::string>> m = {
            {"classify", "OmegaProtocol.Vol18.probabilities_in_unit_interval"},
            {"appa",     "APPA.try_declassify_rejects_iff"},
            {"route",    "CBwK.runCosts_conservation"},
            {"budget",   "CBwK.runCosts_eq_some_iff"},
            {"sandbox",  "OmegaProtocol.Vol12.no_universal_overlap_copy"},
            {"entropy",  "InformationPhysics.informationMass_strictMono"},
            {"born",     "OmegaProtocol.Vol04.born_rule"},
            {"equilibrium", "OmegaProtocol.Vol31.nash_iff_mutual_defection"},
        };
        return m;
    }

    // Stage-specific thinking directives, phrased against the theorems above.
    // These are the deterministic "instructions for thinking" the bridge
    // attaches to its decisions; an agent consuming a route result receives
    // the citation + directive, and may verify them in the corpus.
    static const std::unordered_map<std::string, std::string>& stage_directives() {
        static const std::unordered_map<std::string, std::string> d = {
            {"classify",
             "Classify before you act: probabilities are confined to the unit "
             "interval (Vol18). Any confidence outside [0,1] is a defect; any "
             "certainty below the QSTAR threshold must escalate, never guess."},
            {"appa",
             "Authority is an intersection, never a grant: effective permission "
             "is System∩Declared∩Tenant∩Tool∩Context minus Risk. Risk removes; "
             "it never manufactures. Declassification is a typed gate with "
             "proven rejection cases (APPA.try_declassify_rejects_iff)."},
            {"route",
             "Routing conserves cost: the CBwK run-cost invariant holds — budget "
             "spent must equal budget accounted. A route that loses cost accounting "
             "is invalid regardless of its output quality."},
            {"budget",
             "A budget is a guard, not a suggestion: when the cost bound is Some, "
             "execution must be admitted by the bound; otherwise abstain."},
            {"sandbox",
             "No universal overlap copy: isolation is real — a sandboxed context "
             "cannot be aliased with its parent. If isolation can be bypassed by "
             "aliasing, the tier is void."},
            {"entropy",
             "Information mass is strictly monotone: more claimed knowledge must "
             "mean more information consumed. Claims without information mass are "
             "hallucinations — reject them."},
            {"born",
             "Measurement obeys the Born rule: outcomes are probabilities, not "
             "wishes. Report the distribution; never collapse it into the answer "
             "you wanted."},
            {"equilibrium",
             "Adversarial equilibrium: assume the counterparty defects (Nash). "
             "A verification that only holds when inputs are honest is not a "
             "verification."},
        };
        return d;
    }

    // Load and verify stage->theorem citations against the corpus metadata.
    // Reads only kernel_audit_targets.json + greps the .lean files' names —
    // cheap, no Lean invocation at runtime.
    bool load() {
        by_stage_.clear();
        kernel_checked_count_ = 0;
        std::string audit_path = lean_dir_ + "/kernel_audit_targets.json";
        std::ifstream f(audit_path);
        if (!f) {
            loaded_ = false;
            return false;
        }
        Json targets = Json::parse(f, nullptr, false);
        if (targets.is_discarded() || !targets.is_array()) {
            loaded_ = false;
            return false;
        }
        std::unordered_map<std::string, bool> checked;
        for (const auto& t : targets) {
            if (t.is_string()) {
                checked[t.get<std::string>()] = true;
                ++kernel_checked_count_;
            }
        }

        for (const auto& [stage, decl] : stage_theorems()) {
            // find the file containing this declaration
            GroundingCitation c;
            c.decl = decl;
            c.integrity = checked.count(decl) ? "KERNEL_CHECKED" : "VOLUME_SOURCE";
            // scan .lean files for "<name> " with the last dot-segment
            std::string leaf = decl.substr(decl.find_last_of('.') + 1);
            // file guess: namespace segment before last dot, lowercased
            std::string ns = decl.substr(0, decl.find_last_of('.'));
            std::transform(ns.begin(), ns.end(), ns.begin(),
                           [](unsigned char ch){ return std::tolower(ch); });
            std::vector<std::string> candidates = {
                lean_dir_ + "/" + leaf + ".lean",
                lean_dir_ + "/" + ns + ".lean",
                lean_dir_ + "/OmegaProtocol" + ".lean",
            };
            for (const auto& path : candidates) {
                std::ifstream lf(path);
                if (!lf) continue;
                std::string line;
                int lineno = 0;
                while (std::getline(lf, line)) {
                    ++lineno;
                    if (line.find(leaf) != std::string::npos) {
                        c.file = path.substr(path.find_last_of('/') + 1);
                        c.line = lineno;
                        // kind: crude but honest — read the line's leading token
                        if (line.find("theorem") != std::string::npos) c.kind = "theorem";
                        else if (line.find("lemma") != std::string::npos) c.kind = "lemma";
                        else c.kind = "def";
                        break;
                    }
                }
                if (c.line > 0) break;
            }
            by_stage_[stage] = c;
        }
        loaded_ = !by_stage_.empty();
        return loaded_;
    }

    bool loaded() const { return loaded_; }
    size_t kernel_checked() const { return kernel_checked_count_; }

    // JSON block attached to decisions: citation + directive per stage.
    Json grounding_block(const std::vector<std::string>& stages) const {
        Json out = Json::object();
        out["corpus"] = loaded_ ? "omega_theory_of_everything/lean_proofs" : "unavailable";
        out["kernel_checked_targets"] = kernel_checked_count_;
        if (!loaded_) {
            out["note"] = "Lean corpus not found — decisions proceed WITHOUT "
                          "ToE citations; do not treat this as grounded.";
            return out;
        }
        Json stages_j = Json::object();
        for (const auto& s : stages) {
            auto it = by_stage_.find(s);
            if (it == by_stage_.end()) continue;
            const auto& c = it->second;
            Json cj = Json::object();
            cj["decl"] = c.decl;
            cj["kind"] = c.kind;
            cj["integrity"] = c.integrity;
            if (!c.file.empty()) {
                cj["citation"] = "[" + c.file + ":" + std::to_string(c.line) + "]";
            }
            auto dit = stage_directives().find(s);
            if (dit != stage_directives().end()) cj["directive"] = dit->second;
            stages_j[s] = cj;
        }
        out["stages"] = stages_j;
        return out;
    }
};

} // namespace omni::bridge
