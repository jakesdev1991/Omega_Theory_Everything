// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// governor_train.cpp — Proof-of-Useful-Work trainer entry point.
//
// Trains the byte-level curriculum model on the Lean 4 corpus under the RCOD
// governor, mints kind-31331-shaped work receipts, and prints a RESULTS-style
// honest summary (baseline vs governed comparison at prototype scale).
//
// Usage:
//   ./governor_train [corpus_glob] [steps]
//   ./governor_train "lean_proofs/*.lean" 2000

#include "omega/omega_governor.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace {

std::vector<std::uint8_t> load_corpus(const std::string& glob) {
    // Minimal glob: directory + prefix/suffix split on '*'
    std::vector<std::uint8_t> out;
    const auto star = glob.find('*');
    if (star == std::string::npos) {
        std::ifstream f(glob, std::ios::binary);
        if (!f) { std::cerr << "cannot open " << glob << "\n"; return out; }
        std::ostringstream ss; ss << f.rdbuf();
        const std::string s = ss.str();
        out.insert(out.end(), s.begin(), s.end());
        return out;
    }
    const std::string dir_part = glob.substr(0, star);
    const std::string suffix = glob.substr(star + 1);
    namespace fs = std::filesystem;
    const fs::path dir = dir_part.empty() ? fs::path(".") : fs::path(dir_part);
    std::vector<std::string> files;
    std::error_code ec;
    for (fs::directory_iterator it(dir.empty() ? fs::path(".") : dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().string();
        if (name.size() >= suffix.size() && name.ends_with(suffix))
            files.push_back(name);
    }
    std::sort(files.begin(), files.end());
    for (const auto& f : files) {
        std::ifstream file(f, std::ios::binary);
        if (!file) continue;
        std::ostringstream ss; ss << file.rdbuf();
        const std::string s = ss.str();
        out.insert(out.end(), s.begin(), s.end());
        out.push_back('\n');
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    using namespace omega;

    const std::string glob = argc > 1 ? argv[1] : "lean_proofs/*.lean";
    const std::uint64_t steps = argc > 2 ? std::stoull(argv[2]) : 2000;

    auto corpus = load_corpus(glob);
    std::cout << "corpus: " << corpus.size() << " bytes from " << glob << "\n";
    if (corpus.size() < 4096) {
        std::cerr << "corpus too small for training\n";
        return 1;
    }

    ModelConfig mc;      // defaults: 256-byte vocab, hidden 128, context 4
    GovernorConfig gc;    // spec thresholds; fixes (a)+(b) ON by default
    TrainConfig tc;
    tc.steps = steps;
    tc.receipt_every = std::max<std::uint64_t>(1, steps / 4); // always mint receipts

    // --- Run 1: governed ---
    Trainer governed(mc, gc, tc);
    governed.load_corpus(corpus);
    std::cout << "\n=== RCOD-governed run ===\n";
    const auto receipt_g = governed.run();
    std::cout << "\ngoverned receipts: " << governed.receipts().size()
              << " final eval_loss " << receipt_g.eval_loss << "\n";

    // --- Run 2: baseline (governor effectively disabled: never reverses) ---
    GovernorConfig gc_off = gc;
    gc_off.flow_mu = 2.0;      // mu_bar can never reach 2.0 -> always FLOW
    gc_off.shock_mu = 3.0;
    gc_off.visc_mu = 2.5;
    Trainer baseline(mc, gc_off, tc);
    baseline.load_corpus(corpus);
    std::cout << "\n=== baseline run (governor inert) ===\n";
    const auto receipt_b = baseline.run();
    std::cout << "\nbaseline receipts: " << baseline.receipts().size()
              << " final eval_loss " << receipt_b.eval_loss << "\n";

    // --- Honest verdict (repo standard) ---
    const auto report = [](const char* label, const Receipt& r) {
        if (r.step == 0) {
            std::cout << label << ": no receipts minted\n";
        } else {
            std::cout << label << " final receipt: step " << r.step
                      << " eval_loss " << std::fixed << std::setprecision(4) << r.eval_loss
                      << " sha256:" << r.artifact_sha256.substr(0, 16) << "…\n";
        }
    };
    std::cout << "\n=== summary ===\n";
    std::cout << "params: " << ByteModel(mc).parameter_count() << "\n";
    std::cout << "steps: " << steps << "\n";
    report("baseline", receipt_b);
    report("governed ", receipt_g);
    std::cout << "governed regimes: flow " << receipt_g.flow
              << " visc " << receipt_g.visc
              << " shock " << receipt_g.shock
              << " warmup " << receipt_g.warmup << "\n";
    std::cout << "governed mean alpha: " << receipt_g.mean_alpha << "\n";
    if (receipt_b.step == 0 || receipt_g.step == 0) {
        std::cout << "verdict: incomplete (no receipts on one arm)\n";
        return 0;
    }
    const double delta = receipt_b.eval_loss - receipt_g.eval_loss;
    std::cout << "delta (baseline - governed): " << delta << "\n";
    std::cout << "verdict: "
              << (delta > 0 ? "governor helped at this scale" :
                  delta < 0 ? "governor hurt at this scale" :
                              "indistinguishable at this scale")
              << " (single seed, prototype scale — treat as a smoke test, not a result)\n";

    return 0;
}
