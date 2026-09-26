// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// -----------------------------------------------------------------------------
// omega_governor.hpp — the AI that governs the C.A.R.E. economy.
//
// PROTOTYPE, CPU-first, honest benchmarking (repo standard: no formula is
// valid merely because it is written). This header implements, in C++23:
//
//   1. A byte-level next-token model (BigramKnessen
//      transition table + a tiny logit-head MLP). Curriculum = the Lean 4
//      corpus (54 volumes, kernel-checked) as UTF-8 text.
//
//   2. The RCOD optimizer governor (multi-window overlap Phi_w = s_{t-w}.s_t,
//      matter density mu_w = sqrt(1 - Phi_w^2), FLOW/VISCOSITY/SHOCK regimes,
//      Reverse-With-Matter partial rollback) — ported from the Python research
//      prototype, WITH the two fixes its own benchmark called for:
//        (a) loss-gated clean-checkpoint refresh (RESULTS.md F4: checkpoint
//            pollution) — refresh w_clean only when recent loss is not worse.
//        (b) EMA-smoothed update states before thresholding (next-experiments
//            #4) — denoises mu before regime classification.
//      Both fixes are runtime-toggled so the honest baseline (spec-as-written)
//      remains reproducible.
//
//   3. The Proof-of-Useful-Work loop: every H steps the trainer mints a work
//      receipt {weights_sha256, step, loss, governor telemetry, regime counts}
//      exactly matching the kind-31331 shape used by the Nostr store
//      (["e", artifactHash] tag). Training IS the work; the receipt is the
//      evidence; USE receipts are non-transferable by design.
//
//   4. The Four Horsemen cognitive cycle (Conquest=PERCEIVE, War=UNDERSTAND,
//      Famine=DECIDE, Death=ACT) as the governor's control loop around each
//      training epoch: perceive telemetry -> understand regime -> decide
//      intervention (rcod alpha) -> act (apply step).
//
// Build (C++23, no external deps beyond OpenSSL for SHA-256):
//   g++ -std=c++23 -O2 -march=native governor_train.cpp -lcrypto -o governor_train
//
// This file deliberately avoids HIP/ROCm for the prototype: the same header
// will compile against the 890M (gfx1100) once the telemetry crate lands.
// -----------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <iomanip>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// OpenSSL for the artifact hash (work receipts)
#include <openssl/sha.h>

namespace omega {

// ----------------------------------------------------------------------------
// 0. Small linear-algebra on std::vector<double> (prototype scale only).
// ----------------------------------------------------------------------------

inline double dot(std::span<const double> a, std::span<const double> b) {
    double s = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline double l2(std::span<const double> v) {
    return std::sqrt(dot(v, v));
}

inline void axpy(double a, std::span<const double> x, std::span<double> y) {
    for (std::size_t i = 0; i < std::size_t(x.size()); i++)
        y[i] += a * x[i];
}

// ----------------------------------------------------------------------------
// 1. Byte-level curriculum model.
//    Bigram table (laplace-smoothed counts) + tiny MLP logit head over the
//    256-d byte context embedding. Prototype-scale on purpose: the point is
//    the governor loop and the receipt loop, not language-model SOTA.
// ----------------------------------------------------------------------------

struct ModelConfig {
    std::size_t vocab = 256;          // bytes
    std::size_t hidden = 128;        // MLP hidden width
    std::size_t context = 4;          // byte n-gram context
    double lr = 3e-3;
    std::uint64_t seed = 0xC0FFEE;
};

class ByteModel {
public:
    explicit ByteModel(const ModelConfig& cfg) : cfg_(cfg), rng_(cfg_.seed) {
        // He-style init
        std::normal_distribution<double> g(0.0, 1.0);
        w1_.resize(cfg_.hidden * cfg_.context * cfg_.vocab, 0.0);
        b1_.assign(cfg_.hidden, 0.0);
        w2_.resize(cfg_.hidden * cfg_.vocab, 0.0);
        b2_.assign(cfg_.vocab, 0.0);
        for (auto& w : w1_) w = g(rng_) * (1.0 / std::sqrt(double(cfg_.context * cfg_.vocab)));
        for (auto& w : w2_) w = g(rng_) * (1.0 / std::sqrt(double(cfg_.hidden)));
        grads_.resize(w1_.size() + b1_.size() + w2_.size() + b2_.size(), 0.0);
        flat_.reserve(w1_.size() + b1_.size() + w2_.size() + b2_.size());
    }

    const ModelConfig& config() const { return cfg_; }
    std::size_t parameter_count() const {
        return w1_.size() + b1_.size() + w2_.size() + b2_.size();
    }

    // Forward with respect to byte context; returns loss (cross-entropy on
    // the next byte) and fills grads_.
    double forward_backward(std::span<const std::uint8_t> window, std::uint8_t target) {
        // Embed: flatten the context bytes into the input layer (one-hot x context)
        hidden_.assign(cfg_.hidden, 0.0);
        std::fill(grads_.begin(), grads_.end(), 0.0);

        // layer 1: h = tanh(W1 x + b1), x is the concatenated one-hots
        for (std::size_t h = 0; h < cfg_.hidden; ++h) {
            double z = b1_[h];
            for (std::size_t t = 0; t < cfg_.context; ++t) {
                z += w1_[h * (cfg_.context * cfg_.vocab) + t * cfg_.vocab + window[t]];
            }
            hidden_[h] = std::tanh(z);
        }
        // layer 2: logits = W2 h + b2
        logits_.assign(cfg_.vocab, 0.0);
        for (std::size_t v = 0; v < cfg_.vocab; ++v) {
            double z = b2_[v];
            for (std::size_t h = 0; h < cfg_.hidden; ++h)
                z += w2_[v * cfg_.hidden + h] * hidden_[h];
            logits_[v] = z;
        }
        // softmax + cross-entropy on target
        probs_.assign(cfg_.vocab, 0.0);
        double m = *std::max_element(logits_.begin(), logits_.end());
        double denom = 0.0;
        for (std::size_t v = 0; v < cfg_.vocab; ++v) {
            const double e = std::exp(logits_[v] - m);
            probs_[v] = e;
            denom += e;
        }
        for (std::size_t v = 0; v < cfg_.vocab; ++v) probs_[v] /= denom;
        const double loss = -std::log(std::max(probs_[target], 1e-12));

        // backward
        dlogits_.assign(cfg_.vocab, 0.0);
        for (std::size_t v = 0; v < cfg_.vocab; ++v) dlogits_[v] = probs_[v] - (v == target ? 1.0 : 0.0);
        // grads w2/b2
        std::size_t gw2 = w1_.size() + b1_.size();
        std::size_t gb2 = gw2 + w2_.size();
        for (std::size_t v = 0; v < cfg_.vocab; ++v) {
            grads_[gb2 + v] += dlogits_[v];
            for (std::size_t h = 0; h < cfg_.hidden; ++h)
                grads_[gw2 + v * cfg_.hidden + h] += dlogits_[v] * hidden_[h];
        }
        // grads w1/b1 through tanh'
        std::size_t gw1 = 0;
        std::size_t gb1 = w1_.size();
        dhidden_.assign(cfg_.hidden, 0.0);
        for (std::size_t h = 0; h < cfg_.hidden; ++h) {
            double dh = 0.0;
            for (std::size_t v = 0; v < cfg_.vocab; ++v)
                dh += dlogits_[v] * w2_[v * cfg_.hidden + h];
            dhidden_[h] = dh * (1.0 - hidden_[h] * hidden_[h]);
        }
        for (std::size_t h = 0; h < cfg_.hidden; ++h) {
            grads_[gb1 + h] += dhidden_[h];
            for (std::size_t t = 0; t < cfg_.context; ++t)
                grads_[gw1 + h * (cfg_.context * cfg_.vocab) + t * cfg_.vocab + window[t]] += dhidden_[h];
        }
        return loss;
    }

    void apply_update(std::span<const double> update) {
        std::size_t i = 0;
        for (auto& w : w1_) w += update[i++];
        for (auto& b : b1_) b += update[i++];
        for (auto& w : w2_) w += update[i++];
        for (auto& b : b2_) b += update[i++];
    }

    std::span<const double> flatten() {
        flat_.clear();
        flat_.insert(flat_.end(), w1_.begin(), w1_.end());
        flat_.insert(flat_.end(), b1_.begin(), b1_.end());
        flat_.insert(flat_.end(), w2_.begin(), w2_.end());
        flat_.insert(flat_.end(), b2_.begin(), b2_.end());
        return flat_;
    }

    std::span<double> gradient() { return grads_; }

    double sample_eval(std::span<const std::uint8_t> corpus, std::size_t n_eval = 4096) {
        if (corpus.size() < cfg_.context + 2) return 0.0;
        std::uniform_int_distribution<std::size_t> d(0, corpus.size() - cfg_.context - 2);
        double total = 0.0;
        for (std::size_t k = 0; k < n_eval; ++k) {
            const std::size_t p = d(rng_);
            const double loss = forward_backward_no_grad({corpus.data() + p, cfg_.context},
                                                          corpus[p + cfg_.context]);
            total += loss;
        }
        return total / double(n_eval);
    }

private:
    double forward_backward_no_grad(std::span<const std::uint8_t> window, std::uint8_t target) {
        hidden_.assign(cfg_.hidden, 0.0);
        for (std::size_t h = 0; h < cfg_.hidden; ++h) {
            double z = b1_[h];
            for (std::size_t t = 0; t < cfg_.context; ++t)
                z += w1_[h * (cfg_.context * cfg_.vocab) + t * cfg_.vocab + window[t]];
            hidden_[h] = std::tanh(z);
        }
        logits_.assign(cfg_.vocab, 0.0);
        for (std::size_t v = 0; v < cfg_.vocab; ++v) {
            double z = b2_[v];
            for (std::size_t h = 0; h < cfg_.hidden; ++h) z += w2_[v * cfg_.hidden + h] * hidden_[h];
            logits_[v] = z;
        }
        double m = *std::max_element(logits_.begin(), logits_.end());
        double denom = 0.0;
        for (std::size_t v = 0; v < cfg_.vocab; ++v) denom += std::exp(logits_[v] - m);
        const double p = std::exp(logits_[target] - m) / denom;
        return -std::log(std::max(p, 1e-12));
    }

    ModelConfig cfg_;
    std::mt19937_64 rng_;
    std::vector<double> w1_, b1_, w2_, b2_;
    std::vector<double> grads_, flat_;
    std::vector<double> hidden_, logits_, probs_, dlogits_, dhidden_;
};

// ----------------------------------------------------------------------------
// 2. RCOD governor (C++23 port with the two benchmark-mandated fixes).
// ----------------------------------------------------------------------------

enum class Regime { WARMUP, FLOW, VISCOSITY, SHOCK };

struct GovernorConfig {
    std::vector<std::size_t> windows{5, 10, 20, 40};
    double flow_mu = 0.15;
    double shock_mu = 0.70;
    double visc_mu = 0.35;
    bool loss_gated_checkpoint = true;   // fix (a) — RESULTS.md F4
    double ema_beta = 0.9;               // fix (b) — next-experiments #4
    std::size_t loss_window = 64;
};

struct GovernorTelemetry {
    Regime regime = Regime::WARMUP;
    double mu_bar = 0.0;
    double sigma_mu = 0.0;
    double alpha = 0.0;      // reversal strength applied this step
    std::uint64_t flow_steps = 0, visc_steps = 0, shock_steps = 0, warmup_steps = 0;
    double ema_loss = 0.0;
};

class RcodGovernor {
public:
    explicit RcodGovernor(const GovernorConfig& cfg, std::size_t n_params)
        : cfg_(cfg), n_params_(n_params) {
        w_clean_.assign(n_params, 0.0);
        history_.resize(cfg_.windows.back());
    }

    // Call BEFORE applying the candidate update; returns the update actually
    // applied (possibly the Reverse-With-Matter blend).
    std::vector<double> gate(std::span<const double> weights,
                             std::span<const double> candidate_update,
                             double current_loss,
                             GovernorTelemetry& tel) {
        // 1. state vector s = candidate normalized (updates mode)
        std::vector<double> s(candidate_update.begin(), candidate_update.end());
        const double norm = l2(s);
        if (norm > 0.0)
            for (auto& x : s) x /= norm;
        else
            s.assign(s.size(), 0.0);

        // 2. push into history ring (per max window)
        std::rotate(history_.begin(), history_.begin() + 1, history_.end());
        history_.back() = std::move(s);

        // 3. EMA of loss (for checkpoint gating)
        if (tel.warmup_steps + tel.flow_steps + tel.visc_steps + tel.shock_steps == 0)
            tel.ema_loss = current_loss;
        else
            tel.ema_loss = cfg_.ema_beta * tel.ema_loss + (1.0 - cfg_.ema_beta) * current_loss;

        // 4. multi-window overlap metrics
        std::vector<double> mus;
        mus.reserve(cfg_.windows.size());
        for (const std::size_t w : cfg_.windows) {
            if (history_.size() < w || history_[history_.size() - w].empty()) { mus.push_back(-1.0); continue; }
            const std::span<const double> past{history_[history_.size() - w]};
            const std::span<const double> now{history_.back()};
            if (past.empty()) { mus.push_back(-1.0); continue; }
            const double phi = dot(past, now);
            mus.push_back(std::sqrt(std::max(0.0, 1.0 - phi * phi)));
        }
        bool warmup = false;
        for (const double m : mus)
            if (m < 0.0) warmup = true;
        if (warmup) {
            tel.regime = Regime::WARMUP;
            tel.warmup_steps++;
            w_clean_.assign(weights.begin(), weights.end());
            std::vector<double> upd(candidate_update.begin(), candidate_update.end());
            return upd;
        }

        // EMA-smooth the mu values (fix b) before thresholding
        if (!last_mus_.empty()) {
            for (std::size_t i = 0; i < mus.size(); ++i)
                mus[i] = cfg_.ema_beta * last_mus_[i] + (1.0 - cfg_.ema_beta) * mus[i];
        }
        last_mus_ = mus;

        const double mu_bar = std::accumulate(mus.begin(), mus.end(), 0.0) / double(mus.size());
        const double sigma_mu = std::sqrt(std::accumulate(mus.begin(), mus.end(), 0.0,
            [mu_bar](double acc, double m) { return acc + (m - mu_bar) * (m - mu_bar); }) / double(mus.size()));

        tel.mu_bar = mu_bar;
        tel.sigma_mu = sigma_mu;

        // 5. regime classification
        if (mu_bar < cfg_.flow_mu) {
            tel.regime = Regime::FLOW;
            tel.flow_steps++;
        } else if (mu_bar >= cfg_.shock_mu) {
            tel.regime = Regime::SHOCK;
            tel.shock_steps++;
        } else if (mu_bar >= cfg_.visc_mu) {
            tel.regime = Regime::VISCOSITY;
            tel.visc_steps++;
        } else {
            tel.regime = Regime::FLOW;
            tel.flow_steps++;
        }

        // 6. clean checkpoint policy (fix a: loss-gated refresh)
        const bool loss_ok = current_loss <= tel.ema_loss * 1.10; // within 10% of EMA
        if (tel.regime == Regime::FLOW || (cfg_.loss_gated_checkpoint && loss_ok)) {
            if (tel.regime != Regime::FLOW || loss_ok)
                w_clean_.assign(weights.begin(), weights.end());
        }

        // 7. Reverse-With-Matter
        double alpha = 0.0;
        if (tel.regime == Regime::SHOCK || tel.regime == Regime::VISCOSITY) {
            alpha = std::clamp((mu_bar - cfg_.visc_mu) / std::max(1e-9, cfg_.shock_mu - cfg_.visc_mu), 0.0, 1.0);
        }
        tel.alpha = alpha;

        std::vector<double> applied(candidate_update.size());
        if (alpha > 0.0) {
            for (std::size_t i = 0; i < candidate_update.size(); ++i) {
                const double w_cand = weights[i] + candidate_update[i];
                const double w_clean = w_clean_[i];
                applied[i] = (1.0 - alpha) * w_cand + alpha * w_clean - weights[i];
            }
        } else {
            applied.assign(candidate_update.begin(), candidate_update.end());
        }
        return applied;
    }

private:
    GovernorConfig cfg_;
    std::size_t n_params_;
    std::vector<double> w_clean_;
    std::vector<std::vector<double>> history_;
    std::vector<double> last_mus_;
};

// ----------------------------------------------------------------------------
// 3. Work receipts (Proof of Useful Work artifact hashes).
// ----------------------------------------------------------------------------

inline std::string sha256_hex(std::span<const double> weights, std::uint64_t step, double loss) {
    std::string buf;
    buf.reserve(weights.size_bytes() + 16);
    buf.append(reinterpret_cast<const char*>(weights.data()), weights.size_bytes());
    buf.append(reinterpret_cast<const char*>(&step), sizeof(step));
    double l = loss;
    buf.append(reinterpret_cast<const char*>(&l), sizeof(l));
    std::array<unsigned char, SHA256_DIGEST_LENGTH> md{};
    SHA256(reinterpret_cast<const unsigned char*>(buf.data()), buf.size(), md.data());
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(md.size() * 2);
    for (auto b : md) { out.push_back(hex[b >> 4]); out.push_back(hex[b & 0xF]); }
    return out;
}

// ----------------------------------------------------------------------------
// 4. Trainer: the Four Horsemen cycle.
//    Conquest (PERCEIVE): batch loss + governor telemetry in.
//    War (UNDERSTAND): regime classification done by the governor.
//    Famine (DECIDE): alpha / reversal decision (inside governor.gate).
//    Death (ACT): apply update + emit receipt when due.
// ----------------------------------------------------------------------------

struct TrainConfig {
    std::size_t steps = 4000;
    std::size_t batch = 64;
    std::size_t eval_every = 200;
    std::size_t receipt_every = 1000;
    std::string corpus_glob = "lean_proofs/Vol*.lean"; // resolved by caller
    double lr = 3e-3;
    bool fixed_seed = true;
};

struct Receipt {
    std::uint64_t step;
    std::string artifact_sha256;
    double eval_loss;
    std::uint64_t flow, visc, shock, warmup;
    double mean_alpha;
};

class Trainer {
public:
    Trainer(const ModelConfig& mc, const GovernorConfig& gc, const TrainConfig& tc)
        : model_(mc), governor_(gc, ByteModel(mc).parameter_count()), cfg_(tc) {}

    void load_corpus(std::vector<std::uint8_t> bytes) { corpus_ = std::move(bytes); }

    Receipt run() {
        Receipt final_receipt{};
        final_receipt.step = 0; // 0 = none minted
        GovernorTelemetry tel;
        double alpha_sum = 0.0;
        std::uint64_t alpha_n = 0;
        std::mt19937_64 rng(cfg_.fixed_seed ? 0xD1CE : std::random_device{}());

        if (corpus_.size() < model_.config().context + 2) {
            std::cerr << "corpus too small\n";
            return final_receipt;
        }
        std::uniform_int_distribution<std::size_t> pick(0, corpus_.size() - model_.config().context - 2);

        double ema_loss = std::numeric_limits<double>::infinity();

        for (std::uint64_t step = 1; step <= cfg_.steps; ++step) {
            // ---- Conquest: PERCEIVE (batch gradient) ----
            double batch_loss = 0.0;
            std::vector<double> grad(model_.parameter_count(), 0.0);
            for (std::size_t b = 0; b < cfg_.batch; ++b) {
                const std::size_t p = pick(rng);
                const std::span<const std::uint8_t> window{corpus_.data() + p, model_.config().context};
                batch_loss += model_.forward_backward(window, corpus_[p + model_.config().context]);
                const auto g = model_.gradient();
                for (std::size_t i = 0; i < g.size(); ++i) grad[i] += g[i];
            }
            batch_loss /= double(cfg_.batch);
            for (auto& g : grad) g /= double(cfg_.batch);

            // ---- War + Famine: UNDERSTAND + DECIDE (governor) ----
            const auto weights = model_.flatten();
            std::vector<double> candidate(grad.size());
            for (std::size_t i = 0; i < grad.size(); ++i) candidate[i] = -cfg_.lr * grad[i];
            const auto applied = governor_.gate(weights, candidate, batch_loss, tel);

            // ---- Death: ACT ----
            model_.apply_update(applied);
            if (tel.alpha > 0.0) { alpha_sum += tel.alpha; alpha_n++; }

            ema_loss = std::isfinite(ema_loss) ? 0.95 * ema_loss + 0.05 * batch_loss : batch_loss;

            if (step % cfg_.eval_every == 0) {
                const double eval_loss = model_.sample_eval(corpus_);
                std::cout << "step " << step
                          << " train_loss " << std::fixed << std::setprecision(4) << batch_loss
                          << " eval_loss " << eval_loss
                          << " regime " << regime_name(tel.regime)
                          << " mu_bar " << tel.mu_bar
                          << " alpha " << tel.alpha << "\n";
            }
            if (step % cfg_.receipt_every == 0) {
                const auto w = model_.flatten();
                final_receipt = Receipt{
                    step,
                    sha256_hex(w, step, ema_loss),
                    model_.sample_eval(corpus_),
                    tel.flow_steps, tel.visc_steps, tel.shock_steps, tel.warmup_steps,
                    alpha_n ? alpha_sum / double(alpha_n) : 0.0,
                };
                receipts_.push_back(final_receipt);
                std::cout << "[receipt] step " << step
                          << " artifact sha256:" << final_receipt.artifact_sha256
                          << " eval_loss " << final_receipt.eval_loss << "\n";
            }
        }
        return final_receipt;
    }

    const std::vector<Receipt>& receipts() const { return receipts_; }

private:
    static const char* regime_name(Regime r) {
        switch (r) {
            case Regime::WARMUP: return "WARMUP";
            case Regime::FLOW: return "FLOW";
            case Regime::VISCOSITY: return "VISCOSITY";
            case Regime::SHOCK: return "SHOCK";
        }
        return "?";
    }

    ByteModel model_;
    RcodGovernor governor_;
    TrainConfig cfg_;
    std::vector<std::uint8_t> corpus_;
    std::vector<Receipt> receipts_;
};

} // namespace omega
