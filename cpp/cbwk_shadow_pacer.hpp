// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// cbwk_shadow_pacer.hpp — Contextual-Bandit-with-Knapsacks (CBwK)
// primal-dual shadow-price pacer. C++23, header-only, zero heap
// allocations on the hot path.
//
// SCOPE. This is the PACER, not the bandit. It consumes per-action
// scores from any external model/bandit, applies shadow-price
// penalties, enforces per-resource budgets through pre-execution
// reservations, and updates Lagrange multipliers (shadow prices) by
// projected dual subgradient ascent at interval boundaries. Reward
// learning is out of scope.
//
// MATH. Per resource k (USD, latency, memory, tokens, ...):
//   guard invariant (single atomic per resource):
//       held_k = spent_k + outstanding_k <= B_k          (CAS-enforced)
//   dual ascent at each interval boundary (projected subgradient):
//       lambda_k <- max(0, lambda_k + eta * (u_k - b_k))
//       u_k = realized consumption over the interval, b_k = target rate
//   action scoring under prices:
//       adjusted(a) = score(a) - sum_k lambda_k * c_k(a)
// With lambda in [score]/[unit] and u,b in [unit]/[interval], eta
// carries [score]/[unit]^2. eta MUST be tuned per workload; fixed eta
// oscillates around the budget rate by construction (see README).
// Lineage: Badanidiyuru et al., "Bandits with Knapsacks" (2013);
// budget-pacing duals per Balseiro et al.
//
// CONCURRENCY CONTRACT (read precisely — weaker than a naive reading):
//   * Per-resource reservations are race-free: one CAS on held_k.
//   * Cross-resource acquisition is TWO-PHASE: resources are acquired
//     one at a time and rolled back on later failure. A dispatch never
//     occurs unless ALL K reservations succeeded, but there is no single
//     atomic instant spanning all K resources (impossible without a
//     global lock or packed words). Transient outstanding reservations
//     on a subset of resources are visible while contending.
//   * Shadow prices are published through a seqlock (versioned
//     double-step with lock-free atomic<double> elements); concurrent
//     readers never observe a torn price vector.
//   * on_interval() and all setters are SINGLE-WRITER / setup-thread
//     only. select/try_reserve/commit/settle are multi-thread safe.
//
// BOUNDS. K (resources) and the action table are compile-time bounded
// (kMaxActions) so no hot-path path allocates. Ledger counters are
// uint64 micro-units; the guard uses overflow-safe headroom checks.
//
// ESTIMATES VS ACTUALS. c_k(a) is an ESTIMATE. The pre-execution
// invariant guards estimates; settle() trues-up actual consumption and
// may push held_k above B_k (counted in overshoot_events, never
// silently). The invariant is only as good as the cost model.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

// Cache-line size for ledger isolation. We define our own constant
// instead of std::hardware_destructive_interference_size: GCC warns
// that the library value can vary with -mtune/-mcpu, which would make
// the layout (and any ABI) compiler-flag dependent. 64 bytes is the
// destructive-interference size on every current x86-64/ARM target;
// revisit if targeting a platform where it is not.
inline constexpr std::size_t kPacerCacheLine = 64;

namespace pac {

// ---------------------------------------------------------------------------
// Error type (no exceptions on the hot path)
// ---------------------------------------------------------------------------

enum class PacingError : std::uint8_t {
  NotStarted,        // start() not called
  NotReady,          // start() called but budgets not configured
  InvalidAction,     // unknown action id
  BudgetExhausted,   // reservation would exceed a resource budget
  RegistryFull,      // cold path: action table at capacity
};

[[nodiscard]] constexpr const char* to_string(PacingError e) noexcept {
  switch (e) {
    case PacingError::NotStarted: return "not started";
    case PacingError::NotReady: return "not ready";
    case PacingError::InvalidAction: return "invalid action";
    case PacingError::BudgetExhausted: return "budget exhausted";
    case PacingError::RegistryFull: return "registry full";
  }
  return "?";
}

// `expected` alias. C++23 std::expected when available; a minimal
// API-compatible fallback otherwise (older toolchains, or forced for
// testing by defining PAC_NO_STD_EXPECTED).
#if defined(__cpp_lib_expected) && !defined(PAC_NO_STD_EXPECTED)

using std::expected;
inline constexpr std::unexpect_t unexpect{};

#else

struct unexpect_t {
  explicit unexpect_t() = default;
};
inline constexpr unexpect_t unexpect{};

template <typename T, typename E>
class expected {
 public:
  expected(T&& v) noexcept : ok_(true) {
    ::new (static_cast<void*>(&store_.t)) T(std::move(v));
  }
  expected(unexpect_t, E e) noexcept : ok_(false) {
    ::new (static_cast<void*>(&store_.e)) E(e);
  }
  expected(expected&& o) noexcept : ok_(o.ok_) {
    if (ok_) {
      ::new (static_cast<void*>(&store_.t)) T(std::move(o.store_.t));
    } else {
      ::new (static_cast<void*>(&store_.e)) E(std::move(o.store_.e));
    }
  }
  expected(const expected&) = delete;
  expected& operator=(const expected&) = delete;
  expected& operator=(expected&&) = delete;
  ~expected() {
    if (ok_) {
      store_.t.~T();
    } else {
      store_.e.~E();
    }
  }
  [[nodiscard]] bool has_value() const noexcept { return ok_; }
  [[nodiscard]] explicit operator bool() const noexcept { return ok_; }
  [[nodiscard]] T& operator*() & noexcept { return store_.t; }
  [[nodiscard]] T&& operator*() && noexcept { return std::move(store_.t); }
  [[nodiscard]] T* operator->() noexcept { return &store_.t; }
  [[nodiscard]] E error() const noexcept { return store_.e; }

 private:
  union Store {
    Store() noexcept {}
    ~Store() noexcept {}
    T t;
    E e;
  } store_;
  bool ok_;
};

#endif  // __cpp_lib_expected

// ---------------------------------------------------------------------------
// Reservation token (RAII; releases unless committed)
// ---------------------------------------------------------------------------

template <std::size_t K>
class CBwKShadowPacer;

template <std::size_t K>
class ReservationToken {
 public:
  ReservationToken() noexcept = default;
  ~ReservationToken() { release(); }

  ReservationToken(const ReservationToken&) = delete;
  ReservationToken& operator=(const ReservationToken&) = delete;

  ReservationToken(ReservationToken&& o) noexcept
      : pacer_(o.pacer_), reserved_(o.reserved_) {
    o.pacer_ = nullptr;
    o.reserved_.fill(0);
  }
  ReservationToken& operator=(ReservationToken&& o) noexcept {
    if (this != &o) {
      release();
      pacer_ = o.pacer_;
      reserved_ = o.reserved_;
      o.pacer_ = nullptr;
      o.reserved_.fill(0);
    }
    return *this;
  }

  [[nodiscard]] bool valid() const noexcept { return pacer_ != nullptr; }
  [[nodiscard]] const std::array<std::uint64_t, K>& reserved()
      const noexcept {
    return reserved_;
  }

 private:
  friend class CBwKShadowPacer<K>;

  ReservationToken(CBwKShadowPacer<K>* p,
                   const std::array<std::uint64_t, K>& r) noexcept
      : pacer_(p), reserved_(r) {}

  void release() noexcept;

  CBwKShadowPacer<K>* pacer_ = nullptr;
  std::array<std::uint64_t, K> reserved_{};
};

// ---------------------------------------------------------------------------
// Pacer
// ---------------------------------------------------------------------------

template <std::size_t K>
class CBwKShadowPacer {
 public:
  using Cost = std::array<std::uint64_t, K>;  // micro-units per resource
  static constexpr std::size_t kResources = K;
  static constexpr std::size_t kMaxActions = 64;
  static constexpr std::size_t kInvalidAction = static_cast<std::size_t>(-1);
  static constexpr double kMicro = 1.0e-6;

  struct Settlement {
    Cost reserved{};
  };

  struct Stats {
    std::array<std::uint64_t, K> held{};
    std::array<std::uint64_t, K> spent{};
    std::array<std::uint64_t, K> budget{};
    std::array<double, K> lambda{};
    std::uint64_t reservations_ok = 0;
    std::uint64_t reservations_failed = 0;
    std::uint64_t commits = 0;
    std::uint64_t overshoot_events = 0;
    std::uint64_t intervals = 0;
  };

  CBwKShadowPacer() noexcept {
    for (auto& l : ledgers_) {
      l.held.store(0, std::memory_order_relaxed);
      l.spent.store(0, std::memory_order_relaxed);
    }
    for (auto& v : lambda_) v.store(0.0, std::memory_order_relaxed);
  }

  // ---- cold path (setup thread only, before start()) --------------------

  [[nodiscard]] std::size_t register_action(const Cost& cost) noexcept {
    if (num_actions_ >= kMaxActions) return kInvalidAction;
    costs_[num_actions_] = cost;
    return num_actions_++;
  }

  void set_budget(std::size_t k, std::uint64_t budget_micro) noexcept {
    ledgers_[k].budget = budget_micro;
    budgets_set_ |= (std::uint64_t{1} << k);
  }

  // Target consumption per interval (micro-units). This is b_k.
  void set_interval_target(std::size_t k,
                           std::uint64_t target_micro) noexcept {
    interval_target_[k] = target_micro;
  }

  // Step size for the dual update, in [score]/[unit]^2.
  void set_eta(double eta) noexcept { eta_ = eta; }

  [[nodiscard]] bool start() noexcept {
    if (budgets_set_ != full_mask()) return false;
    started_ = true;
    return true;
  }

  // ---- lock-free read path ----------------------------------------------

  // Current shadow prices (seqlock-consistent snapshot).
  [[nodiscard]] std::array<double, K> prices() const noexcept {
    for (;;) {
      const std::uint64_t s1 =
          price_seq_.load(std::memory_order_acquire);
      if ((s1 & 1u) != 0u) continue;  // writer in progress
      std::array<double, K> out{};
      for (std::size_t k = 0; k < K; ++k) {
        out[k] = lambda_[k].load(std::memory_order_relaxed);
      }
      const std::uint64_t s2 =
          price_seq_.load(std::memory_order_acquire);
      if (s1 == s2) return out;
    }
  }

  [[nodiscard]] std::size_t num_actions() const noexcept {
    return num_actions_;
  }

  // Adjusted score of `action` under explicit prices (evaluation hook).
  [[nodiscard]] double adjusted_score(
      std::span<const double> scores, std::size_t action,
      const std::array<double, K>& lam) const noexcept {
    if (action >= num_actions_ || action >= scores.size()) {
      return 0.0 / 0.0;  // NaN
    }
    double adj = scores[action];
    for (std::size_t k = 0; k < K; ++k) {
      adj -= lam[k] * static_cast<double>(costs_[action][k]) * kMicro;
    }
    return adj;
  }

  // Argmax over adjusted scores. Convention: register an id-0 "decline"
  // action with zero cost and score it 0.0 so refusing is always possible.
  [[nodiscard]] std::size_t select(
      std::span<const double> scores) const noexcept {
    return select_with_prices(scores, prices());
  }

  [[nodiscard]] std::size_t select_with_prices(
      std::span<const double> scores,
      const std::array<double, K>& lam) const noexcept {
    if (num_actions_ == 0) return kInvalidAction;
    const std::size_t n = num_actions_ < scores.size() ? num_actions_
                                                       : scores.size();
    double best = 0.0 / 0.0;  // NaN
    std::size_t best_a = kInvalidAction;
    for (std::size_t a = 0; a < n; ++a) {
      const double adj = adjusted_score(scores, a, lam);
      if (best_a == kInvalidAction || adj > best) {
        best = adj;
        best_a = a;
      }
    }
    return best_a;
  }

  // ---- hot path: reservation / commit / settle --------------------------
  //
  // Two-phase per-resource acquisition (see concurrency contract).
  // On failure every already-acquired resource is rolled back and the
  // caller receives BudgetExhausted; nothing was dispatched.

  [[nodiscard]] expected<ReservationToken<K>, PacingError> try_reserve(
      std::size_t action) noexcept {
    if (!started_) {
      return expected<ReservationToken<K>, PacingError>{
          unexpect, PacingError::NotStarted};
    }
    if (action >= num_actions_) {
      return expected<ReservationToken<K>, PacingError>{
          unexpect, PacingError::InvalidAction};
    }
    const Cost& c = costs_[action];

    for (std::size_t k = 0; k < K; ++k) {
      const std::uint64_t need = c[k];
      if (need == 0) continue;
      Ledger& L = ledgers_[k];
      std::uint64_t cur = L.held.load(std::memory_order_relaxed);
      for (;;) {
        // Overflow-safe headroom (held may exceed budget after a
        // settle() overshoot; then nothing can be reserved).
        const std::uint64_t headroom =
            (cur > L.budget) ? 0u : (L.budget - cur);
        if (need > headroom) {
          rollback(k, c);
          ++reservations_failed_;
          return expected<ReservationToken<K>, PacingError>{
              unexpect, PacingError::BudgetExhausted};
        }
        if (L.held.compare_exchange_weak(cur, cur + need,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed)) {
          break;
        }
      }
    }
    ++reservations_ok_;
    return ReservationToken<K>{this, c};
  }

  // Move `reserved` into `spent`. The token becomes inert. Returns a
  // Settlement for the later settle() true-up.
  Settlement commit(ReservationToken<K>& token) noexcept {
    Settlement s{};
    if (token.pacer_ != this) return s;  // inert or foreign token
    for (std::size_t k = 0; k < K; ++k) {
      if (token.reserved_[k] != 0) {
        ledgers_[k].spent.fetch_add(token.reserved_[k],
                                    std::memory_order_relaxed);
      }
      s.reserved[k] = token.reserved_[k];
    }
    token.pacer_ = nullptr;
    token.reserved_.fill(0);
    ++commits_;
    return s;
  }

  // True-up after execution: `actual` replaces the estimate. Consuming
  // more than estimated can push held above budget (overshoot event,
  // counted); consuming less refunds the difference.
  void settle(const Settlement& s, const Cost& actual) noexcept {
    for (std::size_t k = 0; k < K; ++k) {
      const std::uint64_t r = s.reserved[k];
      const std::uint64_t a = actual[k];
      if (a == r) continue;
      Ledger& L = ledgers_[k];
      if (a > r) {
        const std::uint64_t d = a - r;
        const std::uint64_t h =
            L.held.fetch_add(d, std::memory_order_acq_rel);
        if (h + d > L.budget) ++overshoot_events_;
        L.spent.fetch_add(d, std::memory_order_relaxed);
      } else {
        const std::uint64_t d = r - a;
        L.held.fetch_sub(d, std::memory_order_acq_rel);
        L.spent.fetch_sub(d, std::memory_order_relaxed);
      }
    }
  }

  // ---- interval boundary (single writer: the pacer thread) --------------

  void on_interval() noexcept {
    for (std::size_t k = 0; k < K; ++k) {
      const std::uint64_t s =
          ledgers_[k].spent.load(std::memory_order_relaxed);
      const std::uint64_t u = s - last_spent_[k];  // spent is monotonic
      last_spent_[k] = s;
      const double excess = static_cast<double>(u) * kMicro -
                            static_cast<double>(interval_target_[k]) * kMicro;
      double lam = lambda_writer_[k];
      lam += eta_ * excess;
      if (lam < 0.0) lam = 0.0;
      lambda_writer_[k] = lam;
    }
    publish();
    ++intervals_;
    if (intervals_ <= kLambdaHistory) {
      lambda_history_[static_cast<std::size_t>(intervals_ - 1)] =
          lambda_writer_;
    }
  }

  // ---- telemetry ---------------------------------------------------------

  [[nodiscard]] Stats stats() const noexcept {
    Stats s{};
    for (std::size_t k = 0; k < K; ++k) {
      s.held[k] = ledgers_[k].held.load(std::memory_order_relaxed);
      s.spent[k] = ledgers_[k].spent.load(std::memory_order_relaxed);
      s.budget[k] = ledgers_[k].budget;
    }
    s.lambda = prices();
    s.reservations_ok = reservations_ok_.load(std::memory_order_relaxed);
    s.reservations_failed =
        reservations_failed_.load(std::memory_order_relaxed);
    s.commits = commits_.load(std::memory_order_relaxed);
    s.overshoot_events =
        overshoot_events_.load(std::memory_order_relaxed);
    s.intervals = static_cast<std::uint64_t>(intervals_);
    return s;
  }

  static constexpr std::size_t kLambdaHistory = 256;
  [[nodiscard]] std::size_t history_len() const noexcept {
    return static_cast<std::size_t>(
        intervals_ < kLambdaHistory ? intervals_ : kLambdaHistory);
  }
  [[nodiscard]] std::array<double, K> lambda_at(
      std::size_t interval_index) const noexcept {
    // interval_index in [0, history_len())
    return lambda_history_[interval_index];
  }

 private:
  friend class ReservationToken<K>;

  struct alignas(kPacerCacheLine) Ledger {
    std::atomic<std::uint64_t> held{0};   // spent + outstanding (guard)
    std::atomic<std::uint64_t> spent{0};  // settled consumption
    std::uint64_t budget{0};
  };

  void rollback(std::size_t upto, const Cost& c) noexcept {
    for (std::size_t i = 0; i < upto; ++i) {
      if (c[i] != 0) {
        ledgers_[i].held.fetch_sub(c[i], std::memory_order_acq_rel);
      }
    }
  }

  void release_reservation(const Cost& c) noexcept {
    for (std::size_t k = 0; k < K; ++k) {
      if (c[k] != 0) {
        ledgers_[k].held.fetch_sub(c[k], std::memory_order_acq_rel);
      }
    }
  }

  void publish() noexcept {
    const std::uint64_t s = price_seq_.load(std::memory_order_relaxed);
    price_seq_.store(s + 1, std::memory_order_release);  // odd: writing
    for (std::size_t k = 0; k < K; ++k) {
      lambda_[k].store(lambda_writer_[k], std::memory_order_relaxed);
    }
    price_seq_.store(s + 2, std::memory_order_release);  // even: done
  }

  [[nodiscard]] static constexpr std::uint64_t full_mask() noexcept {
    if (K >= 64) return ~std::uint64_t{0};
    return (std::uint64_t{1} << K) - 1;
  }

  // --- data members --------------------------------------------------
  std::array<Ledger, K> ledgers_{};

  alignas(kPacerCacheLine) std::atomic<std::uint64_t> price_seq_{0};
  alignas(kPacerCacheLine) std::array<std::atomic<double>, K> lambda_{};

  // writer-owned copies (setup/pacer thread only)
  std::array<double, K> lambda_writer_{};
  std::array<std::uint64_t, K> last_spent_{};
  std::array<std::uint64_t, K> interval_target_{};
  std::array<Cost, kMaxActions> costs_{};
  std::array<std::array<double, K>, kLambdaHistory> lambda_history_{};

  std::size_t num_actions_ = 0;
  double eta_ = 0.0;
  std::uint64_t budgets_set_ = 0;
  bool started_ = false;
  std::uint64_t intervals_ = 0;

  alignas(kPacerCacheLine) std::atomic<std::uint64_t> reservations_ok_{0};
  std::atomic<std::uint64_t> reservations_failed_{0};
  std::atomic<std::uint64_t> commits_{0};
  std::atomic<std::uint64_t> overshoot_events_{0};
};

template <std::size_t K>
inline void ReservationToken<K>::release() noexcept {
  if (pacer_ != nullptr) {
    pacer_->release_reservation(reserved_);
    pacer_ = nullptr;
    reserved_.fill(0);
  }
}

}  // namespace pac
