// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
//
// Validation suite for cbwk_shadow_pacer.hpp.
//
// T0  token RAII (uncommitted reservation releases held budget)
// T1  dual-ascent update law (rise on overspend, decay, clamp at zero)
// T2  shadow-price scoring flip (expensive action loses at high lambda)
// T3  concurrency hammer (invariant under 8 threads; ledgers reconcile)
// T4  zero heap allocations on the hot path (global new/delete counter)
// T5  seqlock price publication (no torn vectors under concurrent write)
// T6  end-to-end pacing simulation vs greedy baseline (honest metrics)
//
// Build: see build.sh (also runs TSan/ASan/UBSan builds).

#include "cbwk_shadow_pacer.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <span>
#include <thread>
#include <vector>

using pac::CBwKShadowPacer;
using pac::PacingError;
using pac::ReservationToken;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::printf("CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__,     \
                  #cond);                                             \
      ++g_failures;                                                   \
    }                                                                 \
  } while (0)

bool approx(double a, double b, double tol = 1e-9) {
  return std::fabs(a - b) <= tol;
}

// ---------------------------------------------------------------------------

void t0_token_raii() {
  CBwKShadowPacer<1> p;
  p.set_budget(0, 100'000);  // $0.10 in micro-units (tight, on purpose)
  CHECK(p.start());
  const auto decline = p.register_action({0});          // id 0
  const auto act = p.register_action({100'000});        // id 1, $0.10
  CHECK(decline == 0 && act == 1);

  {
    auto r = p.try_reserve(1);
    CHECK(r.has_value());
    CHECK(p.stats().held[0] == 100'000);  // outstanding
    // No commit: destructor must release.
  }
  CHECK(p.stats().held[0] == 0);
  CHECK(p.stats().spent[0] == 0);
  CHECK(p.stats().reservations_ok == 1);

  // Commit path.
  auto r2 = p.try_reserve(1);
  CHECK(r2.has_value());
  auto settlement = p.commit(*r2);
  CHECK(settlement.reserved[0] == 100'000);
  CHECK(p.stats().held[0] == 100'000);
  CHECK(p.stats().spent[0] == 100'000);

  // Settle refund (actual cheaper than estimate).
  p.settle(settlement, {40'000});
  auto s = p.stats();
  CHECK(s.held[0] == 40'000);
  CHECK(s.spent[0] == 40'000);

  // Budget exhausted.
  auto r3 = p.try_reserve(1);
  CHECK(!r3.has_value());
  CHECK(r3.error() == PacingError::BudgetExhausted);
  std::printf("T0 token RAII ................ OK\n");
}

void t1_dual_law() {
  CBwKShadowPacer<1> p;
  p.set_budget(0, 10'000'000);
  p.set_interval_target(0, 100'000);  // b = $0.10/interval
  p.set_eta(1.0);                     // [score]/[$]^2
  CHECK(p.start());
  CHECK(p.register_action({0}) == 0);
  CHECK(p.register_action({200'000}) == 1);  // $0.20 per use

  auto commit_one = [&p]() {
    auto r = p.try_reserve(1);
    CHECK(r.has_value());
    p.commit(*r);
  };

  commit_one();  // u = $0.20, b = $0.10 -> lambda = 0.1
  p.on_interval();
  CHECK(approx(p.prices()[0], 0.1, 1e-12));

  commit_one();  // lambda = 0.1 + 0.1 = 0.2
  p.on_interval();
  CHECK(approx(p.prices()[0], 0.2, 1e-12));

  p.on_interval();  // u = 0 -> lambda = 0.2 - 0.1 = 0.1
  CHECK(approx(p.prices()[0], 0.1, 1e-12));

  p.on_interval();  // lambda = 0.1 - 0.1 = 0.0
  CHECK(approx(p.prices()[0], 0.0, 1e-12));

  p.on_interval();  // clamp at zero (0 - 0.1 -> 0)
  CHECK(approx(p.prices()[0], 0.0, 1e-12));
  std::printf("T1 dual ascent law .......... OK\n");
}

void t2_scoring_flip() {
  CBwKShadowPacer<1> p;
  p.set_budget(0, 1'000'000'000);
  CHECK(p.start());
  CHECK(p.register_action({0}) == 0);          // decline
  CHECK(p.register_action({100'000}) == 1);    // premium, $0.10
  CHECK(p.register_action({5'000}) == 2);      // local, $0.005

  const double scores[3] = {0.0, 1.0, 0.8};
  const std::array<double, 1> lam_zero{0.0};
  const std::array<double, 1> lam_high{3.0};

  CHECK(p.select_with_prices(scores, lam_zero) == 1);   // premium
  CHECK(p.select_with_prices(scores, lam_high) == 2);   // local
  std::printf("T2 price-gated scoring ..... OK\n");
}

void t3_hammer() {
  constexpr std::size_t K = 2;
  constexpr int kThreads = 8;
  constexpr int kIters = 3000;

  CBwKShadowPacer<K> p;
  p.set_budget(0, 1'000);
  p.set_budget(1, 1'000);
  CHECK(p.start());
  const std::size_t a0 = p.register_action({0, 0});
  const std::size_t a1 = p.register_action({1, 1});
  const std::size_t a2 = p.register_action({3, 0});
  const std::size_t a3 = p.register_action({0, 2});
  CHECK(a0 == 0 && a1 == 1 && a2 == 2 && a3 == 3);

  std::atomic<std::uint64_t> committed_cost[K] = {0, 0};
  std::atomic<int> blocks{0};

  auto worker = [&](int seed) {
    std::mt19937 rng(static_cast<unsigned>(seed));
    for (int i = 0; i < kIters; ++i) {
      const std::size_t a = rng() % 4;  // uniform over decline/a1/a2/a3
      auto r = p.try_reserve(a);
      if (!r.has_value()) {
        if (a != a0) blocks.fetch_add(1);
        continue;
      }
      const auto reserved = r->reserved();
      p.commit(*r);
      for (std::size_t k = 0; k < K; ++k) {
        committed_cost[k].fetch_add(reserved[k]);
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker, t + 1);
  for (auto& t : threads) t.join();

  const auto s = p.stats();
  for (std::size_t k = 0; k < K; ++k) {
    CHECK(s.held[k] <= s.budget[k]);              // invariant
    CHECK(s.spent[k] == committed_cost[k]);       // reconciliation
    CHECK(s.spent[k] <= s.budget[k]);             // never over-dispatched
  }
  CHECK(blocks.load() > 0);  // tiny budgets: contention was real
  std::printf(
      "T3 hammer (8 threads) ...... OK  "
      "[spent %llu/%llu, %llu/%llu; blocked %d]\n",
      static_cast<unsigned long long>(s.spent[0]),
      static_cast<unsigned long long>(s.budget[0]),
      static_cast<unsigned long long>(s.spent[1]),
      static_cast<unsigned long long>(s.budget[1]), blocks.load());
}

std::atomic<std::size_t> g_allocs{0};

void t4_zero_alloc() {
  CBwKShadowPacer<2> p;
  p.set_budget(0, 1'000'000'000);
  p.set_budget(1, 1'000'000'000);
  p.set_eta(1.0);
  CHECK(p.start());
  CHECK(p.register_action({0, 0}) == 0);
  CHECK(p.register_action({100'000, 50'000}) == 1);
  CHECK(p.register_action({5'000, 10'000}) == 2);

  const double scores[3] = {0.0, 0.9, 0.7};
  // Warm up anything lazy, then measure.
  for (int i = 0; i < 100; ++i) {
    const std::size_t a = p.select(scores);
    auto r = p.try_reserve(a);
    if (r.has_value()) {
      auto st = p.commit(*r);
      p.settle(st, st.reserved);
    }
  }
  const std::size_t before = g_allocs.load();
  for (int i = 0; i < 10'000; ++i) {
    const std::size_t a = p.select(scores);
    auto r = p.try_reserve(a);
    if (r.has_value()) {
      auto st = p.commit(*r);
      if ((i % 100) == 0) p.settle(st, st.reserved);
    }
  }
  const std::size_t after = g_allocs.load();
  CHECK(before == after);
  std::printf("T4 hot-path allocations .... OK  [%zu allocations]\n",
              after - before);
}

void t5_seqlock() {
  CBwKShadowPacer<2> p;
  p.set_budget(0, 100'000'000);
  p.set_budget(1, 100'000'000);
  p.set_interval_target(0, 1);  // tiny target -> lambda rises every interval
  p.set_interval_target(1, 1);
  p.set_eta(0.05);
  CHECK(p.start());
  CHECK(p.register_action({0, 0}) == 0);
  CHECK(p.register_action({50'000, 25'000}) == 1);

  std::atomic<bool> stop{false};
  std::atomic<std::size_t> reads{0};
  std::atomic<bool> bad{false};

  auto reader = [&]() {
    double last0 = 0.0;
    double last1 = 0.0;
    while (!stop.load(std::memory_order_relaxed)) {
      const auto lam = p.prices();
      const double l0 = lam[0];
      const double l1 = lam[1];
      if (!std::isfinite(l0) || !std::isfinite(l1) || l0 < last0 - 1e-12 ||
          l1 < last1 - 1e-12 || l0 < 0.0 || l1 < 0.0) {
        bad.store(true);
      }
      last0 = l0;
      last1 = l1;
      reads.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::thread rt(reader);
  for (int i = 0; i < 2000; ++i) {
    auto r = p.try_reserve(1);
    if (r.has_value()) p.commit(*r);
    p.on_interval();
  }
  stop.store(true);
  rt.join();
  CHECK(!bad.load());
  CHECK(reads.load() > 1000);
  std::printf("T5 seqlock prices .......... OK  [%zu concurrent reads]\n",
              reads.load());
}

// ---------------------------------------------------------------------------
// T6: end-to-end pacing simulation vs a greedy baseline.
// ---------------------------------------------------------------------------

struct SimResult {
  double reward = 0.0;
  double spent = 0.0;
  double budget = 0.0;
  std::uint64_t served = 0;
  std::uint64_t premium = 0;
  std::uint64_t local = 0;
  std::uint64_t declined = 0;
  double final_lambda = 0.0;
};

// Actions: 0 decline ($0), 1 premium ($0.10), 2 local NPU ($0.005).
constexpr std::uint64_t kPremiumMicro = 100'000;
constexpr std::uint64_t kLocalMicro = 5'000;
constexpr double kBudget = 200.0;  // dollars
constexpr int kIntervals = 100;
constexpr int kPerInterval = 50;
constexpr int kRequests = kIntervals * kPerInterval;

SimResult run_sim(bool paced, double eta, unsigned seed) {
  CBwKShadowPacer<1> p;
  p.set_budget(0,
               static_cast<std::uint64_t>(kBudget * 1.0e6));
  p.set_interval_target(
      0, static_cast<std::uint64_t>(kBudget * 1.0e6) / kIntervals);
  p.set_eta(eta);
  CHECK(p.start());
  CHECK(p.register_action({0}) == 0);
  CHECK(p.register_action({kPremiumMicro}) == 1);
  CHECK(p.register_action({kLocalMicro}) == 2);

  std::mt19937 rng(seed);
  std::normal_distribution<double> score_premium(0.90, 0.05);
  std::normal_distribution<double> score_local(0.75, 0.05);

  SimResult r;
  r.budget = kBudget;
  const std::array<double, 1> zero{0.0};

  for (int i = 0; i < kRequests; ++i) {
    const double s[3] = {0.0, score_premium(rng), score_local(rng)};
    const std::size_t a =
        paced ? p.select(s) : p.select_with_prices(s, zero);
    auto res = p.try_reserve(a);
    if (!res.has_value()) {
      ++r.declined;  // budget guard refused; nothing dispatched
      continue;
    }
    auto st = p.commit(*res);
    r.spent += static_cast<double>(st.reserved[0]) * 1.0e-6;
    r.reward += s[a];
    ++r.served;
    if (a == 1) ++r.premium;
    if (a == 2) ++r.local;
    if ((i + 1) % kPerInterval == 0) p.on_interval();
  }
  r.final_lambda = p.prices()[0];
  const auto st = p.stats();
  CHECK(st.overshoot_events == 0);  // actual == estimate in this sim
  return r;
}

void t6_simulation() {
  std::printf(
      "T6 pacing simulation (%d requests, $%.0f budget, premium-only "
      "would spend $%.0f)\n",
      kRequests, kBudget, kRequests * 0.10);

  SimResult greedy{};
  {
    std::vector<SimResult> runs;
    for (unsigned seed = 1; seed <= 3; ++seed) {
      runs.push_back(run_sim(false, 0.0, seed));
    }
    for (const auto& r : runs) {
      greedy.reward += r.reward / 3.0;
      greedy.spent += r.spent / 3.0;
      greedy.premium += r.premium / 3;
      greedy.local += r.local / 3;
      greedy.declined += r.declined / 3;
      greedy.final_lambda += r.final_lambda / 3.0;
    }
  }
  std::printf(
      "  greedy (no prices) : reward %8.1f  spent $%7.2f  "
      "premium/local/declined %llu/%llu/%llu\n",
      greedy.reward, greedy.spent,
      static_cast<unsigned long long>(greedy.premium),
      static_cast<unsigned long long>(greedy.local),
      static_cast<unsigned long long>(greedy.declined));

  for (const double eta : {0.5, 1.0, 3.0}) {
    SimResult avg{};
    for (unsigned seed = 1; seed <= 3; ++seed) {
      const SimResult r = run_sim(true, eta, seed);
      avg.reward += r.reward / 3.0;
      avg.spent += r.spent / 3.0;
      avg.premium += r.premium / 3;
      avg.local += r.local / 3;
      avg.declined += r.declined / 3;
      avg.final_lambda += r.final_lambda / 3.0;
    }
    std::printf(
        "  pacer eta=%-4.1f     : reward %8.1f  spent $%7.2f  "
        "premium/local/declined %llu/%llu/%llu  lambda_final %.3f\n",
        eta, avg.reward, avg.spent,
        static_cast<unsigned long long>(avg.premium),
        static_cast<unsigned long long>(avg.local),
        static_cast<unsigned long long>(avg.declined), avg.final_lambda);
    CHECK(avg.spent <= kBudget + 1e-9);  // hard budget never breached
    CHECK(avg.reward > greedy.reward);   // pacing must beat naive greed
  }
}

}  // namespace

// Global allocation counters (test-only).
void* operator new(std::size_t n) {
  g_allocs.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(n)) return p;
  std::abort();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

int main() {
  t0_token_raii();
  t1_dual_law();
  t2_scoring_flip();
  t3_hammer();
  t4_zero_alloc();
  t5_seqlock();
  t6_simulation();

  if (g_failures == 0) {
    std::printf("\nALL CHECKS PASSED\n");
    return 0;
  }
  std::printf("\n%d CHECK(S) FAILED\n", g_failures);
  return 1;
}
