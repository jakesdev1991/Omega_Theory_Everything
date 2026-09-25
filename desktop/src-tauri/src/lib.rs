// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Omega Wallet desktop wrapper.
//!
//! The wallet itself is the synced GUI in `web/public/omega-wallet/` (produced
//! from `app/` by `web/scripts/sync-wallet.mjs`). This crate only provides the
//! native window; it intentionally registers no commands and exposes no
//! additional capabilities, so the desktop build cannot do anything the web
//! build cannot.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running the Omega Wallet desktop wrapper");
}
