// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"use client";

import { Section } from "@/components/Section";

/**
 * The Omega Theory of Everything — 3:37 explainer video (2026-09-26).
 * Served from /media/Omega_Theory_of_Everything.mp4 (720p h264 + AAC).
 */
export function OmegaVideo() {
  return (
    <Section
      title="The Omega Theory of Everything"
      eyebrow="Watch"
      subtitle="The theory and the economy it underwrites, in three minutes and thirty-seven seconds."
    >
      <video
        controls
        preload="metadata"
        playsInline
        style={{
          width: "100%",
          maxWidth: "960px",
          aspectRatio: "16 / 9",
          borderRadius: "12px",
          border: "1px solid var(--color-border)",
          background: "#000",
          display: "block",
        }}
      >
        <source
          src="/media/Omega_Theory_of_Everything.mp4"
          type="video/mp4"
        />
        Your browser does not support HTML video. The video is also available
        at <code>/media/Omega_Theory_of_Everything.mp4</code>.
      </video>
    </Section>
  );
}
