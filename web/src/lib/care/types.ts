// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
export type CarePrivacyLevel = "P0" | "P1" | "P2" | "P3";
export type CareRole = "participant" | "X1" | "X2" | "archangel";
export type CareFeed = "circle" | "requests" | "projects";

export type CarePost = {
  id: string;
  author: string;
  role: CareRole;
  privacyLevel: CarePrivacyLevel;
  circle: string;
  body: string;
  createdAt: string;
  audienceCount: number;
  attentionEvents: number;
  supportLabel: string;
  accent: string;
};

export type CareRequest = {
  id: string;
  title: string;
  detail: string;
  circle: string;
  status: "open" | "covered";
  supportCount: number;
  privacy: "pseudonymous" | "verified-circle";
};

export type CareProfile = {
  displayName: string;
  role: CareRole;
  privacyLevel: CarePrivacyLevel;
  selectedQuotaPercent: number;
  minimumQuotaPercent: number;
  contributedUnits: number;
  requiredUnits: number;
};
