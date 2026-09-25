// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
import type { CarePost, CareProfile, CareRequest } from "./types";

export const demoCareProfile: CareProfile = {
  displayName: "North Star",
  role: "participant",
  privacyLevel: "P0",
  selectedQuotaPercent: 2,
  minimumQuotaPercent: 2,
  contributedUnits: 72,
  requiredUnits: 100,
};

export const demoCarePosts: CarePost[] = [
  {
    id: "post-1",
    author: "Moss & Morning",
    role: "X1",
    privacyLevel: "P1",
    circle: "Quiet Hours",
    body: "Opened a low-pressure listening room for anyone who needs company while starting the day. No advice required; presence is enough.",
    createdAt: "12 min ago",
    audienceCount: 38,
    attentionEvents: 94,
    supportLabel: "Listening room",
    accent: "var(--color-care)",
  },
  {
    id: "post-2",
    author: "Juniper Table",
    role: "participant",
    privacyLevel: "P0",
    circle: "Mutual Aid / South",
    body: "This week's community meal is open to neighbors. Bring what you can, or simply arrive. We are keeping the room quiet and accessible.",
    createdAt: "1 hr ago",
    audienceCount: 64,
    attentionEvents: 171,
    supportLabel: "Community meal",
    accent: "var(--color-twc)",
  },
  {
    id: "post-3",
    author: "Signal Garden",
    role: "X2",
    privacyLevel: "P2",
    circle: "Repair & Practice",
    body: "A short guide for repairing a difficult conversation without demanding immediate forgiveness. Read it, keep what helps, and leave the rest.",
    createdAt: "3 hrs ago",
    audienceCount: 112,
    attentionEvents: 286,
    supportLabel: "Repair practice",
    accent: "var(--color-accent-soft)",
  },
];

export const demoCareRequests: CareRequest[] = [
  {
    id: "request-1",
    title: "Ride-share to the community kitchen",
    detail: "Two seats are needed for Saturday's meal. The request is visible only to the Southside circle.",
    circle: "Mutual Aid / South",
    status: "open",
    supportCount: 1,
    privacy: "verified-circle",
  },
  {
    id: "request-2",
    title: "Quiet company during a paperwork hour",
    detail: "A pseudonymous participant is asking for a low-pressure check-in while completing housing forms.",
    circle: "Quiet Hours",
    status: "open",
    supportCount: 3,
    privacy: "pseudonymous",
  },
  {
    id: "request-3",
    title: "Share a warm meal plan",
    detail: "The resource list is ready. One more person is needed to help distribute it at the neighborhood center.",
    circle: "Mutual Aid / South",
    status: "covered",
    supportCount: 5,
    privacy: "verified-circle",
  },
];
