// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
import { NextRequest, NextResponse } from "next/server";

import { getEngine } from "@/lib/domain/server-engine";
import type { AuditEvent, Plane } from "@/lib/domain/types";

const AUDIT_COLUMNS = [
  "eventId",
  "timestamp",
  "plane",
  "action",
  "policyVersion",
  "actorCommitment",
  "inputCommitments",
  "outputCommitments",
  "evidenceReferences",
] as const;

function toCsv(events: AuditEvent[]): string {
  const escape = (value: string) => `"${value.replace(/"/g, '""')}"`;
  const lines = [AUDIT_COLUMNS.join(",")];
  for (const event of events) {
    lines.push(
      [
        event.eventId,
        event.timestamp,
        event.plane,
        event.action,
        event.policyVersion,
        event.actorCommitment,
        event.inputCommitments.join("|"),
        event.outputCommitments.join("|"),
        event.evidenceReferences.join("|"),
      ]
        .map(escape)
        .join(","),
    );
  }
  return `${lines.join("\n")}\n`;
}

/**
 * GET /api/economy/audit
 * The full audited mutation trail with filters and export formats:
 *   ?plane=CARE|TWC|OMEGA|AMITY  &action=settle_weekly_allocation  &actor=alice
 *   &format=json|ndjson|csv      &limit=200
 */
export async function GET(request: NextRequest) {
  const engine = getEngine();
  const params = request.nextUrl.searchParams;
  const plane = params.get("plane") as Plane | null;
  const action = params.get("action");
  const actor = params.get("actor");
  const format = params.get("format") || "json";
  const limitParam = Number(params.get("limit") || 500);
  const limit = Number.isFinite(limitParam) && limitParam > 0 ? Math.min(limitParam, 10_000) : 500;

  let events = [...engine.auditEvents];
  if (plane) events = events.filter((event) => event.plane === plane);
  if (action) events = events.filter((event) => event.action === action);
  if (actor) events = events.filter((event) => event.actorCommitment === actor);
  events = events.slice(-limit);

  if (format === "csv") {
    return new NextResponse(toCsv(events), {
      headers: {
        "content-type": "text/csv; charset=utf-8",
        "content-disposition": 'attachment; filename="omega-economy-audit.csv"',
      },
    });
  }

  if (format === "ndjson") {
    return new NextResponse(`${events.map((event) => JSON.stringify(event)).join("\n")}\n`, {
      headers: {
        "content-type": "application/x-ndjson; charset=utf-8",
        "content-disposition": 'attachment; filename="omega-economy-audit.ndjson"',
      },
    });
  }

  const actions = Array.from(new Set(engine.auditEvents.map((event) => event.action))).sort();
  const planes = Array.from(new Set(engine.auditEvents.map((event) => event.plane))).sort();

  return NextResponse.json({ ok: true, count: events.length, total: engine.auditEvents.length, actions, planes, events });
}
