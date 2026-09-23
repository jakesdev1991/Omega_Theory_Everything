"use client";

const HEADER_HEIGHT = 64;

export function scrollTo(id: string, behavior: ScrollBehavior = "smooth") {
  const el = document.getElementById(id);
  if (!el) return;

  const y = el.getBoundingClientRect().top + window.scrollY - HEADER_HEIGHT;
  window.scrollTo({ top: y, behavior });
}

export function peek(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  const y = el.getBoundingClientRect().top + window.scrollY - HEADER_HEIGHT;
  window.scrollTo({ top: Math.max(0, y - 60), behavior: "smooth" });
}

export function sectionTopOffset() {
  return HEADER_HEIGHT + 16;
}
