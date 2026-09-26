// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * Relay pool for the Omega nostr client.
 *
 * - dials every configured relay, answers NIP-42 AUTH challenges with a
 *   kind-22242 event tagged ["relay", <the exact URL we dialed>] and
 *   ["challenge", challenge] (chorus rejects a mismatched relay tag with
 *   "AUTH failure: relay is wrong" — verified live against chorus 2.0.2),
 * - publishes events and resolves on the first ["OK", id, true],
 * - subscribes REQs and surfaces EVENT/EOSE/CLOSED,
 * - reconnects with exponential backoff, TLS always verified.
 *
 * Nothing here trusts relay data: events are strictly verified by the
 * consumer (see events.mjs verifyEventStrict) before use.
 */

import { EventEmitter } from "node:events";
import { randomUUID } from "node:crypto";

import WebSocket from "ws";

import { signEvent } from "./events.mjs";

const AUTH_KIND = 22242;

export class RelayConnection extends EventEmitter {
  /**
   * @param {string} url
   * @param {string} secretHex  signing key for NIP-42 AUTH (the client identity)
   */
  constructor(url, secretHex) {
    super();
    this.url = url;
    this.secretHex = secretHex;
    /** @type {WebSocket|null} */
    this.socket = null;
    this.connected = false;
    this.authed = false;
    this.backoffMs = 1000;
    this.maxBackoffMs = 30_000;
    this.pending = new Map(); // eventId -> {resolve, reject, timer}
    this.subs = new Map(); // subId -> {onEvent, onEose}
    this.closedByUs = false;
  }

  connect() {
    this.closedByUs = false;
    const socket = new WebSocket(this.url);
    this.socket = socket;

    socket.on("open", () => {
      this.connected = true;
      this.backoffMs = 1000;
      this.emit("connected");
      // resubscribe after reconnect
      for (const [subId, sub] of this.subs) {
        socket.send(JSON.stringify(["REQ", subId, sub.filter]));
      }
    });

    socket.on("message", (raw) => {
      let msg;
      try {
        msg = JSON.parse(raw.toString());
      } catch {
        return;
      }
      const [type, a, b] = msg;
      switch (type) {
        case "AUTH": {
          // NIP-42: reply with a signed kind-22242 event. The relay tag
          // MUST be exactly the URL we dialed.
          const auth = signEvent(
            {
              kind: AUTH_KIND,
              created_at: Math.floor(Date.now() / 1000),
              tags: [
                ["relay", this.url],
                ["challenge", a],
              ],
              content: "",
            },
            this.secretHex,
          );
          this._lastAuthId = auth.id;
          socket.send(JSON.stringify(["AUTH", auth]));
          break;
        }
        case "OK": {
          // ["OK", eventId, accepted, note]
          const waiter = this.pending.get(a);
          if (waiter) {
            this.pending.delete(a);
            clearTimeout(waiter.timer);
            if (b === true) waiter.resolve({ relay: this.url, accepted: true, note: msg[3] ?? "" });
            else waiter.reject(new Error(`relay ${this.url} rejected event ${a}: ${msg[3] ?? "no note"}`));
          }
          if (b === true && a === this._lastAuthId) this.authed = true;
          break;
        }
        case "EVENT": {
          // ["EVENT", subId, event]
          const sub = this.subs.get(a);
          if (sub) sub.onEvent(b);
          break;
        }
        case "EOSE": {
          const sub = this.subs.get(a);
          if (sub) sub.onEose?.();
          break;
        }
        case "CLOSED": {
          this.emit("closed-notice", a, b);
          break;
        }
        case "NOTICE": {
          this.emit("notice", a);
          break;
        }
        default:
          break;
      }
    });

    socket.on("error", (err) => this.emit("error", err));
    socket.on("close", () => {
      this.connected = false;
      this.authed = false;
      const err = new Error(`relay ${this.url} closed before OK`);
      for (const [, waiter] of this.pending) {
        clearTimeout(waiter.timer);
        waiter.reject(err);
      }
      this.pending.clear();
      this.emit("disconnected");
      if (!this.closedByUs) {
        const delay = this.backoffMs;
        this.backoffMs = Math.min(this.maxBackoffMs, this.backoffMs * 2);
        setTimeout(() => this.connect(), delay).unref?.();
      }
    });
  }

  /** Track our AUTH event id so OK on it flips `authed`. */
  _lastAuthId = null;

  // NIP-42 signing path sends through `publish`-like flow; we instead hook
  // the OK correlation by signing inside the AUTH case and remembering id.
  // To keep it simple we re-derive here: signEvent returns the full event.

  close() {
    this.closedByUs = true;
    this.socket?.close();
  }

  /** Publish an already-signed event; resolves on first OK:true. */
  publish(event, { timeoutMs = 8000 } = {}) {
    if (!this.connected) return Promise.reject(new Error(`relay ${this.url}: not connected`));
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(event.id);
        reject(new Error(`relay ${this.url}: publish timeout for ${event.id}`));
      }, timeoutMs);
      this.pending.set(event.id, { resolve, reject, timer });
      this.socket.send(JSON.stringify(["EVENT", event]));
    });
  }

  /** Subscribe; returns a close function. */
  subscribe(filter, { onEvent, onEose } = {}) {
    const subId = randomUUID().slice(0, 8);
    this.subs.set(subId, { filter, onEvent, onEose });
    if (this.connected) {
      this.socket.send(JSON.stringify(["REQ", subId, filter]));
    }
    return () => {
      this.subs.delete(subId);
      if (this.connected) this.socket.send(JSON.stringify(["CLOSE", subId]));
    };
  }
}

export class RelayPool extends EventEmitter {
  /**
   * @param {string[]} urls
   * @param {string} secretHex
   */
  constructor(urls, secretHex) {
    super();
    this.connections = urls.map((u) => {
      const c = new RelayConnection(u, secretHex);
      c.on("connected", () => this.emit("connected", u));
      c.on("disconnected", () => this.emit("disconnected", u));
      c.on("notice", (n) => this.emit("notice", u, n));
      return c;
    });
    this.secretHex = secretHex;
  }

  connect() {
    for (const c of this.connections) c.connect();
  }

  close() {
    for (const c of this.connections) c.close();
  }

  /** Publish to all relays; resolves when the first relay OKs. */
  publish(event, opts) {
    return Promise.any(this.connections.map((c) => c.publish(event, opts)));
  }

  subscribe(filter, handlers) {
    const closers = this.connections.map((c) => c.subscribe(filter, handlers));
    return () => closers.forEach((close) => close());
  }

  get urls() {
    return this.connections.map((c) => c.url);
  }
}
