// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const { webcrypto } = require("node:crypto");
const ethers = require("./vendor/ethers.umd.min.js");

const WALLET_SOURCE = fs.readFileSync(path.join(__dirname, "wallet.js"), "utf8");
const APP_SOURCE = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");
const HTML_SOURCE = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");

class MemoryStorage {
  constructor(initial = {}) {
    this.values = new Map(Object.entries(initial));
    this.failWriteFor = null;
    this.failRemoveFor = null;
  }

  getItem(key) {
    return this.values.has(key) ? this.values.get(key) : null;
  }

  setItem(key, value) {
    if (key === this.failWriteFor) throw new Error("quota exceeded");
    this.values.set(key, String(value));
  }

  removeItem(key) {
    if (key === this.failRemoveFor) throw new Error("storage unavailable");
    this.values.delete(key);
  }
}

class FakeClassList {
  constructor(...initial) {
    this.classes = new Set(initial);
  }

  add(...names) {
    names.forEach((name) => this.classes.add(name));
  }

  remove(...names) {
    names.forEach((name) => this.classes.delete(name));
  }

  contains(name) {
    return this.classes.has(name);
  }

  toggle(name, force) {
    const enabled = force ?? !this.classes.has(name);
    if (enabled) this.classes.add(name);
    else this.classes.delete(name);
    return enabled;
  }
}

class FakeElement {
  constructor(id, initialClasses = []) {
    this.id = id;
    this.value = "";
    this.textContent = "";
    this.checked = false;
    this.dataset = {};
    this.classList = new FakeClassList(...initialClasses);
    this.listeners = new Map();
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  async dispatch(type, event = {}) {
    for (const listener of this.listeners.get(type) ?? []) {
      await listener(event);
    }
  }

  select() {}
}

function loadApp({ walletOverrides = {} } = {}) {
  const ids = new Set([...HTML_SOURCE.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));
  const elements = new Map(
    [...ids].map((id) => {
      const tag = HTML_SOURCE.match(new RegExp(`<[^>]*\\bid="${id}"[^>]*>`))?.[0] ?? "";
      const classes = tag.match(/\bclass="([^"]*)"/)?.[1].split(/\s+/).filter(Boolean) ?? [];
      return [id, new FakeElement(id, classes)];
    }),
  );
  const wallet = {
    loadActiveAccount: () => null,
    loadSolanaAccount: () => null,
    hasMetaMask: () => false,
    hasPhantom: () => false,
    ...walletOverrides,
  };
  const window = {
    OmegaWallet: wallet,
    location: { origin: "https://wallet.example", href: "https://wallet.example/" },
    sessionStorage: new MemoryStorage(),
    scrollTo() {},
  };
  const document = {
    getElementById: (id) => elements.get(id) ?? null,
    querySelectorAll: () => [],
  };
  const context = vm.createContext({
    window,
    document,
    console: { info() {} },
    setTimeout: () => 1,
    clearTimeout() {},
  });
  vm.runInContext(APP_SOURCE, context, { filename: "app.js" });
  return { context, elements, wallet };
}

function loadWallet({ storage = new MemoryStorage(), sessionStorage = new MemoryStorage(), crypto = webcrypto, ethereum = null } = {}) {
  const window = {
    crypto,
    localStorage: storage,
    sessionStorage,
    location: { origin: "https://wallet.example" },
    ethereum,
    solana: null,
  };
  const context = vm.createContext({
    window,
    localStorage: storage,
    sessionStorage,
    ethers,
    TextEncoder,
    btoa: (value) => Buffer.from(value, "binary").toString("base64"),
  });
  vm.runInContext(WALLET_SOURCE, context, { filename: "wallet.js" });
  return { wallet: window.OmegaWallet, storage, sessionStorage, window };
}

function stubEncryptionEthers(capture) {
  class StubWallet {
    constructor(privateKey) {
      this.privateKey = privateKey;
      this.address = "0x1111111111111111111111111111111111111111";
    }

    async encrypt(password, progress, options) {
      capture.password = password;
      capture.options = options;
      return JSON.stringify({ version: 3, crypto: { kdf: "scrypt", kdfparams: options.scrypt } });
    }

    static async fromEncryptedJson() {
      return new StubWallet("0x" + "1".repeat(64));
    }
  }
  return { ...ethers, Wallet: StubWallet };
}

test("wallet key generation fails closed when Web Crypto entropy is unavailable", () => {
  const { wallet } = loadWallet({ crypto: null });
  assert.throws(() => wallet.createLocalWallet(), /Secure randomness is unavailable/);
});

test("wallet keystore uses ethers' default-strength scrypt parameters", async () => {
  const capture = {};
  const storage = new MemoryStorage();
  // Replace only the constructor used by encryption to avoid spending CI time
  // running scrypt; the real library's requested work-factor options are tested.
  const source = WALLET_SOURCE.replaceAll("ethers.Wallet", "testEthers.Wallet").replaceAll("ethers.HDNodeWallet", "testEthers.HDNodeWallet");
  const testEthers = stubEncryptionEthers(capture);
  const window = { crypto: webcrypto, localStorage: storage, sessionStorage: new MemoryStorage(), location: { origin: "https://wallet.example" } };
  const context = vm.createContext({ window, localStorage: storage, ethers, testEthers, TextEncoder });
  vm.runInContext(source, context, { filename: "wallet-encryption-test.js" });
  const account = context.window.OmegaWallet.importPrivateKey("0x" + "1".repeat(64));
  await context.window.OmegaWallet.encryptAndSave(account, "a strong test password", "Test wallet");

  assert.equal(capture.password, "a strong test password");
  assert.equal(capture.options.scrypt.N, 131072);
  assert.equal(capture.options.scrypt.r, 8);
  assert.equal(capture.options.scrypt.p, 1);
  const saved = JSON.parse(storage.getItem("omega.wallet.evm.keystore.v2"));
  assert.equal(saved.address, account.address);
  assert.equal(saved.label, "Test wallet");
  assert.equal(storage.getItem("omega.wallet.evm.active.v2"), JSON.stringify(account.address));
});

test("keystore storage failures reject and roll back earlier writes", async () => {
  const capture = {};
  const testEthers = stubEncryptionEthers(capture);
  const storage = new MemoryStorage({
    "omega.wallet.evm.keystore.v2": JSON.stringify({ address: "0xold", keystore: "old" }),
    "omega.wallet.evm.accounts.v2": JSON.stringify([{ address: "0xold", label: "Old" }]),
    "omega.wallet.evm.active.v2": JSON.stringify("0xold"),
  });
  storage.failWriteFor = "omega.wallet.evm.accounts.v2";
  const window = { crypto: webcrypto, localStorage: storage, sessionStorage: new MemoryStorage(), location: { origin: "https://wallet.example" } };
  const context = vm.createContext({ window, localStorage: storage, ethers, testEthers, TextEncoder });
  const source = WALLET_SOURCE.replaceAll("ethers.Wallet", "testEthers.Wallet").replaceAll("ethers.HDNodeWallet", "testEthers.HDNodeWallet");
  vm.runInContext(source, context, { filename: "wallet-storage-failure-test.js" });
  const wallet = context.window.OmegaWallet;
  const account = wallet.importPrivateKey("0x" + "1".repeat(64));

  await assert.rejects(wallet.encryptAndSave(account, "password", "New wallet"), /could not be saved/);
  assert.equal(storage.getItem("omega.wallet.evm.keystore.v2"), JSON.stringify({ address: "0xold", keystore: "old" }));
  assert.equal(storage.getItem("omega.wallet.evm.accounts.v2"), JSON.stringify([{ address: "0xold", label: "Old" }]));
  assert.equal(storage.getItem("omega.wallet.evm.active.v2"), JSON.stringify("0xold"));
});

test("legacy proof history is discarded instead of migrated into persistent storage", () => {
  const storage = new MemoryStorage({
    "amity.proofs.v1": JSON.stringify([{ signature: "legacy" }]),
    "omega.wallet.unlock.proofs.v2": JSON.stringify([{ signature: "current" }]),
  });
  loadWallet({ storage });
  assert.equal(storage.getItem("amity.proofs.v1"), null);
  assert.equal(storage.getItem("omega.wallet.unlock.proofs.v2"), null);
});

test("legacy keystore data is retained if a migration write fails", () => {
  const legacy = JSON.stringify({ address: "0xold", keystore: "encrypted" });
  const storage = new MemoryStorage({ "amity.keystore.v1": legacy });
  storage.failWriteFor = "omega.wallet.evm.keystore.v2";
  loadWallet({ storage });

  assert.equal(storage.getItem("omega.wallet.evm.keystore.v2"), null);
  assert.equal(storage.getItem("amity.keystore.v1"), legacy);
});

test("forgetWallet removes keystores, account metadata, connected addresses, proofs, and staged proof", () => {
  const storage = new MemoryStorage({
    "omega.wallet.evm.keystore.v2": "{}",
    "amity.keystore.v1": "{}",
    "omega.wallet.evm.active.v2": "\"0x1\"",
    "amity.active.v1": "\"0x1\"",
    "omega.wallet.evm.accounts.v2": "[]",
    "amity.accounts.v1": "[]",
    "omega.wallet.solana.active.v2": "{}",
    "amity.solana.active.v1": "{}",
    "omega.wallet.unlock.proofs.v2": "[]",
    "amity.proofs.v1": "[]",
  });
  const sessionStorage = new MemoryStorage({ "omega.unlock.proof": "staged" });
  const { wallet } = loadWallet({ storage, sessionStorage });
  wallet.forgetWallet();

  assert.equal(storage.values.size, 0);
  assert.equal(sessionStorage.getItem("omega.unlock.proof"), null);
  assert.equal(wallet.loadActiveAccount(), null);
  assert.equal(wallet.loadSolanaAccount(), null);
  assert.equal("saveProof" in wallet, false);
  assert.equal("listProofs" in wallet, false);
  assert.equal(wallet.forgetWallet(), true);
});

test("forgetWallet reports browser-storage deletion failures", () => {
  const storage = new MemoryStorage({
    "omega.wallet.evm.keystore.v2": "{}",
    "omega.wallet.evm.active.v2": "\"0x1\"",
  });
  storage.failRemoveFor = "omega.wallet.evm.keystore.v2";
  const { wallet } = loadWallet({ storage });

  assert.equal(wallet.forgetWallet(), false);
  assert.notEqual(storage.getItem("omega.wallet.evm.keystore.v2"), null);
  assert.equal(storage.getItem("omega.wallet.evm.active.v2"), null);
});

test("MetaMask proof signing refuses to label a non-Sepolia session as Sepolia", async () => {
  const calls = [];
  const ethereum = {
    async request({ method }) {
      calls.push(method);
      if (method === "eth_accounts") return ["0x1111111111111111111111111111111111111111"];
      if (method === "eth_chainId") return "0x1";
      if (method === "personal_sign") return "0xsignature";
      throw new Error(`Unexpected method ${method}`);
    },
  };
  const { wallet } = loadWallet({ ethereum });
  await assert.rejects(wallet.signUnlockWithMetaMask(), /Switch MetaMask to Ethereum Sepolia/);
  assert.deepEqual(calls, ["eth_accounts", "eth_chainId"]);
});

test("the wallet UI masks import secrets and never logs a recovery mnemonic", () => {
  assert.match(HTML_SOURCE, /id="import-secret" type="password"/);
  assert.match(HTML_SOURCE, /id="backup-confirmed" type="checkbox"/);
  assert.match(HTML_SOURCE, /id="unlock-password" type="password"/);
  assert.match(APP_SOURCE, /if \(els\.recoveryPhrase\?\.value && !clearRecoveryPhrase\(\)\) \{[\s\S]*create-password[\s\S]*return;/);
  assert.doesNotMatch(APP_SOURCE, /console\.(?:log|info|debug)\([^\n]*MNEMONIC/i);
  assert.doesNotMatch(APP_SOURCE, /\b(?:prompt|confirm|alert)\(/);
  assert.match(APP_SOURCE, /showRecoveryPhrase\(account\.mnemonic\)/);
});

test("recovery phrase remains visible until backup is confirmed", async () => {
  const { elements } = loadApp();
  const phrase = elements.get("recovery-phrase");
  const panel = elements.get("recovery-panel");
  const confirmation = elements.get("backup-confirmed");
  phrase.value = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12";
  panel.classList.remove("hidden");

  await elements.get("btn-hide-mnemonic").dispatch("click");
  assert.notEqual(phrase.value, "");
  assert.equal(panel.classList.contains("hidden"), false);
  assert.match(elements.get("toast").textContent, /confirm the backup/);

  confirmation.checked = true;
  await elements.get("btn-hide-mnemonic").dispatch("click");
  assert.equal(phrase.value, "");
  assert.equal(panel.classList.contains("hidden"), true);
  assert.equal(confirmation.checked, false);
});

test("creating another wallet is blocked until the prior recovery phrase is confirmed", async () => {
  let createCalls = 0;
  let saveCalls = 0;
  const { elements } = loadApp({
    walletOverrides: {
      createLocalWallet() {
        createCalls += 1;
        return {
          address: "0x1111111111111111111111111111111111111111",
          mnemonic: "new phrase held only in the page",
        };
      },
      async encryptAndSave() {
        saveCalls += 1;
      },
    },
  });
  const phrase = elements.get("recovery-phrase");
  const panel = elements.get("recovery-panel");
  const confirmation = elements.get("backup-confirmed");
  phrase.value = "existing phrase that still needs confirmation";
  panel.classList.remove("hidden");
  elements.get("create-password").value = "a sufficiently long password";
  elements.get("create-password2").value = "a sufficiently long password";

  await elements.get("btn-create").dispatch("click");
  assert.equal(createCalls, 0);
  assert.equal(saveCalls, 0);
  assert.equal(phrase.value, "existing phrase that still needs confirmation");
  assert.equal(elements.get("create-password").value, "");
  assert.equal(elements.get("create-password2").value, "");

  confirmation.checked = true;
  elements.get("create-password").value = "a sufficiently long password";
  elements.get("create-password2").value = "a sufficiently long password";
  await elements.get("btn-create").dispatch("click");
  assert.equal(createCalls, 1);
  assert.equal(saveCalls, 1);
  assert.equal(phrase.value, "new phrase held only in the page");
  assert.equal(panel.classList.contains("hidden"), false);
});

test("wallet creation displays the recovery phrase if encrypted storage fails", async () => {
  const { elements } = loadApp({
    walletOverrides: {
      createLocalWallet: () => ({
        address: "0x2222222222222222222222222222222222222222",
        mnemonic: "recovery phrase survives storage failure",
      }),
      async encryptAndSave() {
        throw new Error("storage unavailable");
      },
    },
  });
  elements.get("create-password").value = "a sufficiently long password";
  elements.get("create-password2").value = "a sufficiently long password";

  await elements.get("btn-create").dispatch("click");
  assert.equal(elements.get("recovery-phrase").value, "recovery phrase survives storage failure");
  assert.equal(elements.get("recovery-panel").classList.contains("hidden"), false);
  assert.match(elements.get("recovery-notice").textContent, /wallet was not saved/i);
  assert.equal(elements.get("create-password").value, "");
  assert.equal(elements.get("create-password2").value, "");
});

test("local proof signing reads and clears the in-page keystore password", async () => {
  let signedWith = null;
  const { elements } = loadApp({
    walletOverrides: {
      loadActiveAccount: () => ({ hasKeystore: true }),
      async signUnlockWithLocal(password) {
        signedWith = password;
        return {
          signer: "0x1111111111111111111111111111111111111111",
          currency: "OMEGA",
          network: "ethereum-sepolia",
          message: "optional proof",
          signature: "signature",
        };
      },
    },
  });
  elements.get("unlock-currency").value = "OMEGA";
  elements.get("unlock-password").value = "keystore secret";

  await elements.get("btn-sign-unlock").dispatch("click");
  assert.equal(signedWith, "keystore secret");
  assert.equal(elements.get("unlock-password").value, "");
  assert.equal(elements.get("proof-result").classList.contains("hidden"), false);
});

test("opening the wallet dashboard does not automatically query public RPCs", () => {
  const refresh = APP_SOURCE.match(/function refreshWalletView\(\) \{[\s\S]*?\n\}/)?.[0] ?? "";
  assert.notEqual(refresh, "");
  assert.doesNotMatch(refresh, /fetchBalance\(\)/);
});
