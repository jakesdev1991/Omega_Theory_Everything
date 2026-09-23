/* Amity World Citizen Wallet — real wallet core.
 * ethers v6 (vendored UMD) + browser-native crypto.
 *
 * What this actually does:
 * - Generates a REAL BIP-39 mnemonic (12 words) + BIP-32/BIP-44 keypair.
 * - Derives a REAL 0x... Ethereum address (m/44'/60'/0'/0/0).
 * - Stores an encrypted (scrypt) V3 keystore in localStorage, password-locked.
 * - Connects MetaMask (EIP-1193) as an alternative to the local key.
 * - Signs EIP-191 personal_sign messages to produce auditable proofs.
 * - Fetches REAL on-chain ETH balance from public RPCs with failover.
 */

/* global ethers */

const AMITY_KEYSTORE = "amity.keystore.v1";
const AMITY_ACTIVE = "amity.active.v1";
const AMITY_ACCOUNTS = "amity.accounts.v1";
const AMITY_PROOFS = "amity.proofs.v1";

const RPC_URLS = [
  "https://eth.drpc.org",
  "https://ethereum-rpc.publicnode.com",
  "https://rpc.ankr.com/eth",
  "https://1rpc.io/eth",
  "https://cloudflare-eth.com",
];

/* ------------------------------------------------------------------ */
/* Storage helpers (fail soft when storage is unavailable)             */
/* ------------------------------------------------------------------ */

function storeGet(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function storeSet(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

function storeDel(key) {
  try { localStorage.removeItem(key); } catch { /* noop */ }
}

/* ------------------------------------------------------------------ */
/* Wallet creation / import                                            */
/* ------------------------------------------------------------------ */

function strongRandom(bytes) {
  const buf = new Uint8Array(bytes);
  if (window.crypto && window.crypto.getRandomValues) {
    window.crypto.getRandomValues(buf);
  } else {
    for (let i = 0; i < bytes; i++) buf[i] = (Math.random() * 256) | 0;
  }
  return buf;
}

function createLocalWallet() {
  const mnemonic = ethers.Mnemonic.fromEntropy(strongRandom(16)); // 12 words
  const node = ethers.HDNodeWallet.fromPhrase(mnemonic.phrase);
  return {
    provider: "local",
    address: node.address,
    mnemonic: mnemonic.phrase,
    privateKey: node.privateKey,
    createdAt: Date.now(),
  };
}

function importMnemonic(phrase) {
  const mnemonic = ethers.Mnemonic.fromPhrase(phrase.trim());
  const node = ethers.HDNodeWallet.fromPhrase(mnemonic.phrase);
  return {
    provider: "local",
    address: node.address,
    mnemonic: mnemonic.phrase,
    privateKey: node.privateKey,
    createdAt: Date.now(),
  };
}

function importPrivateKey(privateKey) {
  const key = privateKey.trim().startsWith("0x")
    ? privateKey.trim()
    : "0x" + privateKey.trim();
  const wallet = new ethers.Wallet(key);
  return {
    keyonly: true,
    provider: "local",
    address: wallet.address,
    privateKey: key,
    createdAt: Date.now(),
  };
}

/* ------------------------------------------------------------------ */
/* Encrypted keystore (scrypt, V3 JSON)                                */
/* ------------------------------------------------------------------ */

async function encryptAndSave(account, password, label) {
  const sourceWallet = account.keyonly
    ? new ethers.Wallet(account.privateKey)
    : ethers.HDNodeWallet.fromPhrase(account.mnemonic);

  const keystore = await sourceWallet.encrypt(password, undefined, {
    scrypt: { N: 1 << 14, r: 8, p: 1 },
  });

  const record = {
    version: 1,
    label: label || "Amity Wallet",
    address: sourceWallet.address,
    keystore,
    createdAt: Date.now(),
  };

  const accounts = storeGet(AMITY_ACCOUNTS, []).filter(
    (a) => a.address !== record.address
  );
  accounts.push({
    label: record.label,
    address: record.address,
    createdAt: record.createdAt,
  });
  storeSet(AMITY_ACCOUNTS, accounts);
  storeSet(AMITY_KEYSTORE, record);
  storeSet(AMITY_ACTIVE, record.address);
  return record;
}

async function decryptKeystore(password) {
  const record = storeGet(AMITY_KEYSTORE, null);
  if (!record) {
    throw new Error("No keystore found. Create or import a wallet first.");
  }
  const wallet = await ethers.Wallet.fromEncryptedJson(record.keystore, password);
  return { wallet, address: record.address, label: record.label };
}

function loadActiveAccount() {
  const address = storeGet(AMITY_ACTIVE, null);
  if (!address) return null;
  const accounts = storeGet(AMITY_ACCOUNTS, []);
  const record = storeGet(AMITY_KEYSTORE, null);
  const entry = accounts.find(
    (a) => String(a.address).toLowerCase() === String(address).toLowerCase()
  );
  return {
    address,
    label: (entry && entry.label) || (record && record.label) || "Amity Wallet",
    hasKeystore: !!(record && String(record.address).toLowerCase() === String(address).toLowerCase()),
  };
}

function listAccounts() {
  return storeGet(AMITY_ACCOUNTS, []);
}

function forgetWallet() {
  storeDel(AMITY_KEYSTORE);
  storeDel(AMITY_ACTIVE);
}

/* ------------------------------------------------------------------ */
/* MetaMask / EIP-1193                                                 */
/* ------------------------------------------------------------------ */

function hasMetaMask() {
  return (
    typeof window.ethereum !== "undefined" &&
    typeof window.ethereum.request === "function"
  );
}

async function connectMetaMask() {
  if (!hasMetaMask()) throw new Error("MetaMask is not installed.");
  const accounts = await window.ethereum.request({ method: "eth_requestAccounts" });
  if (!accounts || !accounts.length) throw new Error("No accounts authorized.");
  storeSet(AMITY_ACTIVE, accounts[0]);
  return accounts[0];
}

function hexMessage(msg) {
  const bytes = new TextEncoder().encode(msg);
  let hex = "0x";
  for (const b of bytes) hex += b.toString(16).padStart(2, "0");
  return hex;
}

/* ------------------------------------------------------------------ */
/* Balances (real RPC, multi-provider failover)                        */
/* ------------------------------------------------------------------ */

async function fetchEthBalance(address) {
  let lastErr;
  for (const url of RPC_URLS) {
    try {
      const provider = new ethers.JsonRpcProvider(url);
      const balance = await provider.getBalance(address);
      const network = await provider.getNetwork();
      return {
        wei: balance.toString(),
        eth: ethers.formatEther(balance),
        chainId: Number(network.chainId),
      };
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || new Error("All RPC endpoints failed.");
}

/* ------------------------------------------------------------------ */
/* Unlock proofs (signed statements of participation)                  */
/* ------------------------------------------------------------------ */

function buildUnlockMessage(address, tier) {
  return [
    "OMEGA TRI-TOKEN ECONOMY — RELEASE-DAY NOVEL UNLOCK",
    "",
    "I attest that I participate in the Omega tri-token economy.",
    "Address: " + address,
    "Tier: " + tier,
    "Timestamp: " + new Date().toISOString(),
    "",
    "Signature grants access to Genesis Block: The Satoshi Protocol.",
  ].join("\n");
}

async function signUnlockWithLocal(password, tier) {
  const { wallet, address } = await decryptKeystore(password);
  const message = buildUnlockMessage(address, tier);
  const signature = await wallet.signMessage(message);
  return { signer: wallet.address, signature, message };
}

async function signUnlockWithMetaMask(tier) {
  const accounts = await window.ethereum.request({ method: "eth_accounts" });
  if (!accounts || !accounts.length) {
    throw new Error("No MetaMask account connected. Connect first.");
  }
  const address = accounts[0];
  const message = buildUnlockMessage(address, tier);
  const signature = await window.ethereum.request({
    method: "personal_sign",
    params: [hexMessage(message), address],
  });
  return { signer: address, signature, message };
}

function saveProof(proof) {
  const proofs = storeGet(AMITY_PROOFS, []);
  proofs.push({ ...proof, savedAt: Date.now() });
  storeSet(AMITY_PROOFS, proofs);
}

function listProofs() {
  const address = storeGet(AMITY_ACTIVE, null);
  const proofs = storeGet(AMITY_PROOFS, []);
  if (!address) return [];
  return proofs.filter(
    (p) =>
      p.signer &&
      String(p.signer).toLowerCase() === String(address).toLowerCase()
  );
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

const AmityWallet = {
  createLocalWallet,
  importMnemonic,
  importPrivateKey,
  encryptAndSave,
  decryptKeystore,
  loadActiveAccount,
  listAccounts,
  forgetWallet,
  hasMetaMask,
  connectMetaMask,
  fetchEthBalance,
  buildUnlockMessage,
  signUnlockWithLocal,
  signUnlockWithMetaMask,
  saveProof,
  listProofs,
  storage: { get: storeGet, set: storeSet, del: storeDel },
  RPC_URLS,
};

window.AmityWallet = AmityWallet;
