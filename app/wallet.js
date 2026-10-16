/* Omega release wallet prototype — real wallet core.
 * ethers v6 (vendored UMD) + browser-native crypto.
 *
 * What this actually does:
 * - Generates a REAL BIP-39 mnemonic (12 words) + BIP-32/BIP-44 EVM keypair.
 * - Derives a REAL 0x... EVM address (m/44'/60'/0'/0/0).
 * - Stores an encrypted (scrypt) V3 keystore in localStorage, password-locked.
 * - Connects MetaMask (EIP-1193) for the $OMEGA EVM rail.
 * - Connects Phantom for the TWC Solana rail.
 * - Signs optional novel-reader proof statements for the public reader demo.
 * - Fetches REAL on-chain ETH balance from public RPCs with failover.
 *
 * Current chain mapping:
 * - $OMEGA -> EVM rail (current pilot: Ethereum Sepolia)
 * - TWC    -> Solana rail (current pilot: Devnet)
 * - AMITY  -> separate Bitcoin / Lightning / Taproot workstream, not wired here
 */

/* global ethers */

const STORAGE_MISSING = Symbol("storage-missing");

const STORAGE_KEYS = Object.freeze({
  evmKeystore: "omega.wallet.evm.keystore.v2",
  evmActiveAddress: "omega.wallet.evm.active.v2",
  evmAccounts: "omega.wallet.evm.accounts.v2",
  unlockProofs: "omega.wallet.unlock.proofs.v2",
  solanaActiveWallet: "omega.wallet.solana.active.v2",
});

const LEGACY_STORAGE_KEYS = Object.freeze({
  evmKeystore: "amity.keystore.v1",
  evmActiveAddress: "amity.active.v1",
  evmAccounts: "amity.accounts.v1",
  unlockProofs: "amity.proofs.v1",
  solanaActiveWallet: "amity.solana.active.v1",
});

const DEFAULT_EVM_WALLET_LABEL = "Omega Wallet";
const OMEGA_UNLOCK_NETWORK = "ethereum-sepolia";
const TWC_UNLOCK_NETWORK = "solana-devnet";

const EVM_RPC_URLS = [
  "https://eth.drpc.org",
  "https://ethereum-rpc.publicnode.com",
  "https://rpc.ankr.com/eth",
  "https://1rpc.io/eth",
  "https://cloudflare-eth.com",
];

/* ------------------------------------------------------------------ */
/* Storage helpers (fail soft when storage is unavailable)             */
/* ------------------------------------------------------------------ */

function readStorage(key, fallback = STORAGE_MISSING) {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function writeStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

function deleteStorage(key) {
  try {
    localStorage.removeItem(key);
    return true;
  } catch {
    return false;
  }
}

function getStoredValue(primaryKey, legacyKey, fallback) {
  const primary = readStorage(primaryKey, STORAGE_MISSING);
  if (primary !== STORAGE_MISSING) {
    return primary;
  }

  if (!legacyKey) {
    return fallback;
  }

  const legacy = readStorage(legacyKey, STORAGE_MISSING);
  return legacy === STORAGE_MISSING ? fallback : legacy;
}

function setStoredValue(primaryKey, value) {
  return writeStorage(primaryKey, value);
}

function deleteStoredValue(primaryKey, legacyKey) {
  let removed = deleteStorage(primaryKey);
  if (legacyKey) {
    removed = deleteStorage(legacyKey) && removed;
  }
  return removed;
}

function migrateLegacyStorageKey(primaryKey, legacyKey) {
  const primary = readStorage(primaryKey, STORAGE_MISSING);
  if (primary !== STORAGE_MISSING) {
    if (legacyKey && legacyKey !== primaryKey) {
      deleteStorage(legacyKey);
    }
    return;
  }

  const legacy = readStorage(legacyKey, STORAGE_MISSING);
  if (legacy === STORAGE_MISSING) {
    return;
  }

  if (writeStorage(primaryKey, legacy)) {
    deleteStorage(legacyKey);
  }
}

function clearStagedProof() {
  try {
    window.sessionStorage?.removeItem("omega.unlock.proof");
    return true;
  } catch {
    return false;
  }
}

function clearStoredProofs() {
  const persistentProofsCleared = deleteStoredValue(
    STORAGE_KEYS.unlockProofs,
    LEGACY_STORAGE_KEYS.unlockProofs,
  );
  return clearStagedProof() && persistentProofsCleared;
}

function migrateLegacyStorage() {
  migrateLegacyStorageKey(STORAGE_KEYS.evmKeystore, LEGACY_STORAGE_KEYS.evmKeystore);
  migrateLegacyStorageKey(STORAGE_KEYS.evmActiveAddress, LEGACY_STORAGE_KEYS.evmActiveAddress);
  migrateLegacyStorageKey(STORAGE_KEYS.evmAccounts, LEGACY_STORAGE_KEYS.evmAccounts);
  // Proofs contain a public address, signature, origin, and timestamp. Keep
  // them transient: remove old persisted histories instead of migrating them.
  clearStoredProofs();
  migrateLegacyStorageKey(STORAGE_KEYS.solanaActiveWallet, LEGACY_STORAGE_KEYS.solanaActiveWallet);
}

/* ------------------------------------------------------------------ */
/* Wallet creation / import                                            */
/* ------------------------------------------------------------------ */

function strongRandom(bytes) {
  if (!window.crypto || typeof window.crypto.getRandomValues !== "function") {
    throw new Error("Secure randomness is unavailable. Open this wallet in a secure browser context and try again.");
  }

  const buffer = new Uint8Array(bytes);
  try {
    window.crypto.getRandomValues(buffer);
  } catch {
    throw new Error("Secure random generation failed. No wallet was created; try a current browser over HTTPS.");
  }
  return buffer;
}

function createLocalWallet() {
  const mnemonic = ethers.Mnemonic.fromEntropy(strongRandom(16));
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
  const normalized = privateKey.trim().startsWith("0x")
    ? privateKey.trim()
    : `0x${privateKey.trim()}`;
  const wallet = new ethers.Wallet(normalized);
  return {
    keyonly: true,
    provider: "local",
    address: wallet.address,
    privateKey: normalized,
    createdAt: Date.now(),
  };
}

/* ------------------------------------------------------------------ */
/* Encrypted keystore (scrypt, V3 JSON)                                */
/* ------------------------------------------------------------------ */

function persistWalletRecords(entries) {
  let previousValues;
  try {
    previousValues = entries.map(([key]) => localStorage.getItem(key));
  } catch {
    throw new Error("Browser storage is unavailable. The encrypted wallet was not saved.");
  }

  const writtenKeys = [];
  try {
    entries.forEach(([key, value]) => {
      localStorage.setItem(key, JSON.stringify(value));
      writtenKeys.push(key);
    });
  } catch {
    // Best-effort rollback prevents a partial keystore/account update from
    // appearing successful if a quota or browser-storage error interrupts it.
    for (let index = writtenKeys.length - 1; index >= 0; index -= 1) {
      const key = writtenKeys[index];
      const previous = previousValues[index];
      try {
        if (previous === null) localStorage.removeItem(key);
        else localStorage.setItem(key, previous);
      } catch {
        /* Report failure below; rollback is best-effort when storage is full. */
      }
    }
    throw new Error("The encrypted wallet could not be saved to browser storage. Check available storage and retry.");
  }
}

async function encryptAndSave(account, password, label) {
  const sourceWallet = account.keyonly
    ? new ethers.Wallet(account.privateKey)
    : ethers.HDNodeWallet.fromPhrase(account.mnemonic);

  // Use ethers' default V3 scrypt work factor (N=131072, r=8, p=1).
  const keystore = await sourceWallet.encrypt(password, undefined, {
    scrypt: { N: 1 << 17, r: 8, p: 1 },
  });

  const record = {
    version: 1,
    label: label || DEFAULT_EVM_WALLET_LABEL,
    address: sourceWallet.address,
    keystore,
    createdAt: Date.now(),
  };

  const storedAccounts = getStoredValue(
    STORAGE_KEYS.evmAccounts,
    LEGACY_STORAGE_KEYS.evmAccounts,
    [],
  );
  if (!Array.isArray(storedAccounts)) {
    throw new Error("Stored wallet metadata is invalid. Forget the wallet and import it again.");
  }
  const accounts = storedAccounts.filter(
    (entry) => String(entry?.address ?? "").toLowerCase() !== record.address.toLowerCase(),
  );

  accounts.push({
    label: record.label,
    address: record.address,
    createdAt: record.createdAt,
  });

  persistWalletRecords([
    [STORAGE_KEYS.evmKeystore, record],
    [STORAGE_KEYS.evmAccounts, accounts],
    [STORAGE_KEYS.evmActiveAddress, record.address],
  ]);
  return record;
}

async function decryptKeystore(password) {
  const record = getStoredValue(
    STORAGE_KEYS.evmKeystore,
    LEGACY_STORAGE_KEYS.evmKeystore,
    null,
  );
  if (!record) {
    throw new Error("No keystore found. Create or import a wallet first.");
  }

  const wallet = await ethers.Wallet.fromEncryptedJson(record.keystore, password);
  return { wallet, address: record.address, label: record.label };
}

function loadActiveAccount() {
  const address = getStoredValue(
    STORAGE_KEYS.evmActiveAddress,
    LEGACY_STORAGE_KEYS.evmActiveAddress,
    null,
  );
  if (!address) {
    return null;
  }

  const accounts = getStoredValue(
    STORAGE_KEYS.evmAccounts,
    LEGACY_STORAGE_KEYS.evmAccounts,
    [],
  );
  const record = getStoredValue(
    STORAGE_KEYS.evmKeystore,
    LEGACY_STORAGE_KEYS.evmKeystore,
    null,
  );
  const entry = accounts.find(
    (item) => String(item.address).toLowerCase() === String(address).toLowerCase(),
  );

  return {
    address,
    label: (entry && entry.label) || (record && record.label) || DEFAULT_EVM_WALLET_LABEL,
    hasKeystore: !!(
      record &&
      String(record.address).toLowerCase() === String(address).toLowerCase()
    ),
  };
}

function listAccounts() {
  return getStoredValue(
    STORAGE_KEYS.evmAccounts,
    LEGACY_STORAGE_KEYS.evmAccounts,
    [],
  );
}

function forgetWallet() {
  const removals = [
    deleteStoredValue(STORAGE_KEYS.evmKeystore, LEGACY_STORAGE_KEYS.evmKeystore),
    deleteStoredValue(STORAGE_KEYS.evmActiveAddress, LEGACY_STORAGE_KEYS.evmActiveAddress),
    deleteStoredValue(STORAGE_KEYS.evmAccounts, LEGACY_STORAGE_KEYS.evmAccounts),
    deleteStoredValue(STORAGE_KEYS.solanaActiveWallet, LEGACY_STORAGE_KEYS.solanaActiveWallet),
    clearStoredProofs(),
  ];
  return removals.every(Boolean);
}

/* ------------------------------------------------------------------ */
/* Browser wallets                                                     */
/* ------------------------------------------------------------------ */

function hasMetaMask() {
  return (
    typeof window.ethereum !== "undefined" &&
    typeof window.ethereum.request === "function"
  );
}

async function connectMetaMask() {
  if (!hasMetaMask()) {
    throw new Error("MetaMask is not installed.");
  }

  const accounts = await window.ethereum.request({ method: "eth_requestAccounts" });
  if (!accounts || !accounts.length) {
    throw new Error("No accounts authorized.");
  }

  if (!setStoredValue(STORAGE_KEYS.evmActiveAddress, accounts[0])) {
    throw new Error("MetaMask connected, but the account could not be saved in browser storage. Check privacy or storage settings and reconnect.");
  }
  return accounts[0];
}

function hasPhantom() {
  return !!(window.solana && window.solana.isPhantom);
}

function saveConnectedPhantomWallet(address) {
  if (!setStoredValue(STORAGE_KEYS.solanaActiveWallet, {
    provider: "phantom",
    address,
    label: "Phantom",
    connectedAt: Date.now(),
  })) {
    throw new Error("Phantom connected, but the account could not be saved in browser storage. Check privacy or storage settings and reconnect.");
  }
}

async function connectPhantom() {
  if (!hasPhantom()) {
    throw new Error("Phantom is not installed.");
  }

  const response = await window.solana.connect();
  const address = response.publicKey.toString();
  saveConnectedPhantomWallet(address);
  return address;
}

function loadSolanaAccount() {
  const record = getStoredValue(
    STORAGE_KEYS.solanaActiveWallet,
    LEGACY_STORAGE_KEYS.solanaActiveWallet,
    null,
  );
  if (!record || !record.address) {
    return null;
  }
  return record;
}

/* ------------------------------------------------------------------ */
/* Utility helpers                                                     */
/* ------------------------------------------------------------------ */

function hexMessage(message) {
  const bytes = new TextEncoder().encode(message);
  let hex = "0x";
  for (const byte of bytes) {
    hex += byte.toString(16).padStart(2, "0");
  }
  return hex;
}

function bytesToBase64(bytes) {
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

/* ------------------------------------------------------------------ */
/* Balances (real RPC, multi-provider failover)                        */
/* ------------------------------------------------------------------ */

async function fetchEthBalance(address) {
  let lastError;

  for (const url of EVM_RPC_URLS) {
    try {
      const provider = new ethers.JsonRpcProvider(url);
      const balance = await provider.getBalance(address);
      const network = await provider.getNetwork();
      return {
        wei: balance.toString(),
        eth: ethers.formatEther(balance),
        chainId: Number(network.chainId),
      };
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error("All RPC endpoints failed.");
}

/* ------------------------------------------------------------------ */
/* Optional novel-reader proofs (signed statements of participation)  */
/* ------------------------------------------------------------------ */

function buildUnlockMessage(currency, address, network, origin) {
  return [
    "OMEGA OPTIONAL NOVEL READER PROOF",
    `Currency: ${currency}`,
    `Address: ${address}`,
    `Network: ${network}`,
    `Origin: ${origin}`,
    `Timestamp: ${new Date().toISOString()}`,
    "Reader demo: full chapter view",
  ].join("\n");
}

async function signUnlockWithLocal(password) {
  const { wallet, address } = await decryptKeystore(password);
  const message = buildUnlockMessage(
    "OMEGA",
    address,
    OMEGA_UNLOCK_NETWORK,
    window.location.origin || "local-app",
  );
  const signature = await wallet.signMessage(message);
  return {
    signer: wallet.address,
    currency: "OMEGA",
    network: OMEGA_UNLOCK_NETWORK,
    signature,
    message,
  };
}

async function signUnlockWithMetaMask() {
  const accounts = await window.ethereum.request({ method: "eth_accounts" });
  if (!accounts || !accounts.length) {
    throw new Error("No MetaMask account connected. Connect first.");
  }

  const chainId = await window.ethereum.request({ method: "eth_chainId" });
  if (String(chainId).toLowerCase() !== "0xaa36a7") {
    throw new Error("Switch MetaMask to Ethereum Sepolia (chain ID 11155111) before signing this optional proof. No transaction will be sent.");
  }

  const address = accounts[0];
  const message = buildUnlockMessage(
    "OMEGA",
    address,
    OMEGA_UNLOCK_NETWORK,
    window.location.origin || "local-app",
  );
  const signature = await window.ethereum.request({
    method: "personal_sign",
    params: [hexMessage(message), address],
  });

  return {
    signer: address,
    currency: "OMEGA",
    network: OMEGA_UNLOCK_NETWORK,
    signature,
    message,
  };
}

async function signUnlockWithPhantom() {
  if (!hasPhantom()) {
    throw new Error("Phantom is not installed.");
  }

  const response = await window.solana.connect();
  const address = response.publicKey.toString();
  saveConnectedPhantomWallet(address);

  const message = buildUnlockMessage(
    "TWC",
    address,
    TWC_UNLOCK_NETWORK,
    window.location.origin || "local-app",
  );
  const encoded = new TextEncoder().encode(message);
  const signed = await window.solana.signMessage(encoded, "utf8");

  return {
    signer: address,
    currency: "TWC",
    network: TWC_UNLOCK_NETWORK,
    signature: bytesToBase64(signed.signature),
    message,
  };
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

migrateLegacyStorage();

const OmegaWallet = {
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
  hasPhantom,
  connectPhantom,
  loadSolanaAccount,
  fetchEthBalance,
  buildUnlockMessage,
  signUnlockWithLocal,
  signUnlockWithMetaMask,
  signUnlockWithPhantom,
  migrateLegacyStorage,
  storage: {
    get: getStoredValue,
    set: setStoredValue,
    del: deleteStoredValue,
    keys: STORAGE_KEYS,
    legacyKeys: LEGACY_STORAGE_KEYS,
  },
  EVM_RPC_URLS,
};

window.OmegaWallet = OmegaWallet;
window.AmityWallet = OmegaWallet;
