import "server-only";

import { access, readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { Contract, JsonRpcProvider } from "ethers";

import {
  FULL_NOVEL_UNLOCK_CHAPTERS,
  OMEGA_NETWORK,
  TWC_NETWORK,
  type UnlockProofPayload,
  type VerifiedUnlock,
  verifyUnlockProof,
} from "./unlock";

const DEFAULT_OMEGA_RPC_URL = "https://ethereum-sepolia-rpc.publicnode.com";
const DEFAULT_SOLANA_RPC_URL = "https://api.devnet.solana.com";
const OMEGA_CHAIN_ID = BigInt("11155111");
const SOLANA_DEVNET_GENESIS_HASH = "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG";
const SPL_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA";
const DEFAULT_EVM_MANIFEST_PATH = resolve(process.cwd(), "..", "evm", "deployments", "sepolia.json");
const DEFAULT_SOLANA_MANIFEST_PATH = resolve(process.cwd(), "..", "solana", "deployments", "twc-devnet.json");

const omegaGateAbi = [
  "function omega() view returns (address)",
  "function claimThreshold() view returns (uint256)",
  "function claimStart() view returns (uint64)",
  "function claimEnd() view returns (uint64)",
  "function hasClaimed(address) view returns (bool)",
  "function paused() view returns (bool)",
  "function canClaim(address) view returns (bool)",
];

const erc20Abi = [
  "function balanceOf(address) view returns (uint256)",
  "function symbol() view returns (string)",
];

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

async function readJsonIfExists(path: string) {
  try {
    await access(path);
    return JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    if ((error as NodeJS.ErrnoException)?.code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

function firstString(...values: Array<unknown>) {
  for (const value of values) {
    if (typeof value === "string" && value.trim().length > 0) {
      return value.trim();
    }
  }
  return undefined;
}

function formatTokenAmount(baseUnits: bigint, decimals: number) {
  assert(Number.isInteger(decimals) && decimals >= 0, "Token decimals must be a non-negative integer.");
  const divisor = BigInt(10) ** BigInt(decimals);
  const whole = baseUnits / divisor;
  const fraction = baseUnits % divisor;

  if (fraction === BigInt(0) || decimals === 0) {
    return whole.toString();
  }

  return `${whole.toString()}.${fraction.toString().padStart(decimals, "0").replace(/0+$/, "")}`;
}

async function jsonRpcRequest<T>(rpcUrl: string, method: string, params: unknown[]) {
  const response = await fetch(rpcUrl, {
    method: "POST",
    headers: {
      accept: "application/json",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: method,
      method,
      params,
    }),
    signal: AbortSignal.timeout(15_000),
  });

  if (!response.ok) {
    throw new Error(`Solana RPC (${method}) returned HTTP ${response.status}.`);
  }

  const payload = (await response.json()) as { result?: T; error?: { code: number; message: string } };
  if (payload.error) {
    throw new Error(`Solana RPC (${method}) error ${payload.error.code}: ${payload.error.message}`);
  }
  return payload.result as T;
}

async function inspectOmegaConfig() {
  const manifestPath = process.env.OMEGA_SEPOLIA_MANIFEST_PATH?.trim() || DEFAULT_EVM_MANIFEST_PATH;
  const manifest = await readJsonIfExists(manifestPath);

  const rpcUrl = firstString(process.env.OMEGA_SEPOLIA_RPC_URL, manifest?.rpcUrl, DEFAULT_OMEGA_RPC_URL)!;
  const gateAddress = firstString(process.env.OMEGA_NOVEL_GATE_ADDRESS, manifest?.contracts?.omegaNovelGate);
  const tokenAddress = firstString(process.env.OMEGA_TOKEN_ADDRESS, manifest?.contracts?.omegaTestToken);

  return {
    manifestPath,
    manifestPresent: !!manifest,
    rpcUrl,
    gateAddress,
    tokenAddress,
    configured: !!gateAddress,
  };
}

async function loadOmegaConfig() {
  const config = await inspectOmegaConfig();

  assert(
    config.gateAddress,
    "OMEGA on-chain verification is not configured. Provide evm/deployments/sepolia.json or OMEGA_NOVEL_GATE_ADDRESS.",
  );

  return {
    ...config,
    gateAddress: config.gateAddress,
  };
}

async function inspectTwcConfig() {
  const manifestPath = process.env.TWC_DEVNET_MANIFEST_PATH?.trim() || DEFAULT_SOLANA_MANIFEST_PATH;
  const manifest = await readJsonIfExists(manifestPath);

  const rpcUrl = firstString(process.env.SOLANA_DEVNET_RPC_URL, manifest?.deployment?.rpcUrl, DEFAULT_SOLANA_RPC_URL)!;
  const mintAddress = firstString(process.env.TWC_MINT_ADDRESS, manifest?.mint?.address);
  const decimals = Number(manifest?.mint?.decimals ?? 9);
  const symbol = firstString(manifest?.identity?.onChainSymbol, "tTWC")!;

  return {
    manifestPath,
    manifestPresent: !!manifest,
    rpcUrl,
    mintAddress,
    decimals,
    symbol,
    configured: !!mintAddress,
  };
}

async function loadTwcConfig() {
  const config = await inspectTwcConfig();

  assert(
    config.mintAddress,
    "TWC on-chain verification is not configured. Provide solana/deployments/twc-devnet.json or TWC_MINT_ADDRESS.",
  );

  return {
    ...config,
    mintAddress: config.mintAddress,
  };
}

async function verifyOmegaOnChain(proof: VerifiedUnlock) {
  const config = await loadOmegaConfig();
  const provider = new JsonRpcProvider(config.rpcUrl, OMEGA_CHAIN_ID, { staticNetwork: true });
  const network = await provider.getNetwork();
  assert(network.chainId === OMEGA_CHAIN_ID, `Refusing to continue: RPC chainId ${network.chainId} is not Sepolia (${OMEGA_CHAIN_ID}).`);

  const gate = new Contract(config.gateAddress, omegaGateAbi, provider);
  const tokenAddress = config.tokenAddress ?? (await gate.omega()) as string;
  const token = new Contract(tokenAddress, erc20Abi, provider);

  const [claimThreshold, claimStart, claimEnd, hasClaimed, gatePaused, canClaim, balance, symbol, latestBlock] = await Promise.all([
    gate.claimThreshold() as Promise<bigint>,
    gate.claimStart() as Promise<bigint>,
    gate.claimEnd() as Promise<bigint>,
    gate.hasClaimed(proof.address) as Promise<boolean>,
    gate.paused() as Promise<boolean>,
    gate.canClaim(proof.address) as Promise<boolean>,
    token.balanceOf(proof.address) as Promise<bigint>,
    token.symbol().catch(() => "tOMEGA") as Promise<string>,
    provider.getBlock("latest"),
  ]);

  const blockTimestamp = BigInt(latestBlock?.timestamp ?? Math.floor(Date.now() / 1000));
  const withinWindow = blockTimestamp >= claimStart && blockTimestamp <= claimEnd;
  const balanceQualified = balance >= claimThreshold;
  const eligible = hasClaimed || canClaim;

  assert(
    eligible,
    `${proof.address} does not currently satisfy the on-chain $OMEGA gate. It needs either an existing claim receipt or a claimable Sepolia ${symbol} balance at/above the threshold.`,
  );

  const status = hasClaimed ? "claimed" : "claimable";

  return {
    ...proof,
    currency: "OMEGA" as const,
    network: OMEGA_NETWORK,
    unlocked: FULL_NOVEL_UNLOCK_CHAPTERS,
    message: `Verified $OMEGA proof for ${proof.address}. On-chain status: ${status}.`,
    chain: {
      rail: "ethereum-sepolia",
      rpcUrl: config.rpcUrl,
      tokenAddress,
      gateAddress: config.gateAddress,
      tokenSymbol: symbol,
      balanceBaseUnits: balance.toString(),
      balanceTokens: formatTokenAmount(balance, 18),
      claimThresholdBaseUnits: claimThreshold.toString(),
      claimThresholdTokens: formatTokenAmount(claimThreshold, 18),
      claimStart: Number(claimStart),
      claimEnd: Number(claimEnd),
      hasClaimed,
      canClaim,
      gatePaused,
      withinWindow,
      balanceQualified,
    },
  };
}

async function verifyTwcOnChain(proof: VerifiedUnlock) {
  const config = await loadTwcConfig();

  const genesisHash = await jsonRpcRequest<string>(config.rpcUrl, "getGenesisHash", []);
  assert(
    genesisHash === SOLANA_DEVNET_GENESIS_HASH,
    `Refusing to continue: RPC genesis hash ${genesisHash} is not the Solana Devnet genesis hash ${SOLANA_DEVNET_GENESIS_HASH}.`,
  );

  const result = await jsonRpcRequest<{ value?: Array<{ account?: { owner?: string; data?: { parsed?: { type?: string; info?: { owner?: string; mint?: string; tokenAmount?: { amount?: string; decimals?: number } } } } } }> }>(
    config.rpcUrl,
    "getTokenAccountsByOwner",
    [
      proof.address,
      { mint: config.mintAddress },
      { encoding: "jsonParsed", commitment: "confirmed" },
    ],
  );

  const accounts = Array.isArray(result?.value) ? result.value : [];
  let baseUnits = BigInt(0);
  let decimals = config.decimals;

  for (const account of accounts) {
    assert(account?.account?.owner === SPL_TOKEN_PROGRAM, "RPC returned a non-standard SPL token account.");
    const parsed = account.account?.data?.parsed;
    assert(parsed?.type === "account", "RPC returned an unexpected token-account payload.");

    const info = parsed.info ?? {};
    assert(info.owner === proof.address, "RPC returned a token account owned by a different address.");
    assert(info.mint === config.mintAddress, "RPC returned a token account for a different mint.");

    const tokenAmount = info.tokenAmount ?? {};
    assert(typeof tokenAmount.amount === "string" && /^[0-9]+$/.test(tokenAmount.amount), "RPC returned an invalid token amount.");
    const parsedDecimals = Number(tokenAmount.decimals);
    assert(Number.isInteger(parsedDecimals) && parsedDecimals >= 0, "RPC returned invalid token decimals.");

    if (baseUnits === BigInt(0)) {
      decimals = parsedDecimals;
    } else {
      assert(parsedDecimals === decimals, "RPC returned inconsistent token decimals across holder accounts.");
    }

    baseUnits += BigInt(tokenAmount.amount);
  }

  assert(
    baseUnits > BigInt(0),
    `${proof.address} does not currently hold any ${config.symbol} for the configured Devnet mint.`,
  );

  return {
    ...proof,
    currency: "TWC" as const,
    network: TWC_NETWORK,
    unlocked: FULL_NOVEL_UNLOCK_CHAPTERS,
    message: `Verified TWC proof for ${proof.address}. On-chain Devnet balance confirmed.`,
    chain: {
      rail: "solana-devnet",
      rpcUrl: config.rpcUrl,
      mintAddress: config.mintAddress,
      tokenSymbol: config.symbol,
      balanceBaseUnits: baseUnits.toString(),
      balanceTokens: formatTokenAmount(baseUnits, decimals),
      tokenAccounts: accounts.length,
      decimals,
    },
  };
}

export async function getUnlockRailStatus() {
  const [omega, twc] = await Promise.all([inspectOmegaConfig(), inspectTwcConfig()]);

  return {
    ready: omega.configured && twc.configured,
    rails: {
      omega: {
        configured: omega.configured,
        network: OMEGA_NETWORK,
        rpcUrl: omega.rpcUrl,
        manifestPath: omega.manifestPath,
        manifestPresent: omega.manifestPresent,
        gateAddress: omega.gateAddress ?? null,
        tokenAddress: omega.tokenAddress ?? null,
        message: omega.configured
          ? "Configured for on-chain $OMEGA verification."
          : "Missing Sepolia gate configuration. Add evm/deployments/sepolia.json or set OMEGA_NOVEL_GATE_ADDRESS.",
      },
      twc: {
        configured: twc.configured,
        network: TWC_NETWORK,
        rpcUrl: twc.rpcUrl,
        manifestPath: twc.manifestPath,
        manifestPresent: twc.manifestPresent,
        mintAddress: twc.mintAddress ?? null,
        message: twc.configured
          ? "Configured for on-chain TWC verification."
          : "Missing Devnet mint configuration. Add solana/deployments/twc-devnet.json or set TWC_MINT_ADDRESS.",
      },
    },
  };
}

export async function verifyUnlockProofOnChain(payload: UnlockProofPayload) {
  const proof = verifyUnlockProof(payload);

  if (proof.currency === "OMEGA") {
    return verifyOmegaOnChain(proof);
  }

  return verifyTwcOnChain(proof);
}
