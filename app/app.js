/* Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 */
/* Omega release wallet prototype wiring — real wallet flows. */

const wallet = window.OmegaWallet || window.AmityWallet;

const views = document.querySelectorAll('.view');
const nav = document.querySelectorAll('[data-view]');
const toast = document.getElementById('toast');
let toastTimer;

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 4200);
}

function showView(name) {
  views.forEach((view) => view.classList.toggle('hidden', view.id !== `${name}-view`));
  document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.view === name));
  window.scrollTo({ top: 0, behavior: 'smooth' });
  if (name === 'wallet') {
    refreshWalletView();
  }
}

nav.forEach((item) => item.addEventListener('click', () => showView(item.dataset.view)));

document.getElementById('submit-claim')?.addEventListener('click', () => {
  showToast('Claim staged locally for review. Nothing was published.');
});

/* ------------------------------------------------------------------ */
/* Wallet view state                                                   */
/* ------------------------------------------------------------------ */

const els = {
  setup: document.getElementById('setup-panel'),
  live: document.getElementById('wallet-live'),
  address: document.getElementById('account-address'),
  label: document.getElementById('account-label'),
  ethBalance: document.getElementById('eth-balance'),
  balanceStatus: document.getElementById('balance-status'),
  mmStatus: document.getElementById('mm-status'),
  phantomStatus: document.getElementById('phantom-status'),
  solanaAddress: document.getElementById('solana-address'),
  twcStatus: document.getElementById('twc-status'),
  balOmega: document.getElementById('bal-omega'),
  balTwc: document.getElementById('bal-twc'),
  proofResult: document.getElementById('proof-result'),
  proofJson: document.getElementById('proof-json'),
  networkLabel: document.getElementById('network-label'),
};

function refreshWalletView() {
  const account = wallet.loadActiveAccount();
  const solana = wallet.loadSolanaAccount();

  if (!account && !solana) {
    els.setup.classList.remove('hidden');
    els.live.classList.add('hidden');
    els.networkLabel.textContent = 'No wallet connected';
    return;
  }

  els.setup.classList.add('hidden');
  els.live.classList.remove('hidden');

  if (account) {
    els.address.textContent = account.address;
    els.label.textContent = account.label + (account.hasKeystore ? ' · encrypted local keystore' : ' · connected EVM wallet');
    els.networkLabel.textContent = account.hasKeystore ? 'OMEGA · EVM/Sepolia proof rail · local key' : 'OMEGA · EVM/Sepolia proof rail · MetaMask';
    els.balOmega.textContent = 'Connected';
    fetchBalance();
  } else {
    els.address.textContent = 'No OMEGA wallet connected yet';
    els.label.textContent = 'Connect MetaMask or create/import an EVM key to sign $OMEGA unlock proofs.';
    els.networkLabel.textContent = 'TWC only · Solana/Devnet proof rail';
    els.ethBalance.textContent = '—';
    els.balanceStatus.textContent = 'No EVM wallet connected';
    els.balOmega.textContent = 'Not connected';
  }

  if (solana) {
    els.solanaAddress.textContent = solana.address;
    els.twcStatus.textContent = 'Phantom connected · Solana/Devnet proof rail';
    els.balTwc.textContent = 'Connected';
  } else {
    els.solanaAddress.textContent = 'No Phantom wallet connected yet';
    els.twcStatus.textContent = 'Connect Phantom to sign TWC unlock proofs';
    els.balTwc.textContent = 'Not connected';
  }
}

async function fetchBalance() {
  const account = wallet.loadActiveAccount();
  if (!account) {
    return;
  }

  els.balanceStatus.textContent = 'Querying mainnet…';
  try {
    const balance = await wallet.fetchEthBalance(account.address);
    els.ethBalance.textContent = `${parseFloat(balance.eth).toFixed(5)} ETH`;
    els.balanceStatus.textContent = `chainId ${balance.chainId} · live RPC`;
  } catch (error) {
    els.ethBalance.textContent = '—';
    els.balanceStatus.textContent = `RPC unreachable: ${error.message || error}`;
  }
}

document.getElementById('btn-refresh-balance')?.addEventListener('click', fetchBalance);

/* ------------------------------------------------------------------ */
/* Create / import / connect                                           */
/* ------------------------------------------------------------------ */

function passwordCheck(first, second) {
  if (!first || first.length < 8) {
    throw new Error('Password must be at least 8 characters.');
  }
  if (first !== second) {
    throw new Error('Passwords do not match.');
  }
  return first;
}

document.getElementById('btn-create')?.addEventListener('click', async () => {
  try {
    const label = document.getElementById('create-label').value.trim() || 'Omega Wallet';
    const password = passwordCheck(
      document.getElementById('create-password').value,
      document.getElementById('create-password2').value,
    );

    showToast('Generating EVM keypair (BIP-39)…');
    const account = wallet.createLocalWallet();
    await wallet.encryptAndSave(account, password, label);
    showToast(`Wallet created: ${account.address} — WRITE DOWN YOUR MNEMONIC (shown in console once).`);
    console.info('OMEGA WALLET MNEMONIC (write it down, never share):', account.mnemonic);
    console.info('OMEGA EVM ADDRESS:', account.address);
    refreshWalletView();
  } catch (error) {
    showToast(`Create failed: ${error.message || error}`);
  }
});

document.getElementById('btn-import')?.addEventListener('click', async () => {
  try {
    const type = document.getElementById('import-type').value;
    const secret = document.getElementById('import-secret').value.trim();
    const password = document.getElementById('import-password').value;
    if (!secret) {
      throw new Error('Enter your mnemonic or private key.');
    }
    if (!password || password.length < 8) {
      throw new Error('Password must be at least 8 characters.');
    }

    const account = type === 'mnemonic'
      ? wallet.importMnemonic(secret)
      : wallet.importPrivateKey(secret);

    await wallet.encryptAndSave(account, password, 'Imported wallet');
    showToast(`Imported: ${account.address}`);
    refreshWalletView();
  } catch (error) {
    showToast(`Import failed: ${error.message || error}`);
  }
});

document.getElementById('btn-connect-mm')?.addEventListener('click', async () => {
  try {
    els.mmStatus.textContent = 'Requesting EVM accounts…';
    const address = await wallet.connectMetaMask();
    els.mmStatus.textContent = `Connected: ${address}`;
    showToast(`MetaMask connected: ${address}`);
    refreshWalletView();
  } catch (error) {
    els.mmStatus.textContent = `Failed: ${error.message || error}`;
    showToast(`MetaMask connect failed: ${error.message || error}`);
  }
});

document.getElementById('btn-connect-phantom')?.addEventListener('click', async () => {
  try {
    els.phantomStatus.textContent = 'Requesting Solana account…';
    const address = await wallet.connectPhantom();
    els.phantomStatus.textContent = `Connected: ${address}`;
    showToast(`Phantom connected: ${address}`);
    refreshWalletView();
  } catch (error) {
    els.phantomStatus.textContent = `Failed: ${error.message || error}`;
    showToast(`Phantom connect failed: ${error.message || error}`);
  }
});

document.getElementById('btn-forget')?.addEventListener('click', () => {
  wallet.forgetWallet();
  showToast('Wallet forgotten on this device. Your funds remain on-chain.');
  refreshWalletView();
});

/* ------------------------------------------------------------------ */
/* Unlock signing                                                      */
/* ------------------------------------------------------------------ */

document.getElementById('btn-sign-unlock')?.addEventListener('click', async () => {
  try {
    const currency = document.getElementById('unlock-currency').value;
    showToast('Signing unlock statement…');

    let proof;
    if (currency === 'TWC') {
      proof = await wallet.signUnlockWithPhantom();
    } else {
      const account = wallet.loadActiveAccount();
      if (!account) {
        throw new Error('No $OMEGA EVM wallet connected.');
      }
      if (account.hasKeystore) {
        const password = prompt('Enter your keystore password to sign:');
        if (!password) {
          return;
        }
        proof = await wallet.signUnlockWithLocal(password);
      } else {
        proof = await wallet.signUnlockWithMetaMask();
      }
    }

    wallet.saveProof(proof);
    els.proofJson.value = JSON.stringify({
      signer: proof.signer,
      currency: proof.currency,
      network: proof.network,
      message: proof.message,
      signature: proof.signature,
    }, null, 2);
    els.proofResult.classList.remove('hidden');
    showToast(`${proof.currency} unlock proof signed. Verify it at the novel reader.`);
  } catch (error) {
    showToast(`Signing failed: ${error.message || error}`);
  }
});

document.getElementById('btn-copy-proof')?.addEventListener('click', async () => {
  const text = els.proofJson.value;
  try {
    await navigator.clipboard.writeText(text);
    showToast('Proof copied.');
  } catch {
    els.proofJson.select();
    showToast('Select and copy the proof manually.');
  }
});

document.getElementById('btn-goto-novel')?.addEventListener('click', (event) => {
  event.preventDefault();
  const proof = els.proofJson.value;
  sessionStorage.setItem('omega.unlock.proof', proof);
  showToast('Proof staged. Opening the novel reader…');
  setTimeout(() => {
    window.location.href = '/novel';
  }, 600);
});

/* ------------------------------------------------------------------ */
/* Privacy toggles (unchanged)                                         */
/* ------------------------------------------------------------------ */

document.querySelectorAll('.privacy-level').forEach((level) => level.addEventListener('click', () => {
  const toggle = level.querySelector('.toggle');
  if (!toggle || level.classList.contains('active-level')) {
    return;
  }

  toggle.classList.toggle('on');
  showToast(toggle.classList.contains('on')
    ? 'Visibility preference staged locally. Nothing was published.'
    : 'Visibility preference withdrawn locally.');
}));

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */

(function boot() {
  if (wallet && !wallet.loadActiveAccount() && !wallet.loadSolanaAccount()) {
    els.networkLabel.textContent = 'No wallet connected — set up in Wallet';
  }
  if (els.mmStatus && !els.mmStatus.textContent) {
    els.mmStatus.textContent = wallet.hasMetaMask() ? 'MetaMask detected.' : 'MetaMask not detected.';
  }
  if (els.phantomStatus && !els.phantomStatus.textContent) {
    els.phantomStatus.textContent = wallet.hasPhantom() ? 'Phantom detected.' : 'Phantom not detected.';
  }
  console.info('Omega wallet prototype loaded. Live unlock rails: $OMEGA on EVM and TWC on Solana. AMITY remains separate on Bitcoin/Lightning/Taproot.');
})();
