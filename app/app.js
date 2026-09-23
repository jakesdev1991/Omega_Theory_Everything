/* Amity app wiring — real wallet flows. */

const views = document.querySelectorAll('.view');
const nav = document.querySelectorAll('[data-view]');
const toast = document.getElementById('toast');
let toastTimer;

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 4200);
}

function showView(name) {
  views.forEach(view => view.classList.toggle('hidden', view.id !== `${name}-view`));
  document.querySelectorAll('.nav-item').forEach(item => item.classList.toggle('active', item.dataset.view === name));
  window.scrollTo({ top: 0, behavior: 'smooth' });
  if (name === 'wallet') refreshWalletView();
}
nav.forEach(item => item.addEventListener('click', () => showView(item.dataset.view)));

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
  proofResult: document.getElementById('proof-result'),
  proofJson: document.getElementById('proof-json'),
  networkLabel: document.getElementById('network-label'),
};

function refreshWalletView() {
  const account = window.AmityWallet.loadActiveAccount();
  if (!account) {
    els.setup.classList.remove('hidden');
    els.live.classList.add('hidden');
    els.networkLabel.textContent = 'No wallet connected';
    return;
  }
  els.setup.classList.add('hidden');
  els.live.classList.remove('hidden');
  els.address.textContent = account.address;
  els.label.textContent = account.label + (account.hasKeystore ? ' · encrypted keystore' : ' · connected wallet');
  els.networkLabel.textContent = account.hasKeystore ? 'Mainnet · local key' : 'Mainnet · MetaMask';
  fetchBalance();
}

async function fetchBalance() {
  const account = window.AmityWallet.loadActiveAccount();
  if (!account) return;
  els.balanceStatus.textContent = 'Querying mainnet…';
  try {
    const bal = await window.AmityWallet.fetchEthBalance(account.address);
    els.ethBalance.textContent = parseFloat(bal.eth).toFixed(5) + ' ETH';
    els.balanceStatus.textContent = 'chainId ' + bal.chainId + ' · live RPC';
  } catch (err) {
    els.ethBalance.textContent = '—';
    els.balanceStatus.textContent = 'RPC unreachable: ' + (err.message || err);
  }
}

document.getElementById('btn-refresh-balance')?.addEventListener('click', fetchBalance);

/* ------------------------------------------------------------------ */
/* Create / import / connect                                           */
/* ------------------------------------------------------------------ */

function passwordCheck(p1, p2) {
  if (!p1 || p1.length < 8) throw new Error('Password must be at least 8 characters.');
  if (p1 !== p2) throw new Error('Passwords do not match.');
  return p1;
}

document.getElementById('btn-create')?.addEventListener('click', async () => {
  try {
    const label = document.getElementById('create-label').value.trim() || 'Amity Wallet';
    const password = passwordCheck(
      document.getElementById('create-password').value,
      document.getElementById('create-password2').value
    );
    showToast('Generating keypair (BIP-39)…');
    const account = window.AmityWallet.createLocalWallet();
    await window.AmityWallet.encryptAndSave(account, password, label);
    showToast('Wallet created: ' + account.address + ' — WRITE DOWN YOUR MNEMONIC (shown in console once).');
    console.info('AMITY MNEMONIC (write it down, never share):', account.mnemonic);
    console.info('AMITY ADDRESS:', account.address);
    refreshWalletView();
  } catch (err) {
    showToast('Create failed: ' + (err.message || err));
  }
});

document.getElementById('btn-import')?.addEventListener('click', async () => {
  try {
    const type = document.getElementById('import-type').value;
    const secret = document.getElementById('import-secret').value.trim();
    const password = document.getElementById('import-password').value;
    if (!secret) throw new Error('Enter your mnemonic or private key.');
    if (!password || password.length < 8) throw new Error('Password must be at least 8 characters.');
    const account = type === 'mnemonic'
      ? window.AmityWallet.importMnemonic(secret)
      : window.AmityWallet.importPrivateKey(secret);
    await window.AmityWallet.encryptAndSave(account, password, 'Imported wallet');
    showToast('Imported: ' + account.address);
    refreshWalletView();
  } catch (err) {
    showToast('Import failed: ' + (err.message || err));
  }
});

document.getElementById('btn-connect-mm')?.addEventListener('click', async () => {
  try {
    els.mmStatus.textContent = 'Requesting accounts…';
    const address = await window.AmityWallet.connectMetaMask();
    els.mmStatus.textContent = 'Connected: ' + address;
    showToast('MetaMask connected: ' + address);
    refreshWalletView();
  } catch (err) {
    els.mmStatus.textContent = 'Failed: ' + (err.message || err);
    showToast('MetaMask connect failed: ' + (err.message || err));
  }
});

document.getElementById('btn-forget')?.addEventListener('click', () => {
  window.AmityWallet.forgetWallet();
  showToast('Wallet forgotten on this device. Your funds remain on-chain.');
  refreshWalletView();
});

/* ------------------------------------------------------------------ */
/* Unlock signing                                                      */
/* ------------------------------------------------------------------ */

document.getElementById('btn-sign-unlock')?.addEventListener('click', async () => {
  try {
    const account = window.AmityWallet.loadActiveAccount();
    if (!account) throw new Error('No active account.');
    const tier = document.getElementById('unlock-tier').value;
    showToast('Signing unlock statement…');

    let proof;
    if (account.hasKeystore) {
      const password = prompt('Enter your keystore password to sign:');
      if (!password) return;
      proof = await window.AmityWallet.signUnlockWithLocal(password, tier);
    } else {
      proof = await window.AmityWallet.signUnlockWithMetaMask(tier);
    }

    window.AmityWallet.saveProof(proof);
    els.proofJson.value = JSON.stringify({
      signer: proof.signer,
      tier,
      message: proof.message,
      signature: proof.signature,
    }, null, 2);
    els.proofResult.classList.remove('hidden');
    showToast('Unlock proof signed. Verify it at the novel reader.');
  } catch (err) {
    showToast('Signing failed: ' + (err.message || err));
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

document.getElementById('btn-goto-novel')?.addEventListener('click', (e) => {
  e.preventDefault();
  const proof = els.proofJson.value;
  sessionStorage.setItem('amity.unlock.proof', proof);
  showToast('Proof staged. Opening the novel reader…');
  setTimeout(() => window.location.href = '/novel', 600);
});

/* ------------------------------------------------------------------ */
/* Privacy toggles (unchanged)                                         */
/* ------------------------------------------------------------------ */

document.querySelectorAll('.privacy-level').forEach(level => level.addEventListener('click', () => {
  const toggle = level.querySelector('.toggle');
  if (!toggle || level.classList.contains('active-level')) return;
  toggle.classList.toggle('on');
  showToast(toggle.classList.contains('on')
    ? 'Visibility preference staged locally. Nothing was published.'
    : 'Visibility preference withdrawn locally.');
}));

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */

(function boot() {
  if (window.AmityWallet && !window.AmityWallet.loadActiveAccount()) {
    els.networkLabel.textContent = 'No wallet connected — set up in Wallet';
  }
  console.info('Amity wallet loaded. Real keys, local storage, no custodian.');
})();
