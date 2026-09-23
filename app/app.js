/* Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. */

const views = document.querySelectorAll('.view');
const nav = document.querySelectorAll('[data-view]');
const toast = document.getElementById('toast');
let toastTimer;

function showView(name) {
  views.forEach(view => view.classList.toggle('hidden', view.id !== `${name}-view`));
  document.querySelectorAll('.nav-item').forEach(item => item.classList.toggle('active', item.dataset.view === name));
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
nav.forEach(item => item.addEventListener('click', () => showView(item.dataset.view)));

document.getElementById('submit-claim')?.addEventListener('click', () => {
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3200);
});

document.querySelectorAll('.privacy-level').forEach(level => level.addEventListener('click', () => {
  const toggle = level.querySelector('.toggle');
  if (!toggle || level.classList.contains('active-level')) return;
  toggle.classList.toggle('on');
  toast.textContent = toggle.classList.contains('on')
    ? 'Visibility preference staged locally. Nothing was published.'
    : 'Visibility preference withdrawn locally.';
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3200);
}));

// Keep the prototype explicitly offline: no wallet provider, chain, or account API is touched.
console.info('Amity local prototype loaded. No keys, tokens, or personal data are connected.');
