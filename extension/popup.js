// Show model status
chrome.storage.local.get('pnModel', ({ pnModel }) => {
  const el = document.getElementById('status');
  if (pnModel) {
    try {
      const m = JSON.parse(pnModel);
      el.innerHTML = `<span class="dot green"></span>Model loaded &middot; <strong>${m.totalHands} hands</strong> &middot; ${m.carryRate}% carry`;
    } catch (_) {
      el.innerHTML = '<span class="dot grey"></span>No model loaded yet';
    }
  } else {
    el.innerHTML = '<span class="dot grey"></span>No model loaded yet';
  }
});

// Manual inject button — works even if content_scripts didn't fire
document.getElementById('injectBtn').onclick = async () => {
  const btn = document.getElementById('injectBtn');
  btn.textContent = 'Injecting…';
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.url || !tab.url.includes('pokernow')) {
      btn.textContent = '⚠ Open PokerNow first';
      btn.style.background = '#ef4444';
      return;
    }

    // Remove existing overlay if any
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => { const el = document.getElementById('pn-advisor'); if (el) el.remove(); }
    });

    // Inject CSS then JS
    await chrome.scripting.insertCSS({ target: { tabId: tab.id }, files: ['overlay.css'] });
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['lib/model.js'] });
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] });

    btn.textContent = '✓ Overlay injected!';
    btn.style.background = '#22c55e';
    setTimeout(() => window.close(), 800);
  } catch (err) {
    btn.textContent = '⚠ Error: ' + err.message;
    btn.style.background = '#ef4444';
  }
};

document.getElementById('resetBtn').onclick = () => {
  chrome.storage.local.remove('pnModel', () => {
    document.getElementById('status').innerHTML = '<span class="dot grey"></span>Model cleared';
  });
};
