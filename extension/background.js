// PokerNow Advisor — Background Service Worker
// Minimal — model lives on server, no local storage of model data

chrome.runtime.onInstalled.addListener(() => {
  console.log('[PNA] PokerNow Advisor installed/updated v33 — hand log reader + position from DOM');
});
