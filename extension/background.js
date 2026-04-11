// Background service worker — injects overlay when user clicks the extension icon
// This is the fallback if content_scripts doesn't fire automatically

chrome.action.onClicked.addListener((tab) => {
  if (!tab.url) return;
  // Inject manually if content script didn't load
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ['lib/model.js', 'content.js']
  }).catch(() => {});
  chrome.scripting.insertCSS({
    target: { tabId: tab.id },
    files: ['overlay.css']
  }).catch(() => {});
});

chrome.runtime.onInstalled.addListener(() => {
  console.log('PokerNow Advisor v1.1 installed');
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'clearModel') {
    chrome.storage.local.remove('pnModel', () => sendResponse({ ok: true }));
    return true;
  }
});
