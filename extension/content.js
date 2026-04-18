// PokerNow Advisor — Content Script
// Points to Railway server for Mem0 + Claude advice

if (document.getElementById('pn-advisor')) {
  chrome.storage.local.get('pnModel', ({ pnModel }) => {
    try { if (pnModel) window.__pnaModel = JSON.parse(pnModel); } catch(_) {}
  });
} else {

(async () => {
  await new Promise(r => setTimeout(r, 1200));

  // ── YOUR RAILWAY URL — set this after deploying ───────────────────────────
  // It will look like: https://poker-agent-production-xxxx.up.railway.app
const SERVER = 'https://poker-agent-production.up.railway.app';
  // ─────────────────────────────────────────────────────────────────────────

  const SESSION_ID = 'session_' + Date.now();

  // ── Card reading from PokerNow CSS classes ────────────────────────────────
  function cardFromEl(el) {
    const classes = el.classList;
    let suit = null;
    if (classes.contains('card-c'))      suit = '♣';
    else if (classes.contains('card-d')) suit = '♦';
    else if (classes.contains('card-h')) suit = '♥';
    else if (classes.contains('card-s')) suit = '♠';
    if (!suit) return null;
    let rank = null;
    for (const cls of classes) {
      const m = cls.match(/^card-s-([2-9TJQKA]|10)$/i);
      if (m) { rank = m[1].toUpperCase() === 'T' ? '10' : m[1].toUpperCase(); break; }
    }
    return rank ? rank + suit : null;
  }

  function readCards(sel) {
    const cards = [];
    document.querySelectorAll(sel).forEach(el => {
      const c = cardFromEl(el);
      if (c && !cards.includes(c)) cards.push(c);
    });
    return cards;
  }

  function readFromDOM() {
    // Stage from total card-container count — reliable the moment street is dealt,
    // unlike .flipped which lags 350ms–3s during flip animation.
    const boardTotal = document.querySelectorAll('.table-cards .card-container').length;
    const boardStage = boardTotal >= 5 ? 'river'
                    : boardTotal === 4 ? 'turn'
                    : boardTotal === 3 ? 'flop'
                    : 'preflop';
    const holeCards  = readCards('.you-player .card-container.flipped').slice(0, 2);
    const boardCards = readCards('.table-cards .card-container.flipped').slice(0, 5);
    const isHeroTurn = !!document.querySelector('.you-player.decision-current');
    const handNum    = parseInt(document.querySelector('.open-review')?.innerText?.match(/#(\d+)/)?.[1] || '0', 10);
    return {
      holeCards,
      boardCards: boardCards.filter(c => !holeCards.includes(c)),
      boardStage,
      boardTotal,
      isHeroTurn,
      handNum,
    };
  }

  // ── Server communication ──────────────────────────────────────────────────
  let serverUp = false;

  async function checkServer() {
    try {
      const r = await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(5000) });
      if (r.ok) {
        const data = await r.json();
        serverUp = true;
        return data;
      }
    } catch(_) {}
    serverUp = false;
    return null;
  }

  async function postHand(hand) {
    if (!serverUp) return null;
    try {
      const r = await fetch(`${SERVER}/hand`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: SESSION_ID, ...hand })
      });
      return r.ok ? r.json() : null;
    } catch(_) { return null; }
  }

  async function getAgentAdvice(holeCards, boardCards) {
    if (!serverUp) return null;
    try {
      const r = await fetch(`${SERVER}/advice`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id:  SESSION_ID,
          hand_num:    handNum,
          hole_cards:  holeCards,
          board_cards: boardCards,
        }),
        signal: AbortSignal.timeout(15000)
      });
      return r.ok ? r.json() : null;
    } catch(_) { return null; }
  }

  // ── Live hand tracking ────────────────────────────────────────────────────
  let prevHole   = [];
  let handBuffer = { holeCards: [], flop: [], turn: [], river: [] };
  let handNum    = 0;

  function updateHandBuffer(holeCards, boardCards) {
    // Detect new hand
    if (holeCards.length === 2 &&
       (prevHole.length === 0 ||
        (holeCards[0] !== prevHole[0] || holeCards[1] !== prevHole[1]))) {
      // Post completed previous hand
      if (prevHole.length === 2 && handBuffer.holeCards.length) {
        handNum++;
        postHand({
          hand_num:   handNum,
          hole_cards: handBuffer.holeCards,
          flop:       handBuffer.flop,
          turn:       handBuffer.turn,
          river:      handBuffer.river,
        });
      }
      handBuffer = { holeCards, flop: [], turn: [], river: [] };
      prevHole = holeCards;
    }
    if (boardCards.length >= 3 && !handBuffer.flop.length) handBuffer.flop  = boardCards.slice(0,3);
    if (boardCards.length >= 4 && !handBuffer.turn.length)  handBuffer.turn  = [boardCards[3]];
    if (boardCards.length >= 5 && !handBuffer.river.length) handBuffer.river = [boardCards[4]];
  }

  // ── Local fallback model ──────────────────────────────────────────────────
  let localModel = null;
  chrome.storage.local.get('pnModel', ({ pnModel }) => {
    if (pnModel) { try { localModel = JSON.parse(pnModel); } catch(_) {} }
  });

  // ── Build overlay ─────────────────────────────────────────────────────────
  const overlay = document.createElement('div');
  overlay.id = 'pn-advisor';
  overlay.innerHTML = `
    <div id="pna-header">
      <span id="pna-title">♠ Advisor</span>
      <div id="pna-server-dot" title="Railway server"></div>
      <div id="pna-hcontrols">
        <button id="pna-proof-toggle" title="Proof">📋</button>
        <button id="pna-scan">⟳</button>
        <button id="pna-min">−</button>
        <button id="pna-close">✕</button>
      </div>
    </div>
    <div id="pna-body">
      <div id="pna-upload">
        <div id="pna-upload-msg">Upload log CSV to seed the model</div>
        <label id="pna-lbl">Choose log CSV
          <input type="file" id="pna-file" accept=".csv" style="display:none">
        </label>
        <div id="pna-ustat"></div>
      </div>
      <div id="pna-main" style="display:none">
        <div id="pna-detected">
          <div id="pna-hole-row"></div>
          <div id="pna-board-row"></div>
        </div>
        <div id="pna-action-box">
          <div id="pna-action"></div>
          <div id="pna-sizing"></div>
        </div>
        <div id="pna-agent-wrap" style="display:none">
          <div id="pna-agent-label">🧠 Claude says</div>
          <div id="pna-agent-text"></div>
        </div>
        <div id="pna-coming-wrap">
          <div id="pna-clabel"></div>
          <div id="pna-coming"></div>
        </div>
        <div id="pna-reasons"></div>
        <div id="pna-foot">
          <span id="pna-hands"></span>
          <span id="pna-carry"></span>
          <span id="pna-stage"></span>
          <span id="pna-mem-badge" style="display:none">🧠 mem0</span>
        </div>
      </div>
      <div id="pna-waiting">
        <div id="pna-wait-msg">Waiting for cards…</div>
        <div id="pna-model-status" style="font-size:10px;opacity:.5;margin-top:3px"></div>
      </div>
      <div id="pna-proof-panel" style="display:none"></div>
    </div>`;
  document.body.appendChild(overlay);

  // Drag
  let drag=false,ox=0,oy=0;
  overlay.querySelector('#pna-header').addEventListener('mousedown', e => {
    if (e.target.tagName==='BUTTON'||e.target.id==='pna-server-dot') return;
    drag=true; ox=e.clientX-overlay.offsetLeft; oy=e.clientY-overlay.offsetTop;
  });
  document.addEventListener('mousemove', e => {
    if (!drag) return;
    overlay.style.left=(e.clientX-ox)+'px'; overlay.style.top=(e.clientY-oy)+'px'; overlay.style.right='auto';
  });
  document.addEventListener('mouseup', () => drag=false);

  let minimised=false;
  overlay.querySelector('#pna-min').onclick = () => {
    minimised=!minimised;
    overlay.querySelector('#pna-body').style.display=minimised?'none':'block';
    overlay.querySelector('#pna-min').textContent=minimised?'+':'−';
  };
  overlay.querySelector('#pna-close').onclick = () => overlay.remove();
  overlay.querySelector('#pna-scan').onclick  = () => refresh();

  // Proof panel
  let proofOpen = false;
  overlay.querySelector('#pna-proof-toggle').onclick = async () => {
    proofOpen = !proofOpen;
    const panel = overlay.querySelector('#pna-proof-panel');
    panel.style.display = proofOpen ? 'block' : 'none';
    if (!proofOpen) return;
    panel.innerHTML = '<div style="font-size:10px;color:#666;padding:6px">Loading proof…</div>';
    try {
      const r = await fetch(`${SERVER}/proof/${SESSION_ID}`, { signal: AbortSignal.timeout(5000) });
      const data = await r.json();
      let html = '<div style="font-size:10px;color:#888;padding:6px 8px">';
      html += `<div style="color:#22c55e;font-weight:600;margin-bottom:4px">🟢 Mem0 stored memories (${data.mem0_entries.length})</div>`;
      if (data.mem0_entries.length) {
        data.mem0_entries.slice(0,5).forEach(e => {
          html += `<div style="border-left:2px solid #22c55e;padding-left:6px;margin-bottom:5px">
            <b>Hand #${e.hand_num}</b> · ID: ${e.memory_id||'—'}<br>
            <span style="color:#555">${e.memory_text.slice(0,150)}…</span>
          </div>`;
        });
      } else {
        html += '<div style="color:#444">No memories yet — play hands first</div>';
      }
      html += `<div style="color:#a78bfa;font-weight:600;margin:8px 0 4px">🟣 Claude responses (${data.letta_entries.length})</div>`;
      if (data.letta_entries.length) {
        data.letta_entries.slice(0,3).forEach(e => {
          html += `<div style="border-left:2px solid #a78bfa;padding-left:6px;margin-bottom:5px">
            <b>Hand #${e.hand_num}</b><br>
            <span style="color:#555">${(e.response||'none').slice(0,200)}…</span>
          </div>`;
        });
      } else {
        html += '<div style="color:#444">No responses yet — board changes trigger these</div>';
      }
      html += '</div>';
      panel.innerHTML = html;
    } catch(err) {
      panel.innerHTML = `<div style="font-size:10px;color:#666;padding:6px">Server error: ${err.message}</div>`;
    }
  };

  // Server status dot
  const dot = overlay.querySelector('#pna-server-dot');
  const updateDot = (health) => {
    if (!health) { dot.style.background='#ef4444'; dot.title='Railway server offline'; return; }
    const ok = health.mem0 && health.claude;
    dot.style.background = ok ? '#22c55e' : '#f59e0b';
    dot.title = `Railway: mem0=${health.mem0?'✓':'✗'} claude=${health.claude?'✓':'✗'}`;
  };
  checkServer().then(updateDot);
  setInterval(() => checkServer().then(updateDot), 30000);

  // Model load
  let model = null;
  chrome.storage.local.get('pnModel', ({pnModel}) => {
    if (pnModel) {
      try {
        model = JSON.parse(pnModel);
        overlay.querySelector('#pna-model-status').textContent = `${model.totalHands} hands loaded`;
        overlay.querySelector('#pna-upload').style.display = 'none';
      } catch(_) {}
    }
  });

  // File upload
  overlay.querySelector('#pna-file').addEventListener('change', async e => {
    const file = e.target.files[0]; if (!file) return;
    const stat = overlay.querySelector('#pna-ustat');
    stat.textContent = 'Parsing…';
    try {
      const text = await file.text();
      const hands = PokerModel.parseLog(text);
      if (!hands.length) { stat.textContent = '⚠ No hands found'; return; }
      const built = PokerModel.buildModel(hands);
      model = built;
      chrome.storage.local.set({ pnModel: JSON.stringify({
        earlyBias:built.earlyBias, posCarry:built.posCarry, cardGiven:built.cardGiven,
        carryHot:built.carryHot, topCards:built.topCards, carryRate:built.carryRate,
        totalHands:built.totalHands, lastHand:built.lastHand, hands:built.hands.slice(-50)
      })});
      overlay.querySelector('#pna-upload').style.display = 'none';
      overlay.querySelector('#pna-model-status').textContent = `${hands.length} hands seeded`;

      // Sync all historical hands to Railway so Mem0/Claude can learn
      if (serverUp) {
        stat.textContent = `Syncing ${hands.length} hands to Railway…`;
        for (let i = 0; i < hands.length; i++) {
          const h = hands[i];
          await postHand({ hand_num:h.hand_num, hole_cards:h.hole_cards, flop:h.flop, turn:h.turn, river:h.river });
          if (i % 15 === 0) stat.textContent = `Syncing… ${i}/${hands.length}`;
        }
        stat.textContent = `✓ ${hands.length} hands — Railway + Mem0 trained!`;
      } else {
        stat.textContent = `✓ ${hands.length} hands loaded (local only)`;
      }
    } catch(err) { stat.textContent = '⚠ ' + err.message; }
  });

  // ── Render ────────────────────────────────────────────────────────────────
  function cHTML(card, type) {
    const suit=card.slice(-1), red=suit==='♥'||suit==='♦';
    return `<span class="pna-c pna-c-${type}" style="color:${red?'#ff6b6b':'#b3b3ff'}">${card}</span>`;
  }

  let lastBoardStr = '', lastAdviceFetch = 0;

  async function refresh() {
    const { holeCards, boardCards, boardStage, boardTotal, isHeroTurn, handNum: domHandNum } = readFromDOM();
    updateHandBuffer(holeCards, boardCards);

    const main = overlay.querySelector('#pna-main');
    const wait = overlay.querySelector('#pna-waiting');

    if (!holeCards.length) {
      main.style.display = 'none'; wait.style.display = 'block';
      if (!model) overlay.querySelector('#pna-upload').style.display = 'block';
      return;
    }
    if (!model && !serverUp) {
      main.style.display = 'none'; wait.style.display = 'block';
      overlay.querySelector('#pna-wait-msg').textContent = 'Cards found — upload log CSV';
      overlay.querySelector('#pna-upload').style.display = 'block';
      return;
    }

    main.style.display = 'block'; wait.style.display = 'none';

    overlay.querySelector('#pna-hole-row').innerHTML =
      '<span class="pna-dim">You: </span>' + holeCards.map(c => cHTML(c,'hole')).join(' ');
    overlay.querySelector('#pna-board-row').innerHTML = boardCards.length
      ? '<span class="pna-dim">Board: </span>' + boardCards.map(c => cHTML(c,'board')).join(' ')
      : '<span class="pna-dim">Board: —</span>';

    // Local instant advice
    if (model) {
      const adv = PokerModel.getAdvice(model, holeCards, boardCards, boardStage);
      overlay.querySelector('#pna-action').className = 'pna-action pna-action-' + adv.vClass;
      overlay.querySelector('#pna-action').textContent = adv.emoji + ' ' + adv.action;
      const sEl = overlay.querySelector('#pna-sizing');
      if (adv.sizing) { sEl.style.display='block'; sEl.textContent='Size: '+adv.sizing; sEl.className='pna-sizing pna-sizing-'+adv.vClass; }
      else sEl.style.display = 'none';
      overlay.querySelector('#pna-clabel').textContent = adv.nextLabel;
      overlay.querySelector('#pna-coming').innerHTML = adv.top10.slice(0,8).map(({card}) => {
        const suit=card.slice(-1), red=suit==='♥'||suit==='♦';
        const connects = holeCards.map(c=>c.slice(0,-1)).includes(card.slice(0,-1)) || holeCards.map(c=>c.slice(-1)).includes(suit);
        return `<span class="pna-nc ${connects?'pna-nc-hit':''}" style="color:${red?'#ff9999':'#b3b3ff'}">${card}${connects?'★':''}</span>`;
      }).join('');
      overlay.querySelector('#pna-reasons').innerHTML = adv.reasons.slice(0,3).map(r=>`<div class="pna-reason">• ${r}</div>`).join('');
      overlay.querySelector('#pna-hands').textContent = model.totalHands + ' hands';
      overlay.querySelector('#pna-carry').textContent = model.carryRate + '% carry';
      overlay.querySelector('#pna-stage').textContent = adv.stage;
    }

    // Claude advice from Railway (triggered when board changes)
    const boardStr = boardCards.join(',');
    const now = Date.now();
    if (serverUp && boardStr !== lastBoardStr && now - lastAdviceFetch > 5000) {
      lastBoardStr = boardStr;
      lastAdviceFetch = now;
      const agentWrap = overlay.querySelector('#pna-agent-wrap');
      agentWrap.style.display = 'block';
      overlay.querySelector('#pna-agent-text').textContent = '⏳ Claude thinking…';
      overlay.querySelector('#pna-mem-badge').style.display = 'inline';

      getAgentAdvice(holeCards, boardCards).then(result => {
        if (!result || !result.agent_advice) { agentWrap.style.display='none'; return; }
        overlay.querySelector('#pna-agent-text').textContent = result.agent_advice;
        agentWrap.style.display = 'block';
        if (result.carry_rate) overlay.querySelector('#pna-carry').textContent = result.carry_rate+'% carry';
        if (result.hand_count) overlay.querySelector('#pna-hands').textContent = result.hand_count+' hands';
      });
    }
  }

  setInterval(refresh, 1500);
  refresh();

})();

} // end guard
