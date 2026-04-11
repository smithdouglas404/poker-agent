// PokerNow Advisor — Model Engine v2
// Full betting action engine: Raise / Bet / Call / Check / Fold with sizing

const PokerModel = (() => {
  const RANKS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A'];
  const SUITS = ['♠','♥','♦','♣'];
  const ALL_CARDS = RANKS.flatMap(r => SUITS.map(s => r + s));
  const EARLY = new Set(['hole_1','hole_2','flop_1','flop_2','flop_3']);
  const RANK_VAL = {2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,J:11,Q:12,K:13,A:14};

  function parseCards(text) {
    return (text.match(/(?:10|[2-9JQKA])[♠♥♦♣]/g) || []);
  }

  function parseLog(csvText) {
    const lines = csvText.split('\n').slice(1);
    const rows = [];
    for (const line of lines) {
      const m = line.match(/^"?(.*?)"?,(\d{4}-\d{2}-\d{2}T[^,]+),(\d+)/);
      if (m) rows.push({ entry: m[1].replace(/""/g, '"'), order: parseInt(m[3]) });
    }
    rows.sort((a, b) => a.order - b.order);
    const hands = [];
    let cur = null;
    for (const { entry } of rows) {
      if (entry.includes('-- starting hand')) {
        const m = entry.match(/starting hand #(\d+)/);
        if (m) cur = { hand_num: +m[1], hole_cards: [], flop: [], turn: [], river: [] };
      } else if (entry.includes('-- ending hand')) {
        if (cur) { hands.push(cur); cur = null; }
      } else if (cur) {
        if (entry.includes('Your hand is')) cur.hole_cards = parseCards(entry);
        else if (entry.startsWith('Flop:')) cur.flop = parseCards(entry);
        else if (entry.startsWith('Turn:')) {
          const m = entry.match(/\[([^\]]+)\]$/);
          if (m) cur.turn = parseCards(m[1]);
        } else if (entry.startsWith('River:')) {
          const m = entry.match(/\[([^\]]+)\]$/);
          if (m) cur.river = parseCards(m[1]);
        }
      }
    }
    return hands;
  }

  function buildModel(hands) {
    const posFreq = {};
    ALL_CARDS.forEach(c => posFreq[c] = {});
    const incr = (card, pos) => { posFreq[card][pos] = (posFreq[card][pos] || 0) + 1; };
    for (const h of hands) {
      h.hole_cards.forEach((c, j) => incr(c, `hole_${j+1}`));
      h.flop.forEach((c, j) => incr(c, `flop_${j+1}`));
      h.turn.forEach(c => incr(c, 'turn'));
      h.river.forEach(c => incr(c, 'river'));
    }

    const earlyBias = {};
    for (const c of ALL_CARDS) {
      const e = [...EARLY].reduce((s, p) => s + (posFreq[c][p] || 0), 0);
      const t = Object.values(posFreq[c]).reduce((s, n) => s + n, 0);
      earlyBias[c] = t > 0 ? e / t : 0.5;
    }

    const posCarry = {};
    ['hole_1','hole_2','flop_1','flop_2','flop_3','turn','river'].forEach(p => posCarry[p] = [0,0]);
    for (let i = 0; i < hands.length - 1; i++) {
      const h1 = hands[i], h2 = hands[i+1];
      const h2c = new Set([...h2.hole_cards, ...h2.flop, ...h2.turn, ...h2.river]);
      const add = (card, pos) => { posCarry[pos][1]++; if (h2c.has(card)) posCarry[pos][0]++; };
      h1.hole_cards.forEach((c, j) => add(c, `hole_${j+1}`));
      h1.flop.forEach((c, j) => add(c, `flop_${j+1}`));
      h1.turn.forEach(c => add(c, 'turn'));
      h1.river.forEach(c => add(c, 'river'));
    }

    const cardGiven = {};
    for (const h of hands) {
      const cards = [...new Set([...h.hole_cards, ...h.flop, ...h.turn, ...h.river])];
      for (const a of cards) {
        if (!cardGiven[a]) cardGiven[a] = {};
        for (const b of cards) if (a !== b) cardGiven[a][b] = (cardGiven[a][b]||0)+1;
      }
    }

    const carryHot = {};
    let carryPairs = 0, carryTotal = 0;
    for (let i = 0; i < hands.length - 1; i++) {
      const s1 = new Set([...hands[i].hole_cards, ...hands[i].flop, ...hands[i].turn, ...hands[i].river]);
      const s2 = new Set([...hands[i+1].hole_cards, ...hands[i+1].flop, ...hands[i+1].turn, ...hands[i+1].river]);
      if (s1.size && s2.size) {
        carryTotal++;
        const shared = [...s1].filter(c => s2.has(c));
        if (shared.length) { carryPairs++; shared.forEach(c => carryHot[c] = (carryHot[c]||0)+1); }
      }
    }

    const topCards = ALL_CARDS
      .filter(c => Object.values(posFreq[c]).reduce((s,n)=>s+n,0) > 2)
      .sort((a, b) => {
        const sc = c => {
          const e = [...EARLY].reduce((s,p)=>s+(posFreq[c][p]||0),0);
          const t = Object.values(posFreq[c]).reduce((s,n)=>s+n,0)||1;
          return e/t;
        };
        return sc(b)-sc(a);
      }).slice(0,15);

    return { hands, earlyBias, posCarry, cardGiven, carryHot, topCards,
      carryRate: carryTotal ? Math.round(carryPairs/carryTotal*100) : 0,
      totalHands: hands.length, lastHand: hands[hands.length-1]||null };
  }

  function scoreCards(model, boardCards, holeCards, excludeCards) {
    const used = new Set(excludeCards);
    const hands = model.hands;
    const chain = (() => {
      if (hands.length < 2) return 1.0;
      const p = new Set([...hands[hands.length-2].hole_cards,...hands[hands.length-2].flop,...hands[hands.length-2].turn,...hands[hands.length-2].river]);
      const l = new Set([...hands[hands.length-1].hole_cards,...hands[hands.length-1].flop,...hands[hands.length-1].turn,...hands[hands.length-1].river]);
      return [...p].some(x=>l.has(x)) ? 1.8 : 1.0;
    })();

    return ALL_CARDS.map(c => {
      if (used.has(c)) return { card: c, score: -1 };
      let s = (model.earlyBias[c]||0.5)*2.0;
      hands.slice(-3).reverse().forEach((h, j) => {
        const decay = [1.0,0.4,0.15][j];
        const addCard = (card, pos) => {
          if (card !== c) return;
          const [hi,tot] = model.posCarry[pos]||[0,1];
          s += (tot ? hi/tot : 0.1)*decay*chain*3.0;
        };
        h.hole_cards.forEach((card,k) => addCard(card,`hole_${k+1}`));
        h.flop.forEach((card,k) => addCard(card,`flop_${k+1}`));
        h.turn.forEach(card => addCard(card,'turn'));
        h.river.forEach(card => addCard(card,'river'));
      });
      for (const bc of boardCards) {
        const given = model.cardGiven[bc]||{};
        const entries = Object.entries(given).sort((a,b)=>b[1]-a[1]);
        const rank = entries.findIndex(([k])=>k===c);
        if (rank>=0 && rank<8) s += (8-rank)*0.3;
      }
      return { card: c, score: s };
    }).filter(x=>x.score>=0).sort((a,b)=>b.score-a.score);
  }

  function evaluateHand(holeCards, boardCards) {
    const all = [...holeCards, ...boardCards];
    const ranks = all.map(c=>c.slice(0,-1));
    const suits = all.map(c=>c.slice(-1));
    const hRanks = holeCards.map(c=>c.slice(0,-1));
    const hSuits = holeCards.map(c=>c.slice(-1));
    const bRanks = boardCards.map(c=>c.slice(0,-1));
    const bSuits = boardCards.map(c=>c.slice(-1));

    const rankCount = {};
    ranks.forEach(r => rankCount[r] = (rankCount[r]||0)+1);
    const suitCount = {};
    suits.forEach(s => suitCount[s] = (suitCount[s]||0)+1);

    const pairs = Object.entries(rankCount).filter(([,n])=>n>=2).map(([r])=>r);
    const trips = Object.entries(rankCount).filter(([,n])=>n>=3).map(([r])=>r);
    const quads = Object.entries(rankCount).filter(([,n])=>n>=4).map(([r])=>r);
    const flushSuit = Object.entries(suitCount).find(([,n])=>n>=5);

    const allVals = [...new Set(all.map(c=>RANK_VAL[c.slice(0,-1)]||0).filter(Boolean))].sort((a,b)=>a-b);
    let maxStreak=1,streak=1;
    for (let i=1;i<allVals.length;i++){
      if(allVals[i]===allVals[i-1]+1) streak++;
      else streak=1;
      maxStreak=Math.max(maxStreak,streak);
    }

    const made = [];
    if (quads.length)              made.push({ name:`four ${quads[0]}s`,      strength:8 });
    if (trips.length&&pairs.length>1) made.push({ name:'full house',           strength:7 });
    if (flushSuit)                 made.push({ name:`${flushSuit[0]} flush`,   strength:6 });
    if (maxStreak>=5)              made.push({ name:'straight',                strength:5 });
    if (trips.length)              made.push({ name:`three ${trips[0]}s`,      strength:4 });
    if (pairs.length>=2)           made.push({ name:'two pair',                strength:3 });

    const isPocketPair = hRanks[0]===hRanks[1];
    const holePairs = hRanks.filter(r=>bRanks.includes(r));
    if (isPocketPair) made.push({ name:`pocket ${hRanks[0]}s`, strength:2.5 });
    if (holePairs.length>=1 && !trips.length && !isPocketPair)
      made.push({ name:`pair of ${holePairs[0]}s`, strength:2 });

    const draws = [];
    hSuits.forEach((s,i) => {
      const matching = bSuits.filter(bs=>bs===s).length;
      if (matching>=3) draws.push({ name:`flush draw (${s})`, strength:1.5 });
    });
    if (maxStreak===4) draws.push({ name:'open-ended straight draw', strength:1.5 });
    else if (maxStreak===3) draws.push({ name:'gutshot draw', strength:0.8 });

    const best = [...made].sort((a,b)=>b.strength-a.strength)[0]||null;
    const bestDraw = draws[0]||null;
    return { made, draws, best, bestDraw, isPocketPair };
  }

  function getAdvice(model, holeCards, boardCards) {
    const allVisible = [...holeCards, ...boardCards];
    const ranked = scoreCards(model, boardCards, holeCards, allVisible);
    const top10  = ranked.slice(0,10).map(x=>x.card);

    const stage = boardCards.length>=5 ? 'river'
      : boardCards.length===4 ? 'turn'
      : boardCards.length>=2  ? 'flop'
      : 'preflop';

    const { made, draws, best, bestDraw, isPocketPair } = evaluateHand(holeCards, boardCards);
    const hRanks = holeCards.map(c=>c.slice(0,-1));
    const hSuits = holeCards.map(c=>c.slice(-1));

    const connecting = top10.filter(c => hRanks.includes(c.slice(0,-1)) || hSuits.includes(c.slice(-1)));
    const topHole = holeCards.filter(c=>model.topCards.includes(c));
    const highValHole = holeCards.filter(c=>RANK_VAL[c.slice(0,-1)]>=11);
    const hotBoard = boardCards.filter(c=>(model.carryHot[c]||0)>=3);

    let action, sizing, vClass, emoji, reasons=[];

    if (stage === 'preflop') {
      if (topHole.length===2 && isPocketPair) {
        action='RAISE'; sizing='4–5x'; vClass='raise'; emoji='🚀';
        reasons.push(`Pocket ${hRanks[0]}s + both cards are top-bias cards — premium hand`);
        reasons.push('Re-raise if 3-bet, this is your best spot');
      } else if (topHole.length===2) {
        action='RAISE'; sizing='3x'; vClass='raise'; emoji='🚀';
        reasons.push(`${holeCards[0]} + ${holeCards[1]} — both top-bias cards, board will connect`);
        reasons.push('Open raise, call 3-bets in position');
      } else if (isPocketPair && RANK_VAL[hRanks[0]]>=10) {
        action='RAISE'; sizing='3x'; vClass='raise'; emoji='🟢';
        reasons.push(`Pocket ${hRanks[0]}s — strong preflop, raise and protect`);
      } else if (isPocketPair) {
        action='CALL'; sizing='call'; vClass='call'; emoji='🟡';
        reasons.push(`Pocket ${hRanks[0]}s — set mine, see flop cheap`);
        reasons.push('Fold to large raises unless you have odds for implied value');
      } else if (topHole.length===1 && highValHole.length>=1) {
        action='RAISE'; sizing='2.5x'; vClass='raise'; emoji='🟢';
        reasons.push(`${topHole[0]} is a top-bias card with a high kicker — raise`);
      } else if (topHole.length===1) {
        action='CALL'; sizing='call'; vClass='call'; emoji='🟡';
        reasons.push(`${topHole[0]} is a top-bias card but kicker weak — call only`);
      } else if (highValHole.length===2) {
        action='RAISE'; sizing='2.5x'; vClass='raise'; emoji='🟢';
        reasons.push(`${holeCards.join('+')} — both high-value, standard open`);
      } else if (highValHole.length===1) {
        action='CALL'; sizing='call'; vClass='call'; emoji='🟡';
        reasons.push('One high card — call in position, fold to aggression out of position');
      } else {
        action='FOLD'; sizing=null; vClass='fold'; emoji='🔴';
        reasons.push('Neither card is high-value or top-bias for this deck');
        reasons.push('Save your stack — wait for a better hand');
      }

    } else if (stage === 'flop') {
      const s = best ? best.strength : 0;
      if (s >= 7) {
        action='RAISE'; sizing='pot'; vClass='raise'; emoji='🚀';
        reasons.push(`${best.name} on the flop — slow playing risks free draws, build the pot`);
      } else if (s >= 5) {
        action='BET'; sizing='2/3 pot'; vClass='bet'; emoji='🟢';
        reasons.push(`${best.name} — charge anyone drawing against you`);
        if (connecting.length) reasons.push(`${connecting.length} predicted turn cards connect with your hand (★ cards below)`);
      } else if (s >= 3) {
        action='BET'; sizing='1/2 pot'; vClass='bet'; emoji='🟢';
        reasons.push(`${best.name} — bet for value and fold equity`);
        if (topHole.length) reasons.push(`${topHole[0]} likely to stay relevant as board runs out`);
      } else if (s >= 2) {
        if (connecting.length>=2 || topHole.length>=1) {
          action='BET'; sizing='1/3 pot'; vClass='bet'; emoji='🟡';
          reasons.push(`${best.name} — thin bet, predictions show ${connecting.length} turn cards improve you`);
        } else {
          action='CHECK'; sizing=null; vClass='check'; emoji='🟡';
          reasons.push(`${best.name} but turn predictions don't improve you — check/call only`);
        }
      } else if (bestDraw && bestDraw.strength >= 1.5) {
        if (connecting.length>=3) {
          action='BET'; sizing='1/3 pot'; vClass='bet'; emoji='🟡';
          reasons.push(`${bestDraw.name} — semi-bluff, model predicts ${connecting.length} likely outs`);
        } else {
          action='CALL'; sizing='call'; vClass='call'; emoji='🟡';
          reasons.push(`${bestDraw.name} — call cheap streets, don't bloat the pot`);
        }
      } else if (topHole.length>=1 && connecting.length>=2) {
        action='CHECK'; sizing=null; vClass='check'; emoji='🟡';
        reasons.push(`Missed flop but ${topHole[0]} frequently connects on the turn in this game`);
        reasons.push(`Check — ${connecting.length} of top predicted turn cards hit your hand`);
      } else {
        action='FOLD'; sizing=null; vClass='fold'; emoji='🔴';
        reasons.push('Missed the flop, predictions don\'t connect — cut losses');
      }

    } else if (stage === 'turn') {
      const s = best ? best.strength : 0;
      if (s >= 6) {
        action='RAISE'; sizing='pot'; vClass='raise'; emoji='🚀';
        reasons.push(`${best.name} on the turn — push hard, one card left`);
      } else if (s >= 4) {
        action='BET'; sizing='3/4 pot'; vClass='bet'; emoji='🟢';
        reasons.push(`${best.name} — large bet to deny equity, river is last card`);
        if (hotBoard.length) reasons.push(`${hotBoard.join(' ')} is a carry hotspot — opponent may be chasing bleed cards`);
      } else if (s >= 2.5) {
        if (connecting.length>=2) {
          action='BET'; sizing='1/2 pot'; vClass='bet'; emoji='🟡';
          reasons.push(`${best.name} — ${connecting.length} river predictions improve your hand (★ below)`);
        } else {
          action='CHECK'; sizing=null; vClass='check'; emoji='🟡';
          reasons.push(`${best.name} but river predictions don't connect — pot control`);
        }
      } else if (bestDraw) {
        if (connecting.length>=3) {
          action='CALL'; sizing='call'; vClass='call'; emoji='🟡';
          reasons.push(`${bestDraw.name} — ${connecting.length} likely river cards complete you`);
        } else {
          action='FOLD'; sizing=null; vClass='fold'; emoji='🔴';
          reasons.push(`${bestDraw.name} with only ${connecting.length} predicted outs — odds are wrong`);
        }
      } else {
        action='FOLD'; sizing=null; vClass='fold'; emoji='🔴';
        reasons.push('No hand, no draw by the turn — fold to any bet');
      }

    } else {
      // River
      const s = best ? best.strength : 0;
      if (s >= 7) {
        action='RAISE'; sizing='pot'; vClass='raise'; emoji='🚀';
        reasons.push(`${best.name} — max value, go all in if stack sizes allow`);
      } else if (s >= 5) {
        action='BET'; sizing='3/4 pot'; vClass='bet'; emoji='🟢';
        reasons.push(`${best.name} — strong river, value bet big`);
      } else if (s >= 3) {
        action='BET'; sizing='1/3 pot'; vClass='bet'; emoji='🟡';
        reasons.push(`${best.name} — thin value bet, take what you can get`);
      } else if (s >= 2) {
        action='CHECK'; sizing=null; vClass='check'; emoji='🟡';
        reasons.push(`${best.name} — marginal hand, check and call small bets only`);
        reasons.push('Betting gets you called by better hands only');
      } else {
        action='FOLD'; sizing=null; vClass='fold'; emoji='🔴';
        reasons.push('No made hand on the river — fold to any bet');
      }
    }

    if (hotBoard.length && !reasons.some(r=>r.includes('carry')))
      reasons.push(`${hotBoard.join(' ')} on board — carry hotspot, watch next hand`);

    const nextLabel = stage==='preflop'?'Likely flop':stage==='flop'?'Likely turn':stage==='turn'?'Likely river':'Next hand';

    return { action, sizing, vClass, emoji, stage, nextLabel,
      top10: ranked.slice(0,10), connecting, made, draws, best, bestDraw,
      reasons, carryRate: model.carryRate, totalHands: model.totalHands };
  }

  return { parseLog, buildModel, scoreCards, getAdvice, ALL_CARDS };
})();
