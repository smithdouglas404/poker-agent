"""
agents.py — 6-agent LangGraph orchestrator with LIVE weight learning.

The Trend Watcher updates weights AFTER EVERY HAND based on actual win/loss.
No hardcoded weights — initial weights from backtest, then learned online.

Flow per /raw POST (every poll, ~30-100ms):
    DOM_INGEST → SHUFFLE → PLAYER → POSITION → TREND_WATCHER → META_COORDINATOR
    
Flow per /hand POST (after hand completes — triggers learning):
    on_hand_complete() records (signals, hero_won) → Trend Watcher fits new weights
    Next /raw uses the updated weights.

Weights live in agent_weights table. Observations live in learning_observations.
"""
import asyncio, json, os, math, time
from typing import TypedDict, Dict, Any, List, Optional

class CouncilState(TypedDict, total=False):
    game_id: str; session_id: str; hand_num: int
    raw_payload: Dict[str, Any]; player_stats: Dict[str, Any]
    hand_history: List[Dict]; hero_position: Optional[int]; n_seats: int
    dom_facts: Dict[str, Any]; shuffle_signal: Dict[str, Any]
    player_signals: Dict[str, Any]; position_signal: Dict[str, Any]
    trend_update: Dict[str, Any]; current_weights: Dict[str, float]
    final_decision: Dict[str, Any]; timings: Dict[str, float]

# Initial weights from backtest fit on 743 hands (see backtest.py)
INITIAL_WEIGHTS = {"shuffle": 0.1177, "position": -0.0151, "threat": -0.0367, "bias": -0.1471}

def _ensure_tables(db):
    db.execute("""CREATE TABLE IF NOT EXISTS agent_weights (
        game_id TEXT NOT NULL, updated_at TEXT NOT NULL,
        shuffle_w REAL NOT NULL, position_w REAL NOT NULL,
        threat_w REAL NOT NULL, bias_w REAL NOT NULL,
        n_observations INTEGER NOT NULL, avg_loss REAL,
        PRIMARY KEY (game_id, updated_at))""")
    db.execute("""CREATE TABLE IF NOT EXISTS learning_observations (
        game_id TEXT NOT NULL, hand_num INTEGER NOT NULL, ts TEXT NOT NULL,
        shuffle_sig REAL NOT NULL, position_sig REAL NOT NULL,
        threat_sig REAL NOT NULL, hero_won INTEGER NOT NULL, predicted_p REAL,
        PRIMARY KEY (game_id, hand_num))""")
    db.commit()

def get_current_weights(game_id: str) -> Dict[str, float]:
    try:
        from server import get_db
        with get_db() as db:
            _ensure_tables(db)
            row = db.execute(
                "SELECT shuffle_w, position_w, threat_w, bias_w FROM agent_weights "
                "WHERE game_id=? ORDER BY updated_at DESC LIMIT 1", (game_id,)
            ).fetchone()
            if row:
                return {"shuffle": row["shuffle_w"], "position": row["position_w"],
                        "threat": row["threat_w"], "bias": row["bias_w"]}
    except Exception as e:
        print(f"[weights] read failed: {e}")
    return dict(INITIAL_WEIGHTS)

def write_weights(game_id, weights, n_obs, avg_loss):
    try:
        from server import get_db
        from datetime import datetime
        with get_db() as db:
            _ensure_tables(db)
            db.execute(
                "INSERT OR REPLACE INTO agent_weights (game_id, updated_at, shuffle_w, "
                "position_w, threat_w, bias_w, n_observations, avg_loss) VALUES (?,?,?,?,?,?,?,?)",
                (game_id, datetime.utcnow().isoformat(), weights["shuffle"], weights["position"],
                 weights["threat"], weights["bias"], n_obs, avg_loss))
            db.commit()
    except Exception as e:
        print(f"[weights] write failed: {e}")

def record_observation(game_id, hand_num, sig_shuf, sig_pos, sig_threat, hero_won, predicted_p):
    try:
        from server import get_db
        from datetime import datetime
        with get_db() as db:
            _ensure_tables(db)
            db.execute(
                "INSERT OR REPLACE INTO learning_observations (game_id, hand_num, ts, "
                "shuffle_sig, position_sig, threat_sig, hero_won, predicted_p) VALUES (?,?,?,?,?,?,?,?)",
                (game_id, hand_num, datetime.utcnow().isoformat(),
                 sig_shuf, sig_pos, sig_threat, hero_won, predicted_p))
            db.commit()
    except Exception as e:
        print(f"[obs] write failed: {e}")

def _sigmoid(x):
    if x >= 0: return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x); return e / (1.0 + e)

def _norm(raw): return (raw / 50.0) - 1

# ─── AGENT 1: DOM_INGEST ───
async def agent_dom_ingest(state):
    t0 = time.time()
    p = state.get("raw_payload", {})
    facts = {
        "hole_cards": p.get("hole_cards", []), "board_cards": p.get("board_cards", []),
        "board_stage": p.get("board_stage", "preflop"), "pot_size": p.get("pot_size", 0),
        "call_amount": p.get("call_amount", 0), "bet_facing": p.get("bet_facing", 0),
        "hero_stack": p.get("hero_stack", 0), "is_hero_turn": p.get("is_hero_turn", False),
        "can_check": p.get("can_check", False), "aggressor": p.get("aggressor", ""),
        "dealer_pos": p.get("dealer_pos", 0), "blinds": p.get("blinds", {}),
        "active_players": [
            {"name": n, "seat": ps.get("seatPos"), "is_hero": ps.get("isHero", False),
             "stack": ps.get("stackStart", 0), "vpip_count": ps.get("vpipCount", 0),
             "pfr_count": ps.get("pfrCount", 0), "hands_played": ps.get("handsPlayed", 1)}
            for n, ps in (p.get("player_stats") or {}).items()
        ],
    }
    n_seats = len(facts["active_players"]) or 6
    hero = next((ap for ap in facts["active_players"] if ap["is_hero"]), None)
    hero_pos = None
    if hero and facts["dealer_pos"] is not None:
        seats = sorted(ap["seat"] for ap in facts["active_players"] if ap["seat"] is not None)
        if facts["dealer_pos"] in seats and hero["seat"] in seats:
            d_idx = seats.index(facts["dealer_pos"])
            h_idx = seats.index(hero["seat"])
            hero_pos = (h_idx - d_idx) % len(seats)
    state["dom_facts"] = facts; state["hero_position"] = hero_pos; state["n_seats"] = n_seats
    state.setdefault("timings", {})["dom_ingest_ms"] = round((time.time()-t0)*1000, 1)
    return state

# ─── AGENT 2: SHUFFLE_PATTERN ───
async def agent_shuffle_pattern(state):
    t0 = time.time()
    facts = state.get("dom_facts", {})
    history = state.get("hand_history", []) or []
    bleed_score = 0; bled_ranks = []
    if history and facts.get("hole_cards"):
        last_h = history[-1]
        last_cards = (last_h.get("flop") or []) + ([last_h.get("turn")] if last_h.get("turn") else []) \
                   + ([last_h.get("river")] if last_h.get("river") else []) + (last_h.get("hole_cards") or [])
        last_ranks = {c[:-1] for c in last_cards if c}
        for r in [c[:-1] for c in facts["hole_cards"] if c]:
            if r in last_ranks: bled_ranks.append(r); bleed_score += 35
    recent_bleeds = pairs_checked = 0
    for i in range(max(0, len(history)-15), len(history)-1):
        a = ((history[i].get("flop") or []) + ([history[i].get("turn")] if history[i].get("turn") else []) +
             ([history[i].get("river")] if history[i].get("river") else []) + (history[i].get("hole_cards") or []))
        b = ((history[i+1].get("flop") or []) + ([history[i+1].get("turn")] if history[i+1].get("turn") else []) +
             ([history[i+1].get("river")] if history[i+1].get("river") else []) + (history[i+1].get("hole_cards") or []))
        if a and b:
            pairs_checked += 1
            if {x[:-1] for x in a} & {x[:-1] for x in b}: recent_bleeds += 1
    regime_carry = round(100 * recent_bleeds / pairs_checked) if pairs_checked else 0
    state["shuffle_signal"] = {
        "bleed_score": bleed_score, "bled_ranks": bled_ranks, "regime_carry_rate": regime_carry,
        "raw_signal": bleed_score, "confidence": min(100, bleed_score + (20 if regime_carry >= 40 else 0)),
        "recommendation": "PLAY" if bleed_score >= 35 else ("LEAN_PLAY" if regime_carry >= 45 else "NEUTRAL"),
        "reason": f"Bled ranks {bled_ranks} (regime {regime_carry}%)" if bled_ranks else f"No bleed (regime {regime_carry}%)",
    }
    state.setdefault("timings", {})["shuffle_ms"] = round((time.time()-t0)*1000, 1)
    return state

# ─── AGENT 3: PLAYER_BEHAVIOR ───
async def agent_player_behavior(state):
    t0 = time.time()
    facts = state.get("dom_facts", {})
    aggressor = facts.get("aggressor", ""); bet = facts.get("bet_facing", 0)
    bb = (facts.get("blinds") or {}).get("bb", 2) or 2
    profiles = {}
    for ap in facts.get("active_players", []):
        if ap.get("is_hero"): continue
        name = ap.get("name", ""); hands = max(1, ap.get("hands_played", 1))
        vpip = 100 * ap.get("vpip_count", 0) / hands
        pfr = 100 * ap.get("pfr_count", 0) / hands
        if vpip >= 40: style = "loose"
        elif vpip >= 25: style = "balanced"
        elif vpip > 0: style = "tight"
        else: style = "unknown"
        aggression = "aggressive" if pfr >= vpip * 0.6 and pfr > 0 else "passive"
        is_aggr = (name == aggressor); threat = 0
        if is_aggr:
            if style == "tight" and aggression == "aggressive": threat = 80
            elif style == "loose" and aggression == "aggressive": threat = 35
            elif style == "balanced": threat = 55
            elif style == "unknown": threat = 50
            else: threat = 45
        profiles[name] = {"style": style, "aggression": aggression, "is_aggressor_now": is_aggr,
                         "threat_score": threat, "vpip_pct": round(vpip, 1), "pfr_pct": round(pfr, 1),
                         "bet_size_bbs": round(bet / bb, 1) if is_aggr and bet else 0}
    max_threat = max((p["threat_score"] for p in profiles.values()), default=0)
    threat_player = next((n for n, p in profiles.items() if p["threat_score"] == max_threat and max_threat > 0), None)
    state["player_signals"] = {
        "profiles": profiles, "max_threat_score": max_threat, "threat_player": threat_player,
        "raw_signal": max_threat, "confidence": max_threat,
        "recommendation": "FOLD_MARGINAL" if max_threat >= 75 else ("CAUTIOUS" if max_threat >= 50 else "NEUTRAL"),
        "reason": f"{threat_player} → threat {max_threat}" if threat_player else "No active threat",
    }
    state.setdefault("timings", {})["player_ms"] = round((time.time()-t0)*1000, 1)
    return state

# ─── AGENT 4: POSITION_FLOW ───
async def agent_position_flow(state):
    t0 = time.time()
    history = state.get("hand_history", []) or []
    hero_pos = state.get("hero_position"); n_seats = state.get("n_seats", 6)
    last_winner_pos = None
    for h in reversed(history):
        wp = h.get("winner_position")
        if wp is not None: last_winner_pos = wp; break
    in_zone = False; delta = None; raw = 50
    if hero_pos is not None and last_winner_pos is not None and n_seats:
        delta = (hero_pos - last_winner_pos) % n_seats
        in_zone = delta in (0, 1, 2)
        raw = 68 if in_zone else 32
    bb_walkover = (hero_pos == 2 and state.get("dom_facts", {}).get("can_check"))
    if bb_walkover: raw = max(raw, 60)
    state["position_signal"] = {
        "hero_position": hero_pos, "last_winner_position": last_winner_pos,
        "delta_from_winner": delta, "in_zone": in_zone, "bb_walkover": bb_walkover,
        "raw_signal": raw, "confidence": raw,
        "recommendation": "PLAY" if (in_zone or bb_walkover) else (
                          "FOLD_MARGINAL" if delta is not None and delta >= 4 else "NEUTRAL"),
        "reason": (f"Hero +{delta} from last winner (zone: {in_zone})" if delta is not None else "No prior winner")
                  + (" + BB walkover" if bb_walkover else ""),
    }
    state.setdefault("timings", {})["position_ms"] = round((time.time()-t0)*1000, 1)
    return state

# ─── AGENT 5: TREND_WATCHER (loads current weights, reports learning state) ───
async def agent_trend_watcher(state):
    t0 = time.time()
    weights = get_current_weights(state["game_id"])
    state["current_weights"] = weights
    try:
        from server import get_db
        with get_db() as db:
            _ensure_tables(db)
            row = db.execute(
                "SELECT n_observations, avg_loss, updated_at FROM agent_weights "
                "WHERE game_id=? ORDER BY updated_at DESC LIMIT 1", (state["game_id"],)
            ).fetchone()
            if row:
                state["trend_update"] = {
                    "weights": weights, "n_observations": row["n_observations"],
                    "avg_loss": row["avg_loss"], "last_updated": row["updated_at"],
                    "status": "live (auto-updates per hand)" if row["n_observations"] >= 5 else "warming up",
                }
            else:
                state["trend_update"] = {"weights": weights, "n_observations": 0,
                                          "status": "using initial weights from 743-hand backtest"}
    except Exception as e:
        state["trend_update"] = {"weights": weights, "error": str(e)}
    state.setdefault("timings", {})["trend_ms"] = round((time.time()-t0)*1000, 1)
    return state

# ─── AGENT 6: META_COORDINATOR ───
async def agent_meta_coordinator(state):
    t0 = time.time()
    facts = state.get("dom_facts", {})
    shuffle = state.get("shuffle_signal", {}); players = state.get("player_signals", {})
    position = state.get("position_signal", {}); weights = state.get("current_weights", INITIAL_WEIGHTS)
    x_s = _norm(shuffle.get("raw_signal", 0))
    x_p = _norm(position.get("raw_signal", 50))
    x_t = _norm(players.get("raw_signal", 0))
    z = weights["shuffle"]*x_s + weights["position"]*x_p + weights["threat"]*x_t + weights["bias"]
    p_win = _sigmoid(z)
    bet = facts.get("bet_facing", 0); can_check = facts.get("can_check", False)
    bb = (facts.get("blinds") or {}).get("bb", 2) or 2
    if p_win >= 0.40:
        action, vc, em = ("RAISE", "raise", "🚀") if bet == 0 else ("CALL", "call", "🟢")
    elif p_win >= 0.25:
        if can_check: action, vc, em = ("CHECK", "check", "🟡")
        elif bet > 0 and bet <= 4*bb: action, vc, em = ("CALL", "call", "🟡")
        else: action, vc, em = ("FOLD", "fold", "🔴")
    else:
        action, vc, em = ("CHECK", "check", "🟡") if can_check else ("FOLD", "fold", "🔴")
    reasons = [
        f"P(win) = {round(100*p_win,1)}% (council model)",
        f"Shuffle: {shuffle.get('reason','—')}",
        f"Player: {players.get('reason','—')}",
        f"Position: {position.get('reason','—')}",
        f"Weights: shuf:{round(weights['shuffle'],3)} pos:{round(weights['position'],3)} thr:{round(weights['threat'],3)} b:{round(weights['bias'],3)}",
    ]
    state["final_decision"] = {
        "action": action, "vClass": vc, "emoji": em, "p_win": round(p_win, 4),
        "weights_used": weights,
        "agent_outputs": {"shuffle": shuffle, "player": players, "position": position},
        "reasons": reasons,
    }
    state.setdefault("timings", {})["meta_ms"] = round((time.time()-t0)*1000, 1)
    state["timings"]["total_ms"] = sum(state["timings"].values())
    return state

# ─── Build graph ───
def build_council_graph():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        print("[agents] langgraph unavailable — council disabled"); return None
    g = StateGraph(CouncilState)
    g.add_node("dom_ingest", agent_dom_ingest)
    g.add_node("shuffle_pattern", agent_shuffle_pattern)
    g.add_node("player_behavior", agent_player_behavior)
    g.add_node("position_flow", agent_position_flow)
    g.add_node("trend_watcher", agent_trend_watcher)
    g.add_node("meta_coordinator", agent_meta_coordinator)
    g.set_entry_point("dom_ingest")
    g.add_edge("dom_ingest", "shuffle_pattern")
    g.add_edge("shuffle_pattern", "player_behavior")
    g.add_edge("player_behavior", "position_flow")
    g.add_edge("position_flow", "trend_watcher")
    g.add_edge("trend_watcher", "meta_coordinator")
    g.add_edge("meta_coordinator", END)
    return g.compile()

COUNCIL_GRAPH = None
def get_council_graph():
    global COUNCIL_GRAPH
    if COUNCIL_GRAPH is None: COUNCIL_GRAPH = build_council_graph()
    return COUNCIL_GRAPH

# ─── Public entry points (called from server.py) ───
async def run_council(game_id, session_id, hand_num, raw_payload, hand_history):
    g = get_council_graph()
    if not g: return {"ok": False, "error": "langgraph unavailable"}
    initial = {"game_id": game_id, "session_id": session_id, "hand_num": hand_num,
               "raw_payload": raw_payload, "player_stats": raw_payload.get("player_stats") or {},
               "hand_history": hand_history, "n_seats": 6}
    try:
        result = await g.ainvoke(initial)
        return {
            "ok": True, "decision": result.get("final_decision"),
            "agent_outputs": {
                "dom_ingest": result.get("dom_facts"),
                "shuffle_pattern": result.get("shuffle_signal"),
                "player_behavior": result.get("player_signals"),
                "position_flow": result.get("position_signal"),
                "trend_watcher": result.get("trend_update"),
            },
            "current_weights": result.get("current_weights"),
            "timings": result.get("timings"), "hand_num": hand_num,
        }
    except Exception as e:
        print(f"[council] error: {e}"); return {"ok": False, "error": str(e)}

def update_weights_from_history(game_id, lookback=50):
    """Trend Watcher's heavy work — fits weights via gradient descent on recent hands."""
    try:
        from server import get_db
        with get_db() as db:
            _ensure_tables(db)
            rows = db.execute(
                "SELECT shuffle_sig, position_sig, threat_sig, hero_won FROM learning_observations "
                "WHERE game_id=? ORDER BY hand_num DESC LIMIT ?", (game_id, lookback)
            ).fetchall()
        if len(rows) < 5:
            return {"updated": False, "reason": f"need >=5 obs, have {len(rows)}"}
        weights = get_current_weights(game_id)
        lr = 0.01; n = len(rows)
        gs = gp = gt = gb = 0.0; loss = 0.0
        for r in rows:
            xs, xp, xt = _norm(r["shuffle_sig"]), _norm(r["position_sig"]), _norm(r["threat_sig"])
            y = r["hero_won"]
            z = weights["shuffle"]*xs + weights["position"]*xp + weights["threat"]*xt + weights["bias"]
            p = _sigmoid(z); err = p - y
            gs += err*xs; gp += err*xp; gt += err*xt; gb += err
            eps = 1e-9
            loss += -(y*math.log(p+eps) + (1-y)*math.log(1-p+eps))
        weights["shuffle"]  -= lr * gs / n
        weights["position"] -= lr * gp / n
        weights["threat"]   -= lr * gt / n
        weights["bias"]     -= lr * gb / n
        write_weights(game_id, weights, n, loss/n)
        return {"updated": True, "n_observations": n, "avg_loss": round(loss/n, 4),
                "new_weights": {k: round(v, 4) for k, v in weights.items()}}
    except Exception as e:
        return {"updated": False, "error": str(e)}

def on_hand_complete(game_id, hand_num, hero_won, last_council_signals=None):
    """Called from /hand endpoint after each completed hand."""
    if not last_council_signals:
        return {"recorded": False, "reason": "no signals provided"}
    sig_shuf = last_council_signals.get("shuffle_pattern", {}).get("raw_signal", 0)
    sig_pos = last_council_signals.get("position_flow", {}).get("raw_signal", 50)
    sig_threat = last_council_signals.get("player_behavior", {}).get("raw_signal", 0)
    p_win = (last_council_signals.get("decision") or {}).get("p_win", 0.5)
    record_observation(game_id, hand_num, sig_shuf, sig_pos, sig_threat, 1 if hero_won else 0, p_win)
    return {"recorded": True, "hand_num": hand_num, "hero_won": hero_won,
            "weight_update": update_weights_from_history(game_id)}
