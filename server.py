"""
Poker Agent Server v32 — Railway deployment
FastAPI + SQLite + LangGraph + Letta Cloud + Mem0 + Claude
"""

import sqlite3, json, os, asyncio
import httpx
from datetime import datetime
from typing import Dict, Set, TypedDict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import anthropic
from mem0 import MemoryClient

# ── LangGraph — optional, falls back to direct Claude if not available ─────────
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("[startup] langgraph not available — falling back to direct Claude")

# ── Letta Cloud — optional, falls back to direct Claude if not available ───────
try:
    from letta_client import Letta as LettaClient
    LETTA_AVAILABLE = True
except ImportError:
    try:
        from letta import create_client as letta_create_client
        LETTA_AVAILABLE = True
    except ImportError:
        LETTA_AVAILABLE = False
        print("[startup] letta not available — falling back to direct Claude")

from contextlib import asynccontextmanager

_queue_running = False  # must be defined before lifespan

@asynccontextmanager
async def lifespan(app):
    global _queue_running
    if not _queue_running:
        _queue_running = True  # set before creating task to prevent double-start
        asyncio.create_task(_process_claude_queue())
    yield

app = FastAPI(lifespan=lifespan)

def extract_player_id(full_name: str) -> str:
    import re
    m = re.search(r'@\s*([A-Za-z0-9_-]+)\s*$', full_name or '')
    return m.group(1) if m else full_name

def extract_display_name(full_name: str) -> str:
    import re
    m = re.match(r'^(.+?)\s*@', full_name or '')
    return m.group(1).strip() if m else full_name

def upsert_global_player(db, player_name: str, game_id: str, p: dict):
    player_id = extract_player_id(player_name)
    display_name = extract_display_name(player_name)
    existing = db.execute("SELECT all_names, last_game_id FROM global_players WHERE player_id=?", (player_id,)).fetchone()
    if existing:
        all_names = json.loads(existing["all_names"] or "[]")
        is_new_game = existing["last_game_id"] != game_id
    else:
        all_names = []
        is_new_game = True
    if display_name not in all_names:
        all_names.append(display_name)
    db.execute(
        """INSERT INTO global_players
            (player_id,display_name,all_names,first_game_id,last_game_id,games_played,
             total_hands,vpip_count,pfr_count,threebet_count,total_agg,total_passive,
             win_count,showdown_count,showdown_wins,allin_count)
            VALUES(?,?,?,?,?,1,1,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(player_id) DO UPDATE SET
                display_name=excluded.display_name,
                all_names=?,
                last_seen=datetime('now'),
                last_game_id=excluded.last_game_id,
                games_played=global_players.games_played+CASE WHEN ? THEN 1 ELSE 0 END,
                total_hands=global_players.total_hands+1,
                vpip_count=global_players.vpip_count+excluded.vpip_count,
                pfr_count=global_players.pfr_count+excluded.pfr_count,
                threebet_count=global_players.threebet_count+excluded.threebet_count,
                total_agg=global_players.total_agg+excluded.total_agg,
                total_passive=global_players.total_passive+excluded.total_passive,
                win_count=global_players.win_count+excluded.win_count,
                showdown_count=global_players.showdown_count+excluded.showdown_count,
                showdown_wins=global_players.showdown_wins+excluded.showdown_wins,
                allin_count=global_players.allin_count+excluded.allin_count,
                ts_updated=datetime('now')""",
        (player_id, display_name, json.dumps(all_names), game_id, game_id,
         p.get("vpip",0), p.get("pfr",0), p.get("three_bet",0),
         p.get("aggressive_acts",0), p.get("passive_acts",0),
         p.get("won",0), p.get("went_to_showdown",0),
         1 if (p.get("won",0) and p.get("went_to_showdown",0)) else 0,
         p.get("all_in",0),
         json.dumps(all_names), 1 if is_new_game else 0)
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: tighten to Railway domain before public launch
    allow_methods=["*"],
    allow_headers=["*"],
)

MEM0_API_KEY  = os.environ.get("MEM0_API_KEY", "")
ANTHROPIC_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
LETTA_API_KEY    = os.environ.get("LETTA_API_KEY", "")
OPENROUTER_KEY   = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "z-ai/glm-5.1"

def call_glm(prompt: str, max_tokens: int = 400, system: str = "") -> str:
    """Call GLM-5.1 via OpenRouter. Returns empty string on failure."""
    key = OPENROUTER_KEY or ANTHROPIC_KEY
    if not key:
        return ""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://pokernow-advisor.railway.app",
        "X-Title":       "PokerNow Advisor",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        r = httpx.post(OPENROUTER_URL, headers=headers, json={
            "model": OPENROUTER_MODEL,
            "max_tokens": max_tokens,
            "messages": messages,
        }, timeout=20)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[GLM-5.1] call failed: {e}")
        return ""
DB_PATH       = os.getenv("DB_PATH", "poker.db")
# USER_ID is now the game_id — each game has its own isolated Mem0 namespace.
# No global USER_ID. Pass game_id wherever Mem0 is called.

# ── Letta client ──────────────────────────────────────────────────────────────
def get_letta():
    if not LETTA_AVAILABLE:
        return None
    try:
        if 'LettaClient' in dir():
            return LettaClient(token=LETTA_API_KEY)
        else:
            return letta_create_client(token=LETTA_API_KEY)
    except Exception as e:
        print(f"[letta] client error: {e}")
        return None

# Cache of game_id → letta agent_id so we don't recreate on every hand
_letta_agents: Dict[str, str] = {}

def get_or_create_letta_agent(game_id: str) -> str:
    """Get existing Letta agent for this game or create a new one."""
    if not LETTA_AVAILABLE or not LETTA_API_KEY:
        return ""
    if game_id in _letta_agents:
        return _letta_agents[game_id]
    try:
        client = get_letta()
        if not client:
            return ""
        agents = client.agents.list()
        existing = next((a for a in agents if a.name == f"poker_{game_id}"), None)
        if existing:
            _letta_agents[game_id] = existing.id
            # Persist to DB so it survives restarts
            try:
                with get_db() as db:
                    db.execute(
                        "INSERT OR REPLACE INTO letta_agents (game_id, agent_id) VALUES (?,?)",
                        (game_id, existing.id)
                    )
            except Exception as e:
                print(f"[letta] Warning: could not persist agent_id to DB: {e}")
            return existing.id
        agent = client.agents.create(
            name=f"poker_{game_id}",
            model="claude-sonnet-4-5",
            memory_blocks=[
                {"label": "game_context", "value": f"Poker game ID: {game_id}. Tracks patterns, danger players, carry rates for this game."},
                {"label": "player_notes", "value": "No players tracked yet."},
                {"label": "carry_patterns", "value": "No carry patterns detected yet."},
            ],
            system="""You are a persistent poker analysis agent for a PokerNow game.
Track patterns across every hand. After each hand update your memory.
Return structured JSON with weight_updates affecting the betting algorithm.
Always base analysis on actual data only."""
        )
        _letta_agents[game_id] = agent.id
        # Persist to DB so it survives restarts
        try:
            with get_db() as db:
                db.execute(
                    "INSERT OR REPLACE INTO letta_agents (game_id, agent_id) VALUES (?,?)",
                    (game_id, agent.id)
                )
        except Exception as e:
            print(f"[letta] Warning: could not persist agent_id to DB: {e}")
        print(f"[letta] Created agent for game {game_id}: {agent.id}")
        return agent.id
    except Exception as e:
        print(f"[letta] Error getting/creating agent: {e}")
        return ""


async def _auto_narrative(game_id: str, player_name: str):
    """Auto-generate Claude narrative for a player at milestone hand counts."""
    try:
        req = NarrativeRequest(game_id=game_id, player_name=player_name)
        result = await player_narrative(req)
        if result.get("ok"):
            await manager.broadcast(game_id, {
                "type": "player_narrative",
                "player": player_name,
                "narrative": result.get("narrative", ""),
            })
            print(f"[narrative] Auto-generated for {player_name}: {result.get('narrative','')[:80]}")
    except Exception as e:
        print(f"[narrative] Auto-generate error for {player_name}: {e}")

# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, Set[WebSocket]] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        if session_id not in self.active:
            self.active[session_id] = set()
        self.active[session_id].add(ws)

    def disconnect(self, session_id: str, ws: WebSocket):
        if session_id in self.active:
            self.active[session_id].discard(ws)

    async def broadcast(self, session_id: str, data: dict):
        if session_id not in self.active:
            return
        dead = set()
        for ws in self.active[session_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active[session_id].discard(ws)

manager = ConnectionManager()

# Latest raw state per session (for reconnect)
_latest_state:   Dict[str, dict] = {}
_active_game_id: str  = ""   # most recently active game
_model_cache:    dict = {}   # game_id → {model, version, hand_count}
MODEL_REBUILD_EVERY = 1      # rebuild every hand — model.hands=[] so no perf cost

# ── LangGraph — Between-Hands Workflow ───────────────────────────────────────
# Replaces the manual _run_between_hands prompt engineering.
# Each node is a pure function. Edges are conditional on what was detected.

class HandAnalysisState(TypedDict):
    # Input
    game_id:      str
    session_id:   str
    hand_num:     int
    hole_cards:   list
    board:        list
    hero_won:     bool
    hero_folded:  bool
    all_in:       bool
    away_mode:    bool
    pot:          int
    shown_hands:  dict
    player_stats: dict
    # Computed through graph
    live_stats:   dict
    memories:     list
    letta_response: str
    weight_updates: dict
    next_hand:    str
    danger_players: list
    # Control flow
    has_danger:   bool
    has_carry:    bool
    is_allin:     bool

def node_load_context(state: HandAnalysisState) -> dict:
    """Node 1: Load live stats from DB."""
    try:
        with get_db() as db:
            total   = db.execute("SELECT COUNT(*) FROM hands WHERE game_id=?", (state["game_id"],)).fetchone()[0]
            won     = db.execute("SELECT COUNT(*) FROM hands WHERE game_id=? AND hero_won=1", (state["game_id"],)).fetchone()[0]
            allin_t = db.execute("SELECT COUNT(*) FROM hands WHERE game_id=? AND all_in=1", (state["game_id"],)).fetchone()[0]
            allin_w = db.execute("SELECT COUNT(*) FROM hands WHERE game_id=? AND all_in=1 AND hero_won=1", (state["game_id"],)).fetchone()[0]
            recent  = db.execute("SELECT hole_cards, board, hero_won, all_in FROM hands WHERE game_id=? ORDER BY id DESC LIMIT 10", (state["game_id"],)).fetchall()
            recent50 = [dict(r) for r in db.execute("SELECT hole_cards, board FROM hands WHERE game_id=? ORDER BY id DESC LIMIT 50", (state["game_id"],)).fetchall()]
            danger_rows = db.execute(
                """SELECT player_name FROM player_stats
                   WHERE game_id=? AND showdown_count >= 5
                   AND CAST(showdown_wins AS FLOAT)/MAX(showdown_count,1) >= 0.60""",
                (state["game_id"],)
            ).fetchall()

        live_wr  = round(won/total*100, 1)      if total  > 0 else 0
        allin_wr = round(allin_w/allin_t*100,1) if allin_t > 0 else 0

        # Carry rate from last 50 hands (already loaded above)
        carry_pairs = sum(
            1 for i in range(1, len(recent50))
            if (set(json.loads(recent50[i-1].get("hole_cards","[]") or "[]") +
                    json.loads(recent50[i-1].get("board","[]") or "[]")) &  # ALL prev hand cards
               set(json.loads(recent50[i].get("hole_cards","[]") or "[]") +
                   json.loads(recent50[i].get("board","[]") or "[]")) )  # ALL next hand cards
        )
        carry_rate = round(carry_pairs / max(len(recent50)-1, 1) * 100, 1)

        danger = [r["player_name"] for r in danger_rows]

        return {
            "live_stats": {
                "total_hands": total, "win_rate": live_wr,
                "allin_wr": allin_wr, "carry_rate": carry_rate,
                "recent": [dict(r) for r in recent],
            },
            "danger_players": danger,
            "has_danger": len(danger) > 0,
            "has_carry":  carry_rate > 35,
            "is_allin":   state["all_in"],
        }
    except Exception as e:
        print(f"[graph:load_context] {e}")
        return {"live_stats": {}, "danger_players": [], "has_danger": False, "has_carry": False, "is_allin": False}

def node_mem0_retrieve(state: HandAnalysisState) -> dict:
    """Node 2: Retrieve relevant memories from Mem0 for this game."""
    try:
        mem0  = get_mem0()
        # Semantic query — Mem0 needs context not raw cards
        stage = 'preflop' if not state['board'] else ('flop' if len(state['board'])==3 else ('turn' if len(state['board'])==4 else 'river'))
        result_str = 'won' if state['hero_won'] else ('folded' if state['hero_folded'] else 'lost')
        carry = state.get('live_stats', {}).get('carry_rate', 0)
        danger = state.get('danger_players', [])
        query = (f"{stage} hand, hero {result_str}, carry rate {carry}%, "
                 f"{'danger player active' if danger else 'no danger'}, "
                 f"all-in: {state['all_in']}, pot: {state['pot']}")
        results = mem0.search(query=query, filters={"user_id": state["game_id"]}, limit=8)
        memories = [m.get("memory","") for m in (results if isinstance(results, list) else [])]
        return {"memories": memories}
    except Exception as e:
        print(f"[graph:mem0_retrieve] {e}")
        return {"memories": []}

def node_letta_reason(state: HandAnalysisState) -> dict:
    """Node 3: Send hand to Letta agent — falls back to direct Claude if Letta unavailable."""
    # Try Letta first
    if LETTA_AVAILABLE and LETTA_API_KEY:
        try:
            agent_id = get_or_create_letta_agent(state["game_id"])
            if agent_id:
                client = get_letta()
                ls     = state["live_stats"]
                result_str = "WON" if state["hero_won"] else ("FOLDED" if state["hero_folded"] else "LOST")
                shown  = ", ".join(f"{p}: {' '.join(c)}" for p,c in state["shown_hands"].items()) if state["shown_hands"] else "none"

                message = f"""Hand #{state['hand_num']} just completed.

RESULT: {result_str} | Pot: {state['pot']} | All-in: {state['all_in']}
Hero held: {' '.join(state['hole_cards']) if state['hole_cards'] else 'away'}
Board: {' '.join(state['board']) if state['board'] else 'no board'}
Opponents showed: {shown}

LIVE STATS (actual DB values):
- Hands: {ls.get('total_hands',0)} | Win rate: {ls.get('win_rate',0)}%
- All-in win rate: {ls.get('allin_wr',0)}% | Carry rate: {ls.get('carry_rate',0)}%
- Danger players: {state['danger_players'] or 'none'}

MEM0 MEMORIES:
{chr(10).join(f'- {m}' for m in state['memories']) if state['memories'] else '- none yet'}

Update your memory. Respond in JSON only:
{{
  "nextHand": "1-2 sentence forward-looking instruction for next hand",
  "weight_updates": {{"carryBoost": 0.0, "dangerWarning": false, "allinWarning": false, "hotCards": []}}
}}"""

                response = client.agents.messages.create(
                    agent_id=agent_id,
                    messages=[{"role": "user", "content": message}]
                )
                letta_text = ""
                for msg in response.messages:
                    # Try multiple field names — Letta API varies by version
                    for field in ['text', 'content', 'message', 'assistant_message']:
                        val = getattr(msg, field, None)
                        if val and isinstance(val, str) and len(val.strip()) > 0:
                            letta_text = val.strip(); break
                    if letta_text: break
                if letta_text:
                    print(f"[letta] Got response ({len(letta_text)} chars)")
                    return {"letta_response": letta_text}
                else:
                    print(f"[letta] Empty response — {len(response.messages)} messages received")
        except Exception as e:
            print(f"[graph:letta_reason] Letta failed, falling back to Claude: {e}")

    # Fallback: direct Claude call
    try:
        ls = state["live_stats"]
        result_str = "WON" if state["hero_won"] else ("FOLDED" if state["hero_folded"] else "LOST")
        shown = ", ".join(f"{p}: {' '.join(c)}" for p,c in state["shown_hands"].items()) if state["shown_hands"] else "none"
        prompt = f"""Hand #{state['hand_num']}: {result_str} | Pot: {state['pot']} | All-in: {state['all_in']}
Hero: {' '.join(state['hole_cards'])} Board: {' '.join(state['board'])}
Opponents: {shown}
Stats: WR={ls.get('win_rate',0)}% Carry={ls.get('carry_rate',0)}% Danger={state['danger_players']}
Memories: {'; '.join(state['memories'][:3]) if state['memories'] else 'none'}

Respond JSON only: {{"nextHand": "instruction", "weight_updates": {{"carryBoost": 0.0, "dangerWarning": false, "allinWarning": false, "hotCards": []}}}}"""
        text = call_glm(prompt, max_tokens=300)
        return {"letta_response": text}
    except Exception as e:
        print(f"[graph:letta_reason] Claude fallback also failed: {e}")
        return {"letta_response": ""}

def node_parse_updates(state: HandAnalysisState) -> dict:
    """Node 4: Parse Letta response into weight_updates + nextHand."""
    try:
        raw = state["letta_response"].strip().replace('```json','').replace('```','').strip()
        # Find JSON block
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])
        else:
            parsed = {}
        return {
            "next_hand":      parsed.get("nextHand", ""),
            "weight_updates": parsed.get("weight_updates", {
                "carryBoost": 0.0, "dangerWarning": False,
                "allinWarning": False, "hotCards": []
            })
        }
    except Exception as e:
        print(f"[graph:parse_updates] {e}")
        return {
            "next_hand": "",
            "weight_updates": {"carryBoost": 0.0, "dangerWarning": False, "allinWarning": False, "hotCards": []}
        }

def node_apply_danger(state: HandAnalysisState) -> dict:
    """Node 5a: Danger player detected — force dangerWarning on."""
    wu = dict(state.get("weight_updates", {}))
    wu["dangerWarning"] = True
    return {"weight_updates": wu}

def node_apply_carry(state: HandAnalysisState) -> dict:
    """Node 5b: Strong carry pattern — boost carry signal."""
    wu = dict(state.get("weight_updates", {}))
    ls = state.get("live_stats", {})
    carry = ls.get("carry_rate", 0)
    wu["carryBoost"] = max(wu.get("carryBoost", 0), round(carry / 50, 2))
    return {"weight_updates": wu}

def node_apply_allin(state: HandAnalysisState) -> dict:
    """Node 5c: All-in this hand — flag allinWarning."""
    wu = dict(state.get("weight_updates", {}))
    wu["allinWarning"] = True
    return {"weight_updates": wu}

def node_store_memory(state: HandAnalysisState) -> dict:
    """Node 6: Store analysis back to Mem0 and DB.
    Always stores — weight_updates without nextHand are still insights worth keeping.
    """
    try:
        next_hand      = state.get("next_hand", "")
        weight_updates = state.get("weight_updates", {})
        danger         = state.get("danger_players", [])

        # Build a meaningful memory even if nextHand is empty
        wu_summary = []
        if weight_updates.get("carryBoost", 0) > 0:
            wu_summary.append(f"carry boost {weight_updates['carryBoost']:.1f}")
        if weight_updates.get("dangerWarning"):
            wu_summary.append(f"danger: {danger}")
        if weight_updates.get("allinWarning"):
            wu_summary.append("all-in pattern")
        if weight_updates.get("hotCards"):
            wu_summary.append(f"hot cards: {weight_updates['hotCards']}")

        memory_content = next_hand
        if not memory_content and wu_summary:
            memory_content = f"Hand #{state['hand_num']}: {', '.join(wu_summary)}"

        if memory_content:
            mem0 = get_mem0()
            mem0.add(
                f"Hand #{state['hand_num']} insight: {memory_content}",
                user_id=state["game_id"],
                metadata={"type": "between_hands", "hand_num": state["hand_num"],
                          "has_warning": bool(wu_summary)}
            )

        with get_db() as db:
            db.execute(
                """INSERT INTO claude_log
                (game_id,session_id,hand_num,prompt,response,next_hand,
                 weight_updates,carry_rate,live_wr,danger_players)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (state["game_id"], state["session_id"], state["hand_num"],
                 f"LangGraph+Letta hand #{state['hand_num']}",
                 state.get("letta_response", ""),
                 next_hand,
                 json.dumps(state.get("weight_updates", {})),
                 state.get("live_stats", {}).get("carry_rate", 0),
                 state.get("live_stats", {}).get("win_rate", 0),
                 json.dumps(state.get("danger_players", [])))
            )
    except Exception as e:
        print(f"[graph:store_memory] {e}")
    return {}

# ── Build the LangGraph (only if langgraph is installed) ─────────────────────
def build_poker_graph():
    if not LANGGRAPH_AVAILABLE:
        print("[startup] LangGraph not available — graph disabled")
        return None

    graph = StateGraph(HandAnalysisState)

    # Add nodes
    graph.add_node("load_context",   node_load_context)
    graph.add_node("mem0_retrieve",  node_mem0_retrieve)
    graph.add_node("letta_reason",   node_letta_reason)
    graph.add_node("parse_updates",  node_parse_updates)
    graph.add_node("apply_danger",   node_apply_danger)
    graph.add_node("apply_carry",    node_apply_carry)
    graph.add_node("apply_allin",    node_apply_allin)
    graph.add_node("store_memory",   node_store_memory)

    # Linear flow: load → mem0 → letta → parse
    graph.set_entry_point("load_context")
    graph.add_edge("load_context",  "mem0_retrieve")
    graph.add_edge("mem0_retrieve", "letta_reason")
    graph.add_edge("letta_reason",  "parse_updates")

    # Conditional edges after parse — apply modifiers based on what was detected
    def route_after_parse(state):
        if state.get("has_danger"):   return "apply_danger"
        if state.get("has_carry"):    return "apply_carry"
        if state.get("is_allin"):     return "apply_allin"
        return "store_memory"

    graph.add_conditional_edges("parse_updates", route_after_parse, {
        "apply_danger":  "apply_danger",
        "apply_carry":   "apply_carry",
        "apply_allin":   "apply_allin",
        "store_memory":  "store_memory",
    })

    # After each modifier, check if more apply
    def route_after_danger(state):
        if state.get("has_carry"):  return "apply_carry"
        if state.get("is_allin"):   return "apply_allin"
        return "store_memory"

    def route_after_carry(state):
        if state.get("is_allin"): return "apply_allin"
        return "store_memory"

    graph.add_conditional_edges("apply_danger", route_after_danger, {
        "apply_carry":  "apply_carry",
        "apply_allin":  "apply_allin",
        "store_memory": "store_memory",
    })
    graph.add_conditional_edges("apply_carry", route_after_carry, {
        "apply_allin":  "apply_allin",
        "store_memory": "store_memory",
    })
    graph.add_edge("apply_allin",   "store_memory")
    graph.add_edge("store_memory",  END)

    return graph.compile()

# Compile once at startup
poker_graph = build_poker_graph()

# ── DB ────────────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS hands (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id     TEXT,
            session_id  TEXT,
            hand_num    INTEGER,
            hole_cards  TEXT,
            board       TEXT,
            hero_won     INTEGER DEFAULT 0,
            hero_folded  INTEGER DEFAULT 0,
            all_in       INTEGER DEFAULT 0,
            pot          INTEGER DEFAULT 0,
            away_mode    INTEGER DEFAULT 0,
            shown_hands  TEXT    DEFAULT '{}',
            player_stats TEXT    DEFAULT '{}',
            player_count INTEGER DEFAULT 0,
            engine_action TEXT   DEFAULT '',
            engine_confidence INTEGER DEFAULT 0,
            engine_hand_strength REAL DEFAULT 0,
            ts           TEXT    DEFAULT (datetime('now')),
            UNIQUE(game_id, hand_num)
        );

        CREATE TABLE IF NOT EXISTS letta_agents (
            game_id   TEXT PRIMARY KEY,
            agent_id  TEXT NOT NULL,
            ts        TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS mem0_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id     TEXT,
            session_id  TEXT,
            hand_num    INTEGER,
            memory_text TEXT,
            memory_id   TEXT,
            ts          TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS claude_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id        TEXT,
            session_id     TEXT,
            hand_num       INTEGER,
            prompt         TEXT,
            response       TEXT,
            next_hand      TEXT,
            weight_updates TEXT DEFAULT '{}',
            carry_rate     REAL DEFAULT 0,
            live_wr        REAL DEFAULT 0,
            danger_players TEXT DEFAULT '[]',
            ts             TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS analysis_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id         TEXT,
            ts              TEXT DEFAULT (datetime('now')),
            hands_at_time   INTEGER,
            mem0_count      INTEGER,
            win_rate        REAL,
            carry_rate      REAL,
            allin_wr        REAL,
            street_wins     TEXT,
            street_totals   TEXT,
            memories_snapshot TEXT,
            claude_narrative  TEXT
        );

        CREATE TABLE IF NOT EXISTS player_hands (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id          TEXT NOT NULL,
            hand_num         INTEGER,
            player_name      TEXT,
            seat_pos         INTEGER,
            is_hero          INTEGER DEFAULT 0,
            stack_start      INTEGER DEFAULT 0,
            preflop_bet      INTEGER DEFAULT 0,
            flop_bet         INTEGER DEFAULT 0,
            turn_bet         INTEGER DEFAULT 0,
            river_bet        INTEGER DEFAULT 0,
            vpip             INTEGER DEFAULT 0,
            pfr              INTEGER DEFAULT 0,
            three_bet        INTEGER DEFAULT 0,
            four_bet_plus    INTEGER DEFAULT 0,
            aggressive_acts  INTEGER DEFAULT 0,
            passive_acts     INTEGER DEFAULT 0,
            folded           INTEGER DEFAULT 0,
            all_in           INTEGER DEFAULT 0,
            won              INTEGER DEFAULT 0,
            went_to_showdown INTEGER DEFAULT 0,
            saw_flop         INTEGER DEFAULT 0,
            in_steal_position   INTEGER DEFAULT 0,
            attempted_steal     INTEGER DEFAULT 0,
            folded_to_steal     INTEGER DEFAULT 0,
            faced_three_bet     INTEGER DEFAULT 0,
            folded_to_three_bet INTEGER DEFAULT 0,
            made_cbet           INTEGER DEFAULT 0,
            faced_cbet          INTEGER DEFAULT 0,
            folded_to_cbet      INTEGER DEFAULT 0,
            shown_cards      TEXT DEFAULT '[]',
            hand_message     TEXT DEFAULT '',
            dom_wins         INTEGER DEFAULT 0,
            ts               TEXT DEFAULT (datetime('now')),
            UNIQUE(game_id, hand_num, player_name)
        );

        CREATE TABLE IF NOT EXISTS global_players (
            player_id       TEXT PRIMARY KEY,   -- stable @ ID extracted from display name
            display_name    TEXT,               -- most recent display name
            all_names       TEXT DEFAULT '[]',  -- JSON array of all names seen
            first_seen      TEXT DEFAULT (datetime('now')),
            last_seen       TEXT DEFAULT (datetime('now')),
            first_game_id   TEXT,
            last_game_id    TEXT,
            games_played    INTEGER DEFAULT 0,
            total_hands     INTEGER DEFAULT 0,
            vpip_count      INTEGER DEFAULT 0,
            pfr_count       INTEGER DEFAULT 0,
            threebet_count  INTEGER DEFAULT 0,
            total_agg       INTEGER DEFAULT 0,
            total_passive   INTEGER DEFAULT 0,
            win_count       INTEGER DEFAULT 0,
            showdown_count  INTEGER DEFAULT 0,
            showdown_wins   INTEGER DEFAULT 0,
            allin_count     INTEGER DEFAULT 0,
            commentary      TEXT    DEFAULT '',
            notes           TEXT    DEFAULT '',
            ts_updated      TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS player_stats (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id         TEXT NOT NULL,
            player_name     TEXT NOT NULL,
            hands_seen      INTEGER DEFAULT 0,
            vpip_count      INTEGER DEFAULT 0,
            pfr_count       INTEGER DEFAULT 0,
            threebet_count  INTEGER DEFAULT 0,
            fourbet_count   INTEGER DEFAULT 0,
            total_agg       INTEGER DEFAULT 0,
            total_passive   INTEGER DEFAULT 0,
            win_count       INTEGER DEFAULT 0,
            showdown_count  INTEGER DEFAULT 0,
            showdown_wins   INTEGER DEFAULT 0,
            saw_flop_count  INTEGER DEFAULT 0,
            allin_count     INTEGER DEFAULT 0,
            dom_win_count   INTEGER DEFAULT 0,
            steal_opps      INTEGER DEFAULT 0,
            steal_attempts  INTEGER DEFAULT 0,
            fold_to_steal   INTEGER DEFAULT 0,
            fold_to_steal_opps INTEGER DEFAULT 0,
            faced_3bet      INTEGER DEFAULT 0,
            folded_to_3bet  INTEGER DEFAULT 0,
            cbet_opps       INTEGER DEFAULT 0,
            cbet_count      INTEGER DEFAULT 0,
            faced_cbet      INTEGER DEFAULT 0,
            folded_to_cbet  INTEGER DEFAULT 0,
            commentary      TEXT    DEFAULT '',
            ts_updated      TEXT    DEFAULT (datetime('now')),
            UNIQUE(game_id, player_name)
        );
        """)
    # Safe migrations for existing DBs
    migrations = [
        ("hands",        "game_id",           "TEXT"),
        ("mem0_log",     "game_id",           "TEXT"),
        ("claude_log",   "game_id",           "TEXT"),
        ("claude_log",   "weight_updates",    "TEXT DEFAULT '{}'"),
        ("claude_log",   "carry_rate",        "REAL DEFAULT 0"),
        ("claude_log",   "live_wr",           "REAL DEFAULT 0"),
        ("claude_log",   "danger_players",    "TEXT DEFAULT '[]'"),
        ("analysis_log", "game_id",           "TEXT"),
        ("hands",        "player_count",      "INTEGER DEFAULT 0"),
        ("hands",        "shown_hands",       "TEXT DEFAULT '{}'"),
        ("hands",        "player_stats",      "TEXT DEFAULT '{}'"),
        ("claude_log",   "next_hand",         "TEXT"),
        # New player_hands columns
        # Note: UNIQUE(game_id, hand_num, player_name) added in v49 schema — new DBs only
        ("player_hands", "in_steal_position",   "INTEGER DEFAULT 0"),
        ("player_hands", "attempted_steal",      "INTEGER DEFAULT 0"),
        ("player_hands", "folded_to_steal",      "INTEGER DEFAULT 0"),
        ("player_hands", "faced_three_bet",      "INTEGER DEFAULT 0"),
        ("player_hands", "folded_to_three_bet",  "INTEGER DEFAULT 0"),
        ("player_hands", "made_cbet",            "INTEGER DEFAULT 0"),
        ("player_hands", "faced_cbet",           "INTEGER DEFAULT 0"),
        ("player_hands", "folded_to_cbet",       "INTEGER DEFAULT 0"),
        # New player_stats columns
        ("player_stats", "steal_opps",           "INTEGER DEFAULT 0"),
        ("player_stats", "steal_attempts",       "INTEGER DEFAULT 0"),
        ("player_stats", "fold_to_steal",        "INTEGER DEFAULT 0"),
        ("player_stats", "fold_to_steal_opps",   "INTEGER DEFAULT 0"),
        ("player_stats", "faced_3bet",           "INTEGER DEFAULT 0"),
        ("player_stats", "folded_to_3bet",       "INTEGER DEFAULT 0"),
        ("player_stats", "cbet_opps",            "INTEGER DEFAULT 0"),
        ("player_stats", "cbet_count",           "INTEGER DEFAULT 0"),
        ("player_stats", "faced_cbet",           "INTEGER DEFAULT 0"),
        ("player_stats", "folded_to_cbet",       "INTEGER DEFAULT 0"),
        ("player_stats", "player_id",             "TEXT DEFAULT ''"),
        ("player_hands", "player_id",             "TEXT DEFAULT ''"),
        ("hands", "engine_action",         "TEXT DEFAULT ''"),
        ("hands", "engine_confidence",     "INTEGER DEFAULT 0"),
        ("hands", "engine_hand_strength",  "REAL DEFAULT 0"),
    ]
    with get_db() as db:
        for table, col, typedef in migrations:
            try:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")
            except Exception:
                pass

init_db()

# ── Load persisted Letta agent IDs from DB into memory cache ─────────────────
def _load_letta_agents():
    try:
        with get_db() as db:
            rows = db.execute("SELECT game_id, agent_id FROM letta_agents").fetchall()
            for row in rows:
                _letta_agents[row["game_id"]] = row["agent_id"]
            if rows:
                print(f"[letta] Loaded {len(rows)} cached agent IDs from DB")
    except Exception as e:
        print(f"[letta] Could not load agent cache: {e}")

_load_letta_agents()


# ── Model builder — server-side, built from SQLite, no hardcoded thresholds ──
RANKS  = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
SUITS  = ['♠','♥','♦','♣']
ALL_CARDS = [r+s for r in RANKS for s in SUITS]
EARLY_POS = {'hole_1','hole_2','flop_1','flop_2','flop_3'}

def build_model_from_db(game_id: str) -> dict:
    """Build the poker model from THIS game's hands only. Each game is independent."""
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT hole_cards, board, player_count FROM hands "
                "WHERE game_id=? AND hole_cards IS NOT NULL ORDER BY id ASC",
                (game_id,)
            ).fetchall()
    except Exception as e:
        print(f"[model] DB error: {e}")
        return {}

    if not rows:
        return {}

    hands = []
    for row in rows:
        hc = json.loads(row["hole_cards"] or "[]")
        bd = json.loads(row["board"] or "[]")
        pc = row["player_count"] or 0
        if hc:
            hands.append({
                "hole_cards":   hc,
                "flop":         bd[:3] if len(bd) >= 3 else [],
                "turn":         [bd[3]] if len(bd) >= 4 else [],
                "river":        [bd[4]] if len(bd) >= 5 else [],
                "player_count": pc,
            })

    total = len(hands)
    if total == 0:
        return {}

    # ── Build global stats ─────────────────────────────────────────────────────
    pos_freq  = {c: {} for c in ALL_CARDS}
    pos_carry = {p: [0, 0] for p in ["hole_1","hole_2","flop_1","flop_2","flop_3","turn","river"]}
    carry_hot = {}
    carry_pairs = carry_total = 0

    for h in hands:
        for j, c in enumerate(h["hole_cards"]): pos_freq[c][f"hole_{j+1}"] = pos_freq[c].get(f"hole_{j+1}", 0) + 1
        for j, c in enumerate(h["flop"]):       pos_freq[c][f"flop_{j+1}"] = pos_freq[c].get(f"flop_{j+1}", 0) + 1
        for c in h["turn"]:                     pos_freq[c]["turn"]         = pos_freq[c].get("turn",  0) + 1
        for c in h["river"]:                    pos_freq[c]["river"]        = pos_freq[c].get("river", 0) + 1

    for i in range(total - 1):
        h1, h2 = hands[i], hands[i+1]
        h2c = set(h2["hole_cards"] + h2["flop"] + h2["turn"] + h2["river"])
        all_h1 = ([(c, f"hole_{j+1}") for j,c in enumerate(h1["hole_cards"])] +
                  [(c, f"flop_{j+1}") for j,c in enumerate(h1["flop"])] +
                  [(c, "turn")  for c in h1["turn"]] +
                  [(c, "river") for c in h1["river"]])
        s1 = set(c for c,_ in all_h1)
        if s1 and h2c:
            carry_total += 1
            shared = s1 & h2c
            if shared:
                carry_pairs += 1
                for c in shared: carry_hot[c] = carry_hot.get(c, 0) + 1
        for c, pos in all_h1:
            pos_carry[pos][1] += 1
            if c in h2c: pos_carry[pos][0] += 1

    early_bias = {}
    for c in ALL_CARDS:
        e = sum(pos_freq[c].get(p, 0) for p in EARLY_POS)
        t = sum(pos_freq[c].values()) or 1
        early_bias[c] = e / t

    # topCards formula matches model.js — earlyBias ratio + raw count tiebreaker + carry heat
    def _tc_score(c):
        e = sum(pos_freq[c].get(p,0) for p in EARLY_POS)
        t = sum(pos_freq[c].values()) or 1
        return (e/t) + e*0.001 + (carry_hot.get(c,0)*0.0001)
    top_cards  = sorted([c for c in ALL_CARDS if sum(pos_freq[c].values()) > 2],
                        key=_tc_score, reverse=True)[:15]
    carry_rate = round(carry_pairs / carry_total * 100) if carry_total else 0

    # ── Segment by player count (3-4, 5-6, 7+) ────────────────────────────────
    def segment_carry(bucket_hands):
        cp = ct = 0
        for i in range(len(bucket_hands) - 1):
            h1, h2 = bucket_hands[i], bucket_hands[i+1]
            s1 = set(h1["hole_cards"]+h1["flop"]+h1["turn"]+h1["river"])
            s2 = set(h2["hole_cards"]+h2["flop"]+h2["turn"]+h2["river"])
            if s1 and s2:
                ct += 1
                if s1 & s2: cp += 1
        return round(cp/ct*100) if ct else 0

    small_table  = [h for h in hands if 2 <= h["player_count"] <= 4]
    medium_table = [h for h in hands if 5 <= h["player_count"] <= 6]
    large_table  = [h for h in hands if h["player_count"] >= 7]

    # ── Data-driven confidence — rolling carry rate stability ──────────────────
    # No hardcoded thresholds. Confidence = how stable carry rate is
    # over the last 20 hands vs overall. High variance = still learning.
    # Low variance = model has converged.
    confidence = "learning"
    if total >= 20:
        window_hands = hands[-20:]
        wp = wt = 0
        for i in range(len(window_hands)-1):
            s1 = set(window_hands[i]["hole_cards"]+window_hands[i]["flop"]+window_hands[i]["turn"]+window_hands[i]["river"])
            s2 = set(window_hands[i+1]["hole_cards"]+window_hands[i+1]["flop"]+window_hands[i+1]["turn"]+window_hands[i+1]["river"])
            if s1 and s2:
                wt += 1
                if s1 & s2: wp += 1
        window_rate = round(wp/wt*100) if wt else 0
        delta = abs(window_rate - carry_rate)
        # Confidence based on rate stability, not hand count
        if delta <= 3 and total >= 30:
            confidence = "high"
        elif delta <= 8 and total >= 15:
            confidence = "medium"
        else:
            confidence = "learning"

    # Last 3 hands from THIS game — extension uses these to seed carry chain
    # CSV and live DOM are the same deck: prior CSV hands = prior hands this session
    last3 = [{"hole_cards": h["hole_cards"], "flop": h["flop"],
               "turn": h["turn"], "river": h["river"]}
             for h in hands[-3:]]

    # ── Chi deviation — matches model.js logic exactly ─────────────────────────
    import math as _math
    EARLY_RATE = 5 / 7
    chi_deviation  = {}
    deck_frequency = {}
    for c in ALL_CARDS:
        e = sum(pos_freq[c].get(p, 0) for p in EARLY_POS)
        t = sum(pos_freq[c].values())
        if t < 3:
            chi_deviation[c] = 0
            continue
        expected = t * EARLY_RATE
        z = (e - expected) / _math.sqrt(max(expected, 1))
        chi_deviation[c] = round(z, 3)
        deck_frequency[c] = {
            "observed": e, "expected": round(expected, 1),
            "z": round(z, 2), "total": t,
            "status": "hot" if z > 1.0 else ("shadow" if z < -0.3 else "neutral"),
        }

    # Shadow cards — avoid as hole cards
    shadow_cards = sorted(
        [c for c in ALL_CARDS if chi_deviation.get(c, 0) < -0.2
         and sum(pos_freq[c].values()) >= 5],
        key=lambda c: chi_deviation[c]
    )

    # Chi-corrected topCards for 50+ hands — matches model.js buildModel
    deck_assessed = total >= 50
    if deck_assessed:
        top_cards = sorted(
            [c for c in ALL_CARDS if sum(pos_freq[c].values()) > 2],
            key=lambda c: (chi_deviation.get(c, 0) + carry_hot.get(c, 0) * 0.01),
            reverse=True
        )[:15]

    # ── Knowledge graph — card co-occurrence with lift scores ─────────────────
    card_given  = {}
    card_totals = {c: sum(pos_freq[c].values()) for c in ALL_CARDS}
    for h in hands:
        cards = list(set(h["hole_cards"] + h["flop"] + h["turn"] + h["river"]))
        for a in cards:
            if a not in card_given:
                card_given[a] = {}
            for b in cards:
                if a != b:
                    card_given[a][b] = card_given[a].get(b, 0) + 1

    knowledge_graph = {}
    if total >= 20:
        for c1, neighbors in card_given.items():
            knowledge_graph[c1] = {}
            for c2, cooccur in neighbors.items():
                p1 = (card_totals.get(c1, 1)) / max(total, 1)
                p2 = (card_totals.get(c2, 1)) / max(total, 1)
                expected = p1 * p2 * total
                lift = cooccur / expected if expected > 0.5 else 0
                if lift > 1.5:
                    knowledge_graph[c1][c2] = round(lift, 2)

    # ── Markov rank transitions ────────────────────────────────────────────────
    markov_ranks = {}
    if total >= 30:
        rank_totals = {}
        for i in range(len(hands) - 1):
            b1ranks = set(c[:-1] for c in hands[i]["hole_cards"] + hands[i]["flop"] + hands[i]["turn"] + hands[i]["river"])
            b2ranks = set(c[:-1] for c in hands[i+1]["hole_cards"] + hands[i+1]["flop"] + hands[i+1]["turn"] + hands[i+1]["river"])
            for r1 in b1ranks:
                rank_totals[r1] = rank_totals.get(r1, 0) + 1
                if r1 not in markov_ranks:
                    markov_ranks[r1] = {}
                for r2 in b2ranks:
                    markov_ranks[r1][r2] = markov_ranks[r1].get(r2, 0) + 1
        # Normalize to probabilities
        for r1 in markov_ranks:
            tot = rank_totals.get(r1, 1)
            for r2 in markov_ranks[r1]:
                markov_ranks[r1][r2] = round(markov_ranks[r1][r2] / tot, 2)

    # ── Regime carry rate (last 20 hands) ─────────────────────────────────────
    regime_carry_rate = 0
    if total >= 20:
        recent20 = hands[-20:]
        rp = rt = 0
        for i in range(len(recent20) - 1):
            s1 = set(recent20[i]["hole_cards"] + recent20[i]["flop"] + recent20[i]["turn"] + recent20[i]["river"])
            s2 = set(recent20[i+1]["hole_cards"] + recent20[i+1]["flop"] + recent20[i+1]["turn"] + recent20[i+1]["river"])
            rt += 1
            if s1 & s2:
                rp += 1
        regime_carry_rate = round(rp / rt * 100) if rt else 0

    return {
        "earlyBias":        early_bias,
        "posCarry":         pos_carry,
        "carryHot":         carry_hot,
        "topCards":         top_cards,
        "carryRate":        carry_rate,
        "totalHands":       total,
        "bigBlind":         2,
        "dangerPlayers":    [],
        "confidence":       confidence,
        "deckAssessed":     deck_assessed,
        "chiDeviation":     chi_deviation,
        "deckFrequency":    deck_frequency,
        "shadowCards":      shadow_cards,
        "knowledgeGraph":   knowledge_graph,
        "markovRanks":      markov_ranks,
        "regimeCarryRate":  regime_carry_rate,
        "tableSegments": {
            "small":  {"count": len(small_table),  "carryRate": segment_carry(small_table)},
            "medium": {"count": len(medium_table), "carryRate": segment_carry(medium_table)},
            "large":  {"count": len(large_table),  "carryRate": segment_carry(large_table)},
        },
        "hands": last3,
    }

def maybe_rebuild_model(game_id: str, hand_count: int) -> dict:
    """Rebuild model every MODEL_REBUILD_EVERY hands. Return cached otherwise."""
    cached = _model_cache.get(game_id, {})
    last_build = cached.get("hand_count", 0)
    if hand_count - last_build >= MODEL_REBUILD_EVERY or not cached:
        model = build_model_from_db(game_id)
        if model:
            _model_cache[game_id] = {
                "model":      model,
                "version":    hand_count,
                "hand_count": hand_count,
            }
            print(f"[model] Rebuilt for {game_id}: {hand_count} hands, "
                  f"carry={model['carryRate']}%, confidence={model['confidence']}")
        return model
    return cached.get("model", {})


_mem0_client = None
def get_mem0():
    global _mem0_client
    if _mem0_client is None:
        _mem0_client = MemoryClient(api_key=MEM0_API_KEY)
    return _mem0_client

# ── Server-side Claude queue ──────────────────────────────────────────────────
_claude_queue   = asyncio.Queue(maxsize=50)  # max 50 pending analyses — prevents memory growth

async def _process_claude_queue():
    global _queue_running
    _queue_running = True
    while True:
        hand_data = await _claude_queue.get()
        try:
            await _run_between_hands(hand_data)
        except Exception as e:
            print(f"[queue] error processing hand {hand_data.get('hand_num')}: {e}")
        finally:
            _claude_queue.task_done()

# Queue started via lifespan handler above

# ── Models ────────────────────────────────────────────────────────────────────
class HandData(BaseModel):
    session_id:      str
    game_id:         str = ""
    hand_num:        int
    hole_cards:      list
    board:           list = []
    flop:            list = []
    turn:            list = []
    river:           list = []
    hero_won:        bool = False
    hero_folded:     bool = False
    all_in:          bool = False
    away_mode:       bool = False
    pot:             int  = 0
    shown_hands:     dict = {}
    shown_winners:   list = []
    win_counts:      dict = {}
    dealer_pos:      int  = 0
    blinds:          dict = {}
    winner:          str  = ""
    decision:        dict = {}   # engine decision for this hand — stored as engine_action
    players_in_hand: list = []   # full per-player HUD data for this hand

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_stats(hand_rows):
    total = len(hand_rows)
    if not total:
        return {}
    hero_won    = sum(1 for h in hand_rows if h.get("hero_won"))
    went_allin  = [h for h in hand_rows if h.get("all_in")]
    allin_won   = sum(1 for h in went_allin if h.get("hero_won"))
    win_rate    = hero_won / total * 100
    allin_wr    = allin_won / len(went_allin) * 100 if went_allin else 0

    street_wins   = {"preflop":0,"flop":0,"turn":0,"river":0}
    street_totals = {"preflop":0,"flop":0,"turn":0,"river":0}
    carry_count = 0

    for i, h in enumerate(hand_rows):
        board = json.loads(h.get("board","[]")) if h.get("board") else []
        street = "preflop" if len(board)==0 else "flop" if len(board)<=3 else "turn" if len(board)==4 else "river"
        street_totals[street] += 1
        if h.get("hero_won"):
            street_wins[street] += 1
        if i > 0:
            prev_board = json.loads(hand_rows[i-1].get("board","[]")) if hand_rows[i-1].get("board") else []
            hole = json.loads(h.get("hole_cards","[]")) if h.get("hole_cards") else []
            if set(prev_board) & set(board + hole):
                carry_count += 1

    carry_rate = carry_count / (total-1) * 100 if total > 1 else 0

    return {
        "total": total,
        "win_rate": round(win_rate, 1),
        "allin_wr": round(allin_wr, 1),
        "carry_rate": round(carry_rate, 1),
        "street_wins": street_wins,
        "street_totals": street_totals,
    }

def get_analysis_history(db, limit=5):
    rows = db.execute(
        "SELECT ts, hands_at_time, win_rate, carry_rate, allin_wr, claude_narrative "
        "FROM analysis_log ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]

# ── POST /hand ────────────────────────────────────────────────────────────────
@app.post("/hand")
async def log_hand(data: HandData):
    mem0_id = None
    memory_text = ""
    game_id = data.game_id or data.session_id
    try:
        board = data.board if data.board else (data.flop + data.turn + data.river)
        with get_db() as db:
            player_count = len(data.players_in_hand)
            engine_action = data.decision.get('action','') if data.decision else ''
            engine_confidence = data.decision.get('confidence', 0) if data.decision else 0
            engine_hand_strength = data.decision.get('handStrength', 0) if data.decision else 0
            db.execute(
                "INSERT OR IGNORE INTO hands (game_id,session_id,hand_num,hole_cards,board,hero_won,hero_folded,all_in,pot,away_mode,shown_hands,player_stats,player_count,engine_action,engine_confidence,engine_hand_strength) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (game_id, data.session_id, data.hand_num,
                 json.dumps(data.hole_cards), json.dumps(board),
                 int(data.hero_won), int(data.hero_folded), int(data.all_in),
                 data.pot, int(data.away_mode), json.dumps(data.shown_hands),
                 json.dumps(data.players_in_hand), player_count,
                 engine_action, engine_confidence, engine_hand_strength)
            )
            hand_was_new = db.execute("SELECT changes()").fetchone()[0] > 0
            if not hand_was_new:
                return {"ok": True, "duplicate": True, "hand_num": data.hand_num}
            for p in data.players_in_hand:
                db.execute(
                    """INSERT OR IGNORE INTO player_hands
                    (game_id,hand_num,player_name,seat_pos,is_hero,stack_start,
                     preflop_bet,flop_bet,turn_bet,river_bet,
                     vpip,pfr,three_bet,four_bet_plus,aggressive_acts,passive_acts,
                     folded,all_in,won,went_to_showdown,saw_flop,
                     in_steal_position,attempted_steal,folded_to_steal,
                     faced_three_bet,folded_to_three_bet,
                     made_cbet,faced_cbet,folded_to_cbet,
                     shown_cards,hand_message,dom_wins)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (game_id, data.hand_num,
                     p.get("player_name",""), p.get("seat_pos",0), p.get("is_hero",0),
                     p.get("stack_start",0), p.get("preflop_bet",0), p.get("flop_bet",0),
                     p.get("turn_bet",0), p.get("river_bet",0),
                     p.get("vpip",0), p.get("pfr",0), p.get("three_bet",0),
                     p.get("four_bet_plus",0), p.get("aggressive_acts",0), p.get("passive_acts",0),
                     p.get("folded",0), p.get("all_in",0), p.get("won",0),
                     p.get("went_to_showdown",0), p.get("saw_flop",0),
                     p.get("in_steal_position",0), p.get("attempted_steal",0), p.get("folded_to_steal",0),
                     p.get("faced_three_bet",0), p.get("folded_to_three_bet",0),
                     p.get("made_cbet",0), p.get("faced_cbet",0), p.get("folded_to_cbet",0),
                     p.get("shown_cards","[]"), p.get("hand_message",""), p.get("dom_wins",0))
                )
                db.execute(
                    """INSERT INTO player_stats
                    (game_id,player_name,hands_seen,vpip_count,pfr_count,threebet_count,
                     fourbet_count,total_agg,total_passive,win_count,showdown_count,
                     showdown_wins,saw_flop_count,allin_count,dom_win_count,
                     steal_opps,steal_attempts,fold_to_steal,fold_to_steal_opps,
                     faced_3bet,folded_to_3bet,cbet_opps,cbet_count,faced_cbet,folded_to_cbet)
                    VALUES(?,?,1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(game_id,player_name) DO UPDATE SET
                        hands_seen=player_stats.hands_seen+1,
                        vpip_count=player_stats.vpip_count+excluded.vpip_count,
                        pfr_count=player_stats.pfr_count+excluded.pfr_count,
                        threebet_count=player_stats.threebet_count+excluded.threebet_count,
                        fourbet_count=player_stats.fourbet_count+excluded.fourbet_count,
                        total_agg=player_stats.total_agg+excluded.total_agg,
                        total_passive=player_stats.total_passive+excluded.total_passive,
                        win_count=player_stats.win_count+excluded.win_count,
                        showdown_count=player_stats.showdown_count+excluded.showdown_count,
                        showdown_wins=player_stats.showdown_wins+excluded.showdown_wins,
                        saw_flop_count=player_stats.saw_flop_count+excluded.saw_flop_count,
                        allin_count=player_stats.allin_count+excluded.allin_count,
                        dom_win_count=MAX(player_stats.dom_win_count,excluded.dom_win_count),
                        steal_opps=player_stats.steal_opps+excluded.steal_opps,
                        steal_attempts=player_stats.steal_attempts+excluded.steal_attempts,
                        fold_to_steal=player_stats.fold_to_steal+excluded.fold_to_steal,
                        fold_to_steal_opps=player_stats.fold_to_steal_opps+excluded.fold_to_steal_opps,
                        faced_3bet=player_stats.faced_3bet+excluded.faced_3bet,
                        folded_to_3bet=player_stats.folded_to_3bet+excluded.folded_to_3bet,
                        cbet_opps=player_stats.cbet_opps+excluded.cbet_opps,
                        cbet_count=player_stats.cbet_count+excluded.cbet_count,
                        faced_cbet=player_stats.faced_cbet+excluded.faced_cbet,
                        folded_to_cbet=player_stats.folded_to_cbet+excluded.folded_to_cbet,
                        ts_updated=datetime('now')""",
                    (game_id, p.get("player_name",""),
                     p.get("vpip",0), p.get("pfr",0), p.get("three_bet",0),
                     p.get("four_bet_plus",0), p.get("aggressive_acts",0), p.get("passive_acts",0),
                     p.get("won",0), p.get("went_to_showdown",0),
                     1 if (p.get("won",0) and p.get("went_to_showdown",0)) else 0,
                     p.get("saw_flop",0), p.get("all_in",0), p.get("dom_wins",0),
                     p.get("in_steal_position",0), p.get("attempted_steal",0),
                     p.get("folded_to_steal",0), p.get("in_steal_position",0),
                     p.get("faced_three_bet",0), p.get("folded_to_three_bet",0),
                     p.get("in_steal_position",0) if p.get("pfr",0) else 0,  # cbet_opps = was PFR
                     p.get("made_cbet",0), p.get("faced_cbet",0), p.get("folded_to_cbet",0))
                )
                # ── Global player upsert — cross-session tracking ─────────────
                try:
                    upsert_global_player(db, p.get("player_name",""), game_id, p)
                except Exception as gpe:
                    pass  # never let global player upsert break hand storage

        result = "WON" if data.hero_won else ("FOLDED" if data.hero_folded else "LOST")
        mode   = "[AWAY]" if data.away_mode else ""
        memory_text = (f"Hand #{data.hand_num} {mode}: hero held {' '.join(data.hole_cards) if data.hole_cards else 'N/A'}, "
                       f"board: {' '.join(board) if board else 'no board'}, result: {result}, pot: {data.pot}, all_in: {data.all_in}")
        if data.shown_hands:
            memory_text += f", opponents showed: {', '.join(f'{p}: {chr(32).join(c)}' for p,c in data.shown_hands.items())}"

        # Raw hand data goes to SQLite only — NOT Mem0
        # Mem0 receives AI insights from LangGraph node_store_memory only
        mem0_id = None
        with get_db() as db:
            db.execute("INSERT INTO mem0_log (game_id,session_id,hand_num,memory_text,memory_id) VALUES(?,?,?,?,?)",
                       (game_id, data.session_id, data.hand_num, memory_text, "pending"))

        try:
            _claude_queue.put_nowait({
            "session_id": data.session_id, "game_id": game_id,
            "hand_num": data.hand_num, "hole_cards": data.hole_cards,
            "flop": data.flop, "turn": data.turn, "river": data.river,
            "hero_won": data.hero_won, "hero_folded": data.hero_folded,
            "all_in": data.all_in, "away_mode": data.away_mode,
            "pot": data.pot, "shown_hands": data.shown_hands,
            "player_stats": {p["player_name"]: p for p in data.players_in_hand},
            })
        except asyncio.QueueFull:
            print(f"[queue] Full — skipping LangGraph for hand {data.hand_num}")

        # Auto-trigger Claude narrative for players crossing 10/25/50 hand milestones
        NARRATIVE_MILESTONES = {10, 25, 50, 100}
        try:
            with get_db() as db:
                for p in data.players_in_hand:
                    if p.get("is_hero"): continue
                    row = db.execute(
                        "SELECT hands_seen FROM player_stats WHERE game_id=? AND player_name=?",
                        (game_id, p["player_name"])
                    ).fetchone()
                    if row and row["hands_seen"] in NARRATIVE_MILESTONES:
                        print(f"[narrative] Auto-generating for {p['player_name']} at {row['hands_seen']} hands")
                        asyncio.create_task(_auto_narrative(game_id, p["player_name"]))
        except Exception as e:
            print(f"[narrative] Auto-trigger error: {e}")
    except Exception as e:
        return {"ok": False, "error": str(e)}

    # Rebuild model every MODEL_REBUILD_EVERY hands and return to extension
    rebuilt_model = {}
    try:
        with get_db() as db:
            hand_count = db.execute(
                "SELECT COUNT(*) FROM hands WHERE game_id=?", (game_id,)
            ).fetchone()[0]
        rebuilt_model = await asyncio.to_thread(maybe_rebuild_model, game_id, hand_count)
    except Exception as e:
        print(f"[model] rebuild error: {e}")

    return {
        "ok": True,
        "mem0_id": mem0_id,
        "memory": memory_text,
        "model": rebuilt_model if rebuilt_model else None,
        "model_version": rebuilt_model.get("totalHands", 0) if rebuilt_model else 0,
    }

# ── GET /analyze — full analysis, stores result historically ──────────────────
@app.get("/analyze")
async def analyze(game_id: str = ""):
    try:
        mem0   = get_mem0()

        # Use game_id as Mem0 user namespace — fall back to listing all if not provided
        mem_user = game_id if game_id else None
        if mem_user:
            all_mems = mem0.get_all(filters={"user_id": mem_user})
        else:
            return {"ok": False, "error": "game_id required for /analyze"}
        memories = []
        if isinstance(all_mems, list):
            for m in all_mems:
                memories.append(m.get("memory",""))

        # Pull all hands from DB
        with get_db() as db:
            hand_rows = [dict(r) for r in db.execute("SELECT * FROM hands WHERE game_id=? ORDER BY id ASC", (game_id,)).fetchall()]
            prev_analyses = get_analysis_history(db, limit=3)

        stats = compute_stats(hand_rows)

        # Previous analysis context so Claude can track change over time
        prev_ctx = ""
        if prev_analyses:
            prev_ctx = "PREVIOUS ANALYSIS RUNS:\n"
            for a in prev_analyses:
                prev_ctx += f"[{a['ts']}] {a['hands_at_time']} hands, WR:{a['win_rate']}%, narrative: {a['claude_narrative'][:200]}…\n"

        mem_block = "\n".join(f"- {m}" for m in memories) if memories else "No memories yet"

        prompt = f"""You are analyzing a PokerNow game with a suspected broken RNG shuffle.

LIVE STATS FROM THIS SESSION ({stats.get('total',0)} hands logged):
- Win rate: {stats.get('win_rate',0)}%
- All-in win rate: {stats.get('allin_wr',0)}%
- Carry rate (cards bleeding between hands): {stats.get('carry_rate',0)}%
- Street breakdown wins: {stats.get('street_wins',{})}
- Street breakdown totals: {stats.get('street_totals',{})}

ALL MEM0 MEMORIES ({len(memories)} total — every hand recorded):
{mem_block}

{prev_ctx}

Based only on the actual data above (no assumptions), provide analysis covering:
1. Win rate pattern — is it statistically anomalous vs expected ~50%?
2. Carry rate pattern — what does it suggest about the shuffle?
3. All-in outcomes — any pattern?
4. Player-specific tendencies found in memories
5. Specific strategic adjustments for next session based on real patterns observed
6. Confidence level (%) that shuffle anomalies exist, and why

Reference specific hands from memory where possible. Do not assume anything not in the data."""

        narrative = call_glm(prompt, max_tokens=2000,
                             system="You are a poker data analyst. Be specific, cite actual numbers.")

        # ── STORE in analysis_log so it builds historical record ──────────────
        with get_db() as db:
            db.execute(
                """INSERT INTO analysis_log
                   (game_id, hands_at_time, mem0_count, win_rate, carry_rate, allin_wr,
                    street_wins, street_totals, memories_snapshot, claude_narrative)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    game_id,
                    stats.get("total", 0),
                    len(memories),
                    stats.get("win_rate", 0),
                    stats.get("carry_rate", 0),
                    stats.get("allin_wr", 0),
                    json.dumps(stats.get("street_wins", {})),
                    json.dumps(stats.get("street_totals", {})),
                    json.dumps(memories),       # full snapshot of all memories at this point
                    narrative
                )
            )

        # Also push the analysis itself into Mem0 so /advice can reference it
        mem0.add(
            f"ANALYSIS at {datetime.utcnow().isoformat()}: {narrative[:500]}",
            user_id=game_id,
            metadata={"type": "analysis", "hands": stats.get("total",0)}
        )

        return {
            "ok": True,
            "stats": stats,
            "mem0_memories": len(memories),
            "analysis_runs_stored": len(prev_analyses) + 1,
            "claude_analysis": narrative
        }

    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

# ── GET /analysis_history — show all past analysis runs ──────────────────────
@app.get("/analysis_history")
async def analysis_history():
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT id, ts, hands_at_time, mem0_count, win_rate, carry_rate, "
                "allin_wr, street_wins, street_totals, claude_narrative "
                "FROM analysis_log ORDER BY ts DESC"
            ).fetchall()
        return {"ok": True, "count": len(rows), "history": [dict(r) for r in rows]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── GET /memories ─────────────────────────────────────────────────────────────
@app.get("/memories")
async def all_memories(game_id: str = ""):
    try:
        mem0 = get_mem0()
        if not game_id:
            return {"ok": False, "error": "game_id required"}
        mem_user = game_id
        all_mems = mem0.get_all(filters={"user_id": mem_user})
        memories = []
        if isinstance(all_mems, list):
            for m in all_mems:
                memories.append({
                    "id": m.get("id",""),
                    "memory": m.get("memory",""),
                    "created_at": m.get("created_at",""),
                })
        with get_db() as db:
            db_log = [dict(r) for r in db.execute(
                "SELECT hand_num, memory_text, ts FROM mem0_log WHERE game_id=? ORDER BY ts DESC", (game_id,)
            ).fetchall()]
        return {"ok": True, "total": len(memories), "memories": memories, "db_log": db_log}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── GET /proof/{session_id} ───────────────────────────────────────────────────
@app.get("/proof/{session_id}")
async def proof(session_id: str):
    try:
        with get_db() as db:
            mem0_entries = db.execute(
                "SELECT hand_num,memory_text,memory_id,ts FROM mem0_log "
                "WHERE game_id=? ORDER BY ts DESC LIMIT 50", (session_id,)
            ).fetchall()
            claude_entries = db.execute(
                "SELECT hand_num,response,next_hand,ts,weight_updates,carry_rate,live_wr FROM claude_log "
                "WHERE game_id=? ORDER BY ts DESC LIMIT 20", (session_id,)
            ).fetchall()
        return {
            "session_id":    session_id,
            "mem0_entries":  [dict(r) for r in mem0_entries],
            "claude_entries": [dict(r) for r in claude_entries],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    mem0_preview   = MEM0_API_KEY[:8]  + "…" if MEM0_API_KEY  else "NOT SET"
    key_for_preview = OPENROUTER_KEY or ANTHROPIC_KEY
    claude_preview  = f"GLM-5.1 via OR ({key_for_preview[:8]}…)" if key_for_preview else "NOT SET"

    mem0_live = False; mem0_error = ""
    try:
        m = get_mem0()
        m.search(query="test", filters={"user_id": "health_check"}, limit=1)
        mem0_live = True
    except Exception as e:
        mem0_error = str(e)[:100]

    claude_live = False; claude_error = ""
    try:
        # Health check: verify GLM-5.1 / OpenRouter key is set and reachable
        key = OPENROUTER_KEY or ANTHROPIC_KEY
        if key:
            test = call_glm("hi", max_tokens=5)
            claude_live = bool(test)
        else:
            claude_error = "No API key set (OPENROUTER_API_KEY or ANTHROPIC_API_KEY)"
    except Exception as e:
        claude_error = str(e)[:100]

    with get_db() as db:
        hand_count     = db.execute("SELECT COUNT(*) FROM hands").fetchone()[0]
        mem0_log_count = db.execute("SELECT COUNT(*) FROM mem0_log").fetchone()[0]
        analysis_count = db.execute("SELECT COUNT(*) FROM analysis_log").fetchone()[0]

    return {
        "ok": True,
        "mem0_key": mem0_preview, "claude_key": claude_preview,
        "mem0_live": mem0_live,   "claude_live": claude_live,
        "mem0_error": mem0_error, "claude_error": claude_error,
        "db": DB_PATH,
        "hands_logged": hand_count,
        "mem0_memories_logged": mem0_log_count,
        "analysis_runs": analysis_count
    }

# ── Core Claude analysis function (called by queue) ──────────────────────────
def _compute_table_dynamic(data_dict: dict) -> str:
    """Generate one-line table dynamic summary from current state"""
    try:
        game_id = data_dict.get("game_id","")
        with get_db() as db:
            danger = db.execute(
                "SELECT COUNT(*) as n FROM player_stats WHERE game_id=? AND CAST(win_count AS REAL)/MAX(hands_seen,1)>=0.60 AND hands_seen>=5",
                (game_id,)
            ).fetchone()
            carry = db.execute(
                "SELECT AVG(CAST(pot AS REAL)) as avg_pot FROM hands WHERE game_id=? ORDER BY id DESC LIMIT 20",
                (game_id,)
            ).fetchone()
        danger_n = danger["n"] if danger else 0
        if danger_n >= 2:
            return "⚠ Aggressive table — tighten range, avoid thin value bets"
        elif danger_n == 1:
            return "⚠ One danger player — 3bet lighter, call their river overbets"
        else:
            return "✓ Table passive — exploit with steals and thin value bets"
    except:
        return ""

async def _run_between_hands(data_dict: dict):
    """
    Between-hands analysis via LangGraph + Letta + Mem0.
    Graph: load_context → mem0_retrieve → letta_reason → parse_updates
           → [apply_danger] → [apply_carry] → [apply_allin] → store_memory → END
    """
    try:
        game_id    = data_dict.get("game_id", data_dict.get("session_id",""))
        session_id = data_dict.get("session_id","")
        board      = data_dict.get("flop",[]) + data_dict.get("turn",[]) + data_dict.get("river",[])

        state_input = {
                "game_id":      game_id,
                "session_id":   session_id,
                "hand_num":     data_dict.get("hand_num", 0),
                "hole_cards":   data_dict.get("hole_cards", []),
                "board":        board,
                "hero_won":     data_dict.get("hero_won", False),
                "hero_folded":  data_dict.get("hero_folded", False),
                "all_in":       data_dict.get("all_in", False),
                "away_mode":    data_dict.get("away_mode", False),
                "pot":          data_dict.get("pot", 0),
                "shown_hands":  data_dict.get("shown_hands", {}),
                "player_stats": data_dict.get("player_stats", {}),
                "live_stats": {}, "memories": [],
                "letta_response": "", "weight_updates": {},
                "next_hand": "", "danger_players": [],
                "has_danger": False, "has_carry": False, "is_allin": False,
        }

        if poker_graph is not None:
            result = await asyncio.to_thread(poker_graph.invoke, state_input)
        else:
            # LangGraph not available — run nodes directly in sequence
            result = state_input
            result.update(node_load_context(result))
            result.update(node_mem0_retrieve(result))
            result.update(node_letta_reason(result))
            result.update(node_parse_updates(result))
            if result.get("has_danger"): result.update(node_apply_danger(result))
            if result.get("has_carry"):  result.update(node_apply_carry(result))
            if result.get("is_allin"):   result.update(node_apply_allin(result))
            node_store_memory(result)

        next_hand      = result.get("next_hand", "")
        weight_updates = result.get("weight_updates", {})
        live_stats     = result.get("live_stats", {})
        danger_players = result.get("danger_players", [])

        # Regime detection: if rolling carry rate drops significantly → possible seed reset
        rolling_carry = live_stats.get("carry_rate", 0)
        total_hands   = live_stats.get("total_hands", 0)
        regime_shift  = rolling_carry < 25 and total_hands > 30  # carry collapsed

        await manager.broadcast(game_id, {
            "type": "weight_updates",
            "data": {
                "hand_num":       data_dict.get("hand_num", 0),
                "next_hand":      next_hand,
                "weight_updates": weight_updates,
                "carry_rate":     rolling_carry,
                "live_wr":        live_stats.get("win_rate", 0),
                "danger_players": danger_players,
                "hero_won":       data_dict.get("hero_won", False),
                "hero_folded":    data_dict.get("hero_folded", False),
                "regime_shift":   regime_shift,
                "deck_assessed":  total_hands >= 50,
            }
        })
        return {"nextHand": next_hand, "weight_updates": weight_updates}

    except Exception as e:
        print(f"[_run_between_hands] LangGraph error: {e}")
        return {}


# ── POST /between_hands — now just enqueues ───────────────────────────────────
class BetweenHandsData(BaseModel):
    session_id: str
    hand_num: int
    hole_cards: list = []
    flop: list = []
    turn: list = []
    river: list = []
    hero_won: bool = False
    hero_folded: bool = False
    all_in: bool = False
    away_mode: bool = False
    pot: int = 0
    shown_hands: dict = {}

@app.post("/between_hands")
async def between_hands(data: BetweenHandsData):
    try:
        _claude_queue.put_nowait(data.__dict__)
    except asyncio.QueueFull:
        return {"ok": False, "error": "queue full"}
    return {"ok": True, "queued": True}

# ── GET /narratives — all stored Claude next-hand notes ──────────────────────
@app.get("/narratives/{session_id}")
async def narratives(session_id: str):
    try:
        with get_db() as db:
            rows = db.execute(
                """SELECT hand_num, next_hand, ts,
                   weight_updates, carry_rate, live_wr, danger_players
                   FROM claude_log
                   WHERE game_id=? AND next_hand IS NOT NULL
                   ORDER BY hand_num DESC LIMIT 50""",
                (session_id,)
            ).fetchall()
        result = []
        for r in rows:
            row = dict(r)
            # Parse JSON fields for dashboard rendering
            try: row["weight_updates"] = json.loads(row.get("weight_updates") or "{}")
            except: row["weight_updates"] = {}
            try: row["danger_players"] = json.loads(row.get("danger_players") or "[]")
            except: row["danger_players"] = []
            result.append(row)
        return {"ok": True, "narratives": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/narratives")
async def narratives_all():
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT session_id, hand_num, next_hand, ts FROM claude_log "
                "WHERE next_hand IS NOT NULL ORDER BY ts DESC LIMIT 100"
            ).fetchall()
        return {"ok": True, "narratives": [dict(r) for r in rows]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    # Always send status immediately on connect so dashboard pills go green
    try:
        await websocket.send_json({"type": "status", "data": {
            "mem0_live":   bool(MEM0_API_KEY),
            "claude_live": bool(OPENROUTER_KEY or ANTHROPIC_KEY),
            "letta_live":  bool(LETTA_API_KEY),
            "game_id":     session_id,
        }})
    except Exception:
        pass
    # Send latest raw state if available
    state = _latest_state.get(session_id) or _latest_state.get(_active_game_id)
    if state:
        try:
            await websocket.send_json({"type": "raw", "data": state})
        except Exception:
            pass

    # ── Auto-seed model from DB on connect ────────────────────────────────────
    # If we have hands stored for this game_id, send model immediately
    # so dashboard populates without needing a CSV upload
    try:
        with get_db() as db:
            hand_count = db.execute(
                "SELECT COUNT(*) as n FROM hands WHERE game_id=? AND hole_cards IS NOT NULL",
                (session_id,)
            ).fetchone()["n"]
            game_hand_count = db.execute(
                "SELECT COUNT(*) as n FROM hands WHERE game_id=? AND hole_cards IS NOT NULL",
                (session_id,)
            ).fetchone()["n"]
        if hand_count > 0:
            model = await asyncio.to_thread(build_model_from_db, session_id)
            # Also load player stats
            # Load from global_players — cross-session stats, keyed by player_id
            with get_db() as db:
                grows = db.execute(
                    "SELECT * FROM global_players ORDER BY total_hands DESC LIMIT 100"
                ).fetchall()
            def _gpct(r):
                h  = r["total_hands"]   or 1
                pa = r["total_passive"] or 1
                sd = r["showdown_count"] or 1
                return {
                    **dict(r),
                    "all_names":    json.loads(r["all_names"] or "[]"),
                    "vpip_pct":     round(r["vpip_count"]    / h  * 100, 1),
                    "pfr_pct":      round(r["pfr_count"]     / h  * 100, 1),
                    "win_pct":      round(r["win_count"]     / h  * 100, 1),
                    "af":           round(r["total_agg"]     / pa, 2),
                    "wsd_pct":      round(r["showdown_wins"] / sd * 100, 1),
                    "threebet_pct": round(r["threebet_count"]/ h  * 100, 1),
                    "handsTracked": r["total_hands"],
                    "isDanger":     (r["win_count"] / h * 100) >= 60 and h >= 10,
                }
            # Key by display_name so dashboard can match to live player names
            players = {}
            for r in grows:
                p = _gpct(r)
                players[p["display_name"]] = p
                # Also key by all known names
                for name in p["all_names"]:
                    players[name] = p
            await websocket.send_json({
                "type": "model_seed",
                "data": {
                    "model":            model,
                    "hand_count":       hand_count,
                    "game_hand_count":  game_hand_count,
                    "game_id":          session_id,
                    "player_stats":     players,
                }
            })
    except Exception as e:
        pass  # never block connection on model seed failure
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)

# ── POST /raw — extension sends all DOM data + computed decision ───────────────
class RawData(BaseModel):
    session_id:   str
    game_id:      str  = ""   # game ID from URL — primary key for WS routing
    hole_cards:   list  = []
    board_cards:  list  = []
    pot_size:     int   = 0
    bet_facing:   int   = 0
    hero_stack:   int   = 0
    is_hero_turn: bool  = False
    hero_bet:     int   = 0
    max_opp_bet:  int   = 0
    call_amount:  int   = 0
    aggressor:    str   = ""
    is_raise:     bool  = False
    is_cold_call: bool  = False
    can_check:    bool  = False
    player_bets:  dict  = {}
    win_counts:   dict  = {}
    hand_messages: dict = {}
    dealer_pos:   int   = 0
    blinds:       dict  = {}
    shown_hands:  dict  = {}
    shown_winners: list = []
    away_mode:    bool  = False
    decision:     dict  = {}
    carry_alert:  dict  = {}
    player_stats: dict  = {}
    danger_players: list = []
    model_meta:   dict  = {}
    model_ready:  bool  = False
    all_players_snap: dict = {}
    board_stage:  str   = ""
    extension_id: str   = ""

@app.post("/raw")
async def post_raw(data: RawData):
    payload = data.model_dump()

    # Assign route_key FIRST — used everywhere below
    global _active_game_id
    route_key       = data.game_id or data.session_id
    _active_game_id = route_key

    try:
        with get_db() as db:
            hc = db.execute("SELECT COUNT(*) FROM hands WHERE game_id=?", (route_key,)).fetchone()[0]
        payload["server_status"] = {
            "hands_logged": hc,
            "mem0_live":    bool(MEM0_API_KEY),
            "claude_live":  bool(OPENROUTER_KEY or ANTHROPIC_KEY),
        }
    except Exception as e:
        print(f"[raw] DB count failed: {e}")
        payload["server_status"] = {"hands_logged": 0, "mem0_live": False, "claude_live": False}

    _latest_state[route_key] = payload
    await manager.broadcast(route_key, {"type": "raw", "data": payload})
    return {"ok": True}

# ── GET /api/state/{session_id} — full state for dashboard on reconnect ────────
@app.get("/active")
async def get_active():
    """Dashboard calls this on load to get the current game ID — no URL params needed."""
    try:
        return {"game_id": _active_game_id, "ok": bool(_active_game_id)}
    except Exception as e:
        return {"game_id": "", "ok": False, "error": str(e)}

@app.get("/model/{game_id}")
async def get_model(game_id: str):
    """Extension fetches current model on startup — no chrome.storage needed."""
    try:
        with get_db() as db:
            game_count = db.execute(
                "SELECT COUNT(*) FROM hands WHERE game_id=?", (game_id,)
            ).fetchone()[0]
        model = await asyncio.to_thread(maybe_rebuild_model, game_id, game_count)
        return {"ok": bool(model), "model": model, "hand_count": game_count}
    except Exception as e:
        return {"ok": False, "error": str(e), "model": {}}

@app.get("/api/state/{session_id}")
async def get_state(session_id: str):
    try:
        with get_db() as db:
            narratives = db.execute(
                "SELECT hand_num, next_hand, ts FROM claude_log "
                "WHERE game_id=? AND next_hand IS NOT NULL ORDER BY hand_num DESC LIMIT 50",
                (session_id,)
            ).fetchall()
            hands = db.execute(
                "SELECT hand_num, hole_cards, board, hero_won, hero_folded, shown_hands FROM hands "
                "WHERE game_id=? ORDER BY id DESC LIMIT 20",
                (session_id,)
            ).fetchall()
            hc = db.execute("SELECT COUNT(*) FROM hands WHERE game_id=?", (session_id,)).fetchone()[0]

        hand_history = []
        for h in reversed(list(hands)):
            result = "WON" if h["hero_won"] else ("FOLDED" if h["hero_folded"] else "LOST")
            hand_history.append({
                "hand_num": h["hand_num"],
                "hole":     " ".join(json.loads(h["hole_cards"] or "[]")),
                "board":    " ".join(json.loads(h["board"] or "[]")),
                "result":   result,
                "note":     None,
                "pending":  False,
            })

        latest = _latest_state.get(session_id, {})
        return {
            "ok":           True,
            "narratives":   [dict(r) for r in narratives],
            "hand_history": hand_history,
            "player_stats": latest.get("player_stats", {}),
            "server_status": {
                "hands_logged": hc,
                "mem0_live":    bool(MEM0_API_KEY),
                "claude_live":  bool(OPENROUTER_KEY or ANTHROPIC_KEY),
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── GET /dashboard — serves the dashboard HTML ─────────────────────────────────
@app.get("/dashboard")
async def dashboard():
    try:
        html_path = Path(__file__).parent / "dashboard.html"
        if html_path.exists():
            return HTMLResponse(
                content=html_path.read_text(),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                }
            )
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {e}</h1>", status_code=500)

# ── POST /upload_csv — parse CSV from dashboard ───────────────────────────────
from fastapi import Request as FastAPIRequest

@app.post("/upload_csv")
async def upload_csv(request: FastAPIRequest, game_id: str = ""):
    import re
    try:
        text  = (await request.body()).decode("utf-8")
        lines = text.strip().split("\n")
        hands, cur = [], None
        card_re = re.compile(r'(?:10|[2-9JQKA])[♠♥♦♣]')
        # First pass — collect hole cards keyed by order number
        # CSV format: "entry",2026-...,ORDER_NUMBER
        order_re = re.compile(r'^"?(.*?)"?,\d{4}-\d{2}-\d{2}T[^,]+,(\d+)')
        hole_by_order = {}
        for line in lines[1:]:
            if "Your hand is" not in line: continue
            m2 = order_re.match(line)
            if m2:
                cards = card_re.findall(m2.group(1).replace('""','"'))
                if cards: hole_by_order[int(m2.group(2))] = cards
        # Second pass — parse hands, attach nearest hole cards
        for line in lines[1:]:
            m = re.match(r'^"?(.*?)"?,(\d{4}-\d{2}-\d{2}T[^,]+),(\d+)', line)
            if not m: continue
            entry = m.group(1).replace('""', '"')
            if "-- starting hand" in entry:
                hm = re.search(r"starting hand #(\d+)", entry)
                if hm:
                    hand_order = int(m.group(3))
                    # Find hole cards within 20 order numbers of hand start
                    # (PokerNow assigns hole card order slightly above hand start order)
                    hole = []
                    for ho, cards in hole_by_order.items():
                        if 0 <= ho - hand_order <= 20:
                            hole = cards
                            break
                    cur = {"hand_num": int(hm.group(1)), "hole_cards": hole, "flop": [], "turn": [], "river": [], "players": set()}
            elif "-- ending hand" in entry:
                if cur:
                    cur["player_count"] = len(cur["players"])
                    hands.append(cur); cur = None
            elif cur:
                if "Your hand is" in entry: cur["hole_cards"] = card_re.findall(entry)
                elif entry.startswith("Flop:"): cur["flop"] = card_re.findall(entry)
                elif entry.startswith("Turn:"):
                    tm = re.search(r"\[([^\]]+)\]$", entry)
                    if tm: cur["turn"] = card_re.findall(tm.group(1))
                elif entry.startswith("River:"):
                    rm = re.search(r"\[([^\]]+)\]$", entry)
                    if rm: cur["river"] = card_re.findall(rm.group(1))
                # Count unique players from action lines e.g. "Pookie calls 20"
                pm = re.match(r'"?([^"@]+)"? (?:calls|raises|folds|checks|bets|posts)', entry)
                if pm: cur["players"].add(pm.group(1).strip())

        # Use active game_id so hands are visible to LangGraph + Mem0 pipeline
        target_game_id = game_id or _active_game_id
        if not target_game_id:
            return {"ok": False, "error": "game_id required — open dashboard from extension popup first"}

        stored = 0
        with get_db() as db:
            for h in hands:
                if not h["hole_cards"]: continue
                board = h["flop"] + h["turn"] + h["river"]
                try:
                    db.execute(
                        """INSERT OR IGNORE INTO hands
                        (game_id, session_id, hand_num, hole_cards, board, player_count)
                        VALUES (?,?,?,?,?,?)""",
                        (target_game_id, target_game_id,
                         h["hand_num"],
                         json.dumps(h["hole_cards"]),
                         json.dumps(board),
                         h.get("player_count", 0))
                    )
                    stored += 1
                except Exception as e:
                    print(f"[upload_csv] INSERT skipped hand {h.get('hand_num','?')}: {e}")

        # ── Build model JSON from ALL hands in DB for this game ─────────────────
        # Uses full history, not just the uploaded batch. Safe for re-uploads.
        # Build model from full DB history using shared function (off main thread)
        model_json = {}
        try:
            with get_db() as db:
                hc = db.execute("SELECT COUNT(*) FROM hands WHERE game_id=?", (target_game_id,)).fetchone()[0]
            model_json = await asyncio.to_thread(maybe_rebuild_model, target_game_id, hc)
        except Exception as e:
            print(f"[upload_csv] Model build error: {e}")


        # ── Trigger LangGraph immediately ─────────────────────────────────────
        # LangGraph node_store_memory will write AI insights to Mem0 (not raw data)
        if stored > 0 and hands:
            last = hands[-1]
            try:
                _claude_queue.put_nowait({
                "session_id":  target_game_id,
                "game_id":     target_game_id,
                "hand_num":    last["hand_num"],
                "hole_cards":  last["hole_cards"],
                "flop":        last["flop"],
                "turn":        last["turn"],
                "river":       last["river"],
                "hero_won":    False,
                "hero_folded": False,
                "all_in":      False,
                "away_mode":   False,
                "pot":         0,
                "shown_hands": {},
                "player_stats": {},
                })
            except asyncio.QueueFull:
                print("[queue] Full — skipping LangGraph for upload")

        # Broadcast new model to extension via WebSocket — so it uses fresh model immediately
        if model_json and target_game_id:
            await manager.broadcast(target_game_id, {
                "type": "raw",
                "data": {
                    "model_ready": True,
                    "model_meta": {
                        "carry_rate":    model_json.get("carryRate", 0),
                        "total_hands":   model_json.get("totalHands", 0),
                        "top_cards":     model_json.get("topCards", [])[:8],
                        "shadow_cards":  model_json.get("shadowCards", [])[:10],
                        "markovRanks":   model_json.get("markovRanks", {}),
                        "knowledgeGraph":model_json.get("knowledgeGraph", {}),
                        "deckAssessed":  model_json.get("deckAssessed", False),
                        "regimeCarryRate": model_json.get("regimeCarryRate", 0),
                        "chiDeviation":   model_json.get("chiDeviation", {}),
                    },
                    "decision": {},
                }
            })
            # Also send model_seed so extension fetchModel picks it up
            await manager.broadcast(target_game_id, {
                "type": "model_seed",
                "data": {
                    "model":           model_json,
                    "hand_count":      stored,
                    "game_hand_count": stored,
                    "game_id":         target_game_id,
                    "player_stats":    {},
                }
            })

        return {"ok": True, "hands": stored, "game_id": target_game_id, "model": model_json}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── GET /player_stats/{game_id} — HUD stats for all players in a game ─────────
@app.get("/player_stats/{game_id}")
async def get_player_stats(game_id: str):
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT * FROM player_stats WHERE game_id=? ORDER BY hands_seen DESC",
                (game_id,)
            ).fetchall()
        def compute(s):
            h   = s["hands_seen"]     or 1
            fl  = s["saw_flop_count"] or 1
            sd  = s["showdown_count"] or 1
            pa  = s["total_passive"]  or 1
            st  = s["steal_opps"]     or 1
            f3  = s["faced_3bet"]     or 1
            cb  = s["cbet_opps"]      or 1
            fcs = s["fold_to_steal_opps"] or 1
            fcb = s["faced_cbet"]     or 1
            return {
                **dict(s),
                "vpip_pct":       round(s["vpip_count"]      / h   * 100, 1),
                "pfr_pct":        round(s["pfr_count"]       / h   * 100, 1),
                "threebet_pct":   round(s["threebet_count"]  / h   * 100, 1),
                "af":             round(s["total_agg"]       / pa,        2),
                "wtsd_pct":       round(s["showdown_count"]  / fl  * 100, 1),
                "wsd_pct":        round(s["showdown_wins"]   / sd  * 100, 1),
                "win_pct":        round(s["win_count"]       / h   * 100, 1),
                "allin_pct":      round(s["allin_count"]     / h   * 100, 1),
                "ats_pct":        round(s["steal_attempts"]  / st  * 100, 1),
                "fold_to_3b_pct": round(s["folded_to_3bet"]  / f3  * 100, 1),
                "cbet_pct":       round(s["cbet_count"]      / cb  * 100, 1),
                "fold_to_cbet_pct": round(s["folded_to_cbet"] / fcb * 100, 1),
                "fold_to_steal_pct": round(s["fold_to_steal"] / fcs * 100, 1),
            }
        return {"ok": True, "players": [compute(r) for r in rows]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── GET /global_player/{player_id} — Cross-session stats for one player ─────────
@app.get("/global_player/{player_id}")
async def get_global_player(player_id: str):
    try:
        with get_db() as db:
            row = db.execute("SELECT * FROM global_players WHERE player_id=?", (player_id,)).fetchone()
        if not row:
            return {"ok": False, "error": "Player not found"}
        r = dict(row)
        h  = r["total_hands"]     or 1
        pa = r["total_passive"]   or 1
        sd = r["showdown_count"]  or 1
        return {"ok": True, "player": {
            **r,
            "all_names":    json.loads(r.get("all_names","[]")),
            "vpip_pct":     round(r["vpip_count"]     / h  * 100, 1),
            "pfr_pct":      round(r["pfr_count"]       / h  * 100, 1),
            "threebet_pct": round(r["threebet_count"]  / h  * 100, 1),
            "af":           round(r["total_agg"]       / pa, 2),
            "win_pct":      round(r["win_count"]       / h  * 100, 1),
            "wsd_pct":      round(r["showdown_wins"]   / sd * 100, 1),
        }}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── GET /global_players — All known players cross-session ─────────────────────
@app.get("/global_players")
async def get_global_players(limit: int = 50):
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT * FROM global_players ORDER BY total_hands DESC LIMIT ?", (limit,)
            ).fetchall()
        def compute(r):
            r = dict(r)
            h  = r["total_hands"]   or 1
            pa = r["total_passive"] or 1
            sd = r["showdown_count"] or 1
            return {**r,
                "all_names":    json.loads(r.get("all_names","[]")),
                "vpip_pct":     round(r["vpip_count"]    / h  * 100, 1),
                "pfr_pct":      round(r["pfr_count"]     / h  * 100, 1),
                "af":           round(r["total_agg"]     / pa, 2),
                "win_pct":      round(r["win_count"]     / h  * 100, 1),
                "wsd_pct":      round(r["showdown_wins"] / sd * 100, 1),
            }
        return {"ok": True, "players": [compute(r) for r in rows]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── POST /global_player_note — Save note for a player ─────────────────────────
class PlayerNoteRequest(BaseModel):
    player_id: str
    notes: str

@app.post("/global_player_note")
async def save_player_note(data: PlayerNoteRequest):
    try:
        with get_db() as db:
            db.execute(
                "UPDATE global_players SET notes=?, ts_updated=datetime('now') WHERE player_id=?",
                (data.notes, data.player_id)
            )
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── GET /action_win_rates — Live win rates per engine action from all history ─────
@app.get("/action_win_rates")
async def get_action_win_rates():
    try:
        with get_db() as db:
            rows = db.execute(
                """SELECT engine_action, 
                          COUNT(*) as total,
                          SUM(hero_won) as wins,
                          AVG(engine_confidence) as avg_confidence,
                          AVG(pot) as avg_pot
                   FROM hands 
                   WHERE engine_action != '' AND engine_action IS NOT NULL
                   GROUP BY engine_action
                   ORDER BY total DESC"""
            ).fetchall()
        result = {}
        for r in rows:
            total = r["total"] or 1
            result[r["engine_action"]] = {
                "total":          r["total"],
                "wins":           r["wins"] or 0,
                "win_rate":       round((r["wins"] or 0) / total * 100, 1),
                "avg_confidence": round(r["avg_confidence"] or 0, 0),
                "avg_pot":        round(r["avg_pot"] or 0, 0),
            }
        return {"ok": True, "action_win_rates": result, 
                "total_hands": sum(v["total"] for v in result.values())}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── GET /export_session/{game_id} — Export session data as JSON ───────────────
@app.get("/export_session/{game_id}")
async def export_session(game_id: str):
    try:
        with get_db() as db:
            hands = db.execute(
                "SELECT * FROM hands WHERE game_id=? ORDER BY hand_num", (game_id,)
            ).fetchall()
            players = db.execute(
                "SELECT * FROM player_stats WHERE game_id=?", (game_id,)
            ).fetchall()
            narratives = db.execute(
                "SELECT * FROM claude_log WHERE game_id=? ORDER BY ts", (game_id,)
            ).fetchall()
            memories = db.execute(
                "SELECT * FROM mem0_log WHERE game_id=? ORDER BY ts", (game_id,)
            ).fetchall()
        total = len(hands)
        wins  = sum(1 for h in hands if h["hero_won"])
        return {
            "ok": True,
            "game_id": game_id,
            "exported_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_hands": total,
                "wins": wins,
                "win_rate": round(wins/total*100,1) if total else 0,
            },
            "hands":      [dict(h) for h in hands],
            "players":    [dict(p) for p in players],
            "narratives": [dict(n) for n in narratives],
            "memories":   [dict(m) for m in memories],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── GET /deck_assessment/{game_id} ─────────────────────────────────────────
@app.get("/deck_assessment/{game_id}")
async def deck_assessment(game_id: str):
    """
    After 50+ hands: chi-squared assessment of entire deck.
    Finds which cards are over/under-represented in early positions.
    This reveals the MT seed's bias pattern for this specific game.
    """
    try:
        import math
        with get_db() as db:
            rows = db.execute(
                "SELECT hole_cards, board FROM hands WHERE game_id=? AND hole_cards IS NOT NULL ORDER BY id ASC",
                (game_id,)
            ).fetchall()

        if len(rows) < 20:
            return {"ok": False, "error": f"Need 20+ hands, have {len(rows)}"}

        EARLY_POS = {'hole_1','hole_2','flop_1','flop_2','flop_3'}
        EARLY_RATE = 5/7  # 5 early / 7 hero-visible positions
        pos_freq = {c: {} for c in ALL_CARDS}

        for row in rows:
            hc = json.loads(row["hole_cards"] or "[]")
            bd = json.loads(row["board"] or "[]")
            for j,c in enumerate(hc):
                k = f"hole_{j+1}"
                pos_freq[c][k] = pos_freq[c].get(k,0) + 1
            flop = bd[:3]
            for j,c in enumerate(flop):
                k = f"flop_{j+1}"
                pos_freq[c][k] = pos_freq[c].get(k,0) + 1
            if len(bd) >= 4:
                pos_freq[bd[3]]["turn"] = pos_freq[bd[3]].get("turn",0) + 1
            if len(bd) >= 5:
                pos_freq[bd[4]]["river"] = pos_freq[bd[4]].get("river",0) + 1

        chi_results = {}
        for c in ALL_CARDS:
            e = sum(pos_freq[c].get(p,0) for p in EARLY_POS)
            t = sum(pos_freq[c].values())
            if t < 3: continue
            expected = t * EARLY_RATE
            z = (e - expected) / math.sqrt(max(expected, 1))
            chi_results[c] = {
                "card": c, "observed_early": e, "expected_early": round(expected,1),
                "z_score": round(z,2), "total": t,
                "status": "hot" if z > 1.0 else ("shadow" if z < -0.2 else "neutral")
            }

        # Sort by Z-score
        hot_cards    = [v for v in chi_results.values() if v["status"] == "hot"]
        shadow_cards = [v for v in chi_results.values() if v["status"] == "shadow"]
        hot_cards.sort(key=lambda x: -x["z_score"])
        shadow_cards.sort(key=lambda x: x["z_score"])

        # Knowledge graph: top card co-occurrences across hands
        cooccur = {}
        hand_list = []
        for row in rows:
            hc = json.loads(row["hole_cards"] or "[]")
            bd = json.loads(row["board"] or "[]")
            hand_list.append(set(hc + bd))

        edges = {}
        for hand in hand_list:
            cards = list(hand)
            for i in range(len(cards)):
                for j in range(i+1, len(cards)):
                    k = tuple(sorted([cards[i], cards[j]]))
                    edges[k] = edges.get(k,0) + 1

        # Regime detection: rolling carry rate (last 20 hands)
        recent = hand_list[-20:] if len(hand_list) >= 20 else hand_list
        regime_carries = sum(1 for i in range(len(recent)-1) if recent[i] & recent[i+1])
        regime_rate = round(regime_carries / max(len(recent)-1,1) * 100, 1)

        # Markov rank transitions
        rank_trans = {}
        for i in range(len(hand_list)-1):
            r1s = set(c[:-1] for c in hand_list[i])
            r2s = set(c[:-1] for c in hand_list[i+1])
            for r1 in r1s:
                if r1 not in rank_trans: rank_trans[r1] = {}
                for r2 in r2s:
                    rank_trans[r1][r2] = rank_trans[r1].get(r2,0) + 1

        top_edges = sorted(edges.items(), key=lambda x: -x[1])[:20]

        return {
            "ok": True,
            "game_id": game_id,
            "hands_analyzed": len(rows),
            "deck_assessed": len(rows) >= 50,
            "hot_cards": hot_cards[:10],
            "shadow_cards": shadow_cards[:10],
            "regime_carry_rate": regime_rate,
            "regime_stable": regime_rate > 30,
            "knowledge_graph_edges": [
                {"card_a": e[0][0], "card_b": e[0][1], "co_occurs": e[1]}
                for e in top_edges
            ],
            "markov_transitions": {
                r1: sorted(trans.items(), key=lambda x: -x[1])[:5]
                for r1, trans in rank_trans.items()
                if sum(trans.values()) >= 5
            },
            "recommended_topCards": [v["card"] for v in hot_cards[:15]],
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

# ── POST /player_narrative — Claude commentary on a specific player ────────────
class NarrativeRequest(BaseModel):
    game_id:     str
    player_name: str

@app.post("/player_narrative")
async def player_narrative(data: NarrativeRequest):
    try:
        with get_db() as db:
            stats = db.execute(
                "SELECT * FROM player_stats WHERE game_id=? AND player_name=?",
                (data.game_id, data.player_name)
            ).fetchone()
            recent_hands = db.execute(
                "SELECT hand_num, preflop_bet, flop_bet, turn_bet, river_bet, vpip, pfr, three_bet, folded, all_in, won, went_to_showdown, shown_cards, hand_message "
                "FROM player_hands WHERE game_id=? AND player_name=? ORDER BY id DESC LIMIT 15",
                (data.game_id, data.player_name)
            ).fetchall()

        if not stats:
            return {"ok": False, "error": "Player not found"}

        s  = dict(stats)
        h  = s["hands_seen"]    or 1
        fl = s["saw_flop_count"] or 1
        sd = s["showdown_count"] or 1
        pa = s["total_passive"]  or 1

        vpip    = round(s["vpip_count"]     / h  * 100, 1)
        pfr     = round(s["pfr_count"]      / h  * 100, 1)
        threebet= round(s["threebet_count"] / h  * 100, 1)
        af      = round(s["total_agg"]      / pa,       2)
        wtsd    = round(s["showdown_count"] / fl * 100, 1)
        wsd     = round(s["showdown_wins"]  / sd * 100, 1)
        win_pct = round(s["win_count"]      / h  * 100, 1)
        allin   = round(s["allin_count"]    / h  * 100, 1)

        # Player profile label
        if vpip > 40:   ptype = "Loose (fish/recreational)"
        elif vpip < 15: ptype = "Nit (very tight)"
        elif pfr > vpip * 0.8: ptype = "Tight-Aggressive (TAG)"
        else: ptype = "Loose-Aggressive (LAG)"

        af_label = "very passive (bet = strong)" if af < 1.5 else "aggressive (likely bluffing)" if af > 3.0 else "balanced"

        hands_summary = []
        for h_row in list(recent_hands)[:8]:
            r = dict(h_row)
            result = "WON" if r["won"] else ("FOLDED" if r["folded"] else "LOST")
            hands_summary.append(f"#{r['hand_num']}: {result}, preflop={r['preflop_bet']}, flop={r['flop_bet']}, {r.get('hand_message','')}")

        mem0 = get_mem0()
        mem_results = mem0.search(query=f"player {data.player_name}", filters={"user_id": data.game_id}, limit=5)
        memories = [m.get("memory","") for m in (mem_results if isinstance(mem_results, list) else [])]

        prompt = f"""Write a professional 4-5 line poker HUD commentary on this player.

Player: {data.player_name}
Sample size: {s["hands_seen"]} hands

CORE STATS:
- VPIP/PFR: {vpip}% / {pfr}% — Profile: {ptype}
- 3Bet: {threebet}% | AF: {af} ({af_label})
- WTSD: {wtsd}% | W$SD: {wsd}% | Win rate: {win_pct}%
- All-in rate: {allin}%

RECENT HANDS (last 8):
{chr(10).join(hands_summary)}

RELEVANT MEMORIES:
{chr(10).join(f'- {m}' for m in memories) if memories else '- none yet'}

Write 4-5 lines of specific, actionable commentary covering:
1. Player type and general tendencies
2. Pre-flop approach and how to counter
3. Post-flop tendencies and exploits
4. One specific strategic recommendation

Be direct and specific. Avoid generic advice. Base everything on the actual stats above."""

        commentary = call_glm(prompt, max_tokens=300,
                               system="You are a poker coach. Be direct and specific.")

        # Store commentary back to player_stats
        with get_db() as db:
            db.execute(
                "UPDATE player_stats SET commentary=?, ts_updated=datetime('now') WHERE game_id=? AND player_name=?",
                (commentary, data.game_id, data.player_name)
            )

        return {
            "ok":        True,
            "player":    data.player_name,
            "commentary": commentary,
            "stats": {
                "vpip": vpip, "pfr": pfr, "threebet": threebet,
                "af": af, "wtsd": wtsd, "wsd": wsd,
                "win_pct": win_pct, "allin_pct": allin,
                "hands_seen": s["hands_seen"],
                "profile": ptype,
            }
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
