"""
Poker Agent Server v32 — Railway deployment
FastAPI + SQLite + LangGraph + Letta Cloud + Mem0 + Claude
"""

import sqlite3, json, os, asyncio
from datetime import datetime
from collections import Counter, defaultdict
from typing import Optional, Dict, Set, Any, TypedDict
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

@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(_process_claude_queue())
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MEM0_API_KEY  = os.environ.get("MEM0_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LETTA_API_KEY = os.environ.get("LETTA_API_KEY", "")
DB_PATH       = "hands.db"
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
            return existing.id
        agent = client.agents.create(
            name=f"poker_{game_id}",
            model="claude-sonnet-4-5",
            embedding="openai/text-embedding-ada-002",
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
        print(f"[letta] Created agent for game {game_id}: {agent.id}")
        return agent.id
    except Exception as e:
        print(f"[letta] Error getting/creating agent: {e}")
        return ""

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
_active_game_id: str = ""  # most recently active game — dashboard fetches this on load

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

        live_wr  = round(won/total*100, 1)      if total  > 0 else 0
        allin_wr = round(allin_w/allin_t*100,1) if allin_t > 0 else 0

        # Carry rate from last 50 hands
        recent50 = []
        with get_db() as db:
            recent50 = [dict(r) for r in db.execute("SELECT hole_cards, board FROM hands WHERE game_id=? ORDER BY id DESC LIMIT 50", (state["game_id"],)).fetchall()]
        carry_pairs = sum(
            1 for i in range(1, len(recent50))
            if set(json.loads(recent50[i-1].get("board","[]") or "[]")) &
               set(json.loads(recent50[i].get("hole_cards","[]") or "[]") + json.loads(recent50[i].get("board","[]") or "[]"))
        )
        carry_rate = round(carry_pairs / max(len(recent50)-1, 1) * 100, 1)

        # Danger players from player_stats
        danger = [p for p,s in state["player_stats"].items()
                  if s.get("showdowns",0) >= 3 and s.get("winRate",0) >= 60]

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
        query = f"{' '.join(state['hole_cards'])} {' '.join(state['board'])}"
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
  "weight_updates": {{"carryBoost": 0.0, "jeezyWarning": false, "allinWarning": false, "hotCards": []}}
}}"""

                response = client.agents.messages.create(
                    agent_id=agent_id,
                    messages=[{"role": "user", "content": message}]
                )
                letta_text = ""
                for msg in response.messages:
                    if hasattr(msg, 'text') and msg.text:
                        letta_text = msg.text; break
                    if hasattr(msg, 'content') and msg.content:
                        letta_text = msg.content; break
                if letta_text:
                    return {"letta_response": letta_text}
        except Exception as e:
            print(f"[graph:letta_reason] Letta failed, falling back to Claude: {e}")

    # Fallback: direct Claude call
    try:
        ls = state["live_stats"]
        result_str = "WON" if state["hero_won"] else ("FOLDED" if state["hero_folded"] else "LOST")
        shown = ", ".join(f"{p}: {' '.join(c)}" for p,c in state["shown_hands"].items()) if state["shown_hands"] else "none"
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        prompt = f"""Hand #{state['hand_num']}: {result_str} | Pot: {state['pot']} | All-in: {state['all_in']}
Hero: {' '.join(state['hole_cards'])} Board: {' '.join(state['board'])}
Opponents: {shown}
Stats: WR={ls.get('win_rate',0)}% Carry={ls.get('carry_rate',0)}% Danger={state['danger_players']}
Memories: {'; '.join(state['memories'][:3]) if state['memories'] else 'none'}

Respond JSON only: {{"nextHand": "instruction", "weight_updates": {{"carryBoost": 0.0, "jeezyWarning": false, "allinWarning": false, "hotCards": []}}}}"""
        resp = client.messages.create(model="claude-sonnet-4-5", max_tokens=300,
                                      messages=[{"role":"user","content":prompt}])
        return {"letta_response": resp.content[0].text.strip()}
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
                "carryBoost": 0.0, "jeezyWarning": False,
                "allinWarning": False, "hotCards": []
            })
        }
    except Exception as e:
        print(f"[graph:parse_updates] {e}")
        return {
            "next_hand": "",
            "weight_updates": {"carryBoost": 0.0, "jeezyWarning": False, "allinWarning": False, "hotCards": []}
        }

def node_apply_danger(state: HandAnalysisState) -> dict:
    """Node 5a: Danger player detected — force jeezyWarning on."""
    wu = dict(state.get("weight_updates", {}))
    wu["jeezyWarning"] = True
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
    """Node 6: Store analysis back to Mem0 and DB."""
    try:
        next_hand = state.get("next_hand","")
        if next_hand:
            mem0 = get_mem0()
            mem0.add(
                f"Hand #{state['hand_num']} analysis: {next_hand}",
                user_id=state["game_id"],
                metadata={"type":"between_hands","hand_num":state["hand_num"]}
            )
        with get_db() as db:
            db.execute(
                "INSERT INTO claude_log (game_id,session_id,hand_num,prompt,response,next_hand) VALUES(?,?,?,?,?,?)",
                (state["game_id"], state["session_id"], state["hand_num"],
                 f"LangGraph+Letta hand #{state['hand_num']}", state.get("letta_response","")[:500], next_hand)
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
    conn = sqlite3.connect(DB_PATH)
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
            ts           TEXT    DEFAULT (datetime('now'))
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
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id     TEXT,
            session_id  TEXT,
            hand_num    INTEGER,
            prompt      TEXT,
            response    TEXT,
            next_hand   TEXT,
            ts          TEXT DEFAULT (datetime('now'))
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
            ts               TEXT DEFAULT (datetime('now'))
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
        ("analysis_log", "game_id",           "TEXT"),
        ("hands",        "shown_hands",       "TEXT DEFAULT '{}'"),
        ("hands",        "player_stats",      "TEXT DEFAULT '{}'"),
        ("claude_log",   "next_hand",         "TEXT"),
        # New player_hands columns
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
    ]
    with get_db() as db:
        for table, col, typedef in migrations:
            try:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")
            except Exception:
                pass

init_db()

def get_mem0():
    return MemoryClient(api_key=MEM0_API_KEY)

# ── Server-side Claude queue ──────────────────────────────────────────────────
_claude_queue   = asyncio.Queue()
_queue_running  = False

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
    winner:          str  = ""
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
            db.execute(
                "INSERT INTO hands (game_id,session_id,hand_num,hole_cards,board,hero_won,hero_folded,all_in,pot,away_mode,shown_hands,player_stats) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (game_id, data.session_id, data.hand_num,
                 json.dumps(data.hole_cards), json.dumps(board),
                 int(data.hero_won), int(data.hero_folded), int(data.all_in),
                 data.pot, int(data.away_mode), json.dumps(data.shown_hands),
                 json.dumps(data.players_in_hand))
            )
            for p in data.players_in_hand:
                db.execute(
                    """INSERT INTO player_hands
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
                     p.get("folded_to_steal",0), p.get("folded_to_steal",0),
                     p.get("faced_three_bet",0), p.get("folded_to_three_bet",0),
                     p.get("in_steal_position",0) if p.get("pfr",0) else 0,  # cbet_opps = was PFR
                     p.get("made_cbet",0), p.get("faced_cbet",0), p.get("folded_to_cbet",0))
                )

        result = "WON" if data.hero_won else ("FOLDED" if data.hero_folded else "LOST")
        mode   = "[AWAY]" if data.away_mode else ""
        memory_text = (f"Hand #{data.hand_num} {mode}: hero held {' '.join(data.hole_cards) if data.hole_cards else 'N/A'}, "
                       f"board: {' '.join(board) if board else 'no board'}, result: {result}, pot: {data.pot}, all_in: {data.all_in}")
        if data.shown_hands:
            memory_text += f", opponents showed: {', '.join(f'{p}: {chr(32).join(c)}' for p,c in data.shown_hands.items())}"

        mem0 = get_mem0()
        result_mem = mem0.add(memory_text, user_id=game_id,
                              metadata={"hand_num": data.hand_num, "game_id": game_id, "hero_won": data.hero_won})
        mem0_id = result_mem.get("id") if isinstance(result_mem, dict) else None
        with get_db() as db:
            db.execute("INSERT INTO mem0_log (game_id,session_id,hand_num,memory_text,memory_id) VALUES(?,?,?,?,?)",
                       (game_id, data.session_id, data.hand_num, memory_text, str(mem0_id)))

        await _claude_queue.put({
            "session_id": data.session_id, "game_id": game_id,
            "hand_num": data.hand_num, "hole_cards": data.hole_cards,
            "flop": data.flop, "turn": data.turn, "river": data.river,
            "hero_won": data.hero_won, "hero_folded": data.hero_folded,
            "all_in": data.all_in, "away_mode": data.away_mode,
            "pot": data.pot, "shown_hands": data.shown_hands,
            "player_stats": {p["player_name"]: p for p in data.players_in_hand},
        })
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "mem0_id": mem0_id, "memory": memory_text}

# ── GET /analyze — full analysis, stores result historically ──────────────────
@app.get("/analyze")
async def analyze(game_id: str = ""):
    try:
        mem0   = get_mem0()
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

        # Use game_id as Mem0 user namespace — fall back to listing all if not provided
        mem_user = game_id if game_id else None
        if mem_user:
            all_mems = mem0.get_all(filters={"user_id": mem_user})
        else:
            all_mems = mem0.get_all(filters={"user_id": "poker_player"})  # legacy fallback
        memories = []
        if isinstance(all_mems, list):
            for m in all_mems:
                memories.append(m.get("memory",""))

        # Pull all hands from DB
        with get_db() as db:
            hand_rows = [dict(r) for r in db.execute("SELECT * FROM hands ORDER BY id ASC").fetchall()]
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

        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            messages=[{"role":"user","content":prompt}]
        )
        narrative = resp.content[0].text

        # ── STORE in analysis_log so it builds historical record ──────────────
        with get_db() as db:
            db.execute(
                """INSERT INTO analysis_log
                   (hands_at_time, mem0_count, win_rate, carry_rate, allin_wr,
                    street_wins, street_totals, memories_snapshot, claude_narrative)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
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
            user_id=game_id if game_id else "poker_player",
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
        mem_user = game_id if game_id else "poker_player"
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
                "SELECT hand_num, memory_text, ts FROM mem0_log ORDER BY ts DESC"
            ).fetchall()]
        return {"ok": True, "total": len(memories), "memories": memories, "db_log": db_log}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── GET /proof/{session_id} ───────────────────────────────────────────────────
@app.get("/proof/{session_id}")
async def proof(session_id: str):
    with get_db() as db:
        mem0_entries = db.execute(
            "SELECT hand_num,memory_text,memory_id,ts FROM mem0_log "
            "WHERE session_id=? ORDER BY ts DESC", (session_id,)
        ).fetchall()
        claude_entries = db.execute(
            "SELECT hand_num,response,next_hand,ts FROM claude_log "
            "WHERE session_id=? ORDER BY ts DESC LIMIT 10", (session_id,)
        ).fetchall()
    return {
        "session_id":    session_id,
        "mem0_entries":  [dict(r) for r in mem0_entries],
        "claude_entries": [dict(r) for r in claude_entries],
    }

# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    mem0_preview   = MEM0_API_KEY[:8]  + "…" if MEM0_API_KEY  else "NOT SET"
    claude_preview = ANTHROPIC_KEY[:8] + "…" if ANTHROPIC_KEY else "NOT SET"

    mem0_live = False; mem0_error = ""
    try:
        m = get_mem0()
        m.search(query="test", filters={"user_id": "health_check"}, limit=1)
        mem0_live = True
    except Exception as e:
        mem0_error = str(e)[:100]

    claude_live = False; claude_error = ""
    try:
        c = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        c.messages.create(model="claude-sonnet-4-5", max_tokens=10,
                          messages=[{"role":"user","content":"hi"}])
        claude_live = True
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
        "db": "hands.db",
        "hands_logged": hand_count,
        "mem0_memories_logged": mem0_log_count,
        "analysis_runs": analysis_count
    }

# ── Core Claude analysis function (called by queue) ──────────────────────────
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

        await manager.broadcast(session_id, {
            "type": "weight_updates",
            "data": {
                "hand_num":       data_dict.get("hand_num", 0),
                "next_hand":      next_hand,
                "weight_updates": weight_updates,
                "carry_rate":     live_stats.get("carry_rate", 0),
                "live_wr":        live_stats.get("win_rate", 0),
                "danger_players": danger_players,
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
    await _claude_queue.put(data.__dict__)
    return {"ok": True, "queued": True}

# ── GET /narratives — all stored Claude next-hand notes ──────────────────────
@app.get("/narratives/{session_id}")
async def narratives(session_id: str):
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT hand_num, next_hand, ts FROM claude_log "
                "WHERE session_id=? AND next_hand IS NOT NULL "
                "ORDER BY hand_num DESC LIMIT 50",
                (session_id,)
            ).fetchall()
        return {"ok": True, "narratives": [dict(r) for r in rows]}
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
    # Send latest state on connect — check both keys
    state = _latest_state.get(session_id) or _latest_state.get(_active_game_id)
    if state:
        try:
            await websocket.send_json({"type": "raw", "data": state})
        except Exception:
            pass
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

@app.post("/raw")
async def post_raw(data: RawData):
    payload = data.model_dump()
    try:
        with get_db() as db:
            hc = db.execute("SELECT COUNT(*) FROM hands").fetchone()[0]
        payload["server_status"] = {
            "hands_logged": hc,
            "mem0_live":    bool(MEM0_API_KEY),
            "claude_live":  bool(ANTHROPIC_KEY),
        }
    except Exception:
        payload["server_status"] = {"hands_logged": 0, "mem0_live": False, "claude_live": False}

    global _active_game_id
    route_key       = data.game_id or data.session_id
    _active_game_id = route_key
    _latest_state[route_key] = payload
    await manager.broadcast(route_key, {"type": "raw", "data": payload})
    return {"ok": True}

# ── GET /api/state/{session_id} — full state for dashboard on reconnect ────────
@app.get("/active")
async def get_active():
    """Dashboard calls this on load to get the current game ID — no URL params needed."""
    return {"game_id": _active_game_id, "ok": bool(_active_game_id)}

@app.get("/api/state/{session_id}")
async def get_state(session_id: str):
    try:
        with get_db() as db:
            narratives = db.execute(
                "SELECT hand_num, next_hand, ts FROM claude_log "
                "WHERE session_id=? AND next_hand IS NOT NULL ORDER BY hand_num DESC LIMIT 50",
                (session_id,)
            ).fetchall()
            hands = db.execute(
                "SELECT hand_num, hole_cards, board, hero_won, hero_folded, shown_hands FROM hands "
                "WHERE session_id=? ORDER BY id DESC LIMIT 20",
                (session_id,)
            ).fetchall()
            hc = db.execute("SELECT COUNT(*) FROM hands").fetchone()[0]

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
                "claude_live":  bool(ANTHROPIC_KEY),
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── GET /dashboard — serves the dashboard HTML ─────────────────────────────────
@app.get("/dashboard")
async def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Dashboard not found. Deploy dashboard.html alongside server.py</h1>", status_code=404)

# ── POST /upload_csv — parse CSV from dashboard ───────────────────────────────
from fastapi import Request as FastAPIRequest

@app.post("/upload_csv")
async def upload_csv(request: FastAPIRequest, game_id: str = "csv_upload"):
    import re
    try:
        text  = (await request.body()).decode("utf-8")
        lines = text.strip().split("\n")
        hands, cur = [], None
        card_re = re.compile(r'(?:10|[2-9JQKA])[♠♥♦♣]')
        for line in lines[1:]:
            m = re.match(r'^"?(.*?)"?,(\d{4}-\d{2}-\d{2}T[^,]+),(\d+)', line)
            if not m: continue
            entry = m.group(1).replace('""', '"')
            if "-- starting hand" in entry:
                hm = re.search(r"starting hand #(\d+)", entry)
                if hm: cur = {"hand_num": int(hm.group(1)), "hole_cards": [], "flop": [], "turn": [], "river": []}
            elif "-- ending hand" in entry:
                if cur: hands.append(cur); cur = None
            elif cur:
                if "Your hand is" in entry: cur["hole_cards"] = card_re.findall(entry)
                elif entry.startswith("Flop:"): cur["flop"] = card_re.findall(entry)
                elif entry.startswith("Turn:"):
                    tm = re.search(r"\[([^\]]+)\]$", entry)
                    if tm: cur["turn"] = card_re.findall(tm.group(1))
                elif entry.startswith("River:"):
                    rm = re.search(r"\[([^\]]+)\]$", entry)
                    if rm: cur["river"] = card_re.findall(rm.group(1))
        stored = 0
        with get_db() as db:
            for h in hands:
                board = h["flop"] + h["turn"] + h["river"]
                try:
                    db.execute(
                        "INSERT OR IGNORE INTO hands (session_id,hand_num,hole_cards,board) VALUES(?,?,?,?)",
                        ("csv_upload", h["hand_num"], json.dumps(h["hole_cards"]), json.dumps(board))
                    )
                    stored += 1
                except Exception:
                    pass
        try:
            get_mem0().add(
                f"CSV upload: {len(hands)} historical hands loaded",
                user_id=game_id,
                metadata={"type": "csv_upload", "count": len(hands)}
            )
        except Exception:
            pass
        return {"ok": True, "hands": stored}
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

        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        resp   = client.messages.create(
            model="claude-sonnet-4-5", max_tokens=300,
            messages=[{"role":"user","content":prompt}]
        )
        commentary = resp.content[0].text.strip()

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
