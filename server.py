"""
Poker Agent Server — Railway deployment
Mem0 Cloud + Claude API + persistent analysis history
"""

import sqlite3, json, os, asyncio
from datetime import datetime
from collections import Counter, defaultdict
from typing import Optional, Dict, Set
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import anthropic
from mem0 import MemoryClient

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MEM0_API_KEY  = os.environ.get("MEM0_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
USER_ID       = "poker_player"
DB_PATH       = "hands.db"

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
_latest_state: Dict[str, dict] = {}

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
            session_id  TEXT,
            hand_num    INTEGER,
            memory_text TEXT,
            memory_id   TEXT,
            ts          TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS claude_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            hand_num    INTEGER,
            prompt      TEXT,
            response    TEXT,
            next_hand   TEXT,
            ts          TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS analysis_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
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
        """)
    # Migrate existing DBs — safe no-op if columns already exist
    for col, typedef in [
        ("away_mode",    "INTEGER DEFAULT 0"),
        ("shown_hands",  "TEXT DEFAULT '{}'"),
        ("player_stats", "TEXT DEFAULT '{}'"),
    ]:
        try:
            db.execute(f"ALTER TABLE hands ADD COLUMN {col} {typedef}")
        except Exception:
            pass
    try:
        db.execute("ALTER TABLE claude_log ADD COLUMN next_hand TEXT")
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

@app.on_event("startup")
async def start_queue():
    asyncio.create_task(_process_claude_queue())

# ── Models ────────────────────────────────────────────────────────────────────
class HandData(BaseModel):
    session_id: str
    hand_num: int
    hole_cards: list
    board: list = []
    flop: list = []
    turn: list = []
    river: list = []
    hero_won: bool = False
    hero_folded: bool = False
    all_in: bool = False
    away_mode: bool = False
    pot: int = 0
    shown_hands: dict = {}
    winner: str = ""

class AdviceRequest(BaseModel):
    session_id: str
    hand_num: int
    hole_cards: list
    board: list
    pot: int = 0
    players_in: int = 2
    facing_bet: int = 0

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
    try:
        # Merge flop/turn/river into flat board if not already provided
        board = data.board if data.board else (data.flop + data.turn + data.river)

        with get_db() as db:
            db.execute(
                "INSERT INTO hands (session_id,hand_num,hole_cards,board,hero_won,hero_folded,all_in,pot,away_mode,shown_hands,player_stats) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (data.session_id, data.hand_num,
                 json.dumps(data.hole_cards), json.dumps(board),
                 int(data.hero_won), int(data.hero_folded),
                 int(data.all_in), data.pot,
                 int(data.away_mode),
                 json.dumps(data.shown_hands),
                 json.dumps(data.player_stats if hasattr(data, 'player_stats') else {}))
            )

        result    = "WON" if data.hero_won else ("FOLDED" if data.hero_folded else "LOST")
        mode      = "[AWAY]" if data.away_mode else ""
        board_str = " ".join(board) if board else "no board"
        memory_text = (
            f"Hand #{data.hand_num} {mode}: hero held {' '.join(data.hole_cards) if data.hole_cards else 'N/A (away)'}, "
            f"board: {board_str}, result: {result}, pot: {data.pot}, all_in: {data.all_in}"
        )
        if data.shown_hands:
            shown_str = ", ".join(f"{p}: {' '.join(c)}" for p,c in data.shown_hands.items())
            memory_text += f", opponents showed: {shown_str}"

        mem0 = get_mem0()
        result_mem = mem0.add(
            memory_text,
            user_id=USER_ID,
            metadata={"hand_num": data.hand_num, "session": data.session_id,
                      "hero_won": data.hero_won, "all_in": data.all_in}
        )
        mem0_id = result_mem.get("id") if isinstance(result_mem, dict) else None

        with get_db() as db:
            db.execute(
                "INSERT INTO mem0_log (session_id,hand_num,memory_text,memory_id) VALUES(?,?,?,?)",
                (data.session_id, data.hand_num, memory_text, str(mem0_id))
            )

        # Auto-enqueue Claude analysis — server handles it, no need for /between_hands call
        await _claude_queue.put({
            "session_id":  data.session_id,
            "hand_num":    data.hand_num,
            "hole_cards":  data.hole_cards,
            "flop":        data.flop,
            "turn":        data.turn,
            "river":       data.river,
            "hero_won":    data.hero_won,
            "hero_folded": data.hero_folded,
            "all_in":      data.all_in,
            "away_mode":   data.away_mode,
            "pot":         data.pot,
            "shown_hands": data.shown_hands,
        })

    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {"ok": True, "mem0_id": mem0_id, "memory": memory_text}

# ── POST /advice ──────────────────────────────────────────────────────────────
@app.post("/advice")
async def get_advice(data: AdviceRequest):
    try:
        # Search Mem0 for relevant patterns
        mem0   = get_mem0()
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

        query = f"hero {' '.join(data.hole_cards)} board {' '.join(data.board)}"
        mem_results = mem0.search(query=query, filters={"user_id": USER_ID}, limit=6)
        memories = [m.get("memory","") for m in (mem_results if isinstance(mem_results, list) else [])]

        # Pull last analysis for context
        with get_db() as db:
            last_analysis = db.execute(
                "SELECT claude_narrative FROM analysis_log ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        analysis_ctx = last_analysis["claude_narrative"][:500] if last_analysis else "No analysis run yet"

        system = """You are a real-time poker advisor for a PokerNow game with a known broken RNG.

PROVEN FACTS about this game:
- Hero (Poo-PokerNow) wins 67.5% of hands played — 2.7x expected
- Speculative/weak hands win 83% — NEVER fold preflop
- Premium hands only win 50% — do not over-value AA/KK
- All-in = river WILL complete someone's draw (54% rate)
- Carry rate accelerates late session: hands 300+ = 60-67% carry
- Jeezy333 wins 68.9% at showdown — fold to their aggression postflop
- Wick3 wins 69.0% at showdown — fold to their aggression postflop  
- Thanos goes all-in 14% of hands, chip feeder, call them
- 4 cards same suit on board without your flush = CHECK/FOLD
- High card is suppressed — almost everyone has at least a pair

Respond with: ACTION (FOLD/CHECK/CALL/BET/RAISE), sizing if bet/raise, and 1-2 sentence reason."""

        prompt = f"""Current hand:
Hole cards: {data.hole_cards}
Board: {data.board if data.board else 'preflop'}
Pot: {data.pot}, Facing bet: {data.facing_bet}, Players in: {data.players_in}

Relevant Mem0 memories from similar past hands:
{chr(10).join(f'- {m}' for m in memories) if memories else '- none yet'}

Last analysis summary: {analysis_ctx}

What is the best action?"""

        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            system=system,
            messages=[{"role":"user","content":prompt}]
        )
        advice = resp.content[0].text

        with get_db() as db:
            db.execute(
                "INSERT INTO claude_log (session_id,hand_num,prompt,response) VALUES(?,?,?,?)",
                (data.session_id, data.hand_num, prompt, advice)
            )

        return {"ok": True, "advice": advice, "memories_used": len(memories)}

    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── GET /analyze — full analysis, stores result historically ──────────────────
@app.get("/analyze")
async def analyze():
    try:
        mem0   = get_mem0()
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

        # Pull all Mem0 memories
        all_mems = mem0.get_all(filters={"user_id": USER_ID})
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

        prompt = f"""You are analyzing a PokerNow game with a broken RNG shuffle.

CURRENT LIVE STATS ({stats.get('total',0)} hands logged by extension):
- Win rate: {stats.get('win_rate',0)}%
- All-in win rate: {stats.get('allin_wr',0)}%
- Carry rate: {stats.get('carry_rate',0)}%
- Street wins: {stats.get('street_wins',{})}
- Street totals: {stats.get('street_totals',{})}

ALL MEM0 MEMORIES ({len(memories)} total — every hand the extension saw):
{mem_block}

HISTORICAL BASELINE (423-hand CSV pre-extension):
- Hero win rate: 67.5% | Speculative hands: 83% | Premium: 50%
- All-in river completion: 54% | Carry 300+ hands: 60-67%
- Jeezy333: 68.9% showdown WR | Wick3: 69.0% | Thanos: chip feeder

{prev_ctx}

Provide a detailed analysis covering:
1. How live play compares to the 423-hand baseline — is the rigged pattern holding?
2. New patterns emerging from Mem0 memories not visible in the CSV
3. Any player-specific tendencies captured in memories
4. Updated carry/board patterns
5. Specific strategic adjustments for next session
6. Confidence level (%) that the RNG is broken

Be specific, reference actual hands from memory where possible."""

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
            user_id=USER_ID,
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
async def all_memories():
    try:
        mem0 = get_mem0()
        all_mems = mem0.get_all(filters={"user_id": USER_ID})
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
            "SELECT hand_num,response,ts FROM claude_log "
            "WHERE session_id=? ORDER BY ts DESC LIMIT 10", (session_id,)
        ).fetchall()
    return {
        "session_id": session_id,
        "mem0_entries": [dict(r) for r in mem0_entries],
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
        m.search(query="test", filters={"user_id": USER_ID}, limit=1)
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
    try:
        session_id = data_dict.get("session_id","")
        hand_num   = data_dict.get("hand_num", 0)
        hole_cards = data_dict.get("hole_cards", [])
        flop       = data_dict.get("flop", [])
        turn       = data_dict.get("turn", [])
        river      = data_dict.get("river", [])
        hero_won   = data_dict.get("hero_won", False)
        hero_folded= data_dict.get("hero_folded", False)
        all_in     = data_dict.get("all_in", False)
        away_mode  = data_dict.get("away_mode", False)
        pot        = data_dict.get("pot", 0)
        shown_hands= data_dict.get("shown_hands", {})

        board = flop + turn + river
        mem0  = get_mem0()
        client= anthropic.Anthropic(api_key=ANTHROPIC_KEY)

        query = f"{' '.join(hole_cards)} {' '.join(board)}"
        mem_results = mem0.search(query=query, filters={"user_id": USER_ID}, limit=8)
        memories = [m.get("memory","") for m in (mem_results if isinstance(mem_results, list) else [])]

        with get_db() as db:
            last_analysis = db.execute(
                "SELECT claude_narrative, hands_at_time FROM analysis_log ORDER BY ts DESC LIMIT 1"
            ).fetchone()
            recent_hands = db.execute(
                "SELECT hole_cards, board, hero_won, all_in FROM hands ORDER BY id DESC LIMIT 10"
            ).fetchall()

        analysis_ctx = last_analysis["claude_narrative"][:300] if last_analysis else ""
        recent = [dict(r) for r in recent_hands]

        result_str = "WON" if hero_won else ("FOLDED" if hero_folded else "LOST")
        mode_str   = " [HERO AWAY]" if away_mode else ""
        shown_str  = ", ".join(f"{p}: {' '.join(c)}" for p,c in shown_hands.items()) if shown_hands else "none"

        prompt = f"""A poker hand just completed. Analyze it and prepare the player for the NEXT hand.

COMPLETED HAND #{hand_num}{mode_str}:
- Hero held: {' '.join(hole_cards) if hole_cards else 'N/A (away)'}
- Board: {' '.join(board) if board else 'no board'}
- Result: {result_str} | Pot: {pot} | All-in: {all_in}
- Opponents showed: {shown_str}

RELEVANT MEM0 MEMORIES:
{chr(10).join(f'- {m}' for m in memories) if memories else '- none yet'}

RECENT HAND TREND (last 10):
{chr(10).join(f"  hole:{r['hole_cards']} board:{r['board']} won:{r['hero_won']}" for r in recent[:5])}

LAST ANALYSIS SUMMARY: {analysis_ctx}

PROVEN FACTS: Hero wins 67.5% overall. Speculative hands win 83%. Never fold preflop.
Jeezy/Wick3 win 69% at showdown — fold to their postflop aggression.
All-in = river completes draws 54%. Carry rate accelerates 300+ hands.

Respond in JSON only:
{{
  "nextHand": "1-2 sentence instruction for what to do NEXT hand — specific action advice based on carry patterns, danger players, and board trends. Forward looking only.",
  "weight_updates": {{
    "carryBoost": 0.0,
    "jeezyWarning": true,
    "allinWarning": true,
    "hotCards": []
  }}
}}"""

        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            messages=[{"role":"user","content":prompt}]
        )
        raw    = resp.content[0].text.strip().replace('```json','').replace('```','').strip()
        parsed = json.loads(raw)
        next_hand_text = parsed.get("nextHand","")

        mem0.add(
            f"Hand #{hand_num} next hand: {next_hand_text}",
            user_id=USER_ID,
            metadata={"type":"between_hands","hand_num":hand_num}
        )

        with get_db() as db:
            db.execute(
                "INSERT INTO claude_log (session_id,hand_num,prompt,response,next_hand) VALUES(?,?,?,?,?)",
                (session_id, hand_num, prompt[:500], raw, next_hand_text)
            )

        return parsed

    except Exception as e:
        print(f"[_run_between_hands] error: {e}")
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
    # Send latest state on connect
    if session_id in _latest_state:
        try:
            await websocket.send_json({"type": "raw", "data": _latest_state[session_id]})
        except Exception:
            pass
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)

# ── POST /raw — extension sends all DOM data + computed decision ───────────────
class RawData(BaseModel):
    session_id: str
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
    payload = data.dict()
    # Add server status
    try:
        with get_db() as db:
            hc = db.execute("SELECT COUNT(*) FROM hands").fetchone()[0]
        payload["server_status"] = {"hands_logged": hc, "mem0_live": True, "claude_live": True}
    except Exception:
        payload["server_status"] = {"hands_logged": 0, "mem0_live": False, "claude_live": False}

    _latest_state[data.session_id] = payload
    await manager.broadcast(data.session_id, {"type": "raw", "data": payload})
    return {"ok": True}

# ── GET /api/state/{session_id} — full state for dashboard on reconnect ────────
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
            "server_status": {"hands_logged": hc, "mem0_live": True, "claude_live": True},
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

# ── POST /upload_csv — parse and store CSV from dashboard ─────────────────────
@app.post("/upload_csv")
async def upload_csv(request_body: str = ""):
    from fastapi import Request
    return {"ok": False, "error": "Use the Request object — see implementation note"}

# Fix upload_csv to actually read body
from fastapi import Request

@app.post("/upload_csv_v2")
async def upload_csv_v2(request: Request):
    try:
        text = (await request.body()).decode("utf-8")
        lines = text.strip().split("\n")
        # Parse hands from CSV
        hands = []
        cur = None
        for line in lines[1:]:
            m = __import__("re").match(r'^"?(.*?)"?,(\d{4}-\d{2}-\d{2}T[^,]+),(\d+)', line)
            if not m:
                continue
            entry = m.group(1).replace('""', '"')
            if "-- starting hand" in entry:
                hm = __import__("re").search(r"starting hand #(\d+)", entry)
                if hm:
                    cur = {"hand_num": int(hm.group(1)), "hole_cards": [], "flop": [], "turn": [], "river": []}
            elif "-- ending hand" in entry:
                if cur:
                    hands.append(cur)
                    cur = None
            elif cur:
                cards_re = __import__("re").compile(r'(?:10|[2-9JQKA])[♠♥♦♣]')
                if "Your hand is" in entry:
                    cur["hole_cards"] = cards_re.findall(entry)
                elif entry.startswith("Flop:"):
                    cur["flop"] = cards_re.findall(entry)
                elif entry.startswith("Turn:"):
                    tm = __import__("re").search(r"\[([^\]]+)\]$", entry)
                    if tm:
                        cur["turn"] = cards_re.findall(tm.group(1))
                elif entry.startswith("River:"):
                    rm = __import__("re").search(r"\[([^\]]+)\]$", entry)
                    if rm:
                        cur["river"] = cards_re.findall(rm.group(1))

        # Store to DB + Mem0 in background
        mem0 = get_mem0()
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
            await asyncio.gather(
                asyncio.to_thread(
                    mem0.add,
                    f"CSV upload: {len(hands)} historical hands loaded",
                    user_id=USER_ID,
                    metadata={"type": "csv_upload", "count": len(hands)}
                )
            )
        return {"ok": True, "hands": stored}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    if session_id in _latest_state:
        try:
            await websocket.send_json({"type": "raw", "data": _latest_state[session_id]})
        except Exception:
            pass
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)

# ── POST /raw ─────────────────────────────────────────────────────────────────
class RawData(BaseModel):
    session_id:    str
    hole_cards:    list = []
    board_cards:   list = []
    pot_size:      int  = 0
    bet_facing:    int  = 0
    hero_stack:    int  = 0
    is_hero_turn:  bool = False
    hero_bet:      int  = 0
    max_opp_bet:   int  = 0
    call_amount:   int  = 0
    aggressor:     str  = ""
    is_raise:      bool = False
    is_cold_call:  bool = False
    can_check:     bool = False
    win_counts:    dict = {}
    hand_messages: dict = {}
    dealer_pos:    int  = 0
    blinds:        dict = {}
    shown_hands:   dict = {}
    shown_winners: list = []
    away_mode:     bool = False
    decision:      dict = {}
    carry_alert:   dict = {}
    player_stats:  dict = {}
    danger_players: list = []
    model_meta:    dict = {}

@app.post("/raw")
async def post_raw(data: RawData):
    payload = data.dict()
    try:
        with get_db() as db:
            hc = db.execute("SELECT COUNT(*) FROM hands").fetchone()[0]
        payload["server_status"] = {"hands_logged": hc, "mem0_live": bool(MEM0_API_KEY), "claude_live": bool(ANTHROPIC_KEY)}
    except Exception:
        payload["server_status"] = {"hands_logged": 0, "mem0_live": False, "claude_live": False}
    _latest_state[data.session_id] = payload
    await manager.broadcast(data.session_id, {"type": "raw", "data": payload})
    return {"ok": True}

# ── GET /api/state/{session_id} ───────────────────────────────────────────────
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
                "SELECT hand_num, hole_cards, board, hero_won, hero_folded FROM hands "
                "WHERE session_id=? ORDER BY id DESC LIMIT 20", (session_id,)
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
            "ok": True,
            "narratives":   [dict(r) for r in narratives],
            "hand_history": hand_history,
            "player_stats": latest.get("player_stats", {}),
            "server_status": {"hands_logged": hc, "mem0_live": bool(MEM0_API_KEY), "claude_live": bool(ANTHROPIC_KEY)},
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── GET /dashboard ────────────────────────────────────────────────────────────
@app.get("/dashboard")
async def dashboard():
    from pathlib import Path
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)

# ── POST /upload_csv ──────────────────────────────────────────────────────────
from fastapi import Request as FastAPIRequest

@app.post("/upload_csv")
async def upload_csv(request: FastAPIRequest):
    import re
    try:
        text = (await request.body()).decode("utf-8")
        lines = text.strip().split("\n")
        hands = []
        cur = None
        card_re = re.compile(r'(?:10|[2-9JQKA])[\u2660\u2665\u2666\u2663]')
        for line in lines[1:]:
            m = re.match(r'^"?(.*?)"?,(\d{4}-\d{2}-\d{2}T[^,]+),(\d+)', line)
            if not m:
                continue
            entry = m.group(1).replace('\"\"',' \"')
            if "-- starting hand" in entry:
                hm = re.search(r"starting hand #(\d+)", entry)
                if hm:
                    cur = {"hand_num": int(hm.group(1)), "hole_cards": [], "flop": [], "turn": [], "river": []}
            elif "-- ending hand" in entry:
                if cur:
                    hands.append(cur)
                    cur = None
            elif cur:
                if "Your hand is" in entry:
                    cur["hole_cards"] = card_re.findall(entry)
                elif entry.startswith("Flop:"):
                    cur["flop"] = card_re.findall(entry)
                elif entry.startswith("Turn:"):
                    tm = re.search(r"\[([^\]]+)\]$", entry)
                    if tm:
                        cur["turn"] = card_re.findall(tm.group(1))
                elif entry.startswith("River:"):
                    rm = re.search(r"\[([^\]]+)\]$", entry)
                    if rm:
                        cur["river"] = card_re.findall(rm.group(1))
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
        return {"ok": True, "hands": stored}
    except Exception as e:
        return {"ok": False, "error": str(e)}
