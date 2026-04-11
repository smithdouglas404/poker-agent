"""
Poker Agent Server — Railway deployment
Uses: Mem0 Cloud + Claude API directly (no local Letta needed)
Railway provides the hosting, you set env vars in Railway dashboard
"""

import sqlite3, json, os
from datetime import datetime
from collections import Counter, defaultdict
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic        # Claude direct API
from mem0 import MemoryClient  # Mem0 cloud

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config from Railway env vars ──────────────────────────────────────────────
MEM0_API_KEY   = os.environ.get("MEM0_API_KEY",    "")
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_API_KEY","")
USER_ID        = "poker_player"

# ── SQLite (Railway persists this with a volume) ──────────────────────────────
DB_PATH = "/data/hands.db" if os.path.exists("/data") else "hands.db"

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
            flop        TEXT,
            turn        TEXT,
            river       TEXT,
            ts          TEXT DEFAULT (datetime('now'))
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
            ts          TEXT DEFAULT (datetime('now'))
        );
        """)

init_db()

# ── Clients ───────────────────────────────────────────────────────────────────
def get_mem0():
    return MemoryClient(api_key=MEM0_API_KEY)

def get_claude():
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ── Pattern stats from SQLite ─────────────────────────────────────────────────
def compute_stats(session_id: str):
    with get_db() as db:
        rows = db.execute(
            "SELECT hole_cards,flop,turn,river FROM hands WHERE session_id=? ORDER BY hand_num",
            (session_id,)
        ).fetchall()

    hands = [{"hole": json.loads(r["hole_cards"] or "[]"),
              "flop": json.loads(r["flop"] or "[]"),
              "turn": json.loads(r["turn"] or "[]"),
              "river":json.loads(r["river"] or "[]")} for r in rows]

    if len(hands) < 2:
        return {"carry_rate":0,"top_cards":[],"top_pairs":[],"hand_count":len(hands)}

    carry_pairs = carry_total = 0
    for i in range(len(hands)-1):
        s1 = set(hands[i]["hole"]+hands[i]["flop"]+hands[i]["turn"]+hands[i]["river"])
        s2 = set(hands[i+1]["hole"]+hands[i+1]["flop"]+hands[i+1]["turn"]+hands[i+1]["river"])
        if s1 and s2:
            carry_total += 1
            if s1 & s2: carry_pairs += 1

    early = {"hole_1","hole_2","flop_1","flop_2","flop_3"}
    pos_count = defaultdict(Counter)
    for h in hands:
        for j,c in enumerate(h["hole"]): pos_count[c][f"hole_{j+1}"] += 1
        for j,c in enumerate(h["flop"]): pos_count[c][f"flop_{j+1}"] += 1

    top_cards = sorted(
        [c for c in pos_count if sum(pos_count[c].values()) > 2],
        key=lambda c: -sum(pos_count[c][p] for p in early)/max(sum(pos_count[c].values()),1)
    )[:10]

    co = Counter()
    for h in hands:
        cards = list(dict.fromkeys(h["hole"]+h["flop"]+h["turn"]+h["river"]))
        for i in range(len(cards)):
            for j in range(i+1,len(cards)):
                co[tuple(sorted([cards[i],cards[j]]))] += 1

    return {
        "carry_rate":  round(carry_pairs/carry_total*100,1) if carry_total else 0,
        "top_cards":   top_cards,
        "top_pairs":   [(list(p),n) for p,n in co.most_common(8)],
        "hand_count":  len(hands),
    }

# ── API Models ────────────────────────────────────────────────────────────────
class HandData(BaseModel):
    session_id: str
    hand_num:   int
    hole_cards: list
    flop:       list = []
    turn:       list = []
    river:      list = []
    game_id:    Optional[str] = ""

class AdviceRequest(BaseModel):
    session_id:  str
    hand_num:    int
    hole_cards:  list
    board_cards: list

# ── POST /hand ────────────────────────────────────────────────────────────────
@app.post("/hand")
async def record_hand(data: HandData):
    # Store in SQLite
    with get_db() as db:
        db.execute(
            "INSERT INTO hands (session_id,hand_num,hole_cards,flop,turn,river) VALUES(?,?,?,?,?,?)",
            (data.session_id, data.hand_num,
             json.dumps(data.hole_cards), json.dumps(data.flop),
             json.dumps(data.turn),       json.dumps(data.river))
        )

    stats = compute_stats(data.session_id)

    memory_text = (
        f"Hand #{data.hand_num} | Session {data.session_id} | "
        f"Hero held {' '.join(data.hole_cards)} | "
        f"Flop: {' '.join(data.flop) or 'none'} | "
        f"Turn: {' '.join(data.turn) or 'none'} | "
        f"River: {' '.join(data.river) or 'none'} | "
        f"Carry rate: {stats['carry_rate']}% | "
        f"Top bias cards: {', '.join(stats['top_cards'][:5])}"
    )

    # Store in Mem0
    mem0_id = None
    try:
        mem0 = get_mem0()
        result = mem0.add(
            messages=[{"role": "user", "content": memory_text}],
            user_id=USER_ID,
            metadata={
                "session_id": data.session_id,
                "hand_num":   data.hand_num,
                "type":       "hand_record",
            }
        )
        mem0_id = result.get("id") if isinstance(result, dict) else None

        # Store RNG pattern warning if carry rate is high
        if stats["carry_rate"] > 45:
            mem0.add(
                messages=[{"role": "user", "content":
                    f"RNG PATTERN: Session {data.session_id} carry rate is {stats['carry_rate']}% "
                    f"(normal ~13%). Top bias cards: {stats['top_cards'][:5]}. "
                    f"Player should bet these cards aggressively."
                }],
                user_id=USER_ID,
                metadata={"type": "rng_pattern", "session_id": data.session_id}
            )
    except Exception as e:
        print(f"[MEM0] Error: {e}")

    # Log
    with get_db() as db:
        db.execute(
            "INSERT INTO mem0_log (session_id,hand_num,memory_text,memory_id) VALUES(?,?,?,?)",
            (data.session_id, data.hand_num, memory_text, str(mem0_id))
        )

    return {"ok": True, "stats": stats, "mem0_id": mem0_id, "memory": memory_text}


# ── POST /advice ──────────────────────────────────────────────────────────────
@app.post("/advice")
async def get_advice(req: AdviceRequest):
    stats = compute_stats(req.session_id)

    # Search Mem0 for relevant memories
    memories_text = ""
    raw_memories  = []
    try:
        mem0 = get_mem0()
        query = f"hand with {' '.join(req.hole_cards)} board {' '.join(req.board_cards)} carry patterns"
        results = mem0.search(query=query, user_id=USER_ID, limit=6)
        raw_memories = results.get("results", []) if isinstance(results, dict) else []
        memories_text = "\n".join(f"- {r['memory']}" for r in raw_memories)
    except Exception as e:
        print(f"[MEM0] Search error: {e}")

    # Ask Claude directly
    stage = ("river" if len(req.board_cards) >= 5 else
             "turn"  if len(req.board_cards) == 4 else
             "flop"  if len(req.board_cards) >= 2 else "preflop")

    prompt = f"""You are a poker pattern analysis agent. You have access to hand history memories.

Current situation:
- Stage: {stage}
- My hole cards: {' '.join(req.hole_cards)}
- Board: {' '.join(req.board_cards) or 'none yet'}

Session statistics (from {stats['hand_count']} hands played):
- Carry-over rate: {stats['carry_rate']}% (a fair deck would be ~13% — higher means the shuffle is biased)
- Top bias cards this session: {', '.join(stats['top_cards'][:6])}
- Hottest card pairs: {stats['top_pairs'][:3]}

Memories from past hands in this session:
{memories_text or 'No memories yet — early in session.'}

Give me:
1. ONE LINE: RAISE/BET/CALL/CHECK/FOLD + sizing (e.g. "RAISE 3x" or "BET 1/2 pot")
2. 2-3 bullet points explaining why — reference specific cards and patterns you remember
3. One card to watch for on the next street

Be specific. Reference actual hand numbers and cards from the memories above."""

    claude_response = ""
    try:
        claude = get_claude()
        msg = claude.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        claude_response = msg.content[0].text
    except Exception as e:
        print(f"[CLAUDE] Error: {e}")
        claude_response = ""

    # Log
    with get_db() as db:
        db.execute(
            "INSERT INTO claude_log (session_id,hand_num,prompt,response) VALUES(?,?,?,?)",
            (req.session_id, req.hand_num, prompt, claude_response)
        )

    return {
        "stats":         stats,
        "agent_advice":  claude_response,
        "memories_used": raw_memories,
        "carry_rate":    stats["carry_rate"],
        "hand_count":    stats["hand_count"],
    }


# ── GET /proof ────────────────────────────────────────────────────────────────
@app.get("/proof/{session_id}")
async def proof(session_id: str):
    with get_db() as db:
        mem0_entries = db.execute(
            "SELECT hand_num,memory_text,memory_id,ts FROM mem0_log "
            "WHERE session_id=? ORDER BY id DESC LIMIT 20",
            (session_id,)
        ).fetchall()
        claude_entries = db.execute(
            "SELECT hand_num,response,ts FROM claude_log "
            "WHERE session_id=? ORDER BY id DESC LIMIT 10",
            (session_id,)
        ).fetchall()
    return {
        "mem0_entries":   [dict(r) for r in mem0_entries],
        "letta_entries":  [dict(r) for r in claude_entries],
    }


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    # Show first 8 chars of each key so you can verify they loaded
    mem0_preview    = MEM0_API_KEY[:8]   + "…" if MEM0_API_KEY    else "NOT SET"
    claude_preview  = ANTHROPIC_KEY[:8]  + "…" if ANTHROPIC_KEY   else "NOT SET"
    mem0_ok         = bool(MEM0_API_KEY)
    claude_ok       = bool(ANTHROPIC_KEY)

    # Try actually calling Mem0
    mem0_live = False
    mem0_error = ""
    try:
        m = get_mem0()
        m.get_all(user_id=USER_ID)
        mem0_live = True
    except Exception as e:
        mem0_error = str(e)[:100]

    # Try actually calling Claude
    claude_live = False
    claude_error = ""
    try:
        c = get_claude()
        c.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=10,
            messages=[{"role":"user","content":"hi"}]
        )
        claude_live = True
    except Exception as e:
        claude_error = str(e)[:100]

    return {
        "ok":           True,
        "mem0_key":     mem0_preview,
        "claude_key":   claude_preview,
        "mem0_live":    mem0_live,
        "claude_live":  claude_live,
        "mem0_error":   mem0_error,
        "claude_error": claude_error,
        "db":           DB_PATH
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8765))
    uvicorn.run(app, host="0.0.0.0", port=port)
