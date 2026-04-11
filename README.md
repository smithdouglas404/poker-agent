# PokerNow Advisor — Railway + Mem0 + Claude

## Deploy in 5 steps

### Step 1 — Upload to GitHub
1. Go to github.com → New repository → call it `poker-agent`
2. Upload the contents of the `server` folder (server.py, requirements.txt, Procfile, railway.json)

### Step 2 — Deploy to Railway
1. Go to railway.app → New Project → Deploy from GitHub repo
2. Select your `poker-agent` repo
3. Railway auto-detects Python and deploys

### Step 3 — Set environment variables in Railway
In Railway dashboard → your project → Variables tab, add:

```
MEM0_API_KEY        = your_mem0_key
ANTHROPIC_API_KEY   = your_anthropic_key
```

### Step 4 — Get your Railway URL
Railway gives you a URL like:
`https://poker-agent-production-xxxx.up.railway.app`

Open extension/content.js and replace `__RAILWAY_URL__` with your actual URL:
```javascript
const SERVER = 'https://poker-agent-production-xxxx.up.railway.app';
```

### Step 5 — Load the extension
1. Open Chrome → chrome://extensions → Developer mode ON
2. Load unpacked → select the `extension` folder

## Proof it's working

Click the 📋 button in the overlay to see:
- 🟢 Every hand stored in Mem0 (with memory ID)
- 🟣 Every Claude response

The dot in the overlay header shows:
- 🟢 Green = Railway up, Mem0 connected, Claude connected
- 🟡 Yellow = Railway up but missing a key
- 🔴 Red = Railway offline
