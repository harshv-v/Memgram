# Memgram Dashboard

Minimal Next.js UI for the user side of memory: view/edit **instructions**,
browse **semantic memories** (with decay tier + retention), and **approve or
dismiss** agent proposals (the trust gate).

```bash
cd dashboard
npm install
cp .env.local.example .env.local   # optional; values are also editable in the UI header
npm run dev                        # http://localhost:3000
```

The API base URL, API key, project/agent/user IDs are editable live in the
header and persisted to `localStorage`, so you can point it at any Memgram
backend without rebuilding.
