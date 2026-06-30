"use client";

import { useCallback, useEffect, useState } from "react";

// ---- config (persisted in localStorage; defaults from env) -----------------
type Cfg = { apiBase: string; apiKey: string; projectId: string; agentId: string; userId: string };
const DEFAULT_CFG: Cfg = {
  apiBase: process.env.NEXT_PUBLIC_MEMGRAM_API_URL || "http://localhost:8000",
  apiKey: process.env.NEXT_PUBLIC_MEMGRAM_API_KEY || "mgram_dev_key",
  projectId: "demo-project",
  agentId: "demo-agent",
  userId: "u1",
};

function useCfg(): [Cfg, (c: Partial<Cfg>) => void] {
  const [cfg, setCfg] = useState<Cfg>(DEFAULT_CFG);
  useEffect(() => {
    try {
      const saved = localStorage.getItem("memgram.cfg");
      if (saved) setCfg({ ...DEFAULT_CFG, ...JSON.parse(saved) });
    } catch {}
  }, []);
  const update = (c: Partial<Cfg>) =>
    setCfg((prev) => {
      const next = { ...prev, ...c };
      try { localStorage.setItem("memgram.cfg", JSON.stringify(next)); } catch {}
      return next;
    });
  return [cfg, update];
}

// ---- tiny API helper -------------------------------------------------------
function useApi(cfg: Cfg) {
  return useCallback(
    async (path: string, init: RequestInit = {}) => {
      const r = await fetch(cfg.apiBase + path, {
        ...init,
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${cfg.apiKey}`, ...(init.headers || {}) },
      });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      return r.status === 204 ? null : r.json();
    },
    [cfg.apiBase, cfg.apiKey]
  );
}

const q = (cfg: Cfg) => `project_id=${cfg.projectId}&agent_id=${cfg.agentId}&user_id=${cfg.userId}`;

// ===========================================================================
export default function Page() {
  const [cfg, setCfg] = useCfg();
  const [tab, setTab] = useState<"instructions" | "memories" | "proposals">("instructions");
  const [err, setErr] = useState("");

  return (
    <div className="wrap">
      <h1>Memgram</h1>
      <p className="sub">Everything your agent remembers about this user — yours to view, edit, and approve.</p>

      <div className="config">
        {(["apiBase", "apiKey", "projectId", "agentId", "userId"] as (keyof Cfg)[]).map((k) => (
          <input key={k} value={cfg[k]} placeholder={k} onChange={(e) => setCfg({ [k]: e.target.value })} />
        ))}
      </div>

      <div className="tabs">
        {(["instructions", "memories", "proposals"] as const).map((t) => (
          <div key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => { setErr(""); setTab(t); }}>
            {t[0].toUpperCase() + t.slice(1)}
          </div>
        ))}
      </div>

      {err && <div className="err">{err}</div>}
      {tab === "instructions" && <Instructions cfg={cfg} onErr={setErr} />}
      {tab === "memories" && <Memories cfg={cfg} onErr={setErr} />}
      {tab === "proposals" && <Proposals cfg={cfg} onErr={setErr} />}
    </div>
  );
}

// ---- Instructions ----------------------------------------------------------
function Instructions({ cfg, onErr }: { cfg: Cfg; onErr: (s: string) => void }) {
  const api = useApi(cfg);
  const [items, setItems] = useState<any[]>([]);
  const [content, setContent] = useState("");
  const [priority, setPriority] = useState(2);

  const load = useCallback(() => {
    api(`/v1/instructions?${q(cfg)}&status=active`).then((d) => setItems(d.instructions)).catch((e) => onErr(String(e)));
  }, [api, cfg, onErr]);
  useEffect(load, [load]);

  const add = async () => {
    if (!content.trim()) return;
    try {
      await api("/v1/instructions", { method: "POST", body: JSON.stringify({
        project_id: cfg.projectId, agent_id: cfg.agentId, user_id: cfg.userId, content, priority, source: "user",
      }) });
      setContent(""); load();
    } catch (e) { onErr(String(e)); }
  };
  const setPrio = async (id: string, p: number) => { try { await api(`/v1/instructions/${id}`, { method: "PATCH", body: JSON.stringify({ priority: p }) }); load(); } catch (e) { onErr(String(e)); } };
  const del = async (id: string) => { try { await api(`/v1/instructions/${id}`, { method: "DELETE" }); load(); } catch (e) { onErr(String(e)); } };

  return (
    <>
      <div className="addrow">
        <input value={content} placeholder="Add a standing instruction, e.g. “Always answer in TypeScript.”" onChange={(e) => setContent(e.target.value)} onKeyDown={(e) => e.key === "Enter" && add()} />
        <select value={priority} onChange={(e) => setPriority(Number(e.target.value))}>
          <option value={1}>1 · always</option>
          <option value={2}>2 · strong</option>
          <option value={3}>3 · soft</option>
        </select>
        <button onClick={add}>Add</button>
      </div>
      {items.length === 0 && <div className="empty">No instructions yet.</div>}
      {items.map((it) => (
        <div className="card" key={it.id}>
          <div className="row">
            <span>{it.content}</span>
            <span style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <select value={it.priority} onChange={(e) => setPrio(it.id, Number(e.target.value))}>
                <option value={1}>P1</option><option value={2}>P2</option><option value={3}>P3</option>
              </select>
              <span className="tag">{it.source}</span>
              <button className="danger" onClick={() => del(it.id)}>Delete</button>
            </span>
          </div>
        </div>
      ))}
    </>
  );
}

// ---- Memories --------------------------------------------------------------
const SCOPE_LABEL: Record<string, string> = {
  global: "shared · all agents", project: "shared · project", private: "private · this agent",
};

function Memories({ cfg, onErr }: { cfg: Cfg; onErr: (s: string) => void }) {
  const api = useApi(cfg);
  const [items, setItems] = useState<any[]>([]);
  const [live, setLive] = useState(false);
  const load = useCallback(() => {
    api(`/v1/memory?${q(cfg)}&limit=200`).then((d) => setItems(d.memories)).catch((e) => onErr(String(e)));
  }, [api, cfg, onErr]);
  useEffect(load, [load]);
  // real-time: poll every 3s while "live" is on, so memories appear as the worker adds them
  useEffect(() => {
    if (!live) return;
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [live, load]);
  const del = async (id: string) => { try { await api(`/v1/memory/${id}`, { method: "DELETE" }); load(); } catch (e) { onErr(String(e)); } };

  const download = async () => {
    try {
      const data = await api(`/v1/memory/export/${cfg.userId}?project_id=${cfg.projectId}`);
      const blob = new Blob([JSON.stringify({ memgram_export_version: 1, data }, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `memgram-${cfg.projectId}-${cfg.userId}.json`;
      a.click(); URL.revokeObjectURL(a.href);
    } catch (e) { onErr(String(e)); }
  };

  return (
    <>
      <div className="row" style={{ marginBottom: 12 }}>
        <span className="muted">{items.length} memories · everything below lives on your own machine</span>
        <span style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} /> live
          </label>
          <button onClick={download} title="Export this user's entire memory as a portable JSON file">⬇ Download memory</button>
        </span>
      </div>
      {items.length === 0 && <div className="empty">No semantic memories yet. They appear after the extractor runs on a conversation.</div>}
      {items.map((m) => (
        <div className="card" key={m.id} style={m.superseded ? { opacity: 0.5 } : undefined}>
          <div className="row">
            <span>{m.content}</span>
            <button className="danger" onClick={() => del(m.id)}>Forget</button>
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            <span className="muted">
              <span className={`tag tier-${m.memory_tier}`}>{m.memory_tier}</span>{" "}
              <span className="tag" title="who can see this memory">{SCOPE_LABEL[m.scope] || m.scope}</span>{" "}
              {m.memory_type} · reinforced ×{m.reinforcement_count}
              {m.emotional_weight > 1 ? " · ⚑ correction" : ""}
              {m.superseded ? " · ⊘ superseded" : ""}
            </span>
            <span className="muted" title="retention score (Ebbinghaus)">
              <span className="bar"><i style={{ width: `${Math.round(Math.min(1, m.retention_score) * 100)}%` }} /></span>
            </span>
          </div>
        </div>
      ))}
    </>
  );
}

// ---- Proposals -------------------------------------------------------------
function Proposals({ cfg, onErr }: { cfg: Cfg; onErr: (s: string) => void }) {
  const api = useApi(cfg);
  const [items, setItems] = useState<any[]>([]);
  const load = useCallback(() => {
    api(`/v1/proposals?${q(cfg)}`).then((d) => setItems(d.proposals)).catch((e) => onErr(String(e)));
  }, [api, cfg, onErr]);
  useEffect(load, [load]);
  const act = async (id: string, verb: "approve" | "reject") => { try { await api(`/v1/proposals/${id}/${verb}`, { method: "POST" }); load(); } catch (e) { onErr(String(e)); } };

  return (
    <>
      {items.length === 0 && <div className="empty">No pending proposals. The agent proposes a permanent instruction once a pattern repeats enough times.</div>}
      {items.map((p) => (
        <div className="card" key={p.id}>
          <div style={{ marginBottom: 10 }}>
            “{p.content}” <span className="muted">— proposed (priority {p.priority})</span>
          </div>
          <div className="row">
            <span className="muted">Agents can only propose. Promoting to a standing instruction is your call.</span>
            <span style={{ display: "flex", gap: 8 }}>
              <button className="good" onClick={() => act(p.id, "approve")}>Approve</button>
              <button className="ghost" onClick={() => act(p.id, "reject")}>Dismiss</button>
            </span>
          </div>
        </div>
      ))}
    </>
  );
}
