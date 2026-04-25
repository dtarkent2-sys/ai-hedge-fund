"""Minimal single-page webview for the AI Hedge Fund.

Launch with:  poetry run aihedgefund-web
Then open:    http://127.0.0.1:7860
"""
import os
import sys

# Force UTF-8 on stdout/stderr so Rich's ✓/✗/⋯ glyphs and any unicode prints
# from downstream code can never trigger a cp1252 UnicodeEncodeError under
# uvicorn on Windows.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import threading
import traceback

from dateutil.relativedelta import relativedelta
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.llm.models import OLLAMA_MODELS
from src.main import run_hedge_fund
from src.utils.analysts import ANALYST_CONFIG, ANALYST_ORDER
from src.utils.llm import RunCancelledError
from src.utils.progress import progress

# Kill the Rich Live terminal display inside the web process — it is TTY-oriented,
# fragile on Windows encodings, and redundant (the browser has its own ticker).
progress.set_quiet(True)


# Single-user webview: at most ONE run in flight at a time. A duplicate
# /api/run-stream while another is active gets rejected with 409 instead
# of stacking up extra LLM-call fan-outs that all compete for the same
# Ollama Cloud concurrency budget. The active run's cancel_event lives
# here too so a new request can interrupt the previous one if the user
# explicitly opts in via ?force=1.
_run_lock = threading.Lock()
_active_cancel_event: Optional[threading.Event] = None


app = FastAPI(title="AI Hedge Fund — Local Webview")


class RunRequest(BaseModel):
    tickers: List[str]
    model: str = "gpt-oss:20b"
    # Optional per-bucket overrides (Ollama only). When set they override
    # `model` for every analyst/persona agent (analyst_model) or for the
    # portfolio manager (pm_model). Risk management is pure-math and uses
    # neither.
    analyst_model: Optional[str] = None
    pm_model: Optional[str] = None
    analysts: Optional[List[str]] = None
    initial_cash: float = 100000.0
    margin_requirement: float = 0.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    show_reasoning: bool = False


@app.get("/api/config")
def api_config():
    return {
        "models": [
            {"name": m.model_name, "display": m.display_name}
            for m in OLLAMA_MODELS
            if m.model_name != "-"
        ],
        "analysts": [
            {
                "key": key,
                "display": display,
                "kind": ANALYST_CONFIG.get(key, {}).get("kind", "heuristic"),
                "description": ANALYST_CONFIG.get(key, {}).get("description"),
            }
            for display, key in ANALYST_ORDER
        ],
        "ollama_key_set": bool(os.getenv("OLLAMA_API_KEY")),
    }


@app.post("/api/run")
def api_run(req: RunRequest):
    end = req.end_date or datetime.now().strftime("%Y-%m-%d")
    if req.start_date:
        start = req.start_date
    else:
        start = (datetime.strptime(end, "%Y-%m-%d") - relativedelta(months=3)).strftime("%Y-%m-%d")

    portfolio = {
        "cash": req.initial_cash,
        "margin_requirement": req.margin_requirement,
        "margin_used": 0.0,
        "positions": {
            t: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for t in req.tickers
        },
        "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in req.tickers},
    }

    try:
        result = run_hedge_fund(
            tickers=req.tickers,
            start_date=start,
            end_date=end,
            portfolio=portfolio,
            show_reasoning=req.show_reasoning,
            selected_analysts=req.analysts or [],
            model_name=req.model,
            model_provider="Ollama",
            analyst_model=req.analyst_model,
            pm_model=req.pm_model,
        )
    except Exception as e:
        print("run_hedge_fund raised:", type(e).__name__, e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return JSONResponse(
            {"error": f"{type(e).__name__}: {e}"}, status_code=500
        )

    payload = {
        "decisions": result.get("decisions"),
        "analyst_signals": result.get("analyst_signals"),
        "window": {"start": start, "end": end},
        "model": req.model,
    }
    # Walk through jsonable_encoder first so pydantic / datetime / numpy / etc.
    # can't poison FastAPI's response serializer.
    try:
        encoded = jsonable_encoder(payload)
        body_bytes = len(json.dumps(encoded))
    except Exception as e:
        print("response serialization failed:", type(e).__name__, e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return JSONResponse(
            {"error": f"response serialization failed: {type(e).__name__}: {e}"},
            status_code=500,
        )

    n_agents = len(payload["analyst_signals"] or {})
    n_sigs = sum(len(v or {}) for v in (payload["analyst_signals"] or {}).values())
    print(
        f"/api/run OK: tickers={req.tickers} model={req.model} "
        f"agents={n_agents} signals={n_sigs} bytes={body_bytes}",
        file=sys.stderr,
        flush=True,
    )

    # Best-effort: index this run into Onyx's RAG corpus so future chat
    # queries ("what did Buffett say about NVDA last month") can retrieve it.
    # Controlled by ONYX_INDEX_URL — unset to disable. Failures are silent.
    onyx_url = os.getenv("ONYX_INDEX_URL")
    if onyx_url:
        try:
            _index_run_in_onyx(onyx_url, req, payload)
        except Exception as e:
            print(f"[onyx-index] failed (non-fatal): {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)

    return JSONResponse(content=encoded)


def _index_run_in_onyx(onyx_url: str, req: RunRequest, payload: dict) -> None:
    """Post the run as a document to Onyx's ingestion API.

    The document is structured: a YAML-ish header with metadata + per-agent
    sections so retrieval surfaces the right slice when a user queries by
    persona or ticker.
    """
    import urllib.request

    onyx_api_key = os.getenv("ONYX_API_KEY") or os.getenv("ONYX_INDEX_API_KEY")
    tickers = ",".join(req.tickers)
    model = req.model
    window = payload.get("window") or {}
    decisions = payload.get("decisions") or {}
    sigs = payload.get("analyst_signals") or {}

    lines = [
        f"# AI Hedge Fund run — {tickers}",
        f"model: {model}",
        f"window: {window.get('start')} → {window.get('end')}",
        f"timestamp: {datetime.utcnow().isoformat()}Z",
        "",
        "## Portfolio Manager decisions",
    ]
    for ticker, dec in decisions.items():
        lines.append(
            f"- **{ticker}**: {dec.get('action','?').upper()} "
            f"qty {dec.get('quantity','?')}, conf {dec.get('confidence','?')}%. "
            f"{dec.get('reasoning','')}"
        )
    lines.append("")
    lines.append("## Agent signals")
    for agent_id, per_ticker in sigs.items():
        if agent_id == "risk_management_agent" or agent_id.startswith("portfolio_manager"):
            continue
        for ticker, sig in (per_ticker or {}).items():
            sig_label = (sig.get("signal") or "—").upper()
            conf = sig.get("confidence", "")
            reasoning = sig.get("reasoning")
            if not isinstance(reasoning, str):
                reasoning = json.dumps(reasoning) if reasoning is not None else ""
            lines.append(
                f"### {agent_id} · {ticker}\n"
                f"signal={sig_label} confidence={conf}\n\n"
                f"{reasoning}\n"
            )

    document = {
        "title": f"Hedge Fund run · {tickers} · {window.get('end','')}",
        "source": "ai-hedge-fund",
        "metadata": {
            "tickers": req.tickers,
            "model": model,
            "window_start": window.get("start"),
            "window_end": window.get("end"),
            "agent_count": len(sigs),
            "decisions": decisions,
        },
        "content": "\n".join(lines),
    }

    body = json.dumps(document).encode()
    headers = {"Content-Type": "application/json"}
    if onyx_api_key:
        headers["Authorization"] = f"Bearer {onyx_api_key}"
    request = urllib.request.Request(
        onyx_url, data=body, headers=headers, method="POST",
    )
    with urllib.request.urlopen(request, timeout=8) as r:
        print(f"[onyx-index] {r.status} for {tickers}", file=sys.stderr, flush=True)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>AI Hedge Fund</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
  :root {
    --bg:#0b0d12; --panel:#12151c; --panel2:#1a1f2a; --ink:#e8ecf3;
    --muted:#8a93a6; --line:#232a38; --accent:#7ee787; --warn:#f0b429; --err:#ff6b6b;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  header{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid var(--line);background:var(--panel)}
  header h1{font-size:15px;margin:0;letter-spacing:.3px;font-weight:600}
  header .dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}
  header .dot.ok{background:var(--accent)} header .dot.bad{background:var(--err)}
  header .sub{color:var(--muted);font-size:12px;margin-left:auto}
  main{display:grid;grid-template-columns:360px 1fr;gap:0;min-height:calc(100vh - 52px)}
  .form{padding:18px 20px;border-right:1px solid var(--line);background:var(--panel)}
  .form label{display:block;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:14px 0 6px}
  .form input,.form select,.form textarea{
    width:100%;padding:9px 10px;border-radius:8px;border:1px solid var(--line);
    background:var(--panel2);color:var(--ink);font:inherit
  }
  .form input:focus,.form select:focus,.form textarea:focus{outline:none;border-color:var(--accent)}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .analysts{max-height:230px;overflow:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel2);padding:8px}
  .analysts label{display:flex;align-items:center;gap:8px;text-transform:none;letter-spacing:0;color:var(--ink);font-size:13px;margin:4px 0}
  .analysts label input{width:auto}
  .toggle{display:flex;gap:12px;margin-top:8px;color:var(--muted);font-size:12px}
  .toggle a{color:var(--accent);cursor:pointer;text-decoration:underline}
  button{
    margin-top:18px;width:100%;padding:11px;border:0;border-radius:10px;
    background:var(--accent);color:#062616;font-weight:700;letter-spacing:.03em;cursor:pointer;font-size:14px
  }
  button:disabled{opacity:.6;cursor:wait}
  .out{padding:20px 22px;overflow:auto}
  .empty{color:var(--muted);padding:40px 0;text-align:center}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:14px}
  .card h3{margin:0 0 10px;font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
  th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;font-size:13px}
  th{color:var(--muted);font-weight:500;font-size:11px;letter-spacing:.08em;text-transform:uppercase}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;letter-spacing:.04em}
  .pill.buy,.pill.long,.pill.bullish{background:#143522;color:#7ee787}
  .pill.sell,.pill.short,.pill.bearish{background:#3a141a;color:#ff9a9a}
  .pill.hold,.pill.neutral,.pill.cover{background:#1e2435;color:#c7d2fe}
  pre{white-space:pre-wrap;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:10px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#c7d2fe}
  .err{border-color:#5a2028;background:#1d1014;color:#ffb4b4}
  .kv{color:var(--muted);font-size:12px}
  .kv b{color:var(--ink);font-weight:600}
</style>
</head>
<body>
<header>
  <div class="dot" id="keyDot"></div>
  <h1>AI Hedge Fund</h1>
  <span class="sub" id="sub">Ollama Cloud · localhost</span>
</header>
<main>
  <form class="form" onsubmit="return run(event)">
    <label>Tickers (comma-separated)</label>
    <input id="tickers" value="AAPL" placeholder="AAPL,MSFT,NVDA" />

    <label>Model</label>
    <select id="model"></select>

    <div class="row">
      <div>
        <label>Initial cash</label>
        <input id="cash" type="number" value="100000" step="1000" />
      </div>
      <div>
        <label>Margin req.</label>
        <input id="margin" type="number" value="0" step="0.1" />
      </div>
    </div>

    <div class="row">
      <div>
        <label>Start date</label>
        <input id="start" type="date" />
      </div>
      <div>
        <label>End date</label>
        <input id="end" type="date" />
      </div>
    </div>

    <label>Analysts</label>
    <div class="analysts" id="analysts"></div>
    <div class="toggle">
      <a onclick="toggleAll(true)">select all</a>
      <a onclick="toggleAll(false)">none</a>
    </div>

    <button id="go" type="submit">Run</button>
  </form>

  <section class="out" id="out">
    <div class="empty">Pick tickers and click <b>Run</b>. Results appear here.</div>
  </section>
</main>

<script>
const $ = (id) => document.getElementById(id);
let CFG;

async function init() {
  CFG = await (await fetch('/api/config')).json();
  const ms = $('model');
  CFG.models.forEach(m => {
    const o = document.createElement('option');
    o.value = m.name; o.textContent = m.display; ms.appendChild(o);
  });
  const a = $('analysts');
  CFG.analysts.forEach(x => {
    const row = document.createElement('label');
    row.innerHTML = `<input type="checkbox" value="${x.key}" checked /> ${x.display}`;
    a.appendChild(row);
  });
  const dot = $('keyDot');
  if (CFG.ollama_key_set) { dot.classList.add('ok'); $('sub').textContent = 'Ollama Cloud · key loaded'; }
  else { dot.classList.add('bad'); $('sub').textContent = 'OLLAMA_API_KEY missing'; }
}

function toggleAll(on) {
  document.querySelectorAll('#analysts input').forEach(i => i.checked = on);
}

let tickTimer = null;
function startTicker() {
  const t0 = Date.now();
  const el = $('out');
  const tick = () => {
    const s = Math.floor((Date.now() - t0) / 1000);
    el.innerHTML = `<div class="empty">Running… <b>${s}s</b> — watch the terminal for per-agent progress</div>`;
  };
  tick();
  tickTimer = setInterval(tick, 500);
}
function stopTicker() { if (tickTimer) { clearInterval(tickTimer); tickTimer = null; } }

async function run(e) {
  e.preventDefault();
  const btn = $('go');
  btn.disabled = true; btn.textContent = 'Running…';
  startTicker();

  const payload = {
    tickers: $('tickers').value.split(',').map(s=>s.trim()).filter(Boolean),
    model: $('model').value,
    analysts: [...document.querySelectorAll('#analysts input:checked')].map(i=>i.value),
    initial_cash: parseFloat($('cash').value),
    margin_requirement: parseFloat($('margin').value) || 0,
    start_date: $('start').value || null,
    end_date: $('end').value || null,
  };

  let r, text;
  try {
    r = await fetch('/api/run', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    text = await r.text();
  } catch (err) {
    stopTicker();
    render_error('Network error: ' + err.message);
    btn.disabled = false; btn.textContent = 'Run';
    return;
  }

  stopTicker();
  btn.disabled = false; btn.textContent = 'Run';

  let data;
  try { data = JSON.parse(text); }
  catch (err) {
    render_error('Non-JSON response (HTTP ' + r.status + '):\\n' + text.slice(0, 4000));
    return;
  }

  if (!r.ok)         return render_error('HTTP ' + r.status + ': ' + (data.error || JSON.stringify(data).slice(0, 600)));
  if (data.error)    return render_error(data.error);
  if (!data.decisions && !data.analyst_signals)
                     return render_error('Empty result payload:\\n' + JSON.stringify(data, null, 2).slice(0, 4000));
  render(data);
}

function pill(kind, v) {
  const cls = (v || '').toString().toLowerCase();
  return `<span class="pill ${cls}">${v}</span>`;
}

function render_error(msg) {
  $('out').innerHTML = `<div class="card err"><h3>Error</h3><pre>${msg}</pre></div>`;
}

function render(data) {
  const parts = [];
  parts.push(`<div class="kv">Model <b>${data.model}</b> · window <b>${data.window.start} → ${data.window.end}</b></div>`);

  const dec = data.decisions || {};
  const rows = Object.entries(dec).map(([t, d]) => `
    <tr>
      <td><b>${t}</b></td>
      <td>${pill('action', d.action)}</td>
      <td>${d.quantity ?? ''}</td>
      <td>${d.confidence ?? ''}</td>
      <td>${(d.reasoning || '').replace(/</g,'&lt;')}</td>
    </tr>`).join('');
  parts.push(`
    <div class="card">
      <h3>Portfolio decisions</h3>
      <table>
        <thead><tr><th>Ticker</th><th>Action</th><th>Qty</th><th>Conf</th><th>Reasoning</th></tr></thead>
        <tbody>${rows || '<tr><td colspan=5 class="kv">No decisions returned.</td></tr>'}</tbody>
      </table>
    </div>`);

  const sigs = data.analyst_signals || {};
  const sigRows = [];
  for (const [agent, perTicker] of Object.entries(sigs)) {
    for (const [ticker, s] of Object.entries(perTicker || {})) {
      let reason = '';
      if (s && s.reasoning != null) {
        reason = typeof s.reasoning === 'string'
          ? s.reasoning
          : JSON.stringify(s.reasoning);
      } else if (s && s.signal == null && typeof s === 'object') {
        // risk_management shape — show its whole payload compactly
        reason = JSON.stringify(s);
      }
      reason = reason.replace(/</g,'&lt;').slice(0, 400);
      sigRows.push(`<tr>
        <td>${agent}</td>
        <td>${ticker}</td>
        <td>${pill('sig', s.signal || '—')}</td>
        <td>${s.confidence ?? ''}</td>
        <td>${reason}</td>
      </tr>`);
    }
  }
  parts.push(`
    <div class="card">
      <h3>Analyst signals (${sigRows.length})</h3>
      <table>
        <thead><tr><th>Analyst</th><th>Ticker</th><th>Signal</th><th>Conf</th><th>Reasoning</th></tr></thead>
        <tbody>${sigRows.join('') || '<tr><td colspan=5 class="kv">No signals.</td></tr>'}</tbody>
      </table>
    </div>`);

  $('out').innerHTML = parts.join('');
}

init();
</script>
</body>
</html>
"""


@app.post("/api/cancel")
async def api_cancel():
    """Best-effort cancel for the run currently in flight (if any).

    Sets the threading.Event flag the worker thread checks before each
    LLM call, so any in-flight run terminates within the runtime of its
    current LLM invoke (capped by OLLAMA_REQUEST_TIMEOUT).
    """
    global _active_cancel_event
    if _active_cancel_event is not None:
        _active_cancel_event.set()
        return {"cancelled": True}
    return {"cancelled": False, "reason": "no run in flight"}


@app.post("/api/run-stream")
async def api_run_stream(req: RunRequest, request: Request):
    """Streaming variant of /api/run.

    Pushes Server-Sent Events as each agent updates its status (the same
    `progress.update_status(...)` calls that drive the terminal display in
    interactive mode), then a final `complete` event with the full result
    payload. The Next dashboard's reverse proxy passes through SSE verbatim
    so the browser sees:

      data: {"type":"agent_status","agent":"aswath_damodaran_agent","ticker":"AAPL","status":"Calculating intrinsic value (DCF)","timestamp":"2026-04-25T19:32:01Z"}
      data: {"type":"agent_status","agent":"aswath_damodaran_agent","ticker":"AAPL","status":"Done","timestamp":"…"}
      …
      data: {"type":"complete","decisions":{...},"analyst_signals":{...},"window":{...},"model":"…"}
    """
    end = req.end_date or datetime.now().strftime("%Y-%m-%d")
    if req.start_date:
        start = req.start_date
    else:
        start = (datetime.strptime(end, "%Y-%m-%d") - relativedelta(months=3)).strftime("%Y-%m-%d")

    portfolio = {
        "cash": req.initial_cash,
        "margin_requirement": req.margin_requirement,
        "margin_used": 0.0,
        "positions": {
            t: {"long": 0, "short": 0, "long_cost_basis": 0.0,
                "short_cost_basis": 0.0, "short_margin_used": 0.0}
            for t in req.tickers
        },
        "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in req.tickers},
    }

    # Single-run gate. If a run is already in flight, reject with a clear
    # message instead of silently stacking another full fan-out on top of
    # the first (which is what made everything feel "hung").
    global _active_cancel_event
    if not _run_lock.acquire(blocking=False):
        return JSONResponse(
            {
                "error": "another run is already in flight",
                "hint": "POST /api/cancel to abort it, or wait for it to finish",
            },
            status_code=409,
        )

    cancel_event = threading.Event()
    _active_cancel_event = cancel_event

    async def event_generator():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict] = asyncio.Queue()

        # Bridge: sync handler -> async queue. progress.update_status is called
        # from worker threads inside LangGraph's fan-out, so we need a
        # thread-safe push back into the asyncio loop.
        def handler(agent_name, ticker, status, analysis, timestamp):
            event = {
                "type": "agent_status",
                "agent": agent_name,
                "ticker": ticker,
                "status": status,
                "timestamp": timestamp,
            }
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                # Loop already closed; client likely disconnected
                pass

        progress.register_handler(handler)
        try:
            run_task = asyncio.create_task(asyncio.to_thread(
                run_hedge_fund,
                tickers=req.tickers, start_date=start, end_date=end,
                portfolio=portfolio, show_reasoning=req.show_reasoning,
                selected_analysts=req.analysts or [],
                model_name=req.model, model_provider="Ollama",
                analyst_model=req.analyst_model, pm_model=req.pm_model,
                cancel_event=cancel_event,
            ))

            yield f"data: {json.dumps({'type': 'start', 'tickers': req.tickers, 'model': req.model, 'start_date': start, 'end_date': end})}\n\n"

            # Pump status events while the workflow runs. Send a keepalive
            # comment every ~10s of silence so CF tunnels don't idle-close.
            while not run_task.done():
                if await request.is_disconnected():
                    # Real cancel: set the flag the worker thread checks
                    # before each LLM call. Then keep draining the queue so
                    # later "Done" / "Cancelled" events still fire.
                    cancel_event.set()
                    run_task.cancel()
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=8.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"

            # Drain any remaining events queued after run_task finished
            while not queue.empty():
                event = queue.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"

            # Final result
            try:
                result = run_task.result()
            except (asyncio.CancelledError, RunCancelledError):
                yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                return
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': f'{type(exc).__name__}: {exc}'})}\n\n"
                return

            payload = {
                "type": "complete",
                "decisions": result.get("decisions"),
                "analyst_signals": result.get("analyst_signals"),
                "window": {"start": start, "end": end},
                "model": req.model,
            }
            try:
                yield f"data: {json.dumps(jsonable_encoder(payload))}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': f'serialization: {type(exc).__name__}: {exc}'})}\n\n"
        finally:
            progress.unregister_handler(handler)
            global _active_cancel_event
            _active_cancel_event = None
            try:
                _run_lock.release()
            except RuntimeError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable nginx-style proxy buffering
            "Connection": "keep-alive",
        },
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


def main():
    import uvicorn
    host = os.getenv("AIHF_HOST", "127.0.0.1")
    port = int(os.getenv("AIHF_PORT", "7860"))
    banner = "=" * 60
    print(f"\n{banner}\n  AI Hedge Fund webview  http://{host}:{port}\n{banner}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
