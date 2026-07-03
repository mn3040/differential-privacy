"""Tiny stdlib web UI exposing epsilon/delta privacy controls."""

from __future__ import annotations

import html
import json
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .dataset import DATASET_KEYS, load_dataset
from .queries import MECHANISMS, QUERY_TYPES, QueryResult, run_query


def _option_tags(values: tuple[str, ...], selected: str | None, labels: dict[str, str] | None = None) -> str:
    tags = []
    for value in values:
        is_selected = " selected" if value == selected else ""
        text = (labels or {}).get(value, value)
        tags.append(f'<option value="{html.escape(value)}"{is_selected}>{html.escape(text)}</option>')
    return "\n".join(tags)


def _render_result(result: QueryResult | None, error: str | None) -> str:
    if error:
        return (
            '<section class="result error">'
            f'<strong>Could not run that query</strong><p class="explain">{html.escape(error)}</p>'
            "</section>"
        )
    if result is None:
        return (
            '<section class="result">'
            '<p class="placeholder">Run a query to see the true value, released value, sensitivity, '
            'and machine-readable payload.</p>'
            '</section>'
        )

    if result.released_label is not None:
        payload = {
            "true_label": result.true_label,
            "released_label": result.released_label,
            "probabilities": {k: round(v, 6) for k, v in (result.probabilities or {}).items()},
            "rows_used": result.rows_used,
        }
        prob_rows = "".join(
            f'<div class="result-row"><span class="field-label">{html.escape(label)}</span>'
            f'<strong>{prob:.1%}</strong></div>'
            for label, prob in (result.probabilities or {}).items()
        )
        return f"""
        <section class="result">
          <div class="result-row true">
            <span class="field-label">True mode</span>
            <strong>{html.escape(result.true_label)}</strong>
          </div>
          <div class="result-row release">
            <span class="field-label">Released mode</span>
            <strong>{html.escape(result.released_label)}</strong>
          </div>
          {prob_rows}
          <p class="explain">{html.escape(result.explanation)}</p>
          <pre>{html.escape(json.dumps(payload, indent=2))}</pre>
        </section>
        """

    payload = {
        "true_value": round(result.true_value, 6),
        "noisy_value": round(result.noisy_value, 6),
        "sensitivity": round(result.sensitivity, 6),
        "rows_used": result.rows_used,
    }
    return f"""
    <section class="result">
      <div class="result-row true">
        <span class="field-label">True value</span>
        <strong>{result.true_value:.6g}</strong>
      </div>
      <div class="result-row release">
        <span class="field-label">Released value</span>
        <strong>{result.noisy_value:.6g}</strong>
      </div>
      <div class="result-row">
        <span class="field-label">Sensitivity</span>
        <strong>{result.sensitivity:.6g}</strong>
      </div>
      <p class="explain">{html.escape(result.explanation)}</p>
      <pre>{html.escape(json.dumps(payload, indent=2))}</pre>
    </section>
    """


def render_page(params: dict[str, str], result: QueryResult | None, error: str | None) -> bytes:
    dataset_key = params.get("dataset", "iris")
    if dataset_key not in DATASET_KEYS:
        dataset_key = "iris"
    dataset = load_dataset(dataset_key)

    query = params.get("query", "mean")
    mechanism = params.get("mechanism", "laplace")
    column = params.get("column", "")
    if column not in dataset.numeric_columns:
        column = next(iter(dataset.numeric_columns))
    category = params.get("category", params.get("species", ""))
    if category not in dataset.categories:
        category = ""
    epsilon = params.get("epsilon", "1.0")
    delta = params.get("delta", "0.0")
    seed = params.get("seed", "")

    dataset_labels = {key: load_dataset(key).label for key in DATASET_KEYS}

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Differential Privacy Demo</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b0d14;
      --panel: #12151f;
      --panel-alt: #191c2a;
      --border: #252a3d;
      --border-strong: #343b55;
      --text: #d8dce8;
      --muted: #7d849d;
      --faint: #5e6580;
      --accent: #5b9dff;
      --true: #ff6b4a;
      --release: #5fd0c0;
      --warn: #f5c842;
      --danger: #e05252;
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; background: var(--bg); }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    header {{ border-bottom: 1px solid var(--border); padding: 2rem 2.5rem 1.4rem; display: flex; gap: 2rem; align-items: flex-start; justify-content: space-between; }}
    h1 {{ margin: 0 0 0.35rem; font-family: "IBM Plex Mono", monospace; font-size: clamp(1.25rem, 2vw, 1.75rem); line-height: 1.2; font-weight: 600; letter-spacing: 0; }}
    .lede {{ max-width: 760px; margin: 0; color: var(--muted); font-size: 0.92rem; line-height: 1.6; }}
    .header-links {{ display: flex; gap: 1rem; flex-wrap: wrap; justify-content: flex-end; padding-top: 0.15rem; font-family: "IBM Plex Mono", monospace; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.08em; white-space: nowrap; }}
    .signal-strip {{ display: flex; gap: 0.6rem 1.3rem; flex-wrap: wrap; padding: 0.75rem 2.5rem; border-bottom: 1px solid var(--border); background: var(--panel); }}
    .signal {{ display: inline-flex; align-items: center; gap: 0.45rem; font-family: "IBM Plex Mono", monospace; font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }}
    .swatch {{ width: 8px; height: 8px; border-radius: 2px; background: var(--accent); flex: 0 0 auto; }}
    main {{ display: grid; grid-template-columns: minmax(320px, 430px) minmax(0, 1fr); min-height: calc(100vh - 143px); }}
    .query-panel {{ border-right: 1px solid var(--border); background: var(--panel); min-width: 0; }}
    .output-panel {{ min-width: 0; background: linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px), var(--bg); background-size: 28px 28px; }}
    .panel-header {{ padding: 1rem 1.5rem 0.75rem; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; }}
    .panel-label {{ font-family: "IBM Plex Mono", monospace; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.11em; text-transform: uppercase; color: var(--muted); }}
    .panel-note {{ font-family: "IBM Plex Mono", monospace; font-size: 0.68rem; color: var(--faint); }}
    form {{ padding: 1.25rem 1.5rem 1.5rem; display: grid; gap: 1rem; }}
    .field-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.9rem; }}
    label.field {{ display: grid; gap: 0.42rem; min-width: 0; }}
    .field-label {{ display: block; font-family: "IBM Plex Mono", monospace; font-size: 0.66rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }}
    select, input {{ width: 100%; min-height: 2.55rem; border: 1px solid var(--border-strong); border-radius: 4px; background: #0f121b; color: var(--text); padding: 0.62rem 0.7rem; font: inherit; font-size: 0.9rem; }}
    select:focus, input:focus, button:focus {{ outline: 2px solid rgba(91, 157, 255, 0.55); outline-offset: 1px; }}
    button {{ width: 100%; min-height: 2.75rem; border: 0; border-radius: 4px; background: var(--accent); color: var(--bg); font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; cursor: pointer; transition: opacity 0.15s, transform 0.15s; }}
    button:hover {{ opacity: 0.9; transform: translateY(-1px); }}
    .dial {{ margin-top: 0.2rem; border: 1px solid var(--border); background: var(--panel-alt); border-radius: 4px; padding: 0.95rem; }}
    .dial-readout, .dial-legend {{ display: flex; justify-content: space-between; gap: 1rem; align-items: center; font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; color: var(--muted); }}
    .dial-readout strong {{ color: var(--text); font-size: 0.88rem; }}
    canvas {{ display: block; width: 100%; height: 138px; margin: 0.75rem 0; border-radius: 3px; background: #0c0f17; }}
    .dial-slider input {{ min-height: auto; padding: 0; accent-color: var(--accent); }}
    .dial-legend {{ justify-content: flex-start; flex-wrap: wrap; }}
    .dial-legend span {{ display: inline-flex; align-items: center; gap: 0.45rem; }}
    .dot {{ width: 8px; height: 8px; border-radius: 2px; display: inline-block; }}
    .output-body {{ padding: 1.5rem; display: grid; gap: 1rem; }}
    .result, .notes {{ border: 1px solid var(--border); background: rgba(18, 21, 31, 0.92); border-radius: 4px; overflow: hidden; }}
    .result {{ min-height: 222px; display: block; }}
    .result-row {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; padding: 0.9rem 1rem; border-bottom: 1px solid var(--border); }}
    .result-row strong {{ font-family: "IBM Plex Mono", monospace; font-size: clamp(1.05rem, 2vw, 1.45rem); color: var(--text); text-align: right; word-break: break-word; }}
    .result-row.true strong {{ color: var(--true); }}
    .result-row.release strong {{ color: var(--release); }}
    .result .placeholder, .result .explain, .notes p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .result .placeholder {{ padding: 1rem; }}
    .result .explain {{ padding: 1rem; border-bottom: 1px solid var(--border); }}
    pre {{ margin: 0; padding: 1rem; overflow: auto; background: #080a10; color: #a9f4e7; font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; line-height: 1.55; }}
    .error {{ border-color: rgba(224, 82, 82, 0.65); }}
    .error strong {{ display: block; padding: 1rem 1rem 0; color: var(--danger); font-family: "IBM Plex Mono", monospace; }}
    .notes {{ padding: 1rem; display: grid; gap: 0.7rem; }}
    .source-line {{ font-family: "IBM Plex Mono", monospace; font-size: 0.75rem; color: var(--faint); }}
    @media (max-width: 820px) {{
      header {{ padding: 1.35rem 1.25rem 1rem; flex-direction: column; gap: 0.85rem; }}
      .header-links {{ justify-content: flex-start; }}
      .signal-strip {{ padding: 0.7rem 1.25rem; }}
      main {{ grid-template-columns: 1fr; }}
      .query-panel {{ border-right: 0; border-bottom: 1px solid var(--border); }}
      .field-grid {{ grid-template-columns: 1fr; }}
      .output-body, form {{ padding-left: 1rem; padding-right: 1rem; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      * {{ animation-duration: 0.001ms !important; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Differential Privacy Workbench</h1>
      <p class="lede">Run bounded queries over public sample datasets, tune epsilon and delta, and inspect the sensitivity behind each private release.</p>
    </div>
    <nav class="header-links" aria-label="Project links">
      <a href="https://github.com/mn3040/differential-privacy">Source</a>
      <a href="https://github.com/mn3040/differential-privacy/actions/workflows/tests.yml">CI</a>
      <a href="https://github.com/mn3040/differential-privacy/blob/main/LICENSE">MIT</a>
    </nav>
  </header>

  <div class="signal-strip" aria-label="Privacy concepts">
    <span class="signal"><i class="swatch" style="background:var(--accent)"></i>epsilon budget</span>
    <span class="signal"><i class="swatch" style="background:var(--warn)"></i>delta slack</span>
    <span class="signal"><i class="swatch" style="background:var(--true)"></i>true value</span>
    <span class="signal"><i class="swatch" style="background:var(--release)"></i>released draw</span>
  </div>

  <main>
    <section class="query-panel">
      <div class="panel-header">
        <span class="panel-label">Query controls</span>
        <span class="panel-note">local Python server</span>
      </div>
      <form method="get">
        <label class="field">
          <span class="field-label">Dataset</span>
          <select name="dataset" onchange="this.form.submit()">{_option_tags(DATASET_KEYS, dataset_key, dataset_labels)}</select>
        </label>
        <div class="field-grid">
          <label class="field">
            <span class="field-label">Query</span>
            <select name="query">{_option_tags(QUERY_TYPES, query)}</select>
          </label>
          <label class="field">
            <span class="field-label">Mechanism</span>
            <select name="mechanism">{_option_tags(MECHANISMS, mechanism)}</select>
          </label>
        </div>
        <div class="field-grid">
          <label class="field">
            <span class="field-label">Column</span>
            <select name="column">{_option_tags(tuple(dataset.numeric_columns), column)}</select>
          </label>
          <label class="field">
            <span class="field-label">Category filter</span>
            <select name="category">
              <option value="">all</option>
              {_option_tags(dataset.categories, category or None)}
            </select>
          </label>
        </div>
        <div class="field-grid">
          <label class="field">
            <span class="field-label">Epsilon</span>
            <input id="epsilonField" name="epsilon" type="number" min="0.01" step="0.01" value="{html.escape(epsilon)}">
          </label>
          <label class="field">
            <span class="field-label">Delta</span>
            <input name="delta" type="number" min="0" max="0.999999" step="0.000001" value="{html.escape(delta)}">
          </label>
        </div>
        <label class="field">
          <span class="field-label">Seed</span>
          <input name="seed" type="number" step="1" value="{html.escape(seed)}" placeholder="blank = fresh draw">
        </label>
        <div class="dial">
          <div class="dial-readout">
            <span>noise preview</span>
            <strong id="epsReadout">epsilon = 1.00</strong>
          </div>
          <canvas id="dialCanvas" width="440" height="160" aria-label="Preview of noisy releases around a true value"></canvas>
          <div class="dial-slider">
            <input id="epsSlider" type="range" min="0.05" max="4" step="0.05" value="1">
          </div>
          <div class="dial-legend">
            <span><i class="dot" style="background:var(--true)"></i>true</span>
            <span><i class="dot" style="background:var(--release)"></i>released draws</span>
          </div>
        </div>
        <button type="submit">Run private query</button>
      </form>
    </section>

    <section class="output-panel">
      <div class="panel-header">
        <span class="panel-label">Private release</span>
        <span class="panel-note">sensitivity calibrated</span>
      </div>
      <div class="output-body">
        {_render_result(result, error)}
        <section class="notes">
          <p class="field-label">Reading the knobs</p>
          <p>Smaller epsilon means stronger privacy and larger noise. Laplace uses delta = 0. Gaussian requires a nonzero delta and uses a normal distribution calibrated to the query sensitivity. Exponential only applies to the mode_category query and picks a real category instead of a noisy number.</p>
          <p class="source-line">Source: {html.escape(dataset.source)}</p>
        </section>
      </div>
    </section>
  </main>
  <script>
    (function () {{
      var canvas = document.getElementById('dialCanvas');
      var ctx = canvas.getContext('2d');
      var slider = document.getElementById('epsSlider');
      var readout = document.getElementById('epsReadout');
      var epsilonField = document.getElementById('epsilonField');
      var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      var dpr = window.devicePixelRatio || 1;
      canvas.width = 440 * dpr;
      canvas.height = 160 * dpr;
      ctx.scale(dpr, dpr);
      var W = 440, H = 160, cy = H / 2;

      function laplaceSample(scale) {{
        var u = Math.random() - 0.5;
        return -scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
      }}

      var points = [];
      function seedPoints(eps) {{
        var scale = 28 / eps;
        points = [];
        for (var i = 0; i < 26; i++) {{
          points.push({{
            x: W / 2 + laplaceSample(scale),
            phase: Math.random() * Math.PI * 2,
            r: 2.4 + Math.random() * 1.6,
          }});
        }}
      }}
      seedPoints(parseFloat(slider.value));

      function draw(t) {{
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = 'rgba(216,220,232,0.14)';
        ctx.beginPath();
        ctx.moveTo(0, cy);
        ctx.lineTo(W, cy);
        ctx.stroke();

        points.forEach(function (p, i) {{
          var wobble = reduceMotion ? 0 : Math.sin(t / 900 + p.phase) * 3;
          ctx.beginPath();
          ctx.arc(p.x, cy + wobble, p.r, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(95,208,192,0.75)';
          ctx.fill();
        }});

        ctx.beginPath();
        ctx.arc(W / 2, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#ff6b4a';
        ctx.fill();

        if (!reduceMotion) requestAnimationFrame(draw);
      }}
      requestAnimationFrame(draw);

      function setEpsilon(value) {{
        var eps = Math.max(0.05, parseFloat(value) || 1);
        readout.textContent = 'epsilon = ' + eps.toFixed(2);
        seedPoints(eps);
      }}
      slider.addEventListener('input', function () {{
        setEpsilon(slider.value);
        epsilonField.value = slider.value;
      }});
      var initial = parseFloat(epsilonField.value);
      if (!isNaN(initial)) {{
        slider.value = Math.min(4, Math.max(0.05, initial));
        setEpsilon(slider.value);
      }}
    }})();
  </script>
</body>
</html>"""
    return body.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        raw = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
        result = None
        error = None

        if parsed.path not in {"/", "/query"}:
            self.send_error(404)
            return

        if raw:
            try:
                seed = int(raw["seed"]) if raw.get("seed") else None
                dataset = load_dataset(raw.get("dataset", "iris"))

                # Column/category select values may be left over from a
                # different dataset (e.g. switching from Iris to PUMS keeps
                # "petal_length" in the query string), so clamp them to the
                # dataset actually being queried instead of erroring.
                column = raw.get("column")
                if column not in dataset.numeric_columns:
                    column = next(iter(dataset.numeric_columns))
                category = raw.get("category") or raw.get("species") or None
                if category is not None and category not in dataset.categories:
                    category = None

                result = run_query(
                    dataset,
                    query=raw.get("query", "mean"),
                    mechanism=raw.get("mechanism", "laplace"),
                    column=column,
                    category=category,
                    epsilon=float(raw.get("epsilon", "1.0")),
                    delta=float(raw.get("delta", "0.0")),
                    seed=seed,
                )
            except Exception as exc:
                error = str(exc)

        payload = render_page(raw, result, error)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the differential privacy demo UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving DP demo at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
