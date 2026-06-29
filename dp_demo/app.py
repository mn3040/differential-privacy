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
        return ""

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
    category = params.get("category", "")
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
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --ink: #0E1A26;
      --ink-soft: #15273A;
      --paper: #ECE7DA;
      --paper-dim: #DCD5C2;
      --signal: #FF6B4A;
      --release: #5FD0C0;
      --hair: rgba(236, 231, 218, 0.16);
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      background: var(--ink);
      color: var(--paper);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 0 20px 64px; }}

    .hero {{ padding: 56px 0 36px; border-bottom: 1px solid var(--hair); display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 32px; align-items: center; }}
    .eyebrow {{ font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--release); margin: 0 0 14px; }}
    h1 {{ font-family: Fraunces, Georgia, serif; font-weight: 600; font-size: clamp(2.1rem, 4.4vw, 3.4rem); line-height: 1.05; margin: 0 0 16px; letter-spacing: -0.01em; }}
    .hero p {{ line-height: 1.6; color: #C9C2AE; max-width: 46ch; margin: 0; }}
    p {{ line-height: 1.55; }}

    .dial {{ background: var(--ink-soft); border: 1px solid var(--hair); border-radius: 4px; padding: 16px; }}
    .dial-readout {{ display: flex; justify-content: space-between; align-items: baseline; font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; color: #9FB0BE; margin-bottom: 10px; }}
    .dial-readout strong {{ color: var(--paper); font-size: 0.95rem; }}
    canvas {{ display: block; width: 100%; height: 160px; border-radius: 2px; }}
    .dial-slider {{ margin-top: 12px; }}
    .dial-slider input {{ width: 100%; accent-color: var(--signal); }}
    .dial-legend {{ display: flex; gap: 18px; margin-top: 10px; font-family: "IBM Plex Mono", monospace; font-size: 0.7rem; color: #9FB0BE; }}
    .dial-legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}

    .workspace {{ display: grid; grid-template-columns: minmax(280px, 360px) 1fr; gap: 20px; align-items: start; margin-top: 36px; }}

    form, .result, .notes {{ background: var(--paper); color: var(--ink); border-radius: 4px; padding: 22px; }}
    form {{ border-top: 3px solid var(--signal); }}
    .field-label {{ display: block; font-family: "IBM Plex Mono", monospace; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: #5B6B5F; margin-bottom: 6px; }}
    label.field {{ display: block; margin-bottom: 16px; }}
    input, select {{ width: 100%; border: 1px solid #C7BFA8; background: #FBF9F2; border-radius: 3px; padding: 9px 10px; font: inherit; color: var(--ink); }}
    input:focus, select:focus, button:focus {{ outline: 2px solid var(--release); outline-offset: 1px; }}
    button {{ width: 100%; border: 0; border-radius: 3px; padding: 12px 14px; background: var(--ink); color: var(--paper); font-weight: 600; letter-spacing: 0.01em; cursor: pointer; margin-top: 4px; }}
    button:hover {{ background: #060d15; }}

    .result {{ display: grid; gap: 0; border-top: 3px solid var(--release); }}
    .result-row {{ display: flex; justify-content: space-between; align-items: baseline; gap: 16px; border-bottom: 1px solid #D9D1B9; padding: 11px 0; }}
    .result-row:first-of-type {{ padding-top: 0; }}
    .result-row .field-label {{ margin: 0; }}
    .result-row strong {{ font-family: "IBM Plex Mono", monospace; font-size: 1.2rem; }}
    .result-row.true strong {{ color: var(--signal); }}
    .result-row.release strong {{ color: #1c8f7f; }}
    .result p.explain {{ color: #4A5550; font-size: 0.92rem; margin: 14px 0 0; }}
    pre {{ overflow: auto; background: var(--ink); color: #BFD4C9; border-radius: 3px; padding: 12px; font-family: "IBM Plex Mono", monospace; font-size: 0.8rem; margin: 14px 0 0; }}
    .error {{ border-top-color: #C24E3A; }}
    .error strong {{ font-family: "IBM Plex Mono", monospace; color: #C24E3A; }}

    .notes {{ margin-top: 18px; border-top: 3px solid var(--paper-dim); }}
    .notes .field-label {{ margin-bottom: 10px; }}
    .notes p {{ color: #4A5550; margin: 0; }}

    @media (max-width: 760px) {{
      .hero {{ grid-template-columns: 1fr; padding-top: 36px; }}
      .workspace {{ grid-template-columns: 1fr; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      * {{ animation-duration: 0.001ms !important; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div>
        <p class="eyebrow">Epsilon &middot; Delta &middot; Sensitivity</p>
        <h1>Every answer costs a little privacy.</h1>
        <p>Drag epsilon below and watch the released values scatter around the true one. Then run the same trade-off over Fisher's Iris dataset or a real OpenDP test sample of California census microdata.</p>
      </div>
      <div class="dial">
        <div class="dial-readout">
          <span>Privacy dial</span>
          <strong id="epsReadout">&epsilon; = 1.00</strong>
        </div>
        <canvas id="dialCanvas" width="440" height="160" aria-label="Animated preview of noisy draws scattering around a true value as epsilon changes"></canvas>
        <div class="dial-slider">
          <input id="epsSlider" type="range" min="0.05" max="4" step="0.05" value="1">
        </div>
        <div class="dial-legend">
          <span><i class="dot" style="background:var(--signal)"></i>true value</span>
          <span><i class="dot" style="background:var(--release)"></i>released draws</span>
        </div>
      </div>
    </section>

    <section class="workspace">
      <form method="get">
        <label class="field">
          <span class="field-label">Dataset</span>
          <select name="dataset" onchange="this.form.submit()">{_option_tags(DATASET_KEYS, dataset_key, dataset_labels)}</select>
        </label>
        <label class="field">
          <span class="field-label">Query</span>
          <select name="query">{_option_tags(QUERY_TYPES, query)}</select>
        </label>
        <label class="field">
          <span class="field-label">Mechanism</span>
          <select name="mechanism">{_option_tags(MECHANISMS, mechanism)}</select>
        </label>
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
        <label class="field">
          <span class="field-label">Epsilon</span>
          <input id="epsilonField" name="epsilon" type="number" min="0.01" step="0.01" value="{html.escape(epsilon)}">
        </label>
        <label class="field">
          <span class="field-label">Delta</span>
          <input name="delta" type="number" min="0" max="0.999999" step="0.000001" value="{html.escape(delta)}">
        </label>
        <label class="field">
          <span class="field-label">Seed</span>
          <input name="seed" type="number" step="1" value="{html.escape(seed)}">
        </label>
        <button type="submit">Run private query</button>
      </form>
      <div>
        {_render_result(result, error)}
        <section class="notes">
          <p class="field-label">Reading the knobs</p>
          <p>Smaller epsilon means stronger privacy and larger noise. Laplace uses delta = 0. Gaussian requires a nonzero delta and uses a normal distribution calibrated to the query sensitivity. Exponential only applies to the mode_category query and picks a real category instead of a noisy number.</p>
          <p class="explain">Source: {html.escape(dataset.source)}</p>
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
        ctx.strokeStyle = 'rgba(236,231,218,0.14)';
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
        ctx.fillStyle = '#FF6B4A';
        ctx.fill();

        if (!reduceMotion) requestAnimationFrame(draw);
      }}
      requestAnimationFrame(draw);

      function setEpsilon(value) {{
        var eps = Math.max(0.05, parseFloat(value) || 1);
        readout.textContent = 'ε = ' + eps.toFixed(2);
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
                category = raw.get("category") or None
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
