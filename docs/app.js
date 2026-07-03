// Client-side mirror of dp_demo/mechanisms.py, dataset.py, and queries.py.
// Everything here runs in the browser so this page works on GitHub Pages
// with no backend. See the Python package in this repo for the reference
// implementation these formulas are kept in lockstep with.

const DATASETS = {
  iris: {
    label: "Iris flowers (Fisher, 1936)",
    source: "Bundled public-domain excerpt of Fisher's Iris dataset.",
    file: "data/iris_sample.csv",
    numericColumns: {
      sepal_length: [4.0, 8.0],
      sepal_width: [2.0, 4.5],
      petal_length: [1.0, 7.0],
      petal_width: [0.0, 2.5],
    },
    categoryColumn: "species",
    categories: ["setosa", "versicolor", "virginica"],
    categoryMap: null,
  },
  pums: {
    label: "California PUMS demographics (sample of 1000)",
    source: "OpenDP test datasets: opendp/dp-test-datasets, data/PUMS_california_demographics_1000.",
    file: "data/pums_california_1000.csv",
    numericColumns: {
      age: [18.0, 95.0],
      educ: [0.0, 16.0],
      income: [0.0, 420500.0],
    },
    categoryColumn: "married",
    categories: ["married", "unmarried"],
    categoryMap: { "0": "unmarried", "1": "married" },
  },
};

const QUERY_TYPES = ["count", "sum", "mean", "mode_category"];
const MECHANISMS = ["laplace", "gaussian", "exponential"];

const datasetCache = {};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    const row = {};
    header.forEach((key, i) => {
      row[key] = cells[i];
    });
    return row;
  });
}

async function loadDataset(key) {
  if (datasetCache[key]) return datasetCache[key];
  const meta = DATASETS[key];
  const text = await fetch(meta.file).then((r) => {
    if (!r.ok) throw new Error(`could not load ${meta.file}: HTTP ${r.status}`);
    return r.text();
  });
  const dataset = buildDatasetFromCsv(key, text);
  datasetCache[key] = dataset;
  return dataset;
}

function buildDatasetFromCsv(key, text) {
  const meta = DATASETS[key];
  const raw = parseCsv(text);
  const numericFields = Object.keys(meta.numericColumns);
  const rows = raw.map((raw) => {
    const row = {};
    numericFields.forEach((field) => {
      row[field] = parseFloat(raw[field]);
    });
    const label = raw[meta.categoryColumn];
    row[meta.categoryColumn] = meta.categoryMap ? meta.categoryMap[label] || label : label;
    return row;
  });
  return { ...meta, key, rows };
}

function clip(dataset, column, value) {
  const [lower, upper] = dataset.numericColumns[column];
  return Math.min(Math.max(value, lower), upper);
}

// --- Seedable PRNG (mulberry32) so the optional "seed" field is repeatable.
// The Python implementation uses the same seeded RNG + Box-Muller Gaussian,
// and tests compare exact releases across both languages.
function makeRng(seed) {
  if (seed === null || seed === undefined || Number.isNaN(seed)) {
    return Math.random;
  }
  let a = seed >>> 0 || 1;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function laplaceNoise(scale, rng) {
  if (scale === 0) return 0;
  const u = rng() - 0.5;
  return -scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
}

function gaussianNoise(sigma, rng) {
  if (sigma === 0) return 0;
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return z * sigma;
}

function laplaceScale(sensitivity, epsilon) {
  return sensitivity / epsilon;
}

function gaussianSigma(sensitivity, epsilon, delta) {
  return Math.sqrt(2 * Math.log(1.25 / delta)) * sensitivity / epsilon;
}

function exponentialWeights(utilities, sensitivity, epsilon) {
  const scores = utilities.map((u) => (epsilon * u) / (2 * sensitivity));
  const top = Math.max(...scores);
  const weights = scores.map((s) => Math.exp(s - top));
  const total = weights.reduce((a, b) => a + b, 0);
  return weights.map((w) => w / total);
}

function weightedChoice(candidates, probabilities, rng) {
  const r = rng();
  let acc = 0;
  for (let i = 0; i < candidates.length; i++) {
    acc += probabilities[i];
    if (r <= acc) return candidates[i];
  }
  return candidates[candidates.length - 1];
}

function validateBudget(mechanism, epsilon, delta) {
  if (epsilon <= 0) throw new Error("epsilon must be positive");
  if (mechanism === "laplace" && delta !== 0) {
    throw new Error("Laplace mechanism is pure DP, so delta must be 0");
  }
  if (mechanism === "exponential" && delta !== 0) {
    throw new Error("Exponential mechanism is pure DP, so delta must be 0");
  }
  if (mechanism === "gaussian" && !(delta > 0 && delta < 1)) {
    throw new Error("Gaussian mechanism requires 0 < delta < 1");
  }
}

function runQuery(dataset, { query, mechanism, epsilon, delta, column, category, seed }) {
  if (!QUERY_TYPES.includes(query)) throw new Error(`query must be one of: ${QUERY_TYPES.join(", ")}`);
  if (!MECHANISMS.includes(mechanism)) throw new Error(`mechanism must be one of: ${MECHANISMS.join(", ")}`);
  if (category && !dataset.categories.includes(category)) {
    throw new Error(`category must be one of: ${dataset.categories.join(", ")}`);
  }
  if ((query === "sum" || query === "mean") && !(column in dataset.numericColumns)) {
    throw new Error(`${query} requires a numeric column`);
  }
  if (query === "mode_category" && mechanism !== "exponential") {
    throw new Error("mode_category requires the exponential mechanism");
  }
  if (query !== "mode_category" && mechanism === "exponential") {
    throw new Error("the exponential mechanism only applies to the mode_category query");
  }
  validateBudget(mechanism, epsilon, delta);

  const filtered = dataset.rows.filter((row) => !category || row[dataset.categoryColumn] === category);
  const rng = makeRng(seed);

  if (query === "mode_category") {
    const counts = {};
    dataset.categories.forEach((c) => {
      counts[c] = filtered.filter((row) => row[dataset.categoryColumn] === c).length;
    });
    const trueLabel = dataset.categories.reduce((best, c) => (counts[c] > counts[best] ? c : best));
    const sensitivity = 1.0;
    const utilities = dataset.categories.map((c) => counts[c]);
    const probabilities = exponentialWeights(utilities, sensitivity, epsilon);
    const releasedLabel = weightedChoice(dataset.categories, probabilities, rng);
    return {
      query,
      mechanism,
      column: null,
      category,
      epsilon,
      delta,
      sensitivity,
      rowsUsed: filtered.length,
      trueLabel,
      releasedLabel,
      probabilities: Object.fromEntries(dataset.categories.map((c, i) => [c, probabilities[i]])),
      explanation:
        `Exponential mechanism selects a value of '${dataset.categoryColumn}' with probability ` +
        "proportional to exp(epsilon * count / (2 * sensitivity)) instead of adding noise to a " +
        "number, so the released answer is always a real category.",
    };
  }

  let trueValue, sensitivity, explanation, resultColumn = column;

  if (query === "count") {
    resultColumn = null;
    trueValue = filtered.length;
    sensitivity = 1.0;
    explanation = "Count sensitivity is 1 because one changed row can alter a count by at most one.";
  } else {
    const [lower, upper] = dataset.numericColumns[column];
    const values = filtered.map((row) => clip(dataset, column, row[column]));
    if (query === "sum") {
      trueValue = values.reduce((a, b) => a + b, 0);
      sensitivity = Math.max(Math.abs(lower), Math.abs(upper));
      explanation =
        `Sum clips ${column} to [${lower}, ${upper}], so add/remove sensitivity is ` +
        `max(abs(lower), abs(upper)) = ${sensitivity}.`;
    } else {
      trueValue = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
      sensitivity = (upper - lower) / Math.max(values.length, 1);
      explanation =
        `Mean clips ${column} to [${lower}, ${upper}] and uses replacement-neighbor ` +
        `sensitivity (upper - lower) / n = ${sensitivity.toFixed(6)}.`;
    }
  }

  const noisyValue =
    mechanism === "laplace"
      ? trueValue + laplaceNoise(laplaceScale(sensitivity, epsilon), rng)
      : trueValue + gaussianNoise(gaussianSigma(sensitivity, epsilon, delta), rng);

  return {
    query,
    mechanism,
    column: resultColumn,
    category,
    epsilon,
    delta,
    trueValue,
    noisyValue,
    sensitivity,
    rowsUsed: filtered.length,
    explanation,
  };
}

// --- UI wiring -------------------------------------------------------------

function fmt(n) {
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return Number(n.toPrecision(6)).toString();
}

function renderResult(result) {
  const panel = document.getElementById("resultPanel");
  if (result.releasedLabel !== undefined) {
    const payload = {
      true_label: result.trueLabel,
      released_label: result.releasedLabel,
      probabilities: Object.fromEntries(Object.entries(result.probabilities).map(([k, v]) => [k, Number(v.toFixed(6))])),
      rows_used: result.rowsUsed,
    };
    const probRows = Object.entries(result.probabilities)
      .map(([label, prob]) => `
        <div class="result-row"><span class="field-label">${label}</span><strong>${(prob * 100).toFixed(1)}%</strong></div>
      `)
      .join("");
    panel.innerHTML = `
      <div class="result-row true">
        <span class="field-label">True mode</span>
        <strong>${result.trueLabel}</strong>
      </div>
      <div class="result-row release">
        <span class="field-label">Released mode</span>
        <strong>${result.releasedLabel}</strong>
      </div>
      ${probRows}
      <p class="explain">${result.explanation}</p>
      <pre>${JSON.stringify(payload, null, 2)}</pre>
    `;
    return;
  }

  const payload = {
    true_value: Number(result.trueValue.toFixed(6)),
    noisy_value: Number(result.noisyValue.toFixed(6)),
    sensitivity: Number(result.sensitivity.toFixed(6)),
    rows_used: result.rowsUsed,
  };
  panel.innerHTML = `
    <div class="result-row true">
      <span class="field-label">True value</span>
      <strong>${fmt(result.trueValue)}</strong>
    </div>
    <div class="result-row release">
      <span class="field-label">Released value</span>
      <strong>${fmt(result.noisyValue)}</strong>
    </div>
    <div class="result-row">
      <span class="field-label">Sensitivity</span>
      <strong>${fmt(result.sensitivity)}</strong>
    </div>
    <p class="explain">${result.explanation}</p>
    <pre>${JSON.stringify(payload, null, 2)}</pre>
  `;
}

function renderError(message) {
  document.getElementById("resultPanel").innerHTML = `
    <strong>Could not run that query</strong><p class="explain">${message}</p>
  `;
  document.getElementById("resultPanel").classList.add("error");
}

function populateOptions(select, values, labels) {
  select.innerHTML = values
    .map((v) => `<option value="${v}">${labels && labels[v] ? labels[v] : v}</option>`)
    .join("");
}

async function refreshDatasetFields() {
  const datasetField = document.getElementById("datasetField");
  const dataset = await loadDataset(datasetField.value);
  const columnField = document.getElementById("columnField");
  const categoryField = document.getElementById("categoryField");
  populateOptions(columnField, Object.keys(dataset.numericColumns));
  categoryField.innerHTML =
    `<option value="">all</option>` + dataset.categories.map((c) => `<option value="${c}">${c}</option>`).join("");
  document.getElementById("sourceNote").textContent = `Source: ${dataset.source}`;
  return dataset;
}

async function init() {
  const datasetField = document.getElementById("datasetField");
  populateOptions(
    datasetField,
    Object.keys(DATASETS),
    Object.fromEntries(Object.entries(DATASETS).map(([k, v]) => [k, v.label]))
  );
  await refreshDatasetFields();
  datasetField.addEventListener("change", refreshDatasetFields);

  const form = document.getElementById("queryForm");
  const runButton = document.getElementById("runButton");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    document.getElementById("resultPanel").classList.remove("error");
    runButton.disabled = true;
    try {
      const dataset = await loadDataset(datasetField.value);
      const seedRaw = document.getElementById("seedField").value;
      const result = runQuery(dataset, {
        query: document.getElementById("queryField").value,
        mechanism: document.getElementById("mechanismField").value,
        epsilon: parseFloat(document.getElementById("epsilonField").value),
        delta: parseFloat(document.getElementById("deltaField").value || "0"),
        column: document.getElementById("columnField").value,
        category: document.getElementById("categoryField").value || null,
        seed: seedRaw ? parseInt(seedRaw, 10) : null,
      });
      renderResult(result);
    } catch (err) {
      renderError(err.message);
    } finally {
      runButton.disabled = false;
    }
  });

  // --- Privacy dial hero (decorative + epsilon-linked preview) -----------
  const canvas = document.getElementById("dialCanvas");
  const ctx = canvas.getContext("2d");
  const slider = document.getElementById("epsSlider");
  const readout = document.getElementById("epsReadout");
  const epsilonField = document.getElementById("epsilonField");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = 440 * dpr;
  canvas.height = 160 * dpr;
  ctx.scale(dpr, dpr);
  const W = 440, H = 160, cy = H / 2;

  function laplaceSample(scale) {
    const u = Math.random() - 0.5;
    return -scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
  }

  let points = [];
  function seedPoints(eps) {
    const scale = 28 / eps;
    points = [];
    for (let i = 0; i < 26; i++) {
      points.push({ x: W / 2 + laplaceSample(scale), phase: Math.random() * Math.PI * 2, r: 2.4 + Math.random() * 1.6 });
    }
  }
  seedPoints(parseFloat(slider.value));

  function draw(t) {
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(236,231,218,0.14)";
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(W, cy);
    ctx.stroke();

    points.forEach((p) => {
      const wobble = reduceMotion ? 0 : Math.sin(t / 900 + p.phase) * 3;
      ctx.beginPath();
      ctx.arc(p.x, cy + wobble, p.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(95,208,192,0.75)";
      ctx.fill();
    });

    ctx.beginPath();
    ctx.arc(W / 2, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#FF6B4A";
    ctx.fill();

    if (!reduceMotion) requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);

  function setEpsilon(value) {
    const eps = Math.max(0.05, parseFloat(value) || 1);
    readout.textContent = "epsilon = " + eps.toFixed(2);
    seedPoints(eps);
  }
  slider.addEventListener("input", () => {
    setEpsilon(slider.value);
    epsilonField.value = slider.value;
  });
  const initial = parseFloat(epsilonField.value);
  if (!Number.isNaN(initial)) {
    slider.value = Math.min(4, Math.max(0.05, initial));
    setEpsilon(slider.value);
  }
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  init();
}

if (typeof module !== "undefined") {
  module.exports = {
    DATASETS,
    QUERY_TYPES,
    MECHANISMS,
    buildDatasetFromCsv,
    exponentialWeights,
    gaussianSigma,
    laplaceScale,
    parseCsv,
    runQuery,
  };
}
