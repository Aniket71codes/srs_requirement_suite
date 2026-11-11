const form        = document.querySelector("#analyzeForm");
const fileInput   = document.querySelector("#fileInput");
const fileName    = document.querySelector("#fileName");
const textInput   = document.querySelector("#srsText");
const toast       = document.querySelector("#toast");
const exportButton= document.querySelector("#exportButton");
const resultsBody = document.querySelector("#resultsBody");
const tabBar      = document.querySelector("#tabBar");

let latestRows   = [];
let activeFilter = "all";

// Chart instances
let donutChart = null, barChart = null, confChart = null;

const TYPE_SLUGS = {
  Security: "security", Performance: "performance", Availability: "availability",
  Usability: "usability", Maintainability: "maintainability", Portability: "portability",
  Reliability: "reliability", Scalability: "scalability",
};

const TYPE_ORDER = [
  "Security","Performance","Availability","Usability",
  "Maintainability","Portability","Reliability","Scalability",
];

const TYPE_COLORS = {
  Security: "#ef4444", Performance: "#f97316", Availability: "#22c55e",
  Usability: "#3b82f6", Maintainability: "#06b6d4", Portability: "#8b5cf6",
  Reliability: "#ec4899", Scalability: "#10b981",
};

const statusLabels = {
  trained: "Trained", "metadata-only": "Needs training", heuristic: "Heuristic",
  "bert-tiny": "BERT-tiny", "sklearn-fallback": "Fallback", unavailable: "Unavailable",
  transformer: "Transformer", huggingface: "Hugging Face", "load-error": "Load error",
};

// ── Helpers ───────────────────────────────────────────────

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add("show");
  window.clearTimeout(showToast._t);
  showToast._t = window.setTimeout(() => toast.classList.remove("show"), 3600);
}

function pct(v) { return Math.round((Number(v) || 0) * 100); }

function escapeHtml(v) {
  return String(v)
    .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
    .replaceAll('"',"&quot;").replaceAll("'","&#039;");
}

// ── Model status ──────────────────────────────────────────

async function loadStatus() {
  try {
    const res  = await fetch("/api/status");
    const data = await res.json();
    document.querySelector("#model1Status").textContent = statusLabels[data.model1.status] || data.model1.status;
    document.querySelector("#model2Status").textContent = statusLabels[data.model2.status] || data.model2.status;
    document.querySelector("#model3Status").textContent = statusLabels[data.model3.status] || data.model3.status;
  } catch {
    showToast("Could not read model status. Start the backend server first.");
  }
}

// ── Metrics & tabs ────────────────────────────────────────

function updateTabs(summary) {
  document.querySelector("#tabCountAll").textContent    = summary.requirements  || 0;
  document.querySelector("#tabCountFR").textContent     = summary.functional    || 0;
  document.querySelector("#tabCountNFR").textContent    = summary.nonFunctional || 0;
  const rc = summary.needsReview || 0;
  document.querySelector("#tabCountReview").textContent = rc;
  document.querySelector("#tabNeedsReview").style.display = rc > 0 ? "" : "none";
}

function updateMetrics(summary) {
  document.querySelector("#metricCandidates").textContent   = summary.candidates    || 0;
  document.querySelector("#metricRequirements").textContent = summary.requirements  || 0;
  document.querySelector("#metricFunctional").textContent   = summary.functional    || 0;
  document.querySelector("#metricNfr").textContent          = summary.nonFunctional || 0;

  const rc = summary.needsReview || 0;
  document.querySelector("#metricNeedsReviewCount").textContent = rc;
  document.querySelector("#metricNeedsReview").style.display = rc > 0 ? "" : "none";

  updateTabs(summary);

  const stack = document.querySelector("#typeStack");
  const entries = Object.entries(summary.nfrTypes || {});
  if (!entries.length) {
    stack.innerHTML = `<div class="empty-state">No NFR categories detected yet.</div>`;
    return;
  }
  const sorted = [...entries].sort((a, b) => {
    const oa = TYPE_ORDER.indexOf(a[0]), ob = TYPE_ORDER.indexOf(b[0]);
    return (oa < 0 ? 99 : oa) - (ob < 0 ? 99 : ob);
  });
  const max = Math.max(...sorted.map(([,v]) => v), 1);
  stack.innerHTML = sorted.map(([name, value]) => {
    const slug  = TYPE_SLUGS[name] || "usability";
    const width = Math.max(8, Math.round((value / max) * 100));
    return `
      <div class="type-row">
        <header>
          <span class="type-dot type-dot-${slug}"></span>
          <span>${escapeHtml(name)}</span>
          <strong>${value}</strong>
        </header>
        <div class="bar"><span class="bar-fill bar-${slug}" style="width:${width}%"></span></div>
      </div>`;
  }).join("");
}

// ── Charts ────────────────────────────────────────────────

function initCharts() {
  Chart.defaults.font.family = "'Inter', ui-sans-serif, sans-serif";
  Chart.defaults.color = "#64748b";

  // 1. Donut — FR vs NFR split
  donutChart = new Chart(document.getElementById("chartDonut"), {
    type: "doughnut",
    data: {
      labels: ["Functional", "Non-Functional"],
      datasets: [{
        data: [0, 0],
        backgroundColor: ["#6366f1", "#8b5cf6"],
        borderColor: "#fff",
        borderWidth: 3,
        hoverOffset: 5,
      }]
    },
    options: {
      cutout: "68%",
      plugins: {
        legend: {
          position: "bottom",
          labels: { padding: 16, usePointStyle: true, pointStyleWidth: 8, font: { size: 11, weight: "700" } }
        },
        tooltip: {
          callbacks: {
            label: ctx => {
              const total = ctx.dataset.data.reduce((a, b) => a + b, 0) || 1;
              return ` ${ctx.label}: ${ctx.parsed} (${Math.round(ctx.parsed / total * 100)}%)`;
            }
          }
        }
      },
      animation: { duration: 500 },
    }
  });

  // 2. Horizontal bar — NFR categories
  barChart = new Chart(document.getElementById("chartBar"), {
    type: "bar",
    data: {
      labels: [],
      datasets: [{
        label: "Count",
        data: [],
        backgroundColor: [],
        borderRadius: 5,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: {
        x: {
          beginAtZero: true,
          ticks: { stepSize: 1, font: { size: 11 } },
          grid: { color: "#f1f5f9" },
          border: { display: false },
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 11 } },
          border: { display: false },
        },
      },
      animation: { duration: 500 },
    }
  });

  // 3. Vertical bar — Confidence distribution
  confChart = new Chart(document.getElementById("chartConf"), {
    type: "bar",
    data: {
      labels: ["< 50%", "50–70%", "70–85%", "> 85%"],
      datasets: [{
        label: "Requirements",
        data: [0, 0, 0, 0],
        backgroundColor: ["#fca5a5", "#fcd34d", "#86efac", "#6366f1"],
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.y} requirement${ctx.parsed.y !== 1 ? "s" : ""}`,
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { stepSize: 1, font: { size: 11 } },
          grid: { color: "#f1f5f9" },
          border: { display: false },
        },
        x: {
          grid: { display: false },
          ticks: { font: { size: 11 } },
          border: { display: false },
        },
      },
      animation: { duration: 500 },
    }
  });
}

function updateCharts(summary, rows) {
  document.getElementById("chartsSection").style.display = "";

  if (!donutChart) initCharts();

  // 1. Donut
  donutChart.data.datasets[0].data = [summary.functional || 0, summary.nonFunctional || 0];
  document.getElementById("chartDonutSubtitle").textContent =
    `${summary.requirements || 0} total requirements identified`;
  donutChart.update();

  // 2. Bar — NFR types
  const types   = summary.nfrTypes || {};
  const present = TYPE_ORDER.filter(t => (types[t] || 0) > 0);
  barChart.data.labels                            = present;
  barChart.data.datasets[0].data                  = present.map(t => types[t] || 0);
  barChart.data.datasets[0].backgroundColor       = present.map(t => TYPE_COLORS[t] || "#94a3b8");
  document.getElementById("chartBarSubtitle").textContent =
    present.length ? `${present.length} categor${present.length !== 1 ? "ies" : "y"} detected` : "No NFRs yet";
  barChart.update();

  // 3. Confidence buckets
  const buckets = [0, 0, 0, 0];
  for (const row of rows) {
    const c = Number(row.classConfidence) || 0;
    if      (c < 0.50) buckets[0]++;
    else if (c < 0.70) buckets[1]++;
    else if (c < 0.85) buckets[2]++;
    else               buckets[3]++;
  }
  confChart.data.datasets[0].data = buckets;
  confChart.update();
}

// ── Table rendering ───────────────────────────────────────

function filteredRows() {
  if (activeFilter === "all")          return latestRows;
  if (activeFilter === "needs-review") return latestRows.filter(r => r.needsReview);
  if (activeFilter === "FR" || activeFilter === "NFR")
    return latestRows.filter(r => r.class.code === activeFilter);
  return latestRows.filter(r => r.nfrType && r.nfrType.name === activeFilter);
}

function renderRow(row) {
  const isNfr     = row.class.code === "NFR";
  const typeName  = row.nfrType ? row.nfrType.name : "";
  const typeSlug  = TYPE_SLUGS[typeName] || "";
  const borderCls = typeSlug ? `border-${typeSlug}` : (isNfr ? "border-nfr" : "border-fr");
  const fillCls   = typeSlug ? `fill-${typeSlug}`   : (isNfr ? "fill-nfr"   : "fill-fr");

  const reviewBadge = row.needsReview
    ? `<span class="badge needs-review" title="Model 1 confidence below 65% — verify manually">Review</span>`
    : "";

  const typeBadge = typeName
    ? `<span class="badge type type-${typeSlug}">${escapeHtml(typeName)}</span>`
    : `<span class="muted-text">—</span>`;

  const cc = pct(row.classConfidence);
  const rc = pct(row.requirementConfidence);

  return `
    <tr class="${row.needsReview ? "row-needs-review" : ""}">
      <td class="${borderCls}">
        <div class="requirement-text">${escapeHtml(row.text)}</div>
      </td>
      <td>
        <span class="badge ${isNfr ? "nfr" : "fr"}">${escapeHtml(row.class.name)}</span>
        ${reviewBadge}
      </td>
      <td>${typeBadge}</td>
      <td>
        <div class="conf-value">${cc}%</div>
        <div class="conf-bar"><span class="conf-fill ${fillCls}" style="width:${cc}%"></span></div>
        <small class="muted-text">Req ${rc}%</small>
      </td>
    </tr>`;
}

function renderGroupHeader(typeName, count) {
  const slug = TYPE_SLUGS[typeName] || "unclassified";
  return `
    <tr class="group-header gh-${slug}">
      <td colspan="4">
        <span class="gh-dot"></span>${escapeHtml(typeName)}
        <span class="gh-count">${count} requirement${count !== 1 ? "s" : ""}</span>
      </td>
    </tr>`;
}

function renderRows() {
  const rows = filteredRows();
  if (!rows.length) {
    resultsBody.innerHTML = `<tr><td colspan="4" class="empty-table">No rows match this view.</td></tr>`;
    return;
  }

  if (activeFilter === "NFR") {
    const groups = {};
    for (const row of rows) {
      const key = row.nfrType ? row.nfrType.name : "Unclassified";
      (groups[key] = groups[key] || []).push(row);
    }
    const keys = Object.keys(groups).sort((a, b) => {
      const oa = TYPE_ORDER.indexOf(a), ob = TYPE_ORDER.indexOf(b);
      return (oa < 0 ? 99 : oa) - (ob < 0 ? 99 : ob);
    });
    let html = "";
    for (const key of keys) {
      html += renderGroupHeader(key, groups[key].length);
      html += groups[key].map(renderRow).join("");
    }
    resultsBody.innerHTML = html;
    return;
  }

  resultsBody.innerHTML = rows.map(renderRow).join("");
}

// ── Export CSV ────────────────────────────────────────────

function exportCsv() {
  if (!latestRows.length) { showToast("Run an analysis before exporting."); return; }
  const headers = ["Requirement","FR/NFR","NFR Type","Needs Review","Requirement Confidence","Class Confidence"];
  const lines   = [headers.join(",")];
  for (const row of filteredRows()) {
    const vals = [
      row.text, row.class.name,
      row.nfrType ? row.nfrType.name : "",
      row.needsReview ? "Yes" : "No",
      row.requirementConfidence, row.classConfidence,
    ].map(v => `"${String(v).replaceAll('"','""')}"`);
    lines.push(vals.join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = "srs_requirements.csv"; a.click();
  URL.revokeObjectURL(url);
}

// ── Event listeners ───────────────────────────────────────

fileInput.addEventListener("change", () => {
  fileName.textContent = fileInput.files[0] ? fileInput.files[0].name : "No file selected";
});

tabBar.addEventListener("click", e => {
  const tab = e.target.closest(".tab");
  if (!tab) return;
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  tab.classList.add("active");
  activeFilter = tab.dataset.filter;
  renderRows();
});

exportButton.addEventListener("click", exportCsv);

form.addEventListener("submit", async event => {
  event.preventDefault();
  const button = form.querySelector("button[type='submit']");
  const data   = new FormData();
  if (fileInput.files[0]) data.append("file", fileInput.files[0]);
  data.append("text", textInput.value);
  button.disabled = true;
  button.innerHTML = `<span class="btn-icon">…</span> Analyzing`;
  try {
    const res     = await fetch("/api/analyze", { method: "POST", body: data });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "Analysis failed.");
    latestRows = payload.requirements || [];
    updateMetrics(payload.summary || {});
    updateCharts(payload.summary || {}, latestRows);
    // Reset to All tab
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelector('.tab[data-filter="all"]').classList.add("active");
    activeFilter = "all";
    renderRows();
    showToast(`Analysis complete — ${latestRows.length} requirements found.`);
    await loadStatus();
  } catch (err) {
    showToast(err.message || "Analysis failed.");
  } finally {
    button.disabled = false;
    button.innerHTML = `<span class="btn-icon">↗</span> Analyze`;
  }
});

loadStatus();
