let model = null;
let geometryChart = null;
let statusChart = null;

const n = (x, digits=2) =>
  x === null || x === undefined || Number.isNaN(Number(x))
    ? "—"
    : Number(x).toFixed(digits);

const pct = x =>
  x === null || x === undefined || Number.isNaN(Number(x))
    ? "—"
    : `${Number(x).toFixed(1)}%`;

function statusClass(status) {
  if (!status) return "";
  if (status.includes("DEGENERACY")) return "status-degeneracy";
  if (status.includes("SHIFT")) return "status-shift";
  return "status-normal";
}

function statusValue(status) {
  return {
    "NORMAL": 0,
    "GEOMETRY SHIFT": 1,
    "APPROACHING DEGENERACY": 2,
    "HIGH DEGENERACY": 3
  }[status] ?? null;
}

function contributionBars(items) {
  return items.map(item => `
    <div class="bar-row">
      <div class="bar-label">
        <span>${item.name}</span>
        <strong>${(item.value * 100).toFixed(1)}%</strong>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${Math.min(100, item.value * 100)}%"></div>
      </div>
    </div>
  `).join("");
}

function renderDashboard(data) {
  model = data;

  const status = document.getElementById("overallStatus");
  status.textContent = data.overall_status;
  status.className = `status ${statusClass(data.overall_status)}`;

  document.getElementById("asOf").textContent = `As of ${data.as_of}`;
  document.getElementById("confirmation").textContent = data.cross_window_confirmation;

  const generated = data.snapshot_generated_at
    ? new Date(data.snapshot_generated_at).toLocaleString()
    : "—";
  document.getElementById("snapshotTime").textContent = generated;

  const top = data.aggregate_contributions[0];
  document.getElementById("dominantDriver").textContent = top?.name ?? "—";
  document.getElementById("dominantShare").textContent =
    top ? `${(top.value * 100).toFixed(1)}% of average weak-direction composition` : "";

  document.getElementById("windowCards").innerHTML = data.windows.map(w => `
    <article class="window-card">
      <div class="window-head">
        <div class="window-name">${w.window} day</div>
        <div class="badge ${statusClass(w.status)}">${w.status}</div>
      </div>
      <div class="kpi-grid">
        <div class="kpi"><span>Sigma min</span><strong>${n(w.sigma_min,3)}</strong></div>
        <div class="kpi"><span>Sigma ratio</span><strong>${n(w.sigma_ratio,3)}</strong></div>
        <div class="kpi"><span>Condition #</span><strong>${n(w.condition_number,1)}</strong></div>
        <div class="kpi"><span>Rotation</span><strong>${n(w.rotation_deg,1)}°</strong></div>
        <div class="kpi"><span>Condition pctile</span><strong>${pct(w.condition_percentile)}</strong></div>
        <div class="kpi"><span>Rotation pctile</span><strong>${pct(w.rotation_percentile)}</strong></div>
      </div>
    </article>
  `).join("");

  document.getElementById("contributionBars").innerHTML =
    contributionBars(data.aggregate_contributions);

  document.getElementById("interpretation").textContent = data.interpretation;
  document.getElementById("researchNote").textContent = data.research_note;

  const reasons = data.windows
    .filter(w => w.status !== "NORMAL")
    .flatMap(w => w.reasons.map(reason => `${w.window}D: ${reason}`));

  document.getElementById("whyFlagged").innerHTML =
    `<ul>${(reasons.length ? reasons : ["All monitored time scales remain within their recent historical ranges."])
      .map(reason => `<li>${reason}</li>`).join("")}</ul>`;

  const dateSelect = document.getElementById("historyDate");
  const dates = [...new Set(
    data.status_history.map(row => row.date)
  )].reverse();

  dateSelect.innerHTML = dates.map(date =>
    `<option value="${date}">${date}</option>`
  ).join("");

  drawGeometryChart(Number(document.getElementById("windowSelect").value));
  renderDegeneracyDetails(Number(document.getElementById("detailWindow").value));
  renderHistoricalExplorer(dateSelect.value);
  drawStatusChart();
  renderStatusTable();
}

function drawGeometryChart(windowSize) {
  if (!model) return;

  const item = model.windows.find(w => w.window === windowSize);
  if (!item) return;

  if (geometryChart) geometryChart.destroy();

  geometryChart = new Chart(document.getElementById("geometryChart"), {
    type: "line",
    data: {
      labels: item.history.map(p => p.date),
      datasets: [
        {
          label: "Sigma ratio",
          data: item.history.map(p => p.sigma_ratio),
          yAxisID: "y"
        },
        {
          label: "Condition #",
          data: item.history.map(p => p.condition),
          yAxisID: "y1"
        },
        {
          label: "Rotation (deg)",
          data: item.history.map(p => p.rotation),
          yAxisID: "y1"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom" } },
      elements: {
        point: { radius: 0 },
        line: { borderWidth: 1.6 }
      },
      scales: {
        y: {
          position: "left",
          title: { display: true, text: "Sigma ratio" }
        },
        y1: {
          position: "right",
          grid: { drawOnChartArea: false },
          title: { display: true, text: "Condition / Rotation" }
        }
      }
    }
  });
}

function renderDegeneracyDetails(windowSize) {
  if (!model) return;

  const w = model.windows.find(item => item.window === windowSize);
  if (!w) return;

  const reasons = w.reasons.map(r => `<li>${r}</li>`).join("");
  const topDrivers = w.contributions.slice(0, 5);

  document.getElementById("degeneracyDetails").innerHTML = `
    <div class="detail-grid">
      <div class="detail-card">
        <div class="label">STATUS</div>
        <div class="detail-status ${statusClass(w.status)}">${w.status}</div>
        <div class="muted">Degeneracy score: ${w.degeneracy_score}</div>
      </div>

      <div class="detail-card">
        <div class="label">SINGULAR CONTRACTION</div>
        <div class="detail-number">${n(w.sigma_ratio, 4)}</div>
        <div class="muted">Trailing percentile: ${pct(w.sigma_ratio_percentile)}</div>
      </div>

      <div class="detail-card">
        <div class="label">CONDITIONING</div>
        <div class="detail-number">${n(w.condition_number, 2)}</div>
        <div class="muted">Trailing percentile: ${pct(w.condition_percentile)}</div>
      </div>

      <div class="detail-card">
        <div class="label">WEAK-DIRECTION ROTATION</div>
        <div class="detail-number">${n(w.rotation_deg, 1)}°</div>
        <div class="muted">Trailing percentile: ${pct(w.rotation_percentile)}</div>
      </div>
    </div>

    <div class="detail-split">
      <div>
        <h3>Why this status?</h3>
        <ul class="reason-list">${reasons}</ul>
      </div>
      <div>
        <h3>Current weak-direction drivers</h3>
        ${contributionBars(topDrivers)}
      </div>
    </div>
  `;
}

function renderHistoricalExplorer(date) {
  if (!model || !date) return;

  const overall = model.status_history.find(row => row.date === date);

  const rows = model.windows.map(w => {
    const d = w.history_detail.find(row => row.date === date);
    return d ? { window: w.window, ...d } : null;
  }).filter(Boolean);

  document.getElementById("historicalExplorer").innerHTML = `
    <div class="history-summary">
      <div>
        <div class="label">OVERALL STATE</div>
        <div class="detail-status ${statusClass(overall?.overall_status)}">
          ${overall?.overall_status ?? "—"}
        </div>
      </div>
      <div>
        <div class="label">CONFIRMING WINDOWS</div>
        <div class="detail-number">${overall?.confirming_windows ?? "—"} / 3</div>
      </div>
    </div>

    <div class="history-window-grid">
      ${rows.map(row => `
        <article class="history-card">
          <div class="window-head">
            <strong>${row.window} day</strong>
            <span class="badge ${statusClass(row.status)}">${row.status}</span>
          </div>
          <div class="history-metrics">
            <span>Sigma ratio <strong>${n(row.sigma_ratio, 3)}</strong></span>
            <span>Condition <strong>${n(row.condition_number, 1)}</strong></span>
            <span>Rotation <strong>${n(row.rotation_deg, 1)}°</strong></span>
            <span>Score <strong>${row.degeneracy_score}</strong></span>
          </div>
          <div class="top-drivers">
            ${row.contributions.slice(0,3).map(c =>
              `<span>${c.name}: ${(c.value*100).toFixed(1)}%</span>`
            ).join("")}
          </div>
        </article>
      `).join("")}
    </div>
  `;
}

function drawStatusChart() {
  if (!model) return;

  const history = model.status_history;
  if (statusChart) statusChart.destroy();

  statusChart = new Chart(document.getElementById("statusChart"), {
    type: "line",
    data: {
      labels: history.map(row => row.date),
      datasets: [{
        label: "Overall geometry state",
        data: history.map(row => statusValue(row.overall_status)),
        stepped: true,
        tension: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      elements: {
        point: { radius: 0 },
        line: { borderWidth: 1.8 }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: context => {
              const map = [
                "NORMAL",
                "GEOMETRY SHIFT",
                "APPROACHING DEGENERACY",
                "HIGH DEGENERACY"
              ];
              return map[context.raw] ?? "—";
            }
          }
        }
      },
      scales: {
        y: {
          min: 0,
          max: 3,
          ticks: {
            stepSize: 1,
            callback: value => ({
              0: "NORMAL",
              1: "SHIFT",
              2: "APPROACHING",
              3: "HIGH"
            })[value] ?? ""
          }
        }
      }
    }
  });
}

function renderStatusTable() {
  const rows = model.status_history.slice(-15).reverse();

  document.getElementById("statusTable").innerHTML = `
    <table class="status-table">
      <thead>
        <tr>
          <th>Date</th>
          <th>Overall</th>
          <th>63D</th>
          <th>126D</th>
          <th>252D</th>
          <th>Confirming</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(row => `
          <tr>
            <td>${row.date}</td>
            <td class="${statusClass(row.overall_status)}">${row.overall_status}</td>
            <td>${row.window_statuses["63"]}</td>
            <td>${row.window_statuses["126"]}</td>
            <td>${row.window_statuses["252"]}</td>
            <td>${row.confirming_windows}/3</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

async function load() {
  try {
    const response = await fetch("/api/dashboard");

    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        message = body.detail || message;
      } catch {}
      throw new Error(message);
    }

    renderDashboard(await response.json());
  } catch (error) {
    document.getElementById("interpretation").textContent =
      `Unable to load market geometry: ${error.message}`;
  }
}

document.getElementById("windowSelect").addEventListener(
  "change",
  event => drawGeometryChart(Number(event.target.value))
);

document.getElementById("detailWindow").addEventListener(
  "change",
  event => renderDegeneracyDetails(Number(event.target.value))
);

document.getElementById("historyDate").addEventListener(
  "change",
  event => renderHistoricalExplorer(event.target.value)
);

load();
