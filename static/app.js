let model = null;
let chart = null;

const n = (x, digits=2) =>
  x === null || x === undefined || Number.isNaN(Number(x))
    ? "—"
    : Number(x).toFixed(digits);

const percentile = x =>
  x === null || x === undefined || Number.isNaN(Number(x))
    ? "—"
    : `${Number(x).toFixed(1)}%`;

function statusClass(status) {
  if (status.includes("DEGENERACY")) return "status-degeneracy";
  if (status.includes("SHIFT")) return "status-shift";
  return "status-normal";
}

function renderDashboard(data) {
  model = data;

  const status = document.getElementById("overallStatus");
  status.textContent = data.overall_status;
  status.className = `status ${statusClass(data.overall_status)}`;

  document.getElementById("asOf").textContent = `As of ${data.as_of}`;
  document.getElementById("confirmation").textContent = data.cross_window_confirmation;

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
        <div class="kpi"><span>Condition pctile</span><strong>${percentile(w.condition_percentile)}</strong></div>
        <div class="kpi"><span>Rotation pctile</span><strong>${percentile(w.rotation_percentile)}</strong></div>
      </div>
    </article>
  `).join("");

  document.getElementById("contributionBars").innerHTML =
    data.aggregate_contributions.map(item => `
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

  document.getElementById("interpretation").textContent = data.interpretation;
  document.getElementById("researchNote").textContent = data.research_note;

  const reasons = data.windows
    .filter(w => w.status !== "NORMAL")
    .flatMap(w => w.reasons.map(reason => `${w.window}D: ${reason}`));

  document.getElementById("whyFlagged").innerHTML =
    `<ul>${(reasons.length ? reasons : ["All monitored time scales remain within their recent historical ranges."])
      .map(reason => `<li>${reason}</li>`).join("")}</ul>`;

  drawChart(Number(document.getElementById("windowSelect").value));
}

function drawChart(windowSize) {
  if (!model) return;
  const item = model.windows.find(w => w.window === windowSize);
  if (!item) return;

  if (chart) chart.destroy();

  chart = new Chart(document.getElementById("geometryChart"), {
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

async function load(refresh=false) {
  const button = document.getElementById("refreshButton");
  button.disabled = true;
  button.textContent = refresh ? "Refreshing…" : "Loading…";

  try {
    const response = await fetch(refresh ? "/api/refresh" : "/api/dashboard");
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
  } finally {
    button.disabled = false;
    button.textContent = "Refresh data";
  }
}

document.getElementById("refreshButton").addEventListener("click", () => load(true));
document.getElementById("windowSelect").addEventListener(
  "change",
  event => drawChart(Number(event.target.value))
);

load(false);
