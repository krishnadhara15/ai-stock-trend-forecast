const tickerSelect = document.getElementById("ticker");
const predictBtn = document.getElementById("predict-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const stepsEl = document.getElementById("steps");
const metaEl = document.getElementById("meta");
const disclaimerEl = document.getElementById("disclaimer");

async function loadTickers() {
  try {
    const res = await fetch("/tickers");
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    tickerSelect.innerHTML = data.tickers
      .map((t) => `<option value="${t}">${t}</option>`)
      .join("");
    statusEl.textContent = `${data.count} tickers available`;
  } catch (err) {
    statusEl.textContent = "API not ready — train model and start server";
    predictBtn.disabled = true;
  }
}

async function predict() {
  const ticker = tickerSelect.value;
  if (!ticker) return;

  predictBtn.disabled = true;
  statusEl.textContent = "Predicting…";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    metaEl.textContent = `${data.ticker} · window ends ${data.window_end_date}`;
    stepsEl.innerHTML = data.predictions
      .map(
        (p) => `
      <div class="step">
        <span class="label">Step +${p.step}</span>
        <span class="badge ${p.direction}">${p.direction}</span>
        <span class="confidence">${(p.confidence * 100).toFixed(1)}% conf.</span>
      </div>`
      )
      .join("");
    disclaimerEl.textContent = data.disclaimer;
    resultsEl.classList.remove("hidden");
    statusEl.textContent = "Done";
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    predictBtn.disabled = false;
  }
}

predictBtn.addEventListener("click", predict);
loadTickers();
