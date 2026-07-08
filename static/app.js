const $ = (sel) => document.querySelector(sel);

// --- Tab switching -------------------------------------------------------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $("#" + tab.dataset.tab).classList.add("active");
  });
});

// --- Helpers -------------------------------------------------------------
const esc = (s) =>
  (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function jobCard(job, index) {
  const badges = [
    index != null ? `<span class="badge cite">[${index}]</span>` : "",
    job.score != null ? `<span class="badge score">match ${job.score}</span>` : "",
    job.job_type ? `<span class="badge">${esc(job.job_type)}</span>` : "",
    job.category ? `<span class="badge">${esc(job.category)}</span>` : "",
    job.source ? `<span class="badge">${esc(job.source)}</span>` : "",
  ].join("");
  return `
    <article class="card">
      <div class="badges">${badges}</div>
      <h3><a href="${esc(job.url)}" target="_blank" rel="noopener">${esc(job.title)}</a></h3>
      <div class="meta">${esc(job.company)} · ${esc(job.location) || "Remote"}${job.salary ? " · " + esc(job.salary) : ""}</div>
      <div class="snippet">${esc(job.summary)}</div>
    </article>`;
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

// --- Search --------------------------------------------------------------
$("#search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = $("#query").value.trim();
  if (!query) return;
  const btn = e.target.querySelector("button");
  btn.disabled = true;
  $("#results").innerHTML = `<div class="empty">Searching…</div>`;
  $("#overview").classList.add("hidden");

  const data = await postJSON("/api/search", {
    query,
    category: $("#category").value || null,
  });

  if (data.overview) {
    $("#overview").textContent = data.overview;
    $("#overview").classList.remove("hidden");
  }
  const results = data.results || [];
  $("#results").innerHTML = results.length
    ? results.map((j) => jobCard(j, null)).join("")
    : `<div class="empty">${esc(data.message || "No matching jobs found.")}</div>`;
  btn.disabled = false;
});

// --- Ask (RAG) -----------------------------------------------------------
$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("#question").value.trim();
  if (!question) return;
  const btn = e.target.querySelector("button");
  btn.disabled = true;
  $("#answer").classList.remove("hidden");
  $("#answer").textContent = "Thinking…";
  $("#sources").innerHTML = "";

  const data = await postJSON("/api/ask", { question });
  $("#answer").textContent = data.answer || "No answer.";
  $("#sources").innerHTML = (data.sources || []).map((j, i) => jobCard(j, i + 1)).join("");
  btn.disabled = false;
});

// --- Init: load categories + status -------------------------------------
(async () => {
  try {
    const health = await (await fetch("/api/health")).json();
    $("#status").textContent = health.index_ready
      ? `${health.jobs_indexed} jobs indexed · AI ${health.ai_enabled ? "enabled" : "disabled (set GEMINI_API_KEY)"}`
      : "Index not built yet — run: python -m src.ingest";

    const { categories } = await (await fetch("/api/categories")).json();
    const sel = $("#category");
    categories.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      sel.appendChild(opt);
    });
  } catch {
    $("#status").textContent = "Could not reach the API.";
  }
})();
