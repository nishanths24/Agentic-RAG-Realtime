const EXAMPLES = [
  "What is the hotel reimbursement limit in our travel policy?",
  "What time is it in Tokyo right now?",
  "The policy gives a daily meal allowance in USD. What's that in INR today?",
  "What's the weather in Paris, and does our policy allow rebooking for weather delays?",
  "Who won the 2030 World Cup?",
];

const thread = document.getElementById("thread");
const form = document.getElementById("composer");
const question = document.getElementById("question");
const pills = document.getElementById("pills");
const sources = document.getElementById("sources");
const health = document.getElementById("health");

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function showEmpty() {
  thread.innerHTML = `
    <div class="empty">
      <h3>Ask a question</h3>
      <p>Agentic RAG retrieves policy passages, calls live tools when needed, and shows both kinds of evidence.</p>
    </div>`;
}

EXAMPLES.forEach((text) => {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = text;
  btn.addEventListener("click", () => {
    question.value = text;
    question.focus();
  });
  pills.appendChild(btn);
});

async function refreshMeta() {
  try {
    const [h, s] = await Promise.all([
      fetch("/health").then((r) => r.json()),
      fetch("/sources").then((r) => r.json()),
    ]);
    health.textContent = `${h.chunks} chunks indexed · ${h.embedding_model}`;
    sources.innerHTML = (s.documents || []).map(
      (doc) => `<li>${escapeHtml(doc.path)}</li>`
    ).join("") || "<li class='muted'>No documents yet</li>";
  } catch {
    health.textContent = "API unreachable";
  }
}

function renderAnswer(payload) {
  const docs = payload.document_evidence || [];
  const live = payload.live_evidence || [];
  const caveats = payload.caveats || [];
  const steps = payload.trace || [];
  const cls = payload.sufficient_evidence ? "card" : "card insufficient";
  return `
    <article class="${cls}">
      <span class="badge">${payload.sufficient_evidence ? "Grounded" : "Insufficient evidence"} · ${escapeHtml(payload.trace_id)}</span>
      <div class="answer">${escapeHtml(payload.answer)}</div>
      <div class="grid">
        <div class="panel">
          <h4>From your documents</h4>
          ${docs.length ? docs.map((c) => `<p class="cite"><strong>[${escapeHtml(c.label)}]</strong> ${escapeHtml(c.reference)}<br>${escapeHtml(c.preview)}</p>`).join("") : "<p class='muted'>No document passages used.</p>"}
        </div>
        <div class="panel">
          <h4>Live information</h4>
          ${live.length ? live.map((c) => `<p class="cite"><strong>[${escapeHtml(c.label)}]</strong> ${escapeHtml(c.reference)}<br>${escapeHtml(c.preview)}</p>`).join("") : "<p class='muted'>No live tools used.</p>"}
        </div>
      </div>
      ${caveats.length ? `<div class="caveats">${caveats.map(escapeHtml).join("<br>")}</div>` : ""}
      <details class="trace">
        <summary>How I answered</summary>
        <ol class="timeline">
          ${steps.map((step) => `<li><strong>${escapeHtml(step.name)}</strong> — ${escapeHtml(step.detail)} · ${Math.round(step.duration_ms)} ms</li>`).join("")}
        </ol>
      </details>
    </article>`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text) return;
  if (thread.querySelector(".empty")) thread.innerHTML = "";
  thread.insertAdjacentHTML("beforeend", `<div class="msg user"><div class="bubble">${escapeHtml(text)}</div></div>`);
  question.value = "";
  const pending = document.createElement("div");
  pending.className = "msg";
  pending.innerHTML = `<article class="card"><p class="muted">Routing, retrieving, and calling tools…</p></article>`;
  thread.appendChild(pending);
  thread.scrollTop = thread.scrollHeight;
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, trace: true }),
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "Request failed");
    pending.innerHTML = renderAnswer(payload);
  } catch (err) {
    pending.innerHTML = `<article class="card insufficient"><p>${escapeHtml(err.message)}</p></article>`;
  }
  thread.scrollTop = thread.scrollHeight;
});

document.getElementById("rebuild").addEventListener("click", async () => {
  health.textContent = "Rebuilding index…";
  await fetch("/index?rebuild=true", { method: "POST" });
  await refreshMeta();
});

showEmpty();
refreshMeta();
