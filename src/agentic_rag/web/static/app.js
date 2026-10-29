const thread = document.getElementById("thread");
const form = document.getElementById("composer");
const question = document.getElementById("question");
const sendBtn = document.getElementById("send");
const sources = document.getElementById("sources");
const health = document.getElementById("health");
const newChatBtn = document.getElementById("new-chat");
const topbarNewChatBtn = document.getElementById("topbar-new-chat");
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarClose = document.getElementById("sidebar-close");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const searchInput = document.getElementById("sources-search");
const composerContainer = document.getElementById("composer-container");

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function showEmpty() {
  thread.innerHTML = `
    <div class="welcome-screen">
      <div class="welcome-accent" aria-hidden="true">
        <svg class="welcome-orb" width="56" height="56" viewBox="0 0 56 56" fill="none">
          <circle cx="28" cy="28" r="26" fill="url(#orb-grad-bg)" />
          <circle cx="28" cy="28" r="20" fill="url(#orb-grad-inner)" />
          <circle cx="24" cy="22" r="8" fill="#ffffff" fill-opacity="0.25" />
          <defs>
            <radialGradient id="orb-grad-bg" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(24 20) rotate(52) scale(34)">
              <stop stop-color="#ffedd5" />
              <stop offset="0.5" stop-color="#fdba74" />
              <stop offset="1" stop-color="#fb923c" />
            </radialGradient>
            <radialGradient id="orb-grad-inner" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(26 22) rotate(48) scale(24)">
              <stop stop-color="#f97316" />
              <stop offset="0.6" stop-color="#ea580c" />
              <stop offset="1" stop-color="#c2410c" />
            </radialGradient>
          </defs>
        </svg>
      </div>
      <h1 class="welcome-heading">Hello, welcome to Agentic RAG</h1>
      <p class="welcome-lead">Ask questions about your documents or explore real-time information with verifiable evidence.</p>

      <div id="welcome-composer-slot" class="welcome-composer-slot"></div>

      <div class="feature-cards-grid">
        <button type="button" class="feature-card suggestion-chip" data-prompt="What is the hotel reimbursement limit?">
          <div class="card-head">
            <span class="card-icon card-icon-peach" aria-hidden="true">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
              </svg>
            </span>
            <span class="card-tag">Documents</span>
          </div>
          <div class="card-body">
            <h3 class="card-title">Document Q&amp;A</h3>
            <p class="card-desc">Find answers in your indexed company documents.</p>
          </div>
          <div class="card-example">
            <span>&ldquo;What is the hotel reimbursement limit?&rdquo;</span>
          </div>
        </button>

        <button type="button" class="feature-card suggestion-chip" data-prompt="What is the daily meal allowance in INR?">
          <div class="card-head">
            <span class="card-icon card-icon-purple" aria-hidden="true">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="1" y="4" width="22" height="16" rx="2" ry="2"></rect>
                <line x1="1" y1="10" x2="23" y2="10"></line>
              </svg>
            </span>
            <span class="card-tag">Policy</span>
          </div>
          <div class="card-body">
            <h3 class="card-title">Policy &amp; Expenses</h3>
            <p class="card-desc">Explore allowances and reimbursement policies.</p>
          </div>
          <div class="card-example">
            <span>&ldquo;What is the daily meal allowance in INR?&rdquo;</span>
          </div>
        </button>

        <button type="button" class="feature-card suggestion-chip" data-prompt="What time is it in Tokyo right now?">
          <div class="card-head">
            <span class="card-icon card-icon-mint" aria-hidden="true">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
              </svg>
            </span>
            <span class="card-tag">Live</span>
          </div>
          <div class="card-body">
            <h3 class="card-title">Live Information</h3>
            <p class="card-desc">Retrieve current information using available live tools.</p>
          </div>
          <div class="card-example">
            <span>&ldquo;What time is it in Tokyo right now?&rdquo;</span>
          </div>
        </button>
      </div>
    </div>`;

  const slot = document.getElementById("welcome-composer-slot");
  if (slot && form) {
    slot.appendChild(form);
  }
  if (composerContainer) {
    composerContainer.style.display = "none";
  }

  thread.querySelectorAll(".suggestion-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const promptText = chip.getAttribute("data-prompt") || chip.textContent.trim();
      question.value = promptText;
      submitQuestion();
    });
  });
}

function restoreBottomComposer() {
  if (composerContainer && form && form.parentElement !== composerContainer) {
    composerContainer.appendChild(form);
  }
  if (composerContainer) {
    composerContainer.style.display = "";
  }
}

async function refreshMeta() {
  try {
    const [h, s] = await Promise.all([
      fetch("/health").then((r) => r.json()),
      fetch("/sources").then((r) => r.json()),
    ]);
    health.textContent = `${h.chunks} chunks indexed · ${h.embedding_model}`;
    const docs = s.documents || [];
    sources.innerHTML = docs.length
      ? docs.map((doc) => `<li data-path="${escapeHtml(doc.path)}">${escapeHtml(doc.path)}</li>`).join("")
      : "<li class='muted'>No documents yet</li>";
    if (searchInput && searchInput.value.trim()) {
      filterSources(searchInput.value.trim());
    }
  } catch {
    health.textContent = "API unreachable";
  }
}

function filterSources(filterText) {
  const query = filterText.toLowerCase();
  const items = sources.querySelectorAll("li");
  items.forEach((item) => {
    if (item.classList.contains("muted")) return;
    const text = item.textContent.toLowerCase();
    item.style.display = text.includes(query) ? "" : "none";
  });
}

if (searchInput) {
  searchInput.addEventListener("input", (e) => {
    filterSources(e.target.value);
  });
}

window.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    if (searchInput) {
      searchInput.focus();
      searchInput.select();
    }
  }
});

function renderAnswer(payload) {
  const docs = payload.document_evidence || [];
  const live = payload.live_evidence || [];
  const caveats = payload.caveats || [];
  const steps = payload.trace || [];
  const isGrounded = payload.sufficient_evidence;
  const cls = isGrounded ? "card" : "card insufficient";
  return `
    <article class="${cls}">
      <div class="card-header">
        <div class="agent-badge">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="9" stroke="#ea580c" stroke-width="2.5" />
            <circle cx="12" cy="12" r="3.5" fill="#ea580c" />
          </svg>
          <span>Agentic RAG</span>
        </div>
        <span class="badge ${isGrounded ? 'badge-grounded' : 'badge-insufficient'}">
          ${isGrounded ? "Grounded" : "Insufficient evidence"} · ${escapeHtml(payload.trace_id)}
        </span>
      </div>
      <div class="answer">${escapeHtml(payload.answer)}</div>
      <div class="grid">
        <div class="panel">
          <h4>From your documents</h4>
          ${docs.length ? docs.map((c) => `<div class="cite"><strong>[${escapeHtml(c.label)}]</strong> ${escapeHtml(c.reference)}<br>${escapeHtml(c.preview)}</div>`).join("") : "<p class='muted'>No document passages used.</p>"}
        </div>
        <div class="panel">
          <h4>Live information</h4>
          ${live.length ? live.map((c) => `<div class="cite"><strong>[${escapeHtml(c.label)}]</strong> ${escapeHtml(c.reference)}<br>${escapeHtml(c.preview)}</div>`).join("") : "<p class='muted'>No live tools used.</p>"}
        </div>
      </div>
      ${caveats.length ? `<div class="caveats">${caveats.map(escapeHtml).join("<br>")}</div>` : ""}
      <details class="trace">
        <summary>How I answered (${steps.length} steps)</summary>
        <ol class="timeline">
          ${steps.map((step) => `<li><strong>${escapeHtml(step.name)}</strong> — ${escapeHtml(step.detail)} · ${Math.round(step.duration_ms)} ms</li>`).join("")}
        </ol>
      </details>
    </article>`;
}

async function submitQuestion() {
  const text = question.value.trim();
  if (!text) return;

  const isWelcomeScreen = Boolean(thread.querySelector(".welcome-screen"));
  if (isWelcomeScreen) {
    restoreBottomComposer();
    thread.innerHTML = "";
  }

  thread.insertAdjacentHTML("beforeend", `
    <div class="msg user">
      <div class="bubble">${escapeHtml(text)}</div>
    </div>
  `);

  question.value = "";
  question.style.height = "auto";
  if (sendBtn) sendBtn.disabled = true;

  const pending = document.createElement("div");
  pending.className = "msg";
  pending.innerHTML = `
    <article class="card">
      <div class="pending-state">
        <span class="pulse-spinner"></span>
        <span>Routing query, retrieving evidence, and calling tools…</span>
      </div>
    </article>
  `;
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
  } finally {
    if (sendBtn) sendBtn.disabled = false;
  }

  thread.scrollTop = thread.scrollHeight;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion();
});

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitQuestion();
  }
});

question.addEventListener("input", () => {
  question.style.height = "auto";
  question.style.height = Math.min(question.scrollHeight, 180) + "px";
});

function handleNewChat() {
  restoreBottomComposer();
  thread.innerHTML = "";
  showEmpty();
  question.value = "";
  question.style.height = "auto";
  question.focus();
}

if (newChatBtn) {
  newChatBtn.addEventListener("click", handleNewChat);
}

if (topbarNewChatBtn) {
  topbarNewChatBtn.addEventListener("click", handleNewChat);
}

function toggleSidebar() {
  if (sidebar) sidebar.classList.toggle("open");
  if (sidebarBackdrop) sidebarBackdrop.classList.toggle("open");
}

function closeSidebar() {
  if (sidebar) sidebar.classList.remove("open");
  if (sidebarBackdrop) sidebarBackdrop.classList.remove("open");
}

if (sidebarToggle) sidebarToggle.addEventListener("click", toggleSidebar);
if (sidebarClose) sidebarClose.addEventListener("click", closeSidebar);
if (sidebarBackdrop) sidebarBackdrop.addEventListener("click", closeSidebar);

document.getElementById("rebuild").addEventListener("click", async () => {
  health.textContent = "Rebuilding index…";
  await fetch("/index?rebuild=true", { method: "POST" });
  await refreshMeta();
});

showEmpty();
refreshMeta();
