const appRoot = document.getElementById("app");
const thread = document.getElementById("thread");
const form = document.getElementById("composer");
const question = document.getElementById("question");
const sendBtn = document.getElementById("send");
const sources = document.getElementById("sources");
const health = document.getElementById("health");
const newChatBtn = document.getElementById("new-chat");
const topbarNewChatBtn = document.getElementById("topbar-new-chat");
const sidebarAddDocBtn = document.getElementById("sidebar-add-doc");
const uploadModal = document.getElementById("upload-modal");
const modalCloseBtn = document.getElementById("modal-close-btn");
const modalCancelBtn = document.getElementById("modal-cancel-btn");
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const dropzoneEmpty = document.getElementById("dropzone-empty");
const selectedFileCard = document.getElementById("selected-file-card");
const fileNameDisplay = document.getElementById("file-name-display");
const fileSizeDisplay = document.getElementById("file-size-display");
const fileExtBadge = document.getElementById("file-ext-badge");
const removeFileBtn = document.getElementById("remove-file-btn");
const modalSubmitBtn = document.getElementById("modal-submit-btn");
const uploadStatus = document.getElementById("upload-status");
const modalOptions = document.getElementById("modal-options");
const overwriteCheckbox = document.getElementById("overwrite-checkbox");
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarClose = document.getElementById("sidebar-close");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const searchInput = document.getElementById("sources-search");
const composerContainer = document.getElementById("composer-container");
const deleteModal = document.getElementById("delete-modal");
const deleteModalCloseBtn = document.getElementById("delete-modal-close-btn");
const deleteCancelBtn = document.getElementById("delete-cancel-btn");
const deleteConfirmBtn = document.getElementById("delete-confirm-btn");
const deleteDocName = document.getElementById("delete-doc-name");
const deleteStatus = document.getElementById("delete-status");
const topbarBrand = document.getElementById("topbar-brand");

function setHeaderBrandingVisible(visible) {
  if (!topbarBrand) return;
  topbarBrand.classList.toggle("visible", Boolean(visible));
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function formatFileSize(bytes) {
  if (bytes == null || isNaN(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showEmpty() {
  setHeaderBrandingVisible(false);
  thread.innerHTML = `
    <div class="welcome-screen">
      <div class="welcome-accent">
        <img src="/static/robot-logo.png" alt="Agentic RAG Assistant" class="welcome-robot-logo" />
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
    if (!docs.length) {
      sources.innerHTML = "<li class='muted'>No documents yet</li>";
    } else {
      sources.innerHTML = docs
        .map((doc) => {
          const pathStr = typeof doc === "string" ? doc : doc.path || "";
          const filename = (typeof doc === "object" && doc.filename) ? doc.filename : pathStr;
          const ext = ((typeof doc === "object" && doc.extension) ? doc.extension : pathStr.slice(pathStr.lastIndexOf("."))).replace(".", "").toUpperCase();
          const sizeStr = (typeof doc === "object" && doc.size_bytes != null) ? formatFileSize(doc.size_bytes) : "";
          const isMd = ext === "MD";
          return `
            <li class="source-item" data-path="${escapeHtml(pathStr)}">
              <div class="source-item-head">
                <span class="source-ext-badge ${isMd ? 'badge-md' : 'badge-txt'}">${escapeHtml(ext || "TXT")}</span>
                <span class="source-name" title="${escapeHtml(pathStr)}">${escapeHtml(filename)}</span>
                <button type="button" class="source-delete-btn" data-filename="${escapeHtml(filename)}" title="Delete ${escapeHtml(filename)}" aria-label="Delete document ${escapeHtml(filename)}">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <polyline points="3 6 5 6 21 6"></polyline>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    <line x1="10" y1="11" x2="10" y2="17"></line>
                    <line x1="14" y1="11" x2="14" y2="17"></line>
                  </svg>
                </button>
              </div>
              <div class="source-item-meta">
                <span class="source-status-pill">
                  <span class="status-dot-mini"></span>
                  <span>Indexed</span>
                </span>
                ${sizeStr ? `<span class="source-size">${escapeHtml(sizeStr)}</span>` : ""}
              </div>
            </li>
          `;
        })
        .join("");
    }
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
          <img src="/static/robot-logo.png" alt="" aria-hidden="true" class="agent-robot-icon" />
          <span>Agentic RAG</span>
        </div>
        <span class="badge ${isGrounded ? 'badge-grounded' : 'badge-insufficient'}">
          ${isGrounded ? "Grounded" : "Insufficient evidence"} · ${escapeHtml(payload.trace_id)}
        </span>
      </div>
      <div class="answer" data-raw-answer="${escapeHtml(payload.answer)}">${escapeHtml(payload.answer)}</div>
      <div class="answer-actions" role="toolbar" aria-label="Answer actions">
        <button type="button" class="action-btn copy-btn" title="Copy answer" aria-label="Copy answer">
          <svg class="icon-copy" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
          </svg>
          <svg class="icon-check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display: none;" aria-hidden="true">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
          <span class="action-tooltip">Copy</span>
        </button>
        <button type="button" class="action-btn tts-btn" title="Read aloud" aria-label="Read aloud">
          <svg class="tts-icon-idle" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
          </svg>
          <svg class="tts-icon-active" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: none;" aria-hidden="true">
            <rect x="6" y="6" width="12" height="12" rx="2"></rect>
          </svg>
          <span class="action-tooltip">Read aloud</span>
        </button>
      </div>
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

  stopSpeech();

  const isWelcomeScreen = Boolean(thread.querySelector(".welcome-screen"));
  if (isWelcomeScreen) {
    restoreBottomComposer();
    thread.innerHTML = "";
  }
  setHeaderBrandingVisible(true);

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
  stopSpeech();
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

function isMobileView() {
  return window.innerWidth <= 840;
}

function updateToggleAria(expanded) {
  if (sidebarToggle) {
    sidebarToggle.setAttribute("aria-expanded", String(expanded));
    sidebarToggle.setAttribute("title", expanded ? "Collapse sidebar" : "Expand sidebar");
    sidebarToggle.setAttribute("aria-label", expanded ? "Collapse sidebar" : "Expand sidebar");
  }
}

function toggleSidebar() {
  if (isMobileView()) {
    const willOpen = !sidebar.classList.contains("open");
    sidebar.classList.toggle("open", willOpen);
    if (sidebarBackdrop) sidebarBackdrop.classList.toggle("open", willOpen);
    updateToggleAria(willOpen);
  } else {
    const isCurrentlyCollapsed = appRoot ? appRoot.classList.contains("sidebar-collapsed") : false;
    const willCollapse = !isCurrentlyCollapsed;
    if (appRoot) appRoot.classList.toggle("sidebar-collapsed", willCollapse);
    updateToggleAria(!willCollapse);
  }
}

function closeSidebar() {
  if (isMobileView()) {
    if (sidebar) sidebar.classList.remove("open");
    if (sidebarBackdrop) sidebarBackdrop.classList.remove("open");
    updateToggleAria(false);
  } else {
    if (appRoot) appRoot.classList.add("sidebar-collapsed");
    updateToggleAria(false);
  }
}

if (sidebarToggle) sidebarToggle.addEventListener("click", toggleSidebar);
if (sidebarClose) sidebarClose.addEventListener("click", closeSidebar);
if (sidebarBackdrop) sidebarBackdrop.addEventListener("click", closeSidebar);

window.addEventListener("resize", () => {
  if (!isMobileView()) {
    if (sidebar) sidebar.classList.remove("open");
    if (sidebarBackdrop) sidebarBackdrop.classList.remove("open");
    const isCollapsed = appRoot && appRoot.classList.contains("sidebar-collapsed");
    updateToggleAria(!isCollapsed);
  } else {
    const isOpen = Boolean(sidebar && sidebar.classList.contains("open"));
    updateToggleAria(isOpen);
  }
});

document.getElementById("rebuild").addEventListener("click", async () => {
  health.textContent = "Rebuilding index…";
  await fetch("/index?rebuild=true", { method: "POST" });
  await refreshMeta();
});

// Document Upload Modal Handling
let currentSelectedFile = null;

function openModal() {
  if (!uploadModal) return;
  uploadModal.classList.add("active");
  uploadModal.setAttribute("aria-hidden", "false");
  resetUploadForm();
}

function closeModal() {
  if (!uploadModal) return;
  uploadModal.classList.remove("active");
  uploadModal.setAttribute("aria-hidden", "true");
  resetUploadForm();
}

function resetUploadForm() {
  currentSelectedFile = null;
  if (fileInput) fileInput.value = "";
  if (dropzoneEmpty) dropzoneEmpty.style.display = "";
  if (selectedFileCard) selectedFileCard.style.display = "none";
  if (modalOptions) modalOptions.style.display = "none";
  if (overwriteCheckbox) overwriteCheckbox.checked = false;
  if (modalSubmitBtn) {
    modalSubmitBtn.disabled = true;
    const btnText = modalSubmitBtn.querySelector(".btn-text");
    const btnSpinner = modalSubmitBtn.querySelector(".btn-spinner");
    if (btnText) btnText.style.display = "";
    if (btnSpinner) btnSpinner.style.display = "none";
  }
  if (uploadStatus) {
    uploadStatus.style.display = "none";
    uploadStatus.className = "upload-status";
    uploadStatus.innerHTML = "";
  }
  if (dropzone) dropzone.classList.remove("dragover");
}

function showUploadStatus(message, type) {
  if (!uploadStatus) return;
  uploadStatus.className = `upload-status status-${type}`;
  uploadStatus.innerHTML = message;
  uploadStatus.style.display = "";
}

function handleFileSelection(file) {
  if (!file) return;

  const name = file.name || "";
  const ext = name.slice(name.lastIndexOf(".")).toLowerCase();
  const allowedExts = [".txt", ".md"];

  if (!allowedExts.includes(ext)) {
    showUploadStatus(`<strong>Unsupported file type (${escapeHtml(ext || "none")}).</strong> Please select a .txt or .md document.`, "error");
    currentSelectedFile = null;
    if (modalSubmitBtn) modalSubmitBtn.disabled = true;
    return;
  }

  if (file.size === 0) {
    showUploadStatus(`<strong>File is empty.</strong> Uploaded documents must contain text.`, "error");
    currentSelectedFile = null;
    if (modalSubmitBtn) modalSubmitBtn.disabled = true;
    return;
  }

  const maxBytes = 1048576; // 1 MB
  if (file.size > maxBytes) {
    showUploadStatus(`<strong>File exceeds size limit.</strong> Maximum allowed size is 1 MB (selected: ${formatFileSize(file.size)}).`, "error");
    currentSelectedFile = null;
    if (modalSubmitBtn) modalSubmitBtn.disabled = true;
    return;
  }

  currentSelectedFile = file;
  if (fileNameDisplay) fileNameDisplay.textContent = file.name;
  if (fileSizeDisplay) fileSizeDisplay.textContent = formatFileSize(file.size);
  if (fileExtBadge) {
    const isMd = ext === ".md";
    fileExtBadge.textContent = isMd ? "MD" : "TXT";
    fileExtBadge.className = `file-badge ${isMd ? "badge-md" : "badge-txt"}`;
  }
  if (dropzoneEmpty) dropzoneEmpty.style.display = "none";
  if (selectedFileCard) selectedFileCard.style.display = "flex";
  if (modalOptions) modalOptions.style.display = "";
  if (uploadStatus) uploadStatus.style.display = "none";
  if (modalSubmitBtn) modalSubmitBtn.disabled = false;
}

if (sidebarAddDocBtn) sidebarAddDocBtn.addEventListener("click", openModal);
if (modalCloseBtn) modalCloseBtn.addEventListener("click", closeModal);
if (modalCancelBtn) modalCancelBtn.addEventListener("click", closeModal);

if (uploadModal) {
  uploadModal.addEventListener("click", (e) => {
    if (e.target === uploadModal) closeModal();
  });
}

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (deleteModal && deleteModal.classList.contains("active")) {
      closeDeleteModal();
    } else if (uploadModal && uploadModal.classList.contains("active")) {
      closeModal();
    } else if (isMobileView() && sidebar && sidebar.classList.contains("open")) {
      closeSidebar();
    }
  }
});

if (dropzone) {
  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("#remove-file-btn")) return;
    if (fileInput) fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !e.target.closest("#remove-file-btn")) {
      e.preventDefault();
      if (fileInput) fileInput.click();
    }
  });
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });
}

if (fileInput) {
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length) {
      handleFileSelection(e.target.files[0]);
    }
  });
}

if (removeFileBtn) {
  removeFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUploadForm();
  });
}

if (uploadForm) {
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!currentSelectedFile) return;

    const btnText = modalSubmitBtn.querySelector(".btn-text");
    const btnSpinner = modalSubmitBtn.querySelector(".btn-spinner");

    modalSubmitBtn.disabled = true;
    if (btnText) btnText.style.display = "none";
    if (btnSpinner) btnSpinner.style.display = "inline-flex";
    if (removeFileBtn) removeFileBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", currentSelectedFile);
    if (overwriteCheckbox && overwriteCheckbox.checked) {
      formData.append("overwrite", "true");
    }

    try {
      const resp = await fetch("/documents/upload", {
        method: "POST",
        body: formData,
      });
      const data = await resp.json();

      if (!resp.ok) {
        const errMsg = data.detail || data.message || `Upload failed with HTTP ${resp.status}`;
        showUploadStatus(`<strong>Upload failed:</strong> ${escapeHtml(errMsg)}`, "error");
        modalSubmitBtn.disabled = false;
      } else {
        showUploadStatus(
          `<strong>Success!</strong> ${escapeHtml(data.message)} (${data.chunks_total} total chunks indexed).`,
          "success"
        );
        // Refresh knowledge base sources and health counts immediately
        await refreshMeta();
        // Reset selection
        currentSelectedFile = null;
        if (fileInput) fileInput.value = "";
        if (dropzoneEmpty) dropzoneEmpty.style.display = "";
        if (selectedFileCard) selectedFileCard.style.display = "none";
        if (modalOptions) modalOptions.style.display = "none";
      }
    } catch (err) {
      showUploadStatus(`<strong>Network error:</strong> Unable to upload document to server.`, "error");
      modalSubmitBtn.disabled = false;
    } finally {
      if (btnText) btnText.style.display = "";
      if (btnSpinner) btnSpinner.style.display = "none";
      if (removeFileBtn) removeFileBtn.disabled = false;
    }
  });
}

// Document Deletion Modal Handling
let currentDeleteFilename = null;

function openDeleteModal(filename) {
  currentDeleteFilename = filename;
  if (deleteDocName) deleteDocName.textContent = filename;
  if (deleteStatus) {
    deleteStatus.style.display = "none";
    deleteStatus.className = "upload-status";
    deleteStatus.innerHTML = "";
  }
  if (deleteConfirmBtn) {
    deleteConfirmBtn.disabled = false;
    const btnText = deleteConfirmBtn.querySelector(".btn-text");
    const btnSpinner = deleteConfirmBtn.querySelector(".btn-spinner");
    if (btnText) btnText.style.display = "";
    if (btnSpinner) btnSpinner.style.display = "none";
  }
  if (deleteCancelBtn) deleteCancelBtn.disabled = false;
  if (deleteModal) {
    deleteModal.classList.add("active");
    deleteModal.setAttribute("aria-hidden", "false");
  }
}

function closeDeleteModal() {
  if (deleteModal) {
    deleteModal.classList.remove("active");
    deleteModal.setAttribute("aria-hidden", "true");
  }
  currentDeleteFilename = null;
}

if (deleteModalCloseBtn) deleteModalCloseBtn.addEventListener("click", closeDeleteModal);
if (deleteCancelBtn) deleteCancelBtn.addEventListener("click", closeDeleteModal);

if (deleteModal) {
  deleteModal.addEventListener("click", (e) => {
    if (e.target === deleteModal) closeDeleteModal();
  });
}

if (sources) {
  sources.addEventListener("click", (e) => {
    const deleteBtn = e.target.closest(".source-delete-btn");
    if (deleteBtn) {
      e.preventDefault();
      e.stopPropagation();
      const fn = deleteBtn.getAttribute("data-filename");
      if (fn) {
        openDeleteModal(fn);
      }
    }
  });
}

if (deleteConfirmBtn) {
  deleteConfirmBtn.addEventListener("click", async () => {
    if (!currentDeleteFilename) return;
    const btnText = deleteConfirmBtn.querySelector(".btn-text");
    const btnSpinner = deleteConfirmBtn.querySelector(".btn-spinner");

    deleteConfirmBtn.disabled = true;
    if (deleteCancelBtn) deleteCancelBtn.disabled = true;
    if (btnText) btnText.style.display = "none";
    if (btnSpinner) btnSpinner.style.display = "inline-flex";

    try {
      const resp = await fetch(`/documents/${encodeURIComponent(currentDeleteFilename)}`, {
        method: "DELETE",
      });
      const data = await resp.json();

      if (!resp.ok) {
        const errMsg = data.detail || data.message || `Deletion failed with HTTP ${resp.status}`;
        if (deleteStatus) {
          deleteStatus.className = "upload-status status-error";
          deleteStatus.innerHTML = `<strong>Deletion failed:</strong> ${escapeHtml(errMsg)}`;
          deleteStatus.style.display = "block";
        }
        deleteConfirmBtn.disabled = false;
        if (deleteCancelBtn) deleteCancelBtn.disabled = false;
        if (btnText) btnText.style.display = "";
        if (btnSpinner) btnSpinner.style.display = "none";
      } else {
        closeDeleteModal();
        showToast(`Document "${currentDeleteFilename}" deleted successfully.`);
        await refreshMeta();
      }
    } catch (err) {
      if (deleteStatus) {
        deleteStatus.className = "upload-status status-error";
        deleteStatus.innerHTML = `<strong>Network error:</strong> Unable to reach server.`;
        deleteStatus.style.display = "block";
      }
      deleteConfirmBtn.disabled = false;
      if (deleteCancelBtn) deleteCancelBtn.disabled = false;
      if (btnText) btnText.style.display = "";
      if (btnSpinner) btnSpinner.style.display = "none";
    }
  });
}

// Answer Action Handlers: Copy and Read Aloud (Web Speech API)
let activeSpeechBtn = null;
let toastTimeout = null;

function showToast(message) {
  let toast = document.getElementById("toast-notification");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "toast-notification";
    toast.className = "toast-msg";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.style.opacity = "1";
  toast.style.transform = "translateX(-50%) translateY(0)";

  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(-50%) translateY(20px)";
  }, 2400);
}

function stopSpeech() {
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
  if (activeSpeechBtn) {
    setSpeechButtonState(activeSpeechBtn, false);
    activeSpeechBtn = null;
  }
}

function setSpeechButtonState(btn, isSpeaking) {
  if (!btn) return;
  const idleIcon = btn.querySelector(".tts-icon-idle");
  const activeIcon = btn.querySelector(".tts-icon-active");
  const tooltip = btn.querySelector(".action-tooltip");

  if (isSpeaking) {
    btn.classList.add("is-speaking");
    btn.setAttribute("aria-label", "Stop reading");
    btn.setAttribute("title", "Stop reading");
    if (idleIcon) idleIcon.style.display = "none";
    if (activeIcon) activeIcon.style.display = "block";
    if (tooltip) tooltip.textContent = "Stop reading";
  } else {
    btn.classList.remove("is-speaking");
    btn.setAttribute("aria-label", "Read aloud");
    btn.setAttribute("title", "Read aloud");
    if (idleIcon) idleIcon.style.display = "block";
    if (activeIcon) activeIcon.style.display = "none";
    if (tooltip) tooltip.textContent = "Read aloud";
  }
}

function cleanSpokenText(rawText) {
  if (!rawText) return "";
  // Strip bracketed citations like [D1], [T1], [D2], [1], etc.
  let cleaned = rawText.replace(/\[\s*[A-Za-z0-9_-]+\s*\]/g, "");
  // Clean up any extra whitespace
  cleaned = cleaned.replace(/\s+/g, " ").trim();
  return cleaned;
}

function handleTTSClick(btn) {
  if (!("speechSynthesis" in window) || typeof SpeechSynthesisUtterance === "undefined") {
    showToast("Speech synthesis is not supported in this browser.");
    return;
  }

  // If this button is currently speaking, click stops it
  if (activeSpeechBtn === btn) {
    stopSpeech();
    return;
  }

  // Stop any other active speech first
  stopSpeech();

  const card = btn.closest(".card");
  if (!card) return;
  const answerEl = card.querySelector(".answer");
  const rawText = answerEl ? (answerEl.getAttribute("data-raw-answer") || answerEl.textContent) : "";
  const spokenText = cleanSpokenText(rawText);

  if (!spokenText) {
    showToast("No answer text to read.");
    return;
  }

  const utterance = new SpeechSynthesisUtterance(spokenText);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;

  utterance.onstart = () => {
    activeSpeechBtn = btn;
    setSpeechButtonState(btn, true);
  };

  utterance.onend = () => {
    if (activeSpeechBtn === btn) {
      setSpeechButtonState(btn, false);
      activeSpeechBtn = null;
    }
  };

  utterance.onerror = (e) => {
    if (activeSpeechBtn === btn) {
      setSpeechButtonState(btn, false);
      activeSpeechBtn = null;
    }
    // "canceled" or "interrupted" are normal when stopped
    if (e.error && e.error !== "canceled" && e.error !== "interrupted") {
      showToast("Speech playback error.");
    }
  };

  // Chrome speech queue fix: cancel before speaking
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

async function handleCopyClick(btn) {
  const card = btn.closest(".card");
  if (!card) return;
  const answerEl = card.querySelector(".answer");
  const textToCopy = answerEl ? (answerEl.getAttribute("data-raw-answer") || answerEl.textContent) : "";
  if (!textToCopy) return;

  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(textToCopy);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = textToCopy;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
    }

    const iconCopy = btn.querySelector(".icon-copy");
    const iconCheck = btn.querySelector(".icon-check");
    const tooltip = btn.querySelector(".action-tooltip");

    btn.classList.add("is-copied");
    if (iconCopy) iconCopy.style.display = "none";
    if (iconCheck) iconCheck.style.display = "block";
    if (tooltip) tooltip.textContent = "Copied!";

    setTimeout(() => {
      btn.classList.remove("is-copied");
      if (iconCopy) iconCopy.style.display = "block";
      if (iconCheck) iconCheck.style.display = "none";
      if (tooltip) tooltip.textContent = "Copy";
    }, 1800);
  } catch (err) {
    showToast("Failed to copy to clipboard.");
  }
}

// Thread event delegation for answer actions
if (thread) {
  thread.addEventListener("click", (e) => {
    const copyBtn = e.target.closest(".copy-btn");
    if (copyBtn) {
      e.preventDefault();
      handleCopyClick(copyBtn);
      return;
    }

    const ttsBtn = e.target.closest(".tts-btn");
    if (ttsBtn) {
      e.preventDefault();
      handleTTSClick(ttsBtn);
      return;
    }
  });
}

window.addEventListener("beforeunload", () => {
  stopSpeech();
});

showEmpty();
refreshMeta();
