import { renderMarkdown } from "./markdown.js";

const THREAD_KEY = "hf_dataset_assistant_thread_id";

const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const topKInput = document.getElementById("top-k");
const hitsList = document.getElementById("hits-list");
const statusEl = document.getElementById("status");
const searchForm = document.getElementById("search-form");
const searchQuery = document.getElementById("search-query");
const searchResults = document.getElementById("search-results");

function getThreadId() {
  let id = sessionStorage.getItem(THREAD_KEY);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(THREAD_KEY, id);
  }
  return id;
}

function appendMessage(role, text, options = {}) {
  const div = document.createElement("div");
  div.className = `msg ${role}${options.loading ? " loading" : ""}`;
  const body = document.createElement("div");
  body.className = "msg-body";

  if (role === "assistant" && !options.loading && !options.plain) {
    body.classList.add("md");
    body.innerHTML = renderMarkdown(text);
  } else {
    body.textContent = text;
  }

  div.appendChild(body);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function formatDownloads(n) {
  if (n == null || Number.isNaN(Number(n))) return null;
  const x = Number(n);
  if (x >= 1_000_000) return `${(x / 1_000_000).toFixed(1)}M`;
  if (x >= 1_000) return `${(x / 1_000).toFixed(1)}k`;
  return String(x);
}

function renderHits(hits, targetEl) {
  targetEl.innerHTML = "";
  if (!hits || hits.length === 0) {
    targetEl.innerHTML = '<p class="hits-empty">No datasets retrieved for this turn.</p>';
    return;
  }
  for (const h of hits) {
    const card = document.createElement("article");
    card.className = "hit-card";
    const head = document.createElement("div");
    head.className = "hit-head";
    const rank = document.createElement("span");
    rank.className = "hit-rank";
    rank.textContent = h.rank != null ? `#${h.rank}` : "";
    const a = document.createElement("a");
    a.className = "hit-repo";
    a.href = h.url || `https://huggingface.co/datasets/${h.repo_id}`;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = h.repo_id || "dataset";
    head.appendChild(rank);
    head.appendChild(a);
    card.appendChild(head);

    const meta = document.createElement("div");
    meta.className = "hit-meta";
    const bits = [];
    if (h.similarity != null) bits.push(`match ${Number(h.similarity).toFixed(2)}`);
    const dl = formatDownloads(h.downloads);
    if (dl) bits.push(`${dl} downloads`);
    if (h.license) bits.push(h.license);
    meta.textContent = bits.join(" · ");
    card.appendChild(meta);

    if (h.description_snippet) {
      const p = document.createElement("p");
      p.className = "hit-snippet";
      const snip = h.description_snippet;
      p.textContent = snip.length > 140 ? snip.slice(0, 140) + "…" : snip;
      card.appendChild(p);
    }
    targetEl.appendChild(card);
  }
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.chat_available) {
      statusEl.textContent = data.chroma_message + " · Chat ready.";
      statusEl.className = "status ok";
    } else if (data.chroma_ok && !data.openai_configured) {
      statusEl.textContent =
        data.chroma_message + " · Chat disabled (set OPENAI_API_KEY). Search works.";
      statusEl.className = "status warn";
    } else {
      statusEl.textContent = data.chroma_message;
      statusEl.className = "status warn";
    }
  } catch {
    statusEl.textContent = "Cannot reach API. Start: python scripts/run_web.py";
    statusEl.className = "status warn";
  }
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;

  appendMessage("user", text);
  messageInput.value = "";
  sendBtn.disabled = true;
  const pending = appendMessage("assistant", "Searching the catalog and drafting recommendations…", {
    loading: true,
    plain: true,
  });

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        thread_id: getThreadId(),
        top_k: Number(topKInput.value) || 8,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || res.statusText);
    }
    if (data.thread_id) {
      sessionStorage.setItem(THREAD_KEY, data.thread_id);
    }
    pending.remove();
    appendMessage("assistant", data.reply || "(empty reply)");
    renderHits(data.hits, hitsList);
  } catch (err) {
    pending.remove();
    appendMessage("assistant", `Error: ${err.message}`, { plain: true });
  } finally {
    sendBtn.disabled = false;
    messageInput.focus();
  }
});

searchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = searchQuery.value.trim();
  if (!q) return;
  searchResults.innerHTML = '<li class="search-pending">Searching…</li>';
  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, k: Number(topKInput.value) || 8 }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    searchResults.innerHTML = "";
    for (const h of data.results || []) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = h.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = h.repo_id;
      li.appendChild(a);
      if (h.similarity != null) {
        const s = document.createElement("span");
        s.className = "sim";
        s.textContent = ` · ${Number(h.similarity).toFixed(3)}`;
        li.appendChild(s);
      }
      searchResults.appendChild(li);
    }
  } catch (err) {
    searchResults.innerHTML = `<li>Error: ${err.message}</li>`;
  }
});

loadHealth();
