/** Safe subset of Markdown for LLM replies (no raw HTML in source). */

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inlineFormat(escaped) {
  let s = escaped;
  s = s.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );
  s = s.replace(
    /(^|[\s(])((https?:\/\/)[^\s<)\]]+)/g,
    '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>'
  );
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  return s;
}

/**
 * @param {string} raw
 * @returns {string} HTML string
 */
export function renderMarkdown(raw) {
  const lines = String(raw).replace(/\r\n/g, "\n").split("\n");
  const parts = [];
  let i = 0;

  const flushParagraph = (buf) => {
    const t = buf.join(" ").trim();
    if (t) parts.push(`<p>${inlineFormat(escapeHtml(t))}</p>`);
    buf.length = 0;
  };

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i++;
      continue;
    }

    const h3 = trimmed.match(/^###\s+(.+)$/);
    if (h3) {
      parts.push(`<h3 class="md-h3">${inlineFormat(escapeHtml(h3[1]))}</h3>`);
      i++;
      continue;
    }

    const h2 = trimmed.match(/^##\s+(.+)$/);
    if (h2) {
      parts.push(`<h2 class="md-h2">${inlineFormat(escapeHtml(h2[1]))}</h2>`);
      i++;
      continue;
    }

    if (/^\d+\.\s/.test(trimmed)) {
      const items = [];
      while (i < lines.length) {
        const tline = lines[i].trim();
        if (!tline || !/^\d+\.\s/.test(tline)) break;
        const itemText = tline.replace(/^\d+\.\s+/, "");
        const subLines = [];
        i++;
        while (i < lines.length) {
          const n = lines[i];
          const nt = n.trim();
          if (!nt) {
            i++;
            break;
          }
          if (/^\d+\.\s/.test(nt) || /^#{1,3}\s/.test(nt)) break;
          if (/^[-*•]\s/.test(nt) || /^\*\*[^*]+:\*\*/.test(nt)) {
            subLines.push(nt);
            i++;
            continue;
          }
          if (/^\s{2,}\S/.test(n) || n.startsWith("\t")) {
            subLines.push(nt.replace(/^[-*•]\s*/, ""));
            i++;
            continue;
          }
          break;
        }
        items.push(buildListItem(itemText, subLines));
      }
      parts.push(`<ol class="md-ol dataset-cards">${items.join("")}</ol>`);
      continue;
    }

    if (/^[-*•]\s/.test(trimmed)) {
      const items = [];
      while (i < lines.length) {
        const nt = lines[i].trim();
        if (!nt || !/^[-*•]\s/.test(nt)) break;
        items.push(
          `<li class="md-ul-li">${inlineFormat(escapeHtml(nt.replace(/^[-*•]\s+/, "")))}</li>`
        );
        i++;
      }
      parts.push(`<ul class="md-ul">${items.join("")}</ul>`);
      continue;
    }

    const para = [];
    while (i < lines.length) {
      const nt = lines[i].trim();
      if (!nt) {
        i++;
        break;
      }
      if (/^\d+\.\s/.test(nt) || /^[-*•]\s/.test(nt) || /^#{1,3}\s/.test(nt)) break;
      para.push(nt);
      i++;
    }
    flushParagraph(para);
  }

  return parts.join("") || `<p>${inlineFormat(escapeHtml(raw))}</p>`;
}

function buildListItem(titleLine, subLines) {
  const titleHtml = inlineFormat(escapeHtml(titleLine));
  let subs = "";
  if (subLines.length) {
    const rows = subLines.map((sl) => {
      const cleaned = sl.replace(/^[-*•]\s+/, "");
      return `<li>${inlineFormat(escapeHtml(cleaned))}</li>`;
    });
    subs = `<ul class="md-meta">${rows.join("")}</ul>`;
  }
  return `<li class="md-ol-li dataset-card"><div class="dataset-title">${titleHtml}</div>${subs}</li>`;
}
