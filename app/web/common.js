export const $ = (id) => document.getElementById(id);

export function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

export function setStatus(text) {
  $("status").textContent = text;
}

export function showError(message) {
  const box = $("error");
  box.textContent = message;
  box.hidden = false;
}

export function clearError() {
  $("error").hidden = true;
}

export function fillRows(tbody, rows) {
  tbody.replaceChildren(
    ...rows.map((cells) => {
      const tr = document.createElement("tr");
      cells.forEach((c) => tr.append(el("td", String(c))));
      return tr;
    }),
  );
}

export function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function inlineMd(escaped) {
  return escaped.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

/** Minimal safe Markdown renderer (headings, tables, lists, blockquotes, paragraphs). Escapes HTML before adding tags. */
export function renderMarkdown(md) {
  const lines = md.split(/\r?\n/);
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = escapeHtml(lines[i]);
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      out.push(`<h${heading[1].length}>${inlineMd(heading[2])}</h${heading[1].length}>`);
      i += 1;
    } else if (line.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) {
        rows.push(
          escapeHtml(lines[i])
            .replace(/^\||\|$/g, "")
            .split("|")
            .map((c) => c.trim()),
        );
        i += 1;
      }
      const body = rows.filter((r) => !r.every((c) => /^:?-+:?$/.test(c)));
      const [head, ...rest] = body;
      out.push(
        `<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th scope="col">${inlineMd(c)}</th>`).join("")}</tr></thead><tbody>${rest
          .map((r) => `<tr>${r.map((c) => `<td>${inlineMd(c)}</td>`).join("")}</tr>`)
          .join("")}</tbody></table></div>`,
      );
    } else if (/^\s*-\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*-\s+/.test(lines[i])) {
        items.push(`<li>${inlineMd(escapeHtml(lines[i]).replace(/^\s*-\s+/, ""))}</li>`);
        i += 1;
      }
      out.push(`<ul>${items.join("")}</ul>`);
    } else if (line.startsWith("&gt;")) {
      out.push(`<blockquote>${inlineMd(line.replace(/^&gt;\s*/, ""))}</blockquote>`);
      i += 1;
    } else if (line.trim() === "") {
      i += 1;
    } else {
      out.push(`<p>${inlineMd(line)}</p>`);
      i += 1;
    }
  }
  return out.join("\n");
}

// ---- Report enhancement: collapsible sections + TOC, sortable / filterable / colour-coded tables ----------
const num = (text) => (/^-?\d+(\.\d+)?%?$/.test(text.trim()) ? parseFloat(text) : NaN);

const CELL_RULES = {
  IV: (v) => (v >= 0.1 ? "cell-ok" : v < 0.02 ? "cell-muted" : ""),
  KS: (v) => (v >= 0.3 ? "cell-ok" : ""),
  PSI: (v) => (v > 0.25 ? "cell-bad" : v > 0.1 ? "cell-warn" : ""),
  缺失率: (v) => (v >= 0.5 ? "cell-bad" : v >= 0.2 ? "cell-warn" : ""),
  覆盖率: (v) => (v < 0.05 ? "cell-bad" : ""),
};
const COLLAPSED_BY_DEFAULT = /未入选|候选明细表/;

function enhanceTable(table) {
  const heads = [...table.tHead.rows[0].cells];
  const body = table.tBodies[0];
  [...body.rows].forEach((tr) =>
    [...tr.cells].forEach((td, i) => {
      const v = num(td.textContent);
      if (Number.isNaN(v)) return;
      td.classList.add("num");
      const rule = CELL_RULES[heads[i]?.textContent.trim()];
      const cls = rule && rule(v);
      if (cls) td.classList.add(cls);
    }),
  );

  heads.forEach((th, col) => {
    th.tabIndex = 0;
    th.setAttribute("role", "button");
    th.classList.add("sortable");
    const sort = () => {
      const dir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
      heads.forEach((h) => h.removeAttribute("aria-sort"));
      th.setAttribute("aria-sort", dir);
      const sign = dir === "ascending" ? 1 : -1;
      const rows = [...body.rows].sort((a, b) => {
        const x = a.cells[col]?.textContent ?? "";
        const y = b.cells[col]?.textContent ?? "";
        const nx = num(x);
        const ny = num(y);
        const cmp = !Number.isNaN(nx) && !Number.isNaN(ny) ? nx - ny : x.localeCompare(y, "zh");
        return sign * cmp;
      });
      body.append(...rows);
    };
    th.addEventListener("click", sort);
    th.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        sort();
      }
    });
  });

  if (body.rows.length > 8) {
    const input = Object.assign(document.createElement("input"), {
      type: "search",
      placeholder: `筛选 ${body.rows.length} 行（特征名 / 注释）…`,
      className: "table-filter",
    });
    input.setAttribute("aria-label", "筛选表格行");
    input.addEventListener("input", () => {
      const q = input.value.trim().toLowerCase();
      [...body.rows].forEach((tr) => {
        tr.hidden = q !== "" && !tr.textContent.toLowerCase().includes(q);
      });
    });
    table.closest(".table-wrap").before(input);
  }
}

/** Turns a rendered report into collapsible `<details>` sections, fills `toc`, and upgrades every table. */
export function enhanceReport(article, toc) {
  const frag = document.createDocumentFragment();
  const sections = [];
  let current = null;
  for (const node of [...article.childNodes]) {
    if (node.nodeName === "H2") {
      const details = document.createElement("details");
      details.open = !COLLAPSED_BY_DEFAULT.test(node.textContent);
      const summary = el("summary", node.textContent);
      summary.id = `sec-${sections.length + 1}`;
      details.append(summary);
      frag.append(details);
      sections.push({ id: summary.id, title: node.textContent, details });
      current = details;
    } else if (current) {
      current.append(node);
    } else {
      frag.append(node);
    }
  }
  article.replaceChildren(frag);
  article.querySelectorAll("table").forEach(enhanceTable);

  toc.replaceChildren(
    ...sections.map((s) => {
      const a = el("a", s.title);
      a.href = `#${s.id}`;
      a.addEventListener("click", () => {
        s.details.open = true;
      });
      return a;
    }),
  );
  toc.hidden = sections.length === 0;
}

export function setAllSections(article, open) {
  article.querySelectorAll("details").forEach((d) => (d.open = open));
}

export function reveal(node) {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  node.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
}

/** fetch + JSON with readable errors (FastAPI `detail` strings / validation arrays). */
export async function apiFetch(url, options) {
  const resp = await fetch(url, options);
  const text = await resp.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  if (!resp.ok) {
    let detail = body && body.detail !== undefined ? body.detail : text;
    if (Array.isArray(detail)) detail = detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
    throw new Error(`${detail || "请求失败"} (HTTP ${resp.status})`);
  }
  return body;
}
