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
