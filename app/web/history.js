import { $, apiFetch, clearError, el, showError } from "./common.js";

const flag = (value) => (value === null || value === undefined ? "—" : value ? "✔ 通过" : "✖ 未通过");

async function load() {
  const { sessions } = await apiFetch("/api/v1/sessions");
  $("empty-note").hidden = sessions.length > 0;
  $("sessions-table").hidden = sessions.length === 0;
  $("sessions-table").tBodies[0].replaceChildren(
    ...sessions.map((s) => {
      const tr = document.createElement("tr");
      const linkCell = document.createElement("td");
      const link = el("a", s.session_id);
      link.href = `index.html?session=${encodeURIComponent(s.session_id)}`;
      linkCell.append(link);
      tr.append(
        el("td", s.saved_at),
        linkCell,
        el("td", s.query),
        el("td", s.card_type || "—"),
        el("td", flag(s.validation_passed)),
        el("td", flag(s.compliance_passed)),
      );
      const delCell = document.createElement("td");
      const del = el("button", "删除", "secondary small");
      del.type = "button";
      del.setAttribute("aria-label", `删除会话 ${s.session_id}`);
      del.addEventListener("click", () => removeSession(s.session_id));
      delCell.append(del);
      tr.append(delCell);
      return tr;
    }),
  );
  $("status").textContent = `共 ${sessions.length} 条会话。`;
}

async function removeSession(id) {
  if (!confirm(`确认删除会话 ${id}？此操作不可恢复。`)) return;
  clearError();
  try {
    await apiFetch(`/api/v1/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
    await load();
    $("status").textContent = `已删除会话 ${id}。`;
  } catch (err) {
    showError(err.message);
  }
}

load().catch((err) => showError(err.message));
