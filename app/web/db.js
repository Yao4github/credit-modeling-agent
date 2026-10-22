import { $, apiFetch, clearError, el, setStatus, showError } from "./common.js";

let lastBody = null; // request of the last successful mining run (reused for sampling)

const split = (text) =>
  String(text)
    .split(/[,，]/)
    .map((s) => s.trim())
    .filter(Boolean);

function fillSelect(select, names, { blank } = {}) {
  select.replaceChildren(
    ...(blank ? [Object.assign(el("option", blank), { value: "" })] : []),
    ...names.map((n) => Object.assign(el("option", n), { value: n })),
  );
}

async function loadColumns() {
  const table = $("table").value;
  if (!table) return;
  const { columns } = await apiFetch(`/api/v1/db/tables/${encodeURIComponent(table)}/columns`);
  const names = columns.map((c) => c.name);
  fillSelect($("label-col"), names);
  fillSelect($("date-col"), names, { blank: "不使用" });
  const guess = names.find((n) => /label/i.test(n));
  if (guess) $("label-col").value = guess;
  const dateGuess = names.find((n) => /created_at|apply.*(date|time)|date$/i.test(n));
  if (dateGuess) $("date-col").value = dateGuess;
}

async function init() {
  const { configured } = await apiFetch("/api/v1/db/status");
  if (!configured) {
    showError("数据库未配置：请在 .env 中设置 DB_HOST、DB_PORT、DB_USER、DB_PASSWORD、DB_NAME（建议使用只读账号）后重启服务。");
    $("mine-btn").disabled = true;
    return;
  }
  // The profile summary is a local file read: show the multi-table module at once instead of queueing it behind the
  // (slower) table list of the remote database.
  const project = initProject().catch((e) => showError(e.message));
  const { tables } = await apiFetch("/api/v1/db/tables");
  fillSelect(
    $("table"),
    tables.map((t) => t.table),
  );
  $("table").addEventListener("change", () => loadColumns().catch((e) => showError(e.message)));
  await Promise.all([loadColumns(), project]);
}

async function initProject() {
  const info = await apiFetch("/api/v1/db/project");
  if (!info.configured) return;
  $("project-info").textContent =
    `${info.name || info.base_table}：样本表 ${info.base_table}，关联 ${info.joins.join("、") || "无"}；` +
    `总体 ${info.population || "全表"}；${info.label_rule}。`;
  $("project-params").replaceChildren(
    ...$("project-params").querySelectorAll("legend"),
    ...Object.entries(info.params).map(([name, value]) => {
      const input = Object.assign(el("input"), { id: `param-${name}`, type: "text", value: String(value), autocomplete: "off" });
      input.dataset.param = name;
      input.dataset.numeric = typeof value === "number" ? "1" : "";
      const wrap = el("div");
      wrap.className = "field";
      const label = el("label", name);
      label.htmlFor = input.id;
      wrap.append(label, input);
      return wrap;
    }),
  );
  $("project-section").hidden = false;
}

function handleBlock(block, ctx) {
  let event = "data";
  let data = "";
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event: ")) event = line.slice(7);
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  if (!data) return;
  const payload = JSON.parse(data);
  if (event === "result") {
    ctx.finished = true;
    $("report").textContent = payload.report_markdown;
    $("report-section").hidden = false;
    $("report").focus();
    setStatus("挖掘完成。");
  } else if (event === "error") {
    ctx.finished = true;
    showError(`挖掘失败：${payload.message}`);
    setStatus("挖掘失败。");
  } else {
    const li = el("li");
    li.append(el("strong", payload.agent), ` ${payload.elapsed_ms} ms — ${payload.action}`);
    $("timeline").append(li);
  }
}

/** Streams a mining run into the timeline / report. `sampleBody` (single-table runs only) enables dataset sampling. */
async function runStream(url, body, button, sampleBody) {
  button.disabled = true;
  $("report-section").hidden = true;
  $("timeline").replaceChildren();
  $("timeline-section").hidden = false;
  setStatus("挖掘进行中…");
  const ctx = { finished: false };
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok || !resp.body) {
      const detail = await resp.text().catch(() => "");
      throw new Error(`请求失败 (HTTP ${resp.status}) ${detail.slice(0, 200)}`);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop();
      blocks.forEach((b) => handleBlock(b, ctx));
    }
    if (buffer.trim()) handleBlock(buffer, ctx);
    if (!ctx.finished) throw new Error("连接中断：未收到最终结果。");
    lastBody = sampleBody || null;
    $("sample-block").hidden = !sampleBody;
    if (sampleBody) $("sample-uri").value ||= `ds://db_${sampleBody.table}`;
  } catch (err) {
    showError(err.message);
    setStatus("挖掘失败。");
  } finally {
    button.disabled = false;
  }
}

$("project-form").addEventListener("submit", (e) => {
  e.preventDefault();
  clearError();
  const params = {};
  for (const input of $("project-params").querySelectorAll("input[data-param]")) {
    const raw = input.value.trim();
    if (!raw) continue;
    params[input.dataset.param] = input.dataset.numeric ? Number(raw) : raw;
  }
  const body = { params };
  if ($("project-oot").value) body.oot_start_date = $("project-oot").value;
  if ($("project-max-rows").value) body.max_rows = Number($("project-max-rows").value);
  runStream("/api/v1/db/project/mine", body, $("project-btn"), null);
});

$("mine-form").addEventListener("submit", (e) => {
  e.preventDefault();
  clearError();
  const fd = new FormData($("mine-form"));
  const body = {
    table: fd.get("table"),
    label_col: fd.get("label_col"),
    bad_values: split(fd.get("bad_values")),
  };
  const good = split(fd.get("good_values"));
  if (good.length) body.good_values = good;
  for (const key of ["date_col", "oot_start_date"]) if (fd.get(key)) body[key] = fd.get(key);
  if (fd.get("max_rows")) body.max_rows = Number(fd.get("max_rows"));
  runStream("/api/v1/db/mine", body, $("mine-btn"), body);
});

$("sample-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!lastBody) return;
  clearError();
  $("sample-btn").disabled = true;
  try {
    const r = await apiFetch("/api/v1/db/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...lastBody, uri: $("sample-uri").value.trim() }),
    });
    setStatus(`已抽样 ${r.row_count} 行并注册 ${r.uri}（文件 ${r.filename}，观察窗口 ${r.obs_window}）。可在建模页选择。`);
  } catch (err) {
    showError(err.message);
    setStatus("抽样失败。");
  } finally {
    $("sample-btn").disabled = false;
  }
});

init().catch((err) => showError(err.message));
