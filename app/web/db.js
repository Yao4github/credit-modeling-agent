import { $, apiFetch, clearError, el, enhanceReport, renderMarkdown, reveal, setAllSections, setStatus, showError } from "./common.js";

let lastBody = null; // request of the last successful mining run (reused for sampling)
let lastMarkdown = "";
let dbConfigured = false;
let projectConfigured = false;

const PARAM_LABELS = {
  is_usernew: "客群口径（首贷 / 复贷）",
  disbursed_status: "放款成功状态阈值",
  settled_status: "已结清状态值",
  bad_overdue_days: "坏样本逾期天数阈值",
  good_max_overdue_days: "好样本最大允许逾期天数",
};

const USERNEW_OPTIONS = [
  { value: "1,2", text: "全部放款订单（一人可多单: is_usernew in [1, 2]）" },
  { value: "1", text: "仅首贷用户（一人一单: is_usernew = 1）" },
  { value: "2", text: "仅复贷订单（全局复贷: is_usernew = 2）" },
];

function switchMode(mode) {
  const isProject = mode === "project";
  const radio = document.querySelector(`input[name="mine_mode"][value="${mode}"]`);
  if (radio && !radio.checked) radio.checked = true;
  $("project-section").hidden = !isProject;
  $("single-section").hidden = isProject;
  if (!CSS.supports("selector(:has(*))")) {
    const card = $("mining-card");
    card.classList.toggle("is-mode-project", isProject);
    card.classList.toggle("is-mode-single", !isProject);
    document.querySelectorAll("#mode-switcher label").forEach((lbl) => {
      const inp = lbl.querySelector('input[type="radio"]');
      lbl.classList.toggle("is-active", Boolean(inp && inp.checked));
    });
  }
}

/** Disables every control of both mining forms while a run is in flight (prevents double submits). */
function lockForms(locked) {
  if (!locked && !dbConfigured) return;
  for (const id of ["mode-switcher", "project-form", "mine-form"]) {
    const node = $(id);
    if (!node) continue;
    node.querySelectorAll("input, select, textarea, button").forEach((c) => (c.disabled = locked));
  }
  if (!locked && !projectConfigured) {
    $("project-btn").disabled = true;
  }
}

function setLoading(button, loading, text) {
  if (loading) {
    button.dataset.label = button.textContent;
    button.textContent = text;
  } else if (button.dataset.label) {
    button.textContent = button.dataset.label;
  }
  button.classList.toggle("is-loading", loading);
}

function pendingItem() {
  const li = el("li", undefined, "pending");
  li.append(el("span", "", "pulse"), el("span", "正在执行下一阶段…", "thought"));
  return li;
}

const split = (text) =>
  String(text)
    .split(/[,，]/)
    .map((s) => s.trim())
    .filter(Boolean);

function fillSelect(select, items, { blank } = {}) {
  select.replaceChildren(
    ...(blank ? [Object.assign(el("option", blank), { value: "" })] : []),
    ...items.map((item) => {
      const value = typeof item === "string" ? item : item.value;
      const text = typeof item === "string" ? item : item.text;
      return Object.assign(el("option", text), { value });
    }),
  );
}

async function loadColumns() {
  const table = $("table").value;
  if (!table) return;
  const { columns } = await apiFetch(`/api/v1/db/tables/${encodeURIComponent(table)}/columns`);
  const names = columns.map((c) => c.name);
  const items = columns.map((c) => ({ value: c.name, text: c.comment ? `${c.name}（${c.comment}）` : c.name }));
  fillSelect($("label-col"), items);
  fillSelect($("date-col"), items, { blank: "不使用" });
  const guess = names.find((n) => /label/i.test(n));
  if (guess) $("label-col").value = guess;
  const dateGuess = names.find((n) => /created_at|apply.*(date|time)|date$/i.test(n));
  if (dateGuess) $("date-col").value = dateGuess;
}

async function init() {
  $("mode-switcher").addEventListener("change", (e) => {
    if (e.target && e.target.name === "mine_mode") {
      clearError();
      switchMode(e.target.value);
    }
  });
  const { configured } = await apiFetch("/api/v1/db/status");
  if (!configured) {
    showError("数据库未配置：请在 .env 中设置 DB_HOST、DB_PORT、DB_USER、DB_PASSWORD、DB_NAME（建议使用只读账号）后重启服务。");
    $("mine-btn").disabled = true;
    $("project-btn").disabled = true;
    return;
  }
  dbConfigured = true;
  // The profile summary is a local file read: show the multi-table module at once instead of queueing it behind the
  // (slower) table list of the remote database.
  const project = initProject().catch((e) => showError(e.message));
  const { tables } = await apiFetch("/api/v1/db/tables");
  fillSelect(
    $("table"),
    tables.map((t) => ({ value: t.table, text: t.comment ? `${t.table}（${t.comment}）` : t.table })),
  );
  $("table").addEventListener("change", () => loadColumns().catch((e) => showError(e.message)));
  await Promise.all([loadColumns(), project]);
}

async function initProject() {
  const info = await apiFetch("/api/v1/db/project");
  if (!info.configured) {
    projectConfigured = false;
    $("project-unconfigured").hidden = false;
    $("project-summary").hidden = true;
    $("project-form").hidden = true;
    $("project-btn").disabled = true;
    switchMode("single");
    return;
  }
  projectConfigured = true;
  $("project-unconfigured").hidden = true;
  $("project-summary").hidden = false;
  $("project-form").hidden = false;
  $("project-info").textContent =
    `${info.name || info.base_table}：样本表 ${info.base_table}，关联 ${info.joins.join("、") || "无"}；` +
    `总体 ${info.population || "全表"}；${info.label_rule}。`;
  if (info.path) $("project-profile-badge").textContent = info.path;
  if (Array.isArray(info.master_tables) && info.master_tables.length) {
    $("proj-tables").textContent = info.master_tables
      .map((m) => `${m.role} ${m.table} (${m.bridge || m.key})`)
      .join(" + ");
  } else {
    $("proj-tables").textContent =
      `主表 ${info.base_table}` + (info.joins.length ? ` · 关联 ${info.joins.join("、")} + 自动 1~2 跳图谱` : " · 自动 1~2 跳图谱");
  }
  $("proj-population").textContent = info.population || "全表样本";
  $("proj-labels").textContent = info.label_rule || "未配置标签（无监督特征库模式）";
  $("project-params").replaceChildren(
    ...$("project-params").querySelectorAll("legend"),
    ...Object.entries(info.params).map(([name, value]) => {
      const wrap = el("div");
      wrap.className = "field m-0";
      const label = el("label", PARAM_LABELS[name] ? `${PARAM_LABELS[name]} (${name})` : name);
      let control;
      if (name === "is_usernew") {
        control = Object.assign(el("select"), { id: `param-${name}` });
        fillSelect(control, USERNEW_OPTIONS);
        control.value = Array.isArray(value) ? value.join(",") : String(value);
        control.dataset.param = name;
        control.dataset.array = "1";
        control.dataset.numeric = "1";
      } else {
        control = Object.assign(el("input"), {
          id: `param-${name}`,
          type: "text",
          value: Array.isArray(value) ? value.join(",") : String(value),
          autocomplete: "off",
        });
        control.dataset.param = name;
        control.dataset.array = Array.isArray(value) ? "1" : "";
        control.dataset.numeric = typeof value === "number" || (Array.isArray(value) && value.every((v) => typeof v === "number")) ? "1" : "";
      }
      label.htmlFor = control.id;
      wrap.append(label, control);
      return wrap;
    }),
  );
  switchMode("project");
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
    lastMarkdown = payload.report_markdown || "";
    $("report").innerHTML = renderMarkdown(lastMarkdown);
    enhanceReport($("report"), $("report-toc"));
    $("report-section").hidden = false;
    $("report").focus({ preventScroll: true });
    reveal($("report-section"));
    setStatus("挖掘完成。");
  } else if (event === "error") {
    ctx.finished = true;
    showError(`挖掘失败：${payload.message}`);
    setStatus("挖掘失败。");
  } else {
    const li = el("li");
    const head = el("div");
    head.append(el("span", `✔ ${payload.agent}`, "agent ok"), el("span", `${payload.elapsed_ms} ms`, "latency"));
    li.append(head, el("p", payload.action, "thought"));
    ctx.pending.before(li);
  }
}

/** Streams a mining run into the timeline / report. `sampleBody` (single-table runs only) enables dataset sampling. */
async function runStream(url, body, button, sampleBody) {
  lockForms(true);
  setLoading(button, true, "挖掘中…");
  $("progress").hidden = false;
  $("report-section").hidden = true;
  $("timeline").replaceChildren();
  $("timeline-section").hidden = false;
  reveal($("timeline-section"));
  setStatus("挖掘进行中…");
  const ctx = { finished: false, pending: pendingItem() };
  $("timeline").append(ctx.pending);
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
    ctx.pending.remove();
    $("progress").hidden = true;
    setLoading(button, false);
    lockForms(false);
  }
}

$("project-form").addEventListener("submit", (e) => {
  e.preventDefault();
  clearError();
  const params = {};
  for (const input of $("project-params").querySelectorAll("[data-param]")) {
    const raw = input.value.trim();
    if (!raw) continue;
    if (input.dataset.array === "1") {
      const items = split(raw);
      params[input.dataset.param] = input.dataset.numeric ? items.map(Number) : items;
    } else {
      params[input.dataset.param] = input.dataset.numeric ? Number(raw) : raw;
    }
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

$("report-expand").addEventListener("click", () => setAllSections($("report"), true));
$("report-collapse").addEventListener("click", () => setAllSections($("report"), false));
$("report-copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(lastMarkdown);
    setStatus("已复制报告 Markdown 到剪贴板。");
  } catch {
    showError("复制失败：浏览器未授权剪贴板，请使用“下载 .md”。");
  }
});
$("report-download").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([lastMarkdown], { type: "text/markdown;charset=utf-8" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: `db-mining-report-${Date.now()}.md` });
  a.click();
  URL.revokeObjectURL(url);
});

init().catch((err) => showError(err.message));
