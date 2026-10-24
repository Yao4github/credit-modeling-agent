# 🏦 Credit-Modeling Multi-Agent System (MAS)

面向**信贷风控数据建模（A卡申请评分、B卡行为评分、C卡催收评分与授信切点仿真）**的生产级多智能体协同系统（Multi-Agent System）。

本项目完整落地了 **意图路由与动态规划（Planning）、数据指针沙箱工具链（Function Calling）、长短期记忆治理（Memory）、基于 LangGraph 的多智能体反思自愈编排（Self-Reflection Workflow）、混合知识检索（Hybrid RAG）、LLM 深度协同（路由 / 反思诊断 / 评审解读）、冠军挑战者模型对比以及全链路量化评估体系（Agent Eval）**。

---

## 🏗️ 一、 核心架构与代码模块映射

| 职责 | 工程实现模块 | 核心技术亮点 |
| :--- | :--- | :--- |
| **1. 核心模块**<br>意图识别、Planning、Tool Calling、长短期记忆 | • 专家节点与意图路由：[`app/agents/specialists.py`](app/agents/specialists.py)<br>• 卡种配置：[`app/core/cards.py`](app/core/cards.py)<br>• 数据指针注册表：[`app/tools/data_registry.py`](app/tools/data_registry.py)<br>• 风控算子工具箱：[`app/tools/credit_metrics.py`](app/tools/credit_metrics.py)<br>• 双层记忆管理：[`app/memory/memory_manager.py`](app/memory/memory_manager.py) | • **Data-Pointer Pattern**：样本通过 `ds://...` URI 指针流转，DataFrame 不进入 LLM Context<br>• **多卡种**：A / B / C 卡共用同一条流水线，由 `CardProfile` 配置差异（见第三节）<br>• **双层记忆**：短期上下文滑动窗口压缩 + 长期情景记忆库（自愈经验落盘 `data/memory_episodic.json`，原子写入，文件损坏回退内置记忆） |
| **2. 多智能体与工作流编排**<br>Multi-Agent 协同、容错与自愈推理 | • 状态机定义：[`app/graph/state.py`](app/graph/state.py)<br>• LangGraph 编排：[`app/graph/workflow.py`](app/graph/workflow.py) | • **6+1 专家协同**：Orchestrator + 数据探查 + 特征工程 + 算法训练 + 跨期验证 + 合规报告<br>• **Self-Reflection 自愈闭环**：验证 Agent 检测到 OOT 跨期 PSI 漂移超标时，条件边打回特征工程剔除漂移变量并重训，回滚次数受 `max_reflections` 约束（`N` 即最多执行 N 次回滚重训）<br>• **自适应分箱策略**：无可剔除特征时，按 Train/OOT KS 差值与分箱诊断切换分箱（过拟合 → 决策树单调分箱，欠拟合 → 10 等频），每种最多试一次，见 [`app/harness/binning_policy.py`](app/harness/binning_policy.py) |
| **3. LLM 深度协同**<br>路由、反思诊断、评审解读 | • LLM 客户端：[`app/core/llm.py`](app/core/llm.py)、[`app/core/config.py`](app/core/config.py) | • **OpenAI 兼容多供应商**（DeepSeek / OpenAI / Qwen / Moonshot / Zhipu / Gemini / 自定义）<br>• **沙箱计算与推理分离**：数值指标始终由确定性算子产出，LLM 只做路由、根因诊断与评审解读<br>• **护栏**：LLM 约束值有范围钳制、卡种有白名单、提议剔除的特征必须在入模池内并与沙箱标记取并集（统一收敛在 [`app/harness/`](app/harness/)）；LLM 评语 / 解答含报告或参考资料中不存在的数值时整段丢弃，并在日志标注“数值护栏”<br>• **离线回退**：未配置密钥或调用失败时自动回退规则逻辑；LLM 调用的真实 Token 用量回填到执行日志 |
| **4. 冠军挑战者**<br>LR 评分卡 vs GBDT | • 挑战者工具：`tool_train_challenger_model`（[`credit_metrics.py`](app/tools/credit_metrics.py)） | • LR 评分卡始终是**冠军**，也是唯一的上线决策模型（可解释性）<br>• 挑战者为带单调约束的 GBDT，仅作区分度基准，输出 OOT KS/AUC 差值与特征重要性<br>• 仅在冠军通过门禁后训练，结果写入报告“冠军挑战者对比”章节 |
| **5. 评估与反馈体系 (Agent Eval)**<br>任务完成率、Tool 准确率、幻觉率、耗时与成本 | • 黄金评测集：[`eval/benchmark_dataset.json`](eval/benchmark_dataset.json)<br>• 量化评测器：[`eval/evaluators.py`](eval/evaluators.py)<br>• 评测流水线：[`eval/run_eval.py`](eval/run_eval.py) | • **L1/L2/L3 分层评测**（8 个用例，覆盖 A/B/C 卡）：工具调用序列与参数匹配、RAG 忠实度、数值零幻觉校验（报告中的 KS/PSI/切点等与沙箱 JSON 逐项比对）、P50/P95 耗时与 Token 成本；线上报告同样遵守“数值必须出自沙箱”的口径 |
| **6. 混合 RAG 与服务化**<br>Hybrid RAG、FastAPI、SSE 流式推送 | • 混合检索器：[`app/tools/rag_retriever.py`](app/tools/rag_retriever.py)<br>• 微服务网关：[`app/api/server.py`](app/api/server.py) | • **BM25 + TF-IDF 向量 RRF 融合检索**：内置数仓血缘字典与公平信贷/PIPL 合规红线，可通过 `data/knowledge_extra.json` 扩展制度文档<br>• 自动拦截贷后/未来信息穿越字段与性别歧视字段，黑名单聚合整个知识库的元数据 |

---

## 🧩 二、 外部数据挂载

默认使用内置合成数据。接入自有数据时，将 `.csv` / `.parquet` 文件放入 `data/` 目录（**只允许读取该目录内文件**，路径越界会被拒绝），再注册为数据指针：

```bash
curl -X POST http://localhost:8000/api/v1/datasets/register -H 'Content-Type: application/json' -d '{
  "uri": "ds://my_train", "path": "my_train.csv", "target_col": "label",
  "obs_start_date": "2025-01-01", "obs_end_date": "2025-06-30"
}'
```

- 标签列必须只含 0/1，且 **1 代表风险事件**；同理注册 OOT 样本。
- **观察期可选**：`obs_start_date` / `obs_end_date` 可省略，`perf_window_months` 默认 3。可传 `date_col`（日期列）由其最早 / 最晚日期自动推断观察期（显式传入的日期优先；该列不参与建模，含无法解析的日期会拒绝注册）。观察期与日期列都没有时，**跳过“Train/OOT 时间窗口无重叠”校验**，报告会标注“未提供观察期，跨期指标可信度下降”，不会伪造通过结果。
- 建模请求通过 `train_uri` / `oot_uri` 指定数据集；不传则使用所选卡种的默认合成样本。
- **没有跨期集时**：只传 `train_uri`、不传 `oot_uri`，系统按**行顺序**把它切成前 70%（开发集 `<uri>#dev`）与后 30%（跨期集 `<uri>#oot`），切分点两侧都要有足够样本且含正负两类，否则拒绝。**这不是严格的跨期验证**（行序不一定是时间序），报告会醒目标注，OOT KS / PSI / 差值门禁仅供参考；有日期列时请先按时间排序再上传。只传 `oot_uri` 会被拒绝。工作台建模页的“开发集 / 跨期集”下拉框即对应这两个参数。
- 穿越/合规字段**自动识别并隔离**（原因写入报告“拦截原因明细”），分三层：知识库黑名单、命名规则（按 `_` 切词后整词匹配，如 `post` / `fut` / `next` / `gender` / `sex`，`postal_code` 不会误伤）、统计红旗（Train 上单特征 AUC ≥ 0.95 或与标签相关系数 ≥ 0.90 的近乎完美代理）。
- **识别局限**：AUC 低于 0.95 的隐蔽穿越、只能靠事件时序判断的穿越识别不出，需写入 `data/knowledge_extra.json` 的 `leakage_blacklist`。
- **人工豁免**：命名规则或统计红旗的**误杀**可通过 `POST /api/v1/datasets/exemptions`（`uri`、`feature`、`reason`、`operator`，原因必填）按数据集逐字段豁免。知识库黑名单和合规敏感字段**永远不可豁免**；豁免字段仍要经过 IV / PSI / VIF / 跨期门禁，报告会醒目列出（统计红旗字段另有“指标可能虚高”提示）；每次运行都会重新校验，豁免之后才被加入黑名单的字段会自动失效。授予与撤销都写入 `data/leakage_exemptions.json` 的追加式审计历史。`operator` 是调用方自报的标识，**系统不校验身份**。

---

## 🗄️ 二·附、 只读数据库指标挖掘

在 `.env` 配置数据库连接参数 `DB_HOST`、`DB_PORT`、`DB_USER`、`DB_PASSWORD`、`DB_NAME`（见 [`.env.example`](.env.example)，**建议用只授 SELECT 的账号**）后，工作台 `db.html` 提供数据库表的指标挖掘：`GET /api/v1/db/tables` 列表 → `POST /api/v1/db/mine`（SSE）→ `POST /api/v1/db/sample` 抽样为建模数据集。未配置数据库参数时该功能关闭（接口返回 503），其余功能不受影响。

- **只读由代码保证**（账号即使有写权限也不会写）：全部语句由 SQLAlchemy `select()` 构造，调用方与 LLM 都不能写 SQL；表 / 列名必须存在于库元数据（并在 `DB_ALLOWED_TABLES` 内，若设置），取值走绑定变量；连接强制 `SET SESSION TRANSACTION READ ONLY`（SQLite 为 `query_only`），MySQL 另设 `MAX_EXECUTION_TIME`；每次取样有行数上限 `DB_MAX_ROWS`（默认 20 万，取前 N 行）。
- **分工**：行数、各列空值率、基数在**库内聚合**；IV / KS / PSI 在**本地**对取样数据用建模同一套算子计算。
- **类型推断**：很多业务表把数值存成文本（`'100%'`、空串表示缺失）。去掉空串与 `%` 后 ≥95% 可解析为数字的文本列会自动转数值（`numeric_text`），其余文本 / 时间列不参与指标。
- **标签映射**：`label_col` + `bad_values`（风险取值）；标签列含风险取值之外的值时必须显式给 `good_values`，两者之外的行丢弃并计数，映射后只有单一类别则拒绝。
- **切分**：给 `date_col` 时按时间切（`oot_start_date` 之前为开发集，缺省排序后 70% / 30%）；不给则按行序切并在报告标注“非严格跨期”。
- **候选特征建议**：IV ≥ 0.02、PSI ≤ 0.1、缺失率 ≤ 0.95，且与更高 IV 的入选特征相关系数绝对值 ≤ 0.9；合规敏感 / 疑似穿越（复用建模的识别规则）、疑似标识符（命名含 id / number / no / code 且近乎唯一）、文本列均不入选并写明原因。LLM 可用时报告末尾附一段解读（只解读不改数值）。
- **抽样落盘**：清洗后的样本（标签为 0 / 1，按日期排序）写入 `data/db_<表名>.csv`，不覆盖已有文件，并注册为**持久化**数据集，随后可在建模页选择。

### 多表自动探索（项目配置驱动）

真实业务库里标签通常不在特征表里，而是由状态 / 逾期天数等字段按规则推出。所有项目差异都放在 JSON **项目配置**里（默认 [`config/db_profile.json`](config/db_profile.json)，可用环境变量 `DB_PROFILE_PATH` 指向别的文件），换项目只改配置、不改代码。工作台 `db.html` 顶部的“项目自动探索”或 `POST /api/v1/db/project/mine`（SSE）按配置执行：加载配置 → 发现候选明细表 → 取样并打标签 → 指标挖掘与特征分类 → 报告。`GET /api/v1/db/project` 返回配置摘要。

| 配置段 | 作用 |
| :--- | :--- |
| `sample` | 样本表、主键、用户键、用于跨期切分的日期列 |
| `joins` | 左连接的特征表（`table` 或通配 `table_pattern`，`left_key` → `right_key`），特征列以 `<表名>.<列名>` 命名 |
| `population` | 建模总体（如 `status >= 62` 即放款成功） |
| `label.bad` / `label.good` | 坏 / 好样本的条件树；两者之外是不确定样本，被丢弃；两者都命中时坏样本优先 |
| `params` | 命名阈值，条件里用 `"$名称"` 引用（如 `$bad_overdue_days`），页面 / 接口可临时覆盖 |
| `exclude` | 禁止作为特征的列（贷后字段、个人敏感信息…），逐条写明原因；**这些列不会从库里读出**，除非标签规则要用 |
| `performance_checks` | 方向校验：如 `repay_count` 越大越好，则好样本均值应更高，否则报告警告标签规则可能有误 |
| `categories` | 特征分类：按列名通配 / 字段注释关键词的有序规则；未命中的可交给 LLM（只能从已配置类别里选），再退回按来源表的兜底类别 |
| `discovery` | 关联键列名、跳过的表、贷后表通配；贷后表只提示、排在最后，不当作特征来源。候选表**自动分类**：贷后表固定归入 `post_event_category`，其余按 `table_categories`（表名通配 `table_patterns` / 表注释关键词 `comment_keywords`，先匹配先得）归类，仍未命中的在 LLM 可用且 `use_llm` 时由 LLM 在给定类别内选择（非法类别忽略），最后归入“其他业务表”；报告第 7 节按类别汇总并分组列出 |

条件树支持 `all` / `any` 嵌套，叶子为 `{"col", "op", "value"}`，操作符限于 `== != > >= < <= in not_in is_null not_null`（总体条件由 SQLAlchemy 构造并绑定参数，标签条件在本地计算，调用方与 LLM 都不能写 SQL）。

**当前范围**：已支持“样本表 + 一对一特征表”的拼装；候选明细表（如还款计划、设备风控）目前只发现并列出，把它们按用户 / 订单聚合成特征（计数、均值、近 N 天等）尚未实现。

---

## 🃏 三、 卡种支持（A / B / C）

| 卡种 | 适用对象 | 标签（1 = 风险事件） | 默认 KS 门禁 | 默认数据集 |
| :--- | :--- | :--- | :---: | :--- |
| **A 卡**（默认） | 首贷申请客户 | `target_mob3_dpd30`：MOB3 内逾期 30 天以上 | 0.30 | `ds://credit_a_card_*` |
| **B 卡** | 存量客户行为评分 | `target_fut3m_dpd30`：观察点后 3 个月内逾期 30 天以上 | 0.30 | `ds://credit_b_card_*` |
| **C 卡** | 已逾期账户催收评分 | `target_unrecovered_d30`：催收后 30 天内未回款 | 0.25 | `ds://credit_c_card_*` |

- **标签方向统一**：1 为风险事件，WOE / 逻辑回归 / PDO 刻度 / KS / 切点仿真全部复用，分数越高风险越低（C 卡即回款可能性越大）。C 卡报告中的“通过率”改称“策略覆盖率”，只改措辞，不改算法。
- **卡种指定方式**：问题中含“B卡 / 行为评分”或“C卡 / 催收”时自动识别；请求参数 `card_type`（`A` / `B` / `C`）显式指定时优先；默认 A 卡。
- B / C 卡的合成数据同样内置了**跨期漂移、共线、未来信息穿越、性别合规**四类陷阱，用于演示自愈回滚。数值指标偏乐观，仅适合演示流程。

---

## 🚀 四、 快速运行指南

```bash
# 0. 安装依赖（需要 Python >= 3.10）
python3 -m venv .venv && .venv/bin/pip install -e .

# 1. 运行端到端多智能体建模与自愈演示脚本
.venv/bin/python scripts/demo_pipeline.py

# 2. 运行 Agent Eval 黄金基准评测套件 (输出 L1/L2/L3 量化指标大盘)
.venv/bin/python eval/run_eval.py

# 3. 运行全量单元测试与集成测试
.venv/bin/pytest -v

# 4. 启动 FastAPI 微服务，浏览器打开 http://localhost:8000 即为建模工作台
.venv/bin/uvicorn app.api.server:app --reload --port 8000
```

验证门禁包含：OOT KS 下限、Train/OOT KS 差值上限（默认 0.05）、模型总分 PSI、入模特征 PSI、系数符号；失败且存在可剔除的漂移/负系数特征时回滚重训，**没有可剔除特征时先尝试切换分箱（每种最多一次），仍无可行策略才终止回滚并在报告中标注 FAILED**。

### LLM 配置（可选）

复制 [`.env.example`](.env.example) 为 `.env` 并填写密钥；未配置密钥时系统以规则模式运行，全部功能可用。`LLM_ENABLED=0` 可强制关闭 LLM。

### 数据目录

- `data/memory_episodic.json`：运行中沉淀的自愈经验（自动生成，反复运行评测会持续增长）。
- `data/knowledge_extra.json`：可选，扩展 RAG 知识库，每条需含 `doc_id`、`category`、`title`、`content`，与内置文档重复的 `doc_id` 会被忽略。
- `data/dataset_registry.json`：已注册数据集的注册参数（只存 URI、文件路径、标签列、观察期等，不存数据）。服务启动时按它**重新挂载**；文件缺失或校验失败的记录会被跳过（记日志）但保留，文件补回后重启即可恢复；注销 / 删除文件时同步移除；文件损坏时忽略并改名为 `.corrupt` 保留。按行切分产生的 `#dev` / `#oot` 不持久化，建模时重新生成。
- `data/leakage_exemptions.json`：穿越字段人工豁免与审计历史（文件损坏时按“无豁免”处理，并将损坏文件改名为 `.corrupt` 保留）。
- `data/sessions/`：最近 50 次建模结果（含完整报告），超出自动淘汰最旧的。
- 工作台上传的数据文件也保存在 `data/`。**当前没有鉴权**，请只在本机（uvicorn 默认监听 127.0.0.1）使用，对外暴露前需先补上鉴权。
- 环境变量 `CREDIT_DATA_DIR` 可覆盖数据目录（测试使用）。`data/` 下的内容默认不入库。

---

## 🌐 五、 API 概览

| 接口 | 说明 |
| :--- | :--- |
| `GET /` | 建模工作台（`app/web/`，原生 JS，无需构建）：`index.html` 建模（实时 Agent 时间线、自愈回滚、评分分箱图、冠军挑战者对比与报告）、`data.html` 数据上传 / 注册 / 穿越字段探查与豁免、`history.html` 历史会话回放 |
| `GET /health` | 健康检查 |
| `GET /api/v1/datasets` | 列出已注册的数据指针 |
| `DELETE /api/v1/datasets?uri=` | 注销文件挂载的数据指针（文件保留；内置合成数据集不可注销） |
| `PUT /api/v1/datasets/files/{filename}` | 上传原始 CSV / Parquet 到 `data/`（仅限 `.csv` / `.parquet`、最大 50 MB、不覆盖同名文件） |
| `GET /api/v1/datasets/files`、`DELETE /api/v1/datasets/files/{filename}` | 列出 `data/` 内数据文件及其关联指针 / 删除文件（**级联注销**由它挂载的指针，豁免审计历史保留；前端二次确认） |
| `POST /api/v1/datasets/register` | 挂载 `data/` 内的 CSV / Parquet 为数据指针 |
| `POST /api/v1/datasets/inspect` | 数据探查：时间窗口、穿越 / 合规隔离字段及原因、可豁免项与当前豁免 |
| `POST` / `GET` / `DELETE /api/v1/datasets/exemptions` | 授予 / 查看（含审计历史） / 撤销穿越字段人工豁免 |
| `GET /api/v1/sessions`、`GET` / `DELETE /api/v1/sessions/{session_id}` | 历史建模会话列表 / 完整结果 / 删除（保留最近 50 条） |
| `POST /api/v1/modeling/run` | 执行工作流，返回指标、反思轨迹、切点策略、挑战者对比与报告 |
| `POST /api/v1/modeling/stream` | SSE 流式推送各 Agent 的执行状态，结束时追加 `result`（或 `error`）事件 |
| `GET /api/v1/db/status`、`GET /api/v1/db/tables`、`GET /api/v1/db/tables/{table}/columns` | 数据库是否已配置 / 可挖掘的表 / 表结构（见“只读数据库指标挖掘”） |
| `POST /api/v1/db/mine` | SSE：表结构 → 库内画像 → 指标挖掘 → 报告，最后推送 `result`（或 `error`） |
| `POST /api/v1/db/sample` | 把表的清洗样本落盘为 CSV 并注册为数据集 |
| `POST /api/v1/eval/benchmark` | 触发黄金评测集并返回量化指标 |

建模请求（`/modeling/run`、`/modeling/stream`）可选字段：`query`、`session_id`、`max_reflections`、`train_uri`、`oot_uri`、`card_type`。

---

## 🗺️ 六、 后续规划

- 接口鉴权与身份校验（当前上传、豁免等接口无鉴权，`operator` 为自报标识）。
