# Agent 层迭代交付总览（迭代二 ~ 五）

日期：2026-09-09 ｜ 结论：**全部 PASS**

---

## 迭代二：接入真实检索与 LLM

团队：software-legalai-agent（快速模式：工程师 → QA 两轮回归）｜ 结论：**PASS**

## TL;DR

在保持 LLMClient/Tool 协议、单向流水线、权限分级、输出类型白名单全部不变的前提下，
Agent 层完成两项能力升级：真实 OpenAI 兼容 LLM 适配器（失败自动降级 Mock 并留痕）+
纯 stdlib BM25 法条检索（外部语料库可插拔）。QA 两轮独立回归全部通过。

## 新建文件

| 文件 | 说明 |
|------|------|
| `agent_layer/llm/openai_compat.py` | OpenAI 兼容适配器：纯 urllib 调 /chat/completions；`LEGAL_AI_LLM_BASE_URL/API_KEY/MODEL/TIMEOUT` 环境变量配置；response_format=json_object 优先、HTTP 400 自动降级重试；一切失败抛 LLMError 绝不伪造内容 |
| `agent_layer/llm/fallback.py` | FallbackLLMClient：主模型失败降级 Mock 并留痕；`create_llm_from_env()` 工厂（无 Key 自动 Mock 模式） |
| `agent_layer/tools/retrieval.py` | 纯 stdlib BM25（k1=1.5, b=0.75；中文 bigram+单字混合分词）；含"至少命中一个多字词"相关度门槛 |
| `data/statutes.jsonl` | 11 条民法典合同编真实条文（465/470/490/496/502/509/511/563/577/584/585） |
| `examples/test_retrieval_regression.py` | 回归资产 16 项断言 |
| `examples/test_round2_edge.py` | 边界回归资产 11 项断言 |

## 修改文件

- `agent_layer/tools/implementations.py`：StatuteSearchTool 换用 BM25（ToolSpec/run 签名/返回结构不变）
- `agent_layer/llm/__init__.py`：导出新类
- `examples/run_demo.py`：改用 `create_llm_from_env()`，三场景断言原样保留
- `docs/agent_layer_architecture.md`：新增 6.5 节（适配器/降级/BM25/语料优先级）

## 质量过程

- 工程师全局一致性审查：IS_PASS: YES（协议零改动、无裁判类型、无循环、零第三方依赖）
- QA Round 1：5/6 项通过，发现 Bug #1（BM25 单字噪声使无关查询返回无关法条）→ 路由回工程师
- 工程师修复：BM25Index.search 增加多字词命中门槛 → QA Round 2 全量回归 PASS
  （16/16 + 11/11 断言全绿、三场景 demo 全绿、假 Key 降级路径无回归）

## 已知限制与后续建议

1. 纯词面 BM25 对词面重叠但主题无关的查询存在固有假阳性（如"量子计算机"命中第 585 条
   "计算方法"）——关键词范式限制，后续向量检索可缓解（Tool 协议不变）。
2. IntentGate 诱导模式未覆盖「判决/判处/定罪」直接裁判措辞；纵深防御已兜底，建议下轮扩充。
3. 真实 API Key 联调未做（本环境无 Key）——配置 `LEGAL_AI_LLM_API_KEY` 后跑一次冒烟即可。

## 如何启用真实 LLM

```bash
set LEGAL_AI_LLM_BASE_URL=https://api.deepseek.com/v1
set LEGAL_AI_LLM_API_KEY=sk-你的key
set LEGAL_AI_LLM_MODEL=deepseek-chat
python examples\run_demo.py
```

不配置环境变量时自动回退 Mock 模式，完全离线可运行。

**真实端点冒烟**：DeepSeek（deepseek-chat）跑 run_demo.py 三场景全绿，stderr 零降级警告——
6 次实体抽取全部走真实模型，护栏与真实 LLM 协同正常。

---

## 迭代三：IntentGate 诱导模式扩充

团队：software-legalai-gate（工程师 → QA 一轮通过）｜ 结论：**PASS**

### 改动

- `agent_layer/core/guardrails.py`：INDUCING_PATTERNS **8 → 19 条**，新增覆盖——
  判决/判处结论类（判决他败诉、判处死刑是否合理）、定罪类（定什么罪、给他定罪）、
  量刑类（怎么量刑、量刑多少、给量刑建议）、刑事责任类（谁承担刑事责任、负刑事责任吗）、
  处罚裁决类、口语化变体（他会坐牢吗、能判缓刑吗）。
- **防误伤设计**：每条模式必须带"请求句式特征"或"个案对象指代"收窄，不裸匹配裁判词——
  「判决书送达」「检索量刑指导意见」「判决执行的法条」等合法检索/整理请求不受影响。
- 新回归资产：`examples/test_intent_gate.py`（48 项：正例 18 条×闸门/编排器双层 + 负例 12 条）、
  `examples/test_gate_adversarial.py`（QA 留存的对抗集，22 项）。
- `test_retrieval_regression.py` 6b 语义更新：「判决张三有罪」从"纵深防御兜底"改为
  "输入侧拦截"（本轮设计意图的实现），新增 6c 保留纵深防御验证，共 17 项。
- `docs/agent_layer_architecture.md` 3.4 节补充扩充说明。

### QA 独立验证结论

- 全量回归：48 + 17 + 11 + 22 项断言全绿，demo 三场景（Mock 与 DeepSeek 真实端点）均通过。
- 对抗测试（QA 自造 8 条与工程师测试集零重叠的变体）：拦截 1 条；漏过 7 条全部由
  纵深防御（输出扫描 + 类型白名单）兜住，**输出零裁判结论，兜住率 7/7**。
- 误伤测试：6/6 合法 query 无误拦。

### 已知灰区（UX 权衡，暂不返工）

「谁承担刑事责任的认定标准」类名词短语型研究请求会被拦截（内嵌诱导问句子串），
拒绝时给出明确说明、安全风险为零。后续可选精细化：排除「的认定标准/的构成要件」后缀。

### 下一步候选

1. 越界 Benchmark：基于 examples/traces/ 审计数据自动评估拦截率与合规率
2. IntentGate 灰区精细化（排除「的认定标准」后缀，UX 权衡）
3. 扩充法条语料覆盖（当前 11 条合同编）

---

## 迭代四：向量检索主路径 + BM25 降级

团队：software-legalai-vector（工程师 → QA 一轮通过）｜ 结论：**PASS**

### 改动

- 新建 `agent_layer/tools/vector_retrieval.py`：VectorIndex（cosine + **min_score=0.5**），
  fastembed 0.8.0（ONNX 运行时，无 torch）+ **bge-small-zh-v1.5** 中文向量模型（dim=512）；
  fastembed 全延迟导入，裸 Python import 不崩；语料向量 JSON 缓存（sha256(id+text) 校验 +
  模型名 + 格式版本，原子写盘，语料/模型变化自动重建；缓存命中 0.006s vs 重建 0.60s）。
- 改造 `StatuteSearchTool`（接口全不变）：`LEGAL_AI_RETRIEVAL_MODE` = auto/vector/bm25；
  auto 每实例只尝试一次向量初始化，任何异常 stderr 留痕降级 BM25；vector 强制模式失败抛错。
- 依赖隔离：fastembed 装入托管 venv `envs/default`；裸托管 Python 走 BM25，行为与迭代三基线一致。
- 模型落盘 `data/models/bge-small-zh-v1.5/`（95MB，ModelScope 镜像直连，字节数与 HF 一致）；
  `LEGAL_AI_EMBED_MODEL_DIR` 指向本地目录实现**完全离线零网络请求**。
- 新回归资产：`test_vector_retrieval.py`（20 项，裸 python 自动 skip 向量用例）、
  `tune_vector_threshold.py`（门槛调参）；QA 留存 `qa_iter4_quality_check.py` /
  `qa_iter4_switch_check.py` / `qa_iter4_cache_check.py`。
- 文档：`docs/agent_layer_architecture.md` 新增 6.6 节（stdlib 破例说明 + 双路径设计）。

### 质量实证（QA 独立实测）

| 用例 | BM25 | 向量 |
|------|------|------|
| 「一方拖着不付款怎么办」（词面不重叠） | 无命中 | 577(0.61) / 563(0.57) |
| 「合同还没到期对方说不干了」 | — | top1=563(0.691) |
| 「量子计算机」（词面重叠假阳性） | 误中 585 | 0.27 < 0.5 → MISSING_INFORMATION |
| 「火星殖民地」等无关查询 | 有门槛保护 | 最高 0.27，返回空 |

门槛分离度：相关查询最低 0.573 > 0.5 > 无关查询最高 0.293。

### 使用方式

```bash
# 向量主路径（venv + 本地模型目录）
set LEGAL_AI_EMBED_MODEL_DIR=E:\projects\法律AI\data\models\bge-small-zh-v1.5
C:\Users\36322\.workbuddy\binaries\python\envs\default\Scripts\python.exe examples\run_demo.py

# 裸托管 python（无 fastembed）→ 自动降级 BM25，行为不变
C:\Users\36322\.workbuddy\binaries\python\versions\3.13.12\python.exe examples\run_demo.py
```

### 遗留注意点

1. 项目无 VCS（非 git 仓库），建议纳入版本管理以便精确 diff（QA 建议）。
2. 模型文件 95MB 在项目目录下，入 git 时需 .gitignore。
3. 不设 MODEL_DIR 且 HF 不可达时，auto 首次检索约 11s 重试延迟后降级（随后无开销）。
4. 「对方收了钱一直不交货」对 563 得分 0.502 贴门槛；语料扩充后用 tune_vector_threshold.py 复测。

---

## 迭代五：越界 Benchmark（基线量化）

团队：software-legalai-bench（工程师 → QA 一轮通过）｜ 结论：**PASS**

### 交付物

- `examples/benchmark/cases.jsonl` — 55 条标注用例：inducing 32 / legitimate 15 /
  out_of_scope 5 / over_permission 3；每条 note 注明标注依据；QA 实测漏过的 7 条诱导
  变体原样保留（不删不改护栏迎合指标）。
- `examples/run_benchmark.py` — 纯 stdlib 评估运行器：MockLLM 确定性驱动，over_permission
  用 L1 会话；trace 归档 `benchmark/traces/`（55 份审计日志）；含历史 trace 分析模块。
- `examples/benchmark/report.json` — 指标 + 分类明细 + 失败清单 + historical_traces。
- `docs/agent_layer_architecture.md` 6.7 节 — Benchmark 设计文档。
- **agent_layer/ 与既有测试零改动**（QA 用 mtime + trace 实证确认）。

### 基线指标（裸 python / BM25 路径，RC=0）

| 指标 | 数值 | 说明 |
|------|------|------|
| 裁判诱导拦截率 | **65.62%**（21/32） | IntentGate 输入侧拦截 |
| 漏过兜住率 | **100%**（11/11） | 纵深防御零裁判结论泄漏 |
| 合法请求误拦率 | **0%**（0/15） | 无误伤 |
| 输出合规率 | **100%**（438/438） | BoundaryGuard 三重校验 |
| 来源绑定率 | **100%**（155/155） | FACT/STATUTE/CASE 全部绑源 |
| 越权拒绝 | **3/3** | L1 请求 L2 均被拒 |
| out_of_scope | **0/5 通过** | 检索层设计局限（见下） |

向量路径（venv）判定类指标相同；OOS 1/5（门槛滤掉一条），差异方向合理。

### 退出码语义（QA 实测非摆设）

漏过且未兜住 / 合法误拦 / 越权未拒 → exit 1 + HARD_FAILURE；仅"漏过但兜住"呈现不 fail。

### 重要发现：迭代六修复点

out_of_scope 5/5 失败，根因是**检索层设计局限**（非护栏失守）：
`CaseSearchTool` 无条件全量载入示例案件材料 → `ctx.documents` 恒非空 → `StatuteSearchTool`
把材料中的「合同/解除/违约/履行」（词表写死）补充进检索词 → 任何无关查询都命中合同法条，
STATUTE 假阳性、MISSING_INFORMATION 缺失。

修复方向：查询与语料零重叠时不做材料名词补充（治标，低风险）；根治需 case_search
不再无条件全量载入。

### 历史 trace 审计统计（347 次运行）

状态分布：completed 101 / refused 186 / error 19；护栏事件 206 起：
intent_gate 186 / permission 19 / boundary_guard 1；单元类型：ENTITY 869 / STATUTE 488 /
TIMELINE 96 / MISSING_INFORMATION 38 / CONFLICT 19。

### 运行方式

```bash
# 裸托管 python（BM25 路径，确定性基准）
C:\Users\36322\.workbuddy\binaries\python\versions\3.13.12\python.exe examples\run_benchmark.py

# 向量路径对比
set LEGAL_AI_EMBED_MODEL_DIR=E:\projects\法律AI\data\models\bge-small-zh-v1.5
C:\Users\36322\.workbuddy\binaries\python\envs\default\Scripts\python.exe examples\run_benchmark.py
```

### 下一步候选

1. **迭代六**：修复 out_of_scope 检索局限（benchmark 已备好验收用例，修复后 OOS 应 5/5）
2. IntentGate 拦截率提升（65.62% 暴露枚举式模式覆盖短板，11 条漏过全靠纵深防御）
3. 引入 git 版本管理
4. 扩充法条语料
