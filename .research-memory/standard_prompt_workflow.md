# 科研新线程标准提示词工作流

目标：减少重复讲历史、降低 token 消耗、让新线程能直接续上当前主线。

## 使用原则

- 新线程一开始先读 `.research-memory`，不要先全仓库考古。
- 对话里只保留结论、决策、风险和下一步；长分析、长表格、长日志优先写入项目文件。
- 没有必要时不要重复复述历史，不要把已经稳定的结论再讲一遍。
- 每轮结束前，如果有新稳定结论，就更新 `.research-memory`。

## 提示词 1：新开线程启动

复制下面这段，开新窗口时直接发。

```text
你现在接手一个已经有科研记忆的项目。

项目根目录：
<PROJECT_ROOT>

如果存在 `.research-memory`，请先做这几件事：
1. 优先读取 `.research-memory`，不要先扫描全仓库。
2. 先基于项目记忆恢复上下文，再决定是否读取 `artifacts/` 或源码。
3. 控制 token：不要长篇重复历史；把长分析、长表格、长日志写进项目文件，不要堆在聊天里。
4. 输出时只保留：当前主线、当前 blocker、你准备先做什么、关键结论。
5. 如果这一轮得到新的稳定结论，结束前更新 `.research-memory`。

当前任务：
<TASK>

开始时请先给我一段很短的恢复结果，必须明确写出：
- 当前主线是什么
- 当前重点是什么
- 你接下来先做哪一步
然后直接开始执行。
```

## 提示词 2：继续改模型 / 跑实验

当你已经确定主线，只想让 Codex 直接干活时，用这段。

```text
继续当前主线，不要重新做大段历史回顾。

要求：
- 先用已有 `.research-memory` 恢复上下文。
- 当前主线默认不变，除非新证据足够强。
- 直接执行，不先给长计划。
- 修改代码后直接跑最必要的验证。
- 把长实验输出、长结果表、长分析写入项目文件。
- 聊天里只告诉我：改了什么、结果怎样、下一步建议。

本轮具体任务：
<TASK>
```

## 提示词 3：读结果、找瓶颈、定下一步

当你已经有新产物，想让 Codex 少废话直接分析时，用这段。

```text
先读取 `.research-memory`，然后只围绕下面这些产物做分析，不要泛泛讲模型结构。

重点要求：
- 先回答“结果说明了什么”
- 再回答“当前最可能的瓶颈是什么”
- 再回答“下一轮最值得优先验证的 1 到 3 件事”
- 不要长篇复述旧历史
- 如果发现结论已经足够稳定，结束前更新 `.research-memory`

需要分析的产物：
<ARTIFACT_PATHS>

当前问题：
<QUESTION>
```

## 提示词 4：线程收尾 / 交接压缩

当一个线程做了很多事，准备结束并给下个线程交接时，用这段。

```text
请先不要继续扩展任务，先做本轮收尾压缩。

要求：
- 只保留稳定结论，不要写过程噪音。
- 更新 `.research-memory`：
  - 如有必要，更新 `project_brief.md`
  - 更新 `active_context.md`
  - 更新 `next_steps.md`
  - 补充必要的 decision / finding / bottleneck / experiment
- 不要把不稳定猜测写成结论。
- 最后给我一个很短的交接摘要，只保留：
  - 本轮新增的稳定结论
  - 当前未解决 blocker
  - 下个线程最优先做什么
```

## ReK-Surv 现成可用版本

这是你在 `ReK-Surv` 上最省事的启动词，直接复制即可。

```text
你现在接手一个已经有科研记忆的项目。

项目根目录：
C:\Users\21107\Desktop\新建文件夹 (2)\test\ReK-Surv

上一条相关线程：
codex://threads/019dce40-34ba-7cb3-b1b9-6fde3d6a6ea7

如果存在 `.research-memory`，请先做这几件事：
1. 优先读取 `.research-memory`，不要先扫描全仓库。
2. 先基于项目记忆恢复上下文，再决定是否读取 `artifacts/` 或源码。
3. 控制 token：不要长篇重复历史；把长分析、长表格、长日志写进项目文件，不要堆在聊天里。
4. 输出时只保留：当前主线、当前 blocker、你准备先做什么、关键结论。
5. 如果这一轮得到新的稳定结论，结束前更新 `.research-memory`。

当前默认约束：
- 当前主线默认是 `period_ms_tree_query`
- 当前重点默认是下一步实验规划
- 当前评估协议默认是 `5` 次 repeated random hold-out，不是 `k-fold CV`
- 如果没有足够强的新证据，不要擅自切主线

开始时请先给我一段很短的恢复结果，必须明确写出：
- 当前主线是什么
- 当前重点是什么
- 你接下来先做哪一步
然后直接开始执行。
```

## 推荐用法

1. 新开线程先发“提示词 1”或直接发 `ReK-Surv 现成可用版本`。
2. 进入具体工作后，按需再发“提示词 2”或“提示词 3”。
3. 准备结束线程时，发“提示词 4”。

## 你最常用的一句短版

如果你懒得发长提示词，就发这一句：

```text
先按 `.research-memory` 恢复当前主线和 blocker，控制 token，不要长篇复述历史，把长分析写进项目文件；然后直接继续这项任务：<TASK>
```

## 半自动命令入口

如果你不想每次手动复制提示词，现在可以直接用这两个项目内命令入口。

### 1. 自动生成新线程启动词

```powershell
& ".research-memory\tools\start_research_thread.ps1" -Task "继续 period_ms_tree_query 主线"
```

可选参数：

- `-Task`：这次新线程的具体任务
- `-Query`：检索记忆时用的查询词；默认等于 `Task`
- `-ThreadLink`：上一条相关线程链接
- `-Write`：把生成结果写到文件，方便复制或归档

### 2. 自动收尾并更新 memory

```powershell
& ".research-memory\tools\close_research_session.ps1" `
  -Title "3->2 主线结果复核" `
  -Summary "确认 3->2 仍是当前稳定性优先的主推窗口。" `
  -Decision "下一轮优先围绕 3->2 做 tree_query 边际收益比较。" `
  -NextStep "补一组 tree_query vs 非 tree-query 的干净对照消融。"
```

可选参数：

- `-Finding`
- `-Decision`
- `-Bottleneck`
- `-NextStep`
- `-Artifact`
- `-Tag`
- `-Pinned`
- `-ProjectBriefFile`
- `-ActiveContextFile`
- `-NextStepsFile`
- `-Write`

如果你提供了 `ProjectBriefFile / ActiveContextFile / NextStepsFile`，收尾命令会在记录结构化记忆后，顺手覆盖对应 markdown 文本。
