# 银证智答
### 面向企业网银的证据增强检索生成系统

#### EviBank-RAG: Evidence-Grounded Retrieval-Augmented QA System for Enterprise E-Banking


<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12+-blue.svg">
  <img alt="Gradio" src="https://img.shields.io/badge/UI-Gradio-orange.svg">
  <img alt="Vector DB" src="https://img.shields.io/badge/VectorDB-Chroma-brightgreen.svg">
  <img alt="LLM" src="https://img.shields.io/badge/LLM-Qwen3.5--Flash-purple.svg">
  <img alt="Embedding" src="https://img.shields.io/badge/Embedding-text--embedding--v4-red.svg">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey.svg">
</p>

> 一个面向**企业网银交易常见问题**场景的轻量级 RAG 问答机器人 Demo。  
> 项目聚焦“**本地可运行、易于演示、证据可追溯、图文可展示**”的工程目标，适用于企业内部知识问答、业务 FAQ 辅助检索与会议演示场景。
>
> Built a lightweight evidence-grounded RAG system for enterprise e-banking FAQ, featuring multi-source knowledge ingestion, hybrid retrieval, source-aware reranking, and explainable answer generation with visual evidence presentation.


---

## 项目简介

本项目实现了一个面向**企业网银 FAQ 场景**的检索增强生成（RAG）系统。  
系统以企业网银领域知识库为基础，结合**结构化 FAQ、图文说明文档、操作手册**等多源资料，实现对常见问题的自然语言问答，并在回答时同步展示**命中证据、来源信息与相关截图**。

与通用聊天机器人不同，本项目强调以下几个目标：

- **面向垂直业务问题**，聚焦企业网银交易与操作常见问题；
- **回答可追溯**，支持显示命中文档、来源文件、页码/页序、相关图示；
- **本地轻量可运行**，适合会议演示与少量人员单机分发；
- **工程结构清晰**，具备从知识解析、索引构建、检索融合到 UI 展示的完整链路；
- **兼顾学术性与工程性**，可作为 RAG 系统设计、知识工程与智能问答方向的实践案例。

---

## 项目背景

在企业网银、数字金融与业务系统支持场景中，用户提出的问题通常具有如下特点：

1. **问题高度重复**，但表述方式存在差异；
2. **知识源异构**，常同时分散在 Excel FAQ、Word 操作手册、PPT 图文说明中；
3. **用户更关心“如何操作”与“到哪里查看”**，而不仅是抽象解释；
4. **演示与落地要求强**，系统既要能答，还要能够展示“依据是什么”。

基于以上特点，本项目没有追求复杂的智能体编排或重型知识系统，而是采用了**轻量、稳健、可追溯**的技术路线：

- 用结构化解析统一知识格式；
- 用向量检索 + BM25 混合召回增强稳定性；
- 用云端大模型做答案组织与语言表达；
- 用本地图文证据展示强化可信度与演示效果。

---

## 核心特性

- **多源知识接入**
    - Excel：结构化 FAQ
    - Word：流程型操作手册
    - PPT：图文问题说明

- **统一知识抽象**
    - 将不同文件来源统一为标准 `KBChunk` 结构
    - 支持 `title / category / full_text / image_paths / page_no / slide_no / priority`

- **混合检索策略**
    - 向量检索（Chroma）
    - BM25 关键词检索
    - 标题精确/模糊匹配增强
    - 规则重排（分类命中、来源优先级、图片证据偏置等）

- **证据驱动生成**
    - 检索层决定证据
    - PromptBuilder 负责组织上下文
    - LLM 仅基于证据生成答案，降低幻觉风险

- **图文问答体验**
    - 左侧对话区
    - 右侧证据摘要、命中证据表
    - 相关截图画廊展示
    - 调试区输出规范化查询与重排原因

- **适合单机演示**
    - 文档向量离线构建
    - 运行时仅对用户 Query 做云端向量化
    - 无需在终端部署本地大模型

---

## 技术架构

### 整体架构

```text
用户问题
   │
   ▼
Query 预处理 / 术语归一化
   │
   ├──► 云端 Embedding（text-embedding-v4）
   │         │
   │         ▼
   │    Chroma 向量检索
   │
   └──► BM25 关键词检索
             │
             ▼
      混合召回与规则重排
             │
             ▼
       Prompt 构造（证据组织）
             │
             ▼
     云端 LLM（qwen3.5-flash）
             │
             ▼
 回答生成 + 来源依据 + 图文证据展示
```

### 技术选型

| 模块        | 方案                                            |
|-----------|-----------------------------------------------|
| 生成模型      | Qwen3.5-Flash（阿里云百炼 OpenAI 兼容接口）              |
| Embedding | text-embedding-v4                             |
| 向量数据库     | Chroma PersistentClient                       |
| 关键词检索     | BM25                                          |
| 前端界面      | Gradio Blocks                                 |
| 数据解析      | pandas / openpyxl / python-docx / python-pptx |
| 工程配置      | pydantic-settings / yaml / dotenv             |
| 打包分发      | PyInstaller（可选）                               |

------

## 检索与生成设计

### 1. 知识清洗与统一

不同来源文件在进入系统前会被统一转换为标准知识块：

- Excel：一行 FAQ 一条知识块
- PPT：一页 slide 一条知识块
- DOCX：一节或一个流程说明一条知识块

统一字段包括：

- `doc_id`
- `source_file`
- `source_type`
- `title`
- `category`
- `question`
- `answer`
- `full_text`
- `keywords`
- `image_paths`
- `slide_no`
- `page_no`
- `priority`
- `chunk_hash`

### 2. 混合召回

系统同时使用两种召回机制：

- **向量检索**：解决用户表述不规范、语义改写问题；
- **BM25 检索**：强化对业务术语、菜单路径、错误信息、关键词的精确命中。

### 3. 规则重排

候选结果会进一步做规则重排，考虑：

- query 与 title 的整句精确匹配
- 标题高相似 / 包含关系
- 关键词精确命中
- 分类命中
- 来源优先级
- 是否包含图像证据

### 4. 证据驱动回答

LLM 不直接“自由作答”，而是只基于检索证据完成：

- 结论提炼
- 操作步骤整理
- 补充说明归纳
- 来源依据展示

------

## 项目目录

```text
ebank_rag_demo/
├─ app/
│  ├─ __init__.py
│  ├─ config.py                  # 配置加载
│  ├─ models.py                  # 核心数据模型
│  ├─ logging_utils.py           # 日志初始化
│  ├─ clients/
│  │  ├─ embedding_client.py     # Embedding API 封装
│  │  └─ llm_client.py           # LLM API 封装
│  ├─ retrieval/
│  │  ├─ query_normalizer.py     # Query 规范化与术语归一
│  │  ├─ bm25_index.py           # BM25 索引加载与查询
│  │  ├─ vector_store.py         # Chroma 检索封装
│  │  └─ hybrid_retriever.py     # 混合检索与重排
│  ├─ services/
│  │  ├─ prompt_builder.py       # Prompt 构造
│  │  └─ chat_service.py         # 主业务服务层
│  └─ ui/
│     └─ app.py                  # Gradio UI
├─ scripts/
│  ├─ parse_excel_faq.py         # Excel FAQ 解析
│  ├─ parse_ppt_kb.py            # PPT 图文知识解析
│  ├─ parse_docx_manual.py       # Word 手册解析
│  ├─ build_kb.py                # 合并生成 kb.jsonl
│  ├─ build_indexes.py           # 构建 Chroma 与 BM25 索引
│  ├─ smoke_test.py              # 全链路最小冒烟测试
│  └─ test/                      # 其他临时调试脚本
├─ data/
│  ├─ raw/                       # 原始知识文件
│  ├─ parsed/
│  │  ├─ kb.jsonl                # 统一知识库
│  │  └─ images/                 # 提取出的图像资源
│  └─ index/
│     ├─ chroma_db/              # 向量索引
│     └─ bm25/                   # BM25 索引
├─ config/
│  ├─ settings.yaml              # 检索与 UI 配置
│  └─ synonyms.json              # 同义词与术语映射
├─ logs/                         # 运行日志
├─ .env.example                  # 环境变量模板
├─ requirements.txt
├─ environment.yaml
└─ main.py                       # 启动入口
```

------

## 快速开始
> 代码详细介绍见[Intro.md](Intro.md)
### 1. 克隆仓库

```bash
git clone https://github.com/yourname/enterprise-ebank-faq-rag-demo.git
cd enterprise-ebank-faq-rag-demo
```

### 2. 创建虚拟环境并安装依赖

```bash
conda env create -f environment.yaml
```

### 3. 配置环境变量

重命名 `.env.example` 为 `.env`，并填写你的百炼 API Key：

```env
DASHSCOPE_API_KEY=your_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.5-flash
EMBED_MODEL=text-embedding-v4
APP_ENV=development
```

### 4. 准备知识库原始文件

将原始文件放入：

```text
data/raw/
```

例如：

- `企业网银常见问题.xlsx`
- `企业网银问题带图.pptx`
- `操作手册-如何网银代发工资.docx`

------

## 数据构建流程

### Step 1：解析 Excel FAQ

```bash
python scripts/parse_excel_faq.py --input data/raw/企业网银常见问题.xlsx --output data/parsed/excel_kb.jsonl
```

### Step 2：解析 PPT 图文知识

```bash
python scripts/parse_ppt_kb.py --input data/raw/企业网银问题带图.pptx --output data/parsed/ppt_kb.jsonl
```

### Step 3：解析 Word 手册

```bash
python scripts/parse_docx_manual.py --input data/raw/操作手册-如何网银代发工资.docx --output data/parsed/docx_kb.jsonl
```

### Step 4：合并知识库

```bash
python scripts/build_kb.py --inputs data/parsed/excel_kb.jsonl data/parsed/ppt_kb.jsonl data/parsed/docx_kb.jsonl --output data/parsed/kb.jsonl
```

### Step 5：构建索引

```bash
python scripts/build_indexes.py --input data/parsed/kb.jsonl --rebuild
```

------

## 启动 Demo

```bash
python main.py
```

启动后会自动打开本地 Gradio 页面。
你将看到：

- 左侧：聊天对话区
- 右侧：命中证据表、证据摘要、相关截图
- 下方：调试信息区域

------

## 测试与验收

### 1. 检索层测试

可使用临时脚本验证：

- Chroma 可读性
- BM25 可加载性
- `HybridRetriever.retrieve()` 的 top-k 命中质量

### 2. LLM 最小调用测试

验证 `LLMClient.ask()` 是否可正常访问模型接口。

### 3. PromptBuilder 测试

验证 Prompt 是否正确拼装了用户问题与证据上下文。

### 4. 全链路冒烟测试

```bash
python scripts/smoke_test.py
```

建议测试问题包括：

- 代发工资失败了在哪里看？
- UKey 插上没反应怎么办？
- 用户暂无权限怎么办？
- 电子回单印章显示红叉怎么办？
- 为什么每次登录都要重新下载控件？
- 代发工资报 undefined message 怎么办？

------

## 界面预览

> 你可以在这里替换为项目截图或录屏 GIF。

```text
[左侧] 聊天区
[右侧上] 证据摘要
[右侧中] 命中证据表
[右侧下] 图像证据画廊
[底部] 调试信息
```

建议你在 GitHub 仓库中加入：

- `assets/demo_home.png`
- `assets/demo_answer.png`
- `assets/demo_gallery.png`

并在 README 中插图展示。

------

## 设计亮点

### 1. 不是“只会聊天”的模型 Demo，而是可追溯的业务问答系统

系统不仅给出答案，还会展示：

- 命中的知识来源
- 文件类型
- 页码 / 页序
- 图文证据

这使得回答更可信，也更适合业务演示。

### 2. 轻量而稳健

项目没有引入复杂 Agent 框架，而是采用明确的工程分层：

- 数据解析
- 索引构建
- 混合检索
- Prompt 构造
- 生成与展示

这种设计更容易调试、更适合落地、更利于单机演示部署。

### 3. 兼顾语义理解与业务精确匹配

企业网银 FAQ 中存在大量：

- 菜单路径
- 报错短语
- 固定表述
- 相似标题

仅依赖向量检索容易漏掉精确问题，仅依赖关键词检索又不够鲁棒。
本项目通过**向量检索 + BM25 + 标题匹配 + 规则重排**，在效果与稳定性之间取得平衡。

------

## 技术实施路线

本项目的实施路线可概括为五个阶段：

### 第一阶段：知识工程

- 统一清洗 Excel / Word / PPT 三类资料
- 抽取图文资源
- 设计统一知识块结构

### 第二阶段：离线索引构建

- 文档向量离线生成
- Chroma 持久化
- BM25 本地索引建立

### 第三阶段：检索与重排

- Query 规范化
- 混合召回
- 标题匹配增强
- 规则重排优化

### 第四阶段：生成与展示

- PromptBuilder 证据组织
- LLM 输出答案
- UI 侧证据回显

### 第五阶段：演示与封装

- 冒烟测试
- 单机演示验证
- PyInstaller 打包（可选）

![Technical Roadmap](assets/technical_roadmap.png)

<p align="center">
 Technical Roadmap
</p>


------

## 技术优势

### 1. 面向演示与交付优化

项目从设计之初就不是纯研究原型，而是考虑了：

- 单机运行
- 少量人员分发
- 会议稳定展示
- 轻量终端适配

### 2. 证据导向，降低幻觉

模型并不直接“凭感觉回答”，而是受到检索证据严格约束。
这对业务 FAQ 场景至关重要。

### 3. 模块边界清晰

不同模块职责明确，便于：

- 单独测试
- 替换模型
- 调整检索策略
- 后续扩展更多知识源

### 4. 可扩展性强

当前系统面向企业网银 FAQ，但整体架构可迁移到：

- 银行客服知识库
- 企业内部 IT 支持 FAQ
- 产品帮助中心
- 操作手册问答
- 政务/教育/医疗等垂直知识服务

------

## 应用前景

本项目虽然以 Demo 形式实现，但具有明确的工程延展性。
其后续演进方向包括：

### 1. 企业级知识服务助手

可扩展到更大范围的企业网银帮助中心、内部知识门户或智能客服前台。

### 2. 多模态知识问答

当前系统已经具备图像证据展示能力，未来可进一步增强 OCR、图文联合理解、截图定位等功能。

### 3. 更丰富的证据治理

未来可增加：

- 文档版本管理
- 知识热度分析
- 问题命中统计
- 失败问题反向补库

### 4. 学术与工程研究价值

该项目不仅是一个工程 Demo，也适合作为以下方向的实践案例：

- 检索增强生成（RAG）
- 垂直领域知识问答
- 轻量知识系统设计
- 混合检索与规则重排
- 可解释问答系统

------

## Roadmap

-  多源知识接入（Excel / DOCX / PPTX）
-  本地 Chroma + BM25 混合检索
-  Prompt 驱动证据生成
-  Gradio 图文问答界面
-  单机演示可运行版本
-  标题精确匹配直通优化
-  中文 BM25 分词增强
-  更完善的检索评估脚本
-  一键打包与安装脚本
-  Docker 化部署支持
-  更丰富的知识库增量更新机制

------

## 适合谁

本项目适合以下读者和使用者：

- 想学习 **RAG 系统工程落地** 的开发者
- 想实现 **垂直 FAQ 智能问答 Demo** 的学生或研究人员
- 关注 **知识工程、信息检索、智能客服、业务问答** 的工程团队
- 需要构建 **演示型、单机可运行、证据可展示** 的智能问答系统的开发者

------

## 许可证

本项目采用 `MIT License`。
如用于商业项目或二次开发，请根据实际情况补充版权与数据使用说明。

------

## 致谢

本项目的实现参考并受益于以下技术生态：

- [Gradio](https://github.com/gradio-app/gradio)
- [Chroma](https://github.com/chroma-core/chroma)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [RapidFuzz](https://github.com/maxbachmann/RapidFuzz)

同时感谢大模型、检索增强生成与开源工具生态为智能问答工程实践提供的基础能力。

------

## Star History

如果这个项目对你有帮助，欢迎点一个 Star。