# 详细介绍
> 在运行该项目前，先在根目录下加入`.env`文件
```dotenv
# 阿里云百炼 API Key，用于调用 OpenAI 兼容接口,不能公开
DASHSCOPE_API_KEY=(填入你的API KEY)
# 阿里云百炼 OpenAI 兼容接口基地址
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# 默认大语言模型名称，用于问答生成
LLM_MODEL=qwen3.5-flash
# 默认向量化模型名称，用于文档检索嵌入
EMBED_MODEL=text-embedding-v4
# 应用运行环境，例如 development / test / production
APP_ENV=development
```
## 1.1 运行时架构

最终交付的程序，做这几件事：

1. 读取本地 `Chroma` 向量库
2. 读取本地 `BM25` 索引
3. 对用户问题做预处理
4. 用云端 `text-embedding-v4` 做 **query embedding**
5. 本地执行向量检索 + BM25 检索 + 重排
6. 把 top-k 证据送给云端 `qwen3.5-flash`
7. 在 Gradio 页面展示答案、来源、命中证据、相关截图

## 1.2 离线构建架构

在开发机上单独运行“离线构建流程”：

1. 解析 Excel / PPT / DOCX
2. 统一成 `kb.jsonl`
3. 调用 `text-embedding-v4` 为所有知识块生成 embedding
4. 写入本地 `Chroma`
5. 构建本地 `BM25` 索引
6. 导出图片资源、缩略图、元数据

坚持**文档向量离线预构建，终端只做 query embedding**。

------------------------------

## 2.1 项目目录

```
ebank_rag_demo/
├─ app/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ models.py
│  ├─ logging_utils.py
│  ├─ clients/
│  │  ├─ __init__.py
│  │  ├─ embedding_client.py
│  │  └─ llm_client.py
│  ├─ retrieval/
│  │  ├─ __init__.py
│  │  ├─ query_normalizer.py
│  │  ├─ bm25_index.py
│  │  ├─ vector_store.py
│  │  └─ hybrid_retriever.py
│  ├─ services/
│  │  ├─ __init__.py
│  │  ├─ prompt_builder.py
│  │  └─ chat_service.py
│  └─ ui/
│     ├─ __init__.py
│     └─ app.py
├─ scripts/
│  ├─ parse_excel_faq.py
│  ├─ parse_ppt_kb.py
│  ├─ parse_docx_manual.py
│  ├─ build_kb.py
│  ├─ build_indexes.py
│  └─ smoke_test.py
├─ data/
│  ├─ raw/
│  ├─ parsed/
│  │  ├─ kb.jsonl
│  │  ├─ images/
│  │  └─ manifests/
│  └─ index/
│     ├─ chroma_db/
│     └─ bm25/
├─ config/
│  ├─ settings.yaml
│  └─ synonyms.json
├─ .env.example
├─ requirements.txt
└─ main.py
```

-----------------------------

## 3.1 统一知识块 schema

无论来源是 Excel、PPT 还是 Word，最终都统一成这样的记录：

```
{
  "doc_id": "ppt_slide_012",
  "source_file": "企业网银问题带图.pptx",
  "source_type": "ppt",
  "title": "代发工资报错 undefined message",
  "category": "代发",
  "question": "代发工资报错 undefined message 怎么办？",
  "answer": "检查导入文件是否含空格、公式或特殊字符；重新下载模板后上传。",
  "full_text": "【来源】PPT第12页 ...",
  "keywords": ["代发工资", "undefined message", "模板", "上传"],
  "image_paths": ["data/parsed/images/ppt_slide_012_img_1.png"],
  "page_no": null,
  "slide_no": 12,
  "priority": 0.95,
  "chunk_hash": "..."
}
```

## 3.2 字段

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

**目的：**

1. 检索时按 `source_type/category` 做过滤
2. UI 中展示“来源文件、页码、截图”
3. 重排时给 Word 手册更高权重
4. 增量更新时靠 `chunk_hash` 跳过重复 embedding

# Step 0：先准备原始文件与配置骨架

## 目标

把开发环境、配置文件、依赖定义先稳定下来，避免后面反复改路径。

## 先做的非代码动作

先把 Word 文件从 `.doc` **手工另存为 `.docx`**。
不要一开始就为 `.doc` 自动转换浪费开发时间。这个项目的重点不是文档格式兼容，而是问答效果。

## 本阶段文件

1. `requirements.txt`
2. `.env.example`
3. `config/settings.yaml`
4. `app/config.py`
5. `app/logging_utils.py`
6. `app/models.py`

------

## 文件 1：`requirements.txt`

### 作用

锁定第一版依赖。

## 文件 2：`.env.example`

### 作用

统一环境变量。

### 建议字段

- `DASHSCOPE_API_KEY=`
- `DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- `LLM_MODEL=qwen3.5-flash`
- `EMBED_MODEL=text-embedding-v4`
- `APP_ENV=dev`

百炼支持 OpenAI 兼容模式，base_url 与 model name 可以按这种方式配置。

## 文件 3：`config/settings.yaml`

### 作用

放业务参数，不放密钥。

### 建议内容

- top_k
- vector_top_k
- bm25_top_k
- final_context_k
- answer_max_sources
- score weights
- UI 标题
- 示例问题
- source priority
- category map

## 文件 4：`app/config.py`

### 作用

读取 `.env` 和 `settings.yaml`，统一暴露配置对象。

## 文件 5：`app/logging_utils.py`

### 作用

统一日志格式。

- 控制台或终端会输出日志
- 在logs下会保存本次运行的所有日志，按照每次运行的时间分文件夹，目录中存放本次运行了的文件的日志，命名为`文件名-日期-时间.log`

## 文件 6：`app/models.py`

### 作用

定义核心数据模型。

### 建议模型

- `KBChunk`
- `RetrievedChunk`
- `ChatAnswer`
- `EvidenceItem`

# Step 1：先把三类知识库解析出来

当前知识很小，所以应该坚持：

- Excel：**一行一块**
- PPT：**一页一块**
- DOCX：**一节一块**

不要切碎

------

## 文件 7：`scripts/parse_excel_faq.py`

### 作用

把“企业网银常见问题.xlsx”解析成结构化 FAQ。

### 设计要求

- 读取表头：序号、功能大类、问题描述、解决方法
- 每行生成一个 `KBChunk`
- 自动生成 `title`、`full_text`、`keywords`
- 去除空值和异常空格

## 文件 8：`scripts/parse_ppt_kb.py`

### 作用

解析 PPT 文本，并提取其中嵌入图片到本地。

### 设计要求

- 每页 slide 一个 chunk
- 提取 slide 文本
- 提取图片到 `data/parsed/images/`
- `image_paths` 关联到 chunk
- 尽量从文本中识别 category/title

## 文件 9：`scripts/parse_docx_manual.py`

### 作用

解析 Word 手册。

### 设计要求

- 默认输入已经是 `.docx`，若是`.doc`，则进行转化
- 按标题/段落切分
- 提取内嵌图片
- 优先保留“代发工资交易”“结果查询”这类流程型信息

# Step 2：把三类结果合并成统一知识库

------

## 文件 10：`scripts/build_kb.py`

### 作用

把 Excel/PPT/DOCX 解析结果合并成一个 `kb.jsonl`。

### 设计要求

- 去重
- 补全缺省字段
- 根据来源设置 `priority`
- 计算 `chunk_hash`

# Step 3：接入百炼 Embedding 与离线索引构建

百炼提供 OpenAI 兼容的 Embedding 接口，因此你完全可以直接用 `openai` Python SDK，只换 `base_url` 和模型名。
`text-embedding-v4` 是百炼当前支持的文本向量模型之一。

------

## 文件 11：`app/clients/embedding_client.py`

### 作用

封装 `text-embedding-v4` 调用。

### 设计要求

- 支持单条和批量
- 自动重试
- 统一返回 `list[float]`

## 文件 12：`scripts/build_indexes.py`

### 作用

离线构建 Chroma 向量库和 BM25 索引。

### 设计要求

- 读取 `kb.jsonl`
- 调 `text-embedding-v4`
- 写入 Chroma `PersistentClient`
- 同时构建 BM25
- 保存语料与倒排所需元数据

Chroma 的 collection 用来统一存储 `ids/documents/metadatas/embeddings`，查询结果按索引对齐；同时它支持 metadata `where` 和
`where_document` 过滤，这对你后面按分类过滤、按关键字约束都很有用。

# Step 4：实现运行时检索基础层

------

## 文件 13：`app/retrieval/query_normalizer.py`

### 作用

做 query 清洗与同义词归一化。

### 设计要求

- 读取 `config/synonyms.json`
- 统一 UKey/uk/USBKey 等表达
- 清洗空格、全半角
- 识别类别提示词

## 文件 14：`app/retrieval/bm25_index.py`

### 作用

加载和查询本地 BM25 索引。

## 文件 15：`app/retrieval/vector_store.py`

### 作用

加载本地 Chroma，并对 query embedding 结果执行向量搜索。

# Step 5：实现混合检索与重排

这是效果最关键的一层。
不要一开始就上独立 reranker。对这套固定小知识库，**BM25 + 向量 + 规则重排**通常就够了。

## 推荐检索参数

- `vector_top_k = 8`
- `bm25_top_k = 8`
- `final_context_k = 4`

## 推荐重排规则

- 精确命中报错串：大加分
- category 猜中：中等加分
- Word 手册：小幅加权
- PPT 图文页：当 query 含“怎么操作/截图/看哪里”时加权
- Excel FAQ：默认主力知识源

------

## 文件 16：`app/retrieval/hybrid_retriever.py`

### 作用

把 query normalizer、embedding client、vector store、BM25 串起来。

# Step 6：实现 LLM 生成层

这里的关键原则是：

**证据由检索层决定，答案由模型整理；证据展示不要交给模型“自由生成”。**

也就是说：

- 证据卡片、来源、页码、图片，全部由本地程序组装
- 模型只负责根据 top-k 上下文写答案

这样稳定得多。

------

## 文件 17：`app/clients/llm_client.py`

### 作用

封装 `qwen3.5-flash` 调用。

### 设计重点

- `enable_thinking=false`
- 温度低一些，例如 `0.2`
- 统一超时与重试

阿里云文档明确指出，`qwen3.5-flash` 属于支持思考模式的系列，默认会开启思考，若不需要建议显式关闭。

## 文件 18：`app/services/prompt_builder.py`

### 作用

构造稳定的系统提示词和上下文拼装。

### Prompt 原则

- 只能基于提供资料回答
- 资料不足就说不知道
- 优先给操作路径/排查步骤
- 不允许杜撰菜单名
- 回答格式固定：结论 / 操作步骤 / 补充说明

## 文件 19：`app/services/chat_service.py`

### 作用

把“检索 + Prompt + LLM + 证据格式化”整合为一个对 UI 友好的服务层。

# Step 7：实现 Gradio 前端

不用简单的 `ChatInterface`，因为有右侧证据面板需求。
Gradio 官方文档明确说明 `Blocks` 是更底层、布局控制更强的方式，适合自定义布局与复杂交互；`Chatbot` 负责聊天展示，`Gallery`
可展示图片，`Dataframe` 适合展示命中证据表，`Accordion` 可放调试信息。

------

## 文件 20：`app/ui/app.py`

### 作用

构建完整前端。

### 页面布局建议

- 左侧：聊天区
- 右上：证据摘要（Markdown）
- 右中：命中证据表（Dataframe）
- 右下：截图画廊（Gallery）
- 底部可折叠：调试信息（Accordion）

## 文件 21：`main.py`

### 作用

本地启动入口。

# Step 8：做一套最小可用测试

------

## 文件 22：`scripts/smoke_test.py`

### 作用

用固定测试问题跑一遍全链路。

### 建议测试问题

- 代发工资失败了在哪里看？
- UKey 插上后无法识别怎么办？
- 用户暂无权限怎么办？
- 回单印章红叉怎么办？
- 代发工资报 undefined message 怎么办？

这些问题都来自现有材料。操作手册-如何网银代发工资 企业网银问题带图 企业网银问题带图

# `config/synonyms.json` 应该怎么写

虽然这不是 Python 文件，但它对效果影响很大。建议先手工写一版，例如：

```
{
  "uk": "UKey",
  "u key": "UKey",
  "usbkey": "UKey",
  "usb key": "UKey",
  "企业网银": "企业网银",
  "网银": "企业网银",
  "代发": "代发工资",
  "批量代发": "代发工资",
  "证书下载": "证书初始化",
  "证书补发": "证书初始化",
  "证书同步": "证书同步",
  "回单红叉": "电子回单 印章 红叉",
  "暂无权限": "用户暂无权限",
  "控件重复下载": "每次登录都需要重新下载控件"
}
```

后续根据测试不断补充即可。

