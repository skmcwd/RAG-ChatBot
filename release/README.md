# EviBank-RAG 发布版使用说明

## 一、项目简介

EviBank-RAG 是一个面向企业网银常见问题场景的问答演示系统。  
系统基于本地知识库与云端大模型接口实现问答能力，并支持展示命中证据、图文说明及相关截图。

本发布版本适合少量内部人员使用，支持：

1. 修改 `.env.example`为`.env`，并填入其中的API
   Key。获取阿里云百炼API: [链接](https://help.aliyun.com/zh/model-studio/get-api-key)；
2. (可选)替换 `data\raw` 中的原始知识文件；
3. 一键重建知识库与索引；
4. 启动本地问答服务并在浏览器中使用。默认启动ssh映射，可供远程访问。

---

## 二、目录结构

请保持 release 目录结构完整，不要随意移动文件夹位置。

标准结构如下：

```
release
├─ EviBank-RAG
│  └─ EviBank-RAG.exe
├─ EviBank-RAG-Rebuild
│  └─ EviBank-RAG-Rebuild.exe
├─ config
│  ├─ settings.yaml
│  └─ synonyms.json
├─ data
│  ├─ raw
│  ├─ parsed
│  └─ index
├─ .env.example
├─ logs
├─ 启动助手.bat
├─ 重建知识库.bat
└─ README.md
```

说明：

- `EviBank-RAG\`          主程序目录
- `EviBank-RAG-Rebuild\`  知识库重建程序目录
- `config\`               配置目录
- `data\raw\`             原始知识文件目录
- `data\parsed\`          解析后的中间文件目录
- `data\index\`           向量索引与 BM25 索引目录
- `logs\`                 日志目录

> 其中`EviBank-RAG`目录和`EviBank-RAG-Rebuild`
> 目录由于细碎文件较多，我不会放进git仓库，请到release中下载压缩包。具体见[文末](#section1)。
---

## 三、首次使用步骤

1. 打开 release 根目录下的 `.env.example` 文件。
2. 填写或修改以下内容：

   ```ini
   DASHSCOPE_API_KEY=你的APIKey
   ```

   注意：
    - `DASHSCOPE_API_KEY` 必须填写为可用的阿里云百炼 API Key。
    - `DASHSCOPE_BASE_URL` 必须保持为纯 URL，不要写成 Markdown 链接格式。
3. 重命名`.env.example`为`.env`。
4. 确认 `config\settings.yaml` 存在。
5. 双击“启动助手.bat”。
6. 程序启动后，浏览器通常会自动打开。
7. 若浏览器未自动打开，请手动访问：

   ```
   http://127.0.0.1:7860
   ```

---

## 四、如何修改 API Key

当需要更换 API Key 时：

1. 打开 release 根目录下的 `.env` 文件；
2. 修改 `DASHSCOPE_API_KEY` 的值；
3. 保存文件；
4. 重新双击“启动助手.bat”。

说明：

- 仅修改 API Key 时，不需要重建知识库。
- 修改后重新启动主程序即可生效。

---

## 五、如何更新知识库文件

原始知识文件位于：

```
data\raw\
```

你可以根据需要替换或更新其中的文件，例如：

- Excel FAQ 文件
- PPT 图文说明文件
- Word 操作手册文件

注意事项：

1. 可存在 0 个、1 个或多个同类型文件；
2. 若某一类文件不存在，系统会尽量处理其余可用文件；
3. 建议不要在文件名中使用过于特殊的字符；
4. 更新原始文件后，必须重新执行“知识库重建”。

---

## 六、如何重建知识库

当你修改了 `data\raw` 中的文件后，请执行以下步骤：

1. 双击“重建知识库.bat”；
2. 等待终端窗口执行完成；
3. 若未出现错误提示，则说明重建流程已基本完成；
4. 然后返回 release 根目录，双击“启动助手.bat”重新启动系统。

说明：

- 重建知识库会重新解析原始文件，并生成新的 `data\parsed` 与 `data\index` 内容；
- 若原始文件内容有变化，只有完成重建后，问答系统才会使用新的知识库；
- 重建过程中请不要关闭终端窗口。

---

## 七、常见问题

1. **双击“启动助手.bat”后没有打开页面怎么办？**
    - 请稍候几秒钟；
    - 如果迟迟没有启动浏览器，
    - 若浏览器未自动打开，可能是ssh映射网络不畅，请先手动访问本地链接，待网络环境好转后再试：
      ```
      http://127.0.0.1:7860
      ```  
    - 若仍无法访问，请查看 `logs` 目录中的日志文件。

2. **启动时报错“未找到 config/settings.yaml”怎么办？**
    - 请确认 release 根目录下存在 `config\settings.yaml`；
    - 不要只复制 exe 文件，必须保持整个 release 目录结构完整。

3. **修改了 `data\raw` 中的文件，但问答结果没有变化怎么办？**
    - 请确认已执行“重建知识库.bat”；
    - 请在重建完成后重新启动主程序。

4. **启动时报 API Key 或接口错误怎么办？**
    - 请检查 `.env` 中 `DASHSCOPE_API_KEY` 是否正确；
    - 请检查 `DASHSCOPE_BASE_URL` 是否为：
      ```
      https://dashscope.aliyuncs.com/compatible-mode/v1
      ```

5. **程序运行失败如何排查？**
    - 优先查看 `logs` 目录中的最新日志文件；
    - 确认 config、data、.env 是否都位于 release 根目录；
    - 确认不要直接移动 `EviBank-RAG.exe` 到其他目录单独运行。

---

## 八、使用建议

1. 建议将整个 release 目录放置在本地磁盘路径下，例如：
   ```
   D:\EviBank-RAG\
   ```
2. 不建议放在桌面临时目录、聊天软件缓存目录或只读目录中；
3. 不建议随意修改目录层级；
4. 若需长期维护，请保留 logs 目录以便排查问题。

---

## 九、说明

本版本为面向企业网银常见问题演示场景的本地发布版。  
系统依赖本地知识库与云端模型接口共同完成问答与证据展示。

如需更新功能、修复问题或重新打包，请提交issue。

## 十、(附加内容,面向源码使用者)自行打包说明<a id="section1"></a>

**说明**：  
由于 `EviBank-RAG\` 与 `EviBank-RAG-Rebuild\` 目录中包含较多打包生成文件，不适合完整提交到 GitHub 仓库中。因此，仓库中通常仅保留源码、配置文件、
`.spec` 文件及必要脚本；如需获得可执行程序，使用者可以根据以下步骤自行打包。

本项目根目录下已提供以下两个 PyInstaller 配置文件：

- `EviBank-RAG.spec`
- `EviBank-RAG-Rebuild.spec`

它们分别用于：

- 主程序打包（问答服务与前端界面）
- 知识库重建程序打包

---

### 10.1 打包前准备

1. 请先确保已获取完整源码仓库；
2. 请在本机安装 Python，并创建虚拟环境（建议与项目开发环境保持一致）；
3. 请在项目根目录下安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

4. 请确认已安装 PyInstaller：

   ```bash
   pip install pyinstaller
   ```

5. 请确认项目根目录中存在以下内容：

    - `EviBank-RAG.spec`
    - `EviBank-RAG-Rebuild.spec`
    - `app\`
    - `scripts\`
    - `config\`
    - `data\`
    - `main.py`

**说明**：

- 若缺少 `.spec` 文件，则不能直接按本说明打包；
- 若缺少 `config` 或 `data` 目录，程序即使打包成功，运行时也可能无法正常工作。

---

### 10.2 建议的打包目录约定

为保持项目目录清晰，建议使用以下目录约定：

- `release\`                最终可执行程序输出目录
- `release_build\`          PyInstaller 中间构建目录

**说明**：

- `release\` 中存放最终生成的可执行程序目录；
- `release_build\` 为中间构建目录，可在打包后保留，也可按需删除；
- 不建议将 `build\`、`dist\` 与 `release\` 混用，以免混淆最终产物与中间文件。

---

### 10.3 打包前清理旧产物

若你之前已经执行过打包，建议先删除旧的输出目录，避免旧文件残留导致运行异常。

建议至少删除以下目录（若存在）：

- `release\EviBank-RAG\`
- `release\EviBank-RAG-Rebuild\`
- `release_build\`

也可以删除旧的 `build\` / `dist\` 目录（若此前使用过默认输出路径）。

---

### 10.4 打包主程序

在项目根目录下执行以下命令：

```bash
pyinstaller --noconfirm --clean --distpath release --workpath release_build EviBank-RAG.spec
```

**说明**：

- `--distpath release`  
  表示最终打包产物输出到 `release\` 目录下；
- `--workpath release_build`  
  表示中间构建文件输出到 `release_build\` 目录下；
- `EviBank-RAG.spec`  
  为主程序的打包配置文件。

打包完成后，主程序目录应为：

```
release\EviBank-RAG\
```

其中应至少包含：

```
release\EviBank-RAG\EviBank-RAG.exe
```

---

### 10.5 打包知识库重建程序

在项目根目录下执行以下命令：

```bash
pyinstaller --noconfirm --clean --distpath release --workpath release_build EviBank-RAG-Rebuild.spec
```

**说明**：

- 该命令用于构建知识库重建程序；
- 运行后会生成独立的 Builder 程序目录。

打包完成后，重建程序目录应为：

```
release\EviBank-RAG-Rebuild\
```

其中应至少包含：

```
release\EviBank-RAG-Rebuild\EviBank-RAG-Rebuild.exe
```

---

### 10.6 打包完成后的目录检查

成功打包后，release 目录结构应类似如下：

```
release
├─ EviBank-RAG
│  └─ EviBank-RAG.exe
├─ EviBank-RAG-Rebuild
│  └─ EviBank-RAG-Rebuild.exe
├─ config
├─ data
├─ .env
├─ .env.example
├─ logs
├─ 启动助手.bat
├─ 重建知识库.bat
└─ README_使用说明.txt
```

**注意**：

1. `release\EviBank-RAG\` 和 `release\EviBank-RAG-Rebuild\` 为打包生成目录；
2. `config\`、`data\`、`.env`、`logs` 等仍应位于 release 根目录；
3. 不要仅复制 exe 文件单独运行，必须保持目录结构完整。

---

### 10.7 如何验证打包结果

建议不要一开始直接双击 exe，而应先在终端中测试运行。

1. **测试主程序**：

   ```bash
   cd release\EviBank-RAG
   EviBank-RAG.exe
   ```

   若启动成功，程序通常会启动本地服务，并在浏览器中打开页面。  
   若浏览器未自动打开，请手动访问：

   ```
   http://127.0.0.1:7860
   ```

2. **测试知识库重建程序**：

   ```bash
   cd release\EviBank-RAG-Rebuild
   EviBank-RAG-Rebuild.exe
   ```

   若执行正常，程序会扫描 `release\data\raw\` 中的知识文件，并重建 `release\data\parsed\` 与 `release\data\index\` 中的内容。

**说明**：

- 使用终端运行可以保留错误信息，便于定位打包问题；
- 若直接双击 exe，某些错误可能只会“一闪而过”，不便排查。

---

### 10.8 打包后常见问题

1. **启动时报“未找到 `config/settings.yaml`”**  
   **原因**：
    - 程序运行目录缺少 `config\settings.yaml`；
    - 或 release 目录结构不完整。  
      **处理方法**：
    - 确认 `config\` 目录位于 release 根目录；
    - 不要只复制 exe 文件单独运行。

2. **启动时报缺少 `gradio_client/types.json` 或 `safehttpx/version.txt`**  
   **原因**：
    - 依赖资源文件未被正确收集到打包产物中。  
      **处理方法**：
    - 请优先使用仓库自带的 `.spec` 文件重新打包；
    - 不建议改用临时、简化的 pyinstaller 命令；
    - 若问题仍存在，请检查 `.spec` 文件与 hooks 是否完整。

3. **`Builder`报找不到`scripts.`模块**
   **原因**：
    - 打包环境未正确包含 `scripts` 包；
    - 或使用了不完整的打包配置。  
      **处理方法**：
    - 请确认使用的是项目根目录下提供的 `EviBank-RAG-Rebuild.spec`；
    - 请不要自行省略 `.spec` 中的依赖收集配置。

4. **修改了 `data\raw` 后，系统内容没有变化**  
   **原因**：
    - 修改原始文件后未重新运行 Builder。  
      **处理方法**：
    - 运行 `release\EviBank-RAG-Rebuild\EviBank-RAG-Rebuild.exe`；
    - 或双击“重建知识库.bat”。

---

### 10.9 推荐的打包后使用方式

打包完成并验证无误后，建议按以下方式使用：

1. 保留整个 release 目录；
2. 在 release 根目录下维护：
    - `config\`
    - `data\`
    - `.env`
    - `logs\`
3. 使用：
    - `启动助手.bat` 启动主程序
    - `重建知识库.bat` 重建知识库

**说明**：

- 若只是修改 API Key，只需编辑 `.env` 后重新启动主程序；
- 若修改了 `data\raw` 中的原始知识文件，则需先重建知识库，再启动主程序。

---

### 10.10 补充说明

1. `release\EviBank-RAG\` 与 `release\EviBank-RAG-Rebuild\` 属于打包生成目录，文件数量较多，通常不建议完整提交到 GitHub
   仓库；
2. 仓库中保留源码、配置模板、`.spec` 文件和必要脚本，便于他人在本地自行构建；
3. 若仅用于内部演示分发，建议由维护者打包完成后，直接发送整个 release 压缩包给使用者；
4. 若面向开发者共享源码，则建议同时保留本“自行打包说明”，便于其他用户从源码恢复可执行版本。