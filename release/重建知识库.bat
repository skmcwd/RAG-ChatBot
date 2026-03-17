@echo off
setlocal

cd /d "%~dp0"

echo ==========================================
echo   EviBank-RAG 知识库重建工具
echo ==========================================
echo.

if not exist ".env" (
    echo [错误] 当前目录下未找到 .env 文件。
    echo 请先根据 .env.example 创建并填写 API Key。
    echo.
    pause
    exit /b 1
)

if not exist "config\settings.yaml" (
    echo [错误] 未找到配置文件：config\settings.yaml
    echo 请确认 release 目录结构完整。
    echo.
    pause
    exit /b 1
)

if not exist "data\raw" (
    echo [错误] 未找到原始知识库目录：data\raw
    echo 请确认原始知识文件已放入该目录。
    echo.
    pause
    exit /b 1
)

if not exist "EviBank-RAG-Rebuild\EviBank-RAG-Rebuild.exe" (
    echo [错误] 未找到重建程序：EviBank-RAG-Rebuild\EviBank-RAG-Rebuild.exe
    echo 请确认程序文件已正确放置。
    echo.
    pause
    exit /b 1
)

if not exist "logs" (
    mkdir "logs"
)

echo [信息] 即将启动知识库重建程序。
echo [说明] 若你已替换 data\raw 中的文件，请等待重建完成后再启动主程序。
echo.

cd /d "%~dp0EviBank-RAG-Rebuild"
"EviBank-RAG-Rebuild.exe"

echo.
echo [信息] 若终端未出现错误信息，则知识库重建流程已执行完成。
echo [建议] 完成后请返回 release 根目录，双击“启动助手.bat”启动系统。
echo.
pause

endlocal
exit /b 0