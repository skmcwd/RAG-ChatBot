@echo off
setlocal

cd /d "%~dp0"

echo ==========================================
echo   EviBank-RAG 启动助手
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

if not exist "EviBank-RAG\EviBank-RAG.exe" (
    echo [错误] 未找到主程序：EviBank-RAG\EviBank-RAG.exe
    echo 请确认程序文件已正确放置。
    echo.
    pause
    exit /b 1
)

if not exist "logs" (
    mkdir "logs"
)

echo [信息] 正在启动 EviBank-RAG ...
echo [信息] 若浏览器未自动打开，请稍后手动访问：
echo         http://127.0.0.1:7860
echo.

cd /d "%~dp0EviBank-RAG"
start "" "EviBank-RAG.exe"

endlocal
exit /b 0