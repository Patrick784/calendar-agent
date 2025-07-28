@echo off
REM Multi-Agent Calendar System Setup Script for Windows
REM This script automates setup, checks, and launches the agent UI

echo 🚀 Multi-Agent Calendar System Setup
echo =====================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is required but not installed
    echo Please install Python 3.10 from https://www.python.org/downloads/release/python-31011/
    pause
    exit /b 1
) else (
    echo ✅ Python found
)

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is required but not installed
    pause
    exit /b 1
) else (
    echo ✅ pip found
)

REM Create virtual environment if it doesn't exist
if not exist "env" (
    echo ℹ️ Creating virtual environment with Python 3.10...
    py -3.10 -m venv env
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Create necessary directories
echo ℹ️ Creating required directories...
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "chroma_db" mkdir chroma_db
echo ✅ Directories created: models/, logs/, chroma_db/

REM Copy environment template if .env doesn't exist
if not exist ".env" (
    if exist ".env.template" (
        copy ".env.template" ".env" >nul
        echo ✅ Created .env file from template
        echo ⚠️  IMPORTANT: Edit .env file with your actual API keys!
    ) else (
        echo ❌ .env.template not found
    )
) else (
    echo ✅ .env file already exists
)

REM Check for credentials.json
if exist "credentials.json" (
    echo ✅ Found Google Calendar credentials.json
) else (
    echo ❌ Missing credentials.json
    echo   Please download from: https://console.cloud.google.com/
    echo   Enable Google Calendar API → Create OAuth 2.0 → Download as credentials.json
)

REM Check .env file content
if exist ".env" (
    findstr /C:"your_openai_api_key_here" ".env" >nul
    if %errorlevel% equ 0 (
        echo ❌ OpenAI API key not set in .env file
        echo   Please edit .env and add your actual
