@echo off
REM Multi-Agent Calendar System Setup Script for Windows
REM This script automates the setup process where possible

echo 🚀 Multi-Agent Calendar System Setup
echo =====================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is required but not installed
    echo Please install Python 3.9+ from https://python.org
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
    echo ℹ️ Creating virtual environment...
    python -m venv env
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
        echo ⚠️ IMPORTANT: Edit .env file with your actual API keys!
    ) else (
        echo ❌ .env.template not found
    )
) else (
    echo ✅ .env file already exists
)

REM Check for critical files
echo ℹ️ Checking for required files...

if exist "credentials.json" (
    echo ✅ Found Google Calendar credentials.json
) else (
    echo ❌ Missing credentials.json
    echo   Please download from: https://console.cloud.google.com/
    echo   1. Create a new project or select existing
    echo   2. Enable Google Calendar API
    echo   3. Create OAuth 2.0 credentials
    echo   4. Download as credentials.json
)

REM Check .env file content
if exist ".env" (
    findstr /C:"your_openai_api_key_here" ".env" >nul
    if %errorlevel% equ 0 (
        echo ❌ OpenAI API key not set in .env file
        echo   Please edit .env and add your actual OpenAI API key
    ) else (
        echo ✅ OpenAI API key appears to be set
    )
)

REM Install Python dependencies
echo ℹ️ Installing Python dependencies...
if exist "requirements.txt" (
    REM Activate virtual environment and install
    call env\Scripts\activate.bat
    pip install -r requirements.txt
    if %errorlevel% equ 0 (
        echo ✅ Dependencies installed successfully
    ) else (
        echo ❌ Failed to install dependencies
    )
) else (
    echo ❌ requirements.txt not found
)

REM Test scripts setup
echo ✅ Test scripts ready: test_openai.py, test_google_auth.py

echo.
echo 🎯 Setup Summary
echo ================

REM Check what's ready and what needs manual setup
set /a checks_passed=0
set /a total_checks=4

echo Checking setup status...

if exist ".env" (
    findstr /C:"your_openai_api_key_here" ".env" >nul
    if %errorlevel% neq 0 (
        echo ✅ OpenAI API key configured
        set /a checks_passed+=1
    ) else (
        echo ❌ OpenAI API key needs manual setup
    )
) else (
    echo ❌ OpenAI API key needs manual setup
)

if exist "credentials.json" (
    echo ✅ Google Calendar credentials ready
    set /a checks_passed+=1
) else (
    echo ❌ Google Calendar credentials need manual setup
)

if exist "env" (
    if exist "requirements.txt" (
        echo ✅ Python environment ready
        set /a checks_passed+=1
    ) else (
        echo ❌ Python environment needs attention
    )
) else (
    echo ❌ Python environment needs attention
)

if exist "models" (
    if exist "logs" (
        if exist "chroma_db" (
            echo ✅ Directories created
            set /a checks_passed+=1
        ) else (
            echo ❌ Directory setup incomplete
        )
    ) else (
        echo ❌ Directory setup incomplete
    )
) else (
    echo ❌ Directory setup incomplete
)

echo.
echo Setup Progress: %checks_passed%/%total_checks% checks passed

if %checks_passed% equ %total_checks% (
    echo.
    echo ✅ 🎉 Setup complete! Ready to test.
    echo.
    echo Next steps:
    echo 1. Run: python test_openai.py
    echo 2. Run: python test_google_auth.py
    echo 3. Run: streamlit run main.py
) else (
    echo.
    echo ⚠️ Setup partially complete. Please address the issues above.
    echo.
    echo Manual steps remaining:
    if not exist ".env" echo • Create .env file with your OpenAI API key
    if not exist "credentials.json" echo • Download credentials.json from Google Cloud Console
)

echo.
echo For help, see: README.md
pause 