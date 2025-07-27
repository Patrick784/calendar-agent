#!/bin/bash

# Multi-Agent Calendar System Setup Script
# This script automates the setup process where possible

echo "🚀 Multi-Agent Calendar System Setup"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
else
    print_status "Python 3 found: $(python3 --version)"
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is required but not installed"
    exit 1
else
    print_status "pip3 found: $(pip3 --version)"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv env
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Create necessary directories
print_info "Creating required directories..."
mkdir -p models logs chroma_db
print_status "Directories created: models/, logs/, chroma_db/"

# Copy environment template if .env doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.template ]; then
        cp .env.template .env
        print_status "Created .env file from template"
        print_warning "IMPORTANT: Edit .env file with your actual API keys!"
    else
        print_error ".env.template not found"
    fi
else
    print_status ".env file already exists"
fi

# Check for critical files
print_info "Checking for required files..."

if [ -f credentials.json ]; then
    print_status "Found Google Calendar credentials.json"
else
    print_error "Missing credentials.json"
    echo "  Please download from: https://console.cloud.google.com/"
    echo "  1. Create a new project or select existing"
    echo "  2. Enable Google Calendar API"
    echo "  3. Create OAuth 2.0 credentials"
    echo "  4. Download as credentials.json"
fi

# Check .env file content
if [ -f .env ]; then
    if grep -q "your_openai_api_key_here" .env; then
        print_error "OpenAI API key not set in .env file"
        echo "  Please edit .env and add your actual OpenAI API key"
    else
        print_status "OpenAI API key appears to be set"
    fi
fi

# Install Python dependencies
print_info "Installing Python dependencies..."
if [ -f requirements.txt ]; then
    # Activate virtual environment and install
    source env/bin/activate 2>/dev/null || source env/Scripts/activate 2>/dev/null
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        print_status "Dependencies installed successfully"
    else
        print_error "Failed to install dependencies"
    fi
else
    print_error "requirements.txt not found"
fi

# Test scripts setup
print_status "Test scripts ready: test_openai.py, test_google_auth.py"

echo ""
echo "🎯 Setup Summary"
echo "================"

# Check what's ready and what needs manual setup
checks_passed=0
total_checks=4

echo "Checking setup status..."

if [ -f .env ] && ! grep -q "your_openai_api_key_here" .env; then
    print_status "OpenAI API key configured"
    ((checks_passed++))
else
    print_error "OpenAI API key needs manual setup"
fi

if [ -f credentials.json ]; then
    print_status "Google Calendar credentials ready"
    ((checks_passed++))
else
    print_error "Google Calendar credentials need manual setup"
fi

if [ -d env ] && [ -f requirements.txt ]; then
    print_status "Python environment ready"
    ((checks_passed++))
else
    print_error "Python environment needs attention"
fi

if [ -d models ] && [ -d logs ] && [ -d chroma_db ]; then
    print_status "Directories created"
    ((checks_passed++))
else
    print_error "Directory setup incomplete"
fi

echo ""
echo "Setup Progress: $checks_passed/$total_checks checks passed"

if [ $checks_passed -eq $total_checks ]; then
    echo ""
    print_status "🎉 Setup complete! Ready to test."
    echo ""
    echo "Next steps:"
    echo "1. Run: python test_openai.py"
    echo "2. Run: python test_google_auth.py"  
    echo "3. Run: streamlit run main.py"
else
    echo ""
    print_warning "Setup partially complete. Please address the issues above."
    echo ""
    echo "Manual steps remaining:"
    if ! [ -f .env ] || grep -q "your_openai_api_key_here" .env; then
        echo "• Edit .env file with your OpenAI API key"
    fi
    if ! [ -f credentials.json ]; then
        echo "• Download credentials.json from Google Cloud Console"
    fi
fi

echo ""
echo "For help, see: README.md" 