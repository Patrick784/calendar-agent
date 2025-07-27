#!/usr/bin/env python3
"""
Setup Status Checker

This script checks your Multi-Agent Calendar System setup and tells you
exactly what's missing and what needs to be fixed.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def print_status(message, status="info"):
    """Print colored status messages"""
    colors = {
        "success": "\033[92m✅",
        "error": "\033[91m❌", 
        "warning": "\033[93m⚠️",
        "info": "\033[94mℹ️"
    }
    end_color = "\033[0m"
    print(f"{colors.get(status, colors['info'])} {message}{end_color}")

def check_python_environment():
    """Check Python environment setup"""
    print("\n🐍 Checking Python Environment...")
    
    # Check Python version
    if sys.version_info >= (3, 9):
        print_status(f"Python version: {sys.version.split()[0]}", "success")
    else:
        print_status(f"Python {sys.version.split()[0]} - Need 3.9+", "error")
        return False
    
    # Check if we're in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_status("Virtual environment: Active", "success")
    else:
        print_status("Virtual environment: Not active (recommended)", "warning")
    
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n📦 Checking Dependencies...")
    
    required_packages = [
        'streamlit', 'openai', 'google-api-python-client', 
        'google-auth-oauthlib', 'python-dotenv', 'psycopg2-binary',
        'redis', 'chromadb', 'scikit-learn', 'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print_status(f"{package}: Installed", "success")
        except ImportError:
            print_status(f"{package}: Missing", "error") 
            missing.append(package)
    
    if missing:
        print_status(f"Run: pip install {' '.join(missing)}", "info")
        return False
    
    return True

def check_environment_file():
    """Check .env file configuration"""
    print("\n🔧 Checking Environment Configuration...")
    
    if not os.path.exists('.env'):
        print_status(".env file: Missing", "error")
        if os.path.exists('.env.template'):
            print_status("Run: cp .env.template .env", "info")
        return False
    
    print_status(".env file: Found", "success")
    
    # Load and check environment variables
    load_dotenv()
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print_status("OPENAI_API_KEY: Not set", "error")
        return False
    elif openai_key == "your_openai_api_key_here":
        print_status("OPENAI_API_KEY: Placeholder value - needs real key", "error")
        return False
    elif len(openai_key) < 20:
        print_status("OPENAI_API_KEY: Too short - check format", "error") 
        return False
    else:
        print_status("OPENAI_API_KEY: Configured", "success")
    
    # Check other important settings
    settings_to_check = [
        ('USER_TIMEZONE', 'UTC'),
        ('GOOGLE_CREDENTIALS_FILE', 'credentials.json'),
        ('GOOGLE_TOKEN_FILE', 'token.json')
    ]
    
    for setting, default in settings_to_check:
        value = os.getenv(setting, default)
        print_status(f"{setting}: {value}", "success")
    
    return True

def check_google_credentials():
    """Check Google Calendar API setup"""
    print("\n📅 Checking Google Calendar Setup...")
    
    creds_file = os.getenv('GOOGLE_CREDENTIALS_FILE', 'credentials.json')
    
    if not os.path.exists(creds_file):
        print_status(f"{creds_file}: Missing", "error")
        print("  Get credentials from: https://console.cloud.google.com/")
        print("  1. Create project & enable Calendar API")
        print("  2. Create OAuth 2.0 credentials")
        print("  3. Download as credentials.json")
        return False
    
    print_status(f"{creds_file}: Found", "success")
    
    # Check if token.json exists and validate calendar connection
    token_file = os.getenv('GOOGLE_TOKEN_FILE', 'token.json')
    
    calendar_auth_status = "unknown"
    try:
        from adapters.google_calendar import GoogleCalendarAdapter
        
        # Test Google Calendar adapter
        google_config = {
            "client_secrets_file": creds_file,
            "token_file": token_file
        }
        adapter = GoogleCalendarAdapter(google_config)
        auth_status = adapter.get_auth_status()
        
        if os.path.exists(token_file) and not auth_status["needs_reauth"]:
            print_status(f"{token_file}: Found (testing connection...)", "success")
            
            # Test calendar list fetch
            try:
                import asyncio
                calendars = asyncio.run(adapter._get_calendars())
                print_status(f"Calendar Access: Connected - found {calendars['count']} calendars", "success")
                calendar_auth_status = "connected"
            except Exception as e:
                error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
                print_status(f"Calendar Access: Auth expired - {error_msg}", "warning")
                print("  Run: python test_google_auth.py")
                calendar_auth_status = "auth_expired"
                
        else:
            print_status(f"{token_file}: Not found (authentication needed)", "warning")
            print("  Run: python test_google_auth.py")
            calendar_auth_status = "not_authenticated"
            
    except ImportError as e:
        print_status(f"Google Calendar: Import error - {e}", "error")
        calendar_auth_status = "import_error"
    except Exception as e:
        error_msg = str(e)[:60] + "..." if len(str(e)) > 60 else str(e)
        print_status(f"Google Calendar: {error_msg}", "warning")
        print("  Run: python test_google_auth.py")
        calendar_auth_status = "error"
    
    return calendar_auth_status in ["connected", "not_authenticated"]

def check_directories():
    """Check required directories exist"""
    print("\n📁 Checking Directories...")
    
    required_dirs = ['models', 'logs', 'chroma_db', 'agents', 'adapters']
    all_exist = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print_status(f"{directory}/: Exists", "success")
        else:
            print_status(f"{directory}/: Missing", "error")
            all_exist = False
    
    return all_exist

def check_optional_services():
    """Check optional database services"""
    print("\n🗄️ Checking Optional Services...")
    
    # Check PostgreSQL
    postgres_url = os.getenv('POSTGRES_URL')
    if postgres_url:
        print_status("PostgreSQL: Configured", "success")
        # Could add connection test here
    else:
        print_status("PostgreSQL: Not configured (will use in-memory)", "warning")
    
    # Check Redis
    redis_url = os.getenv('REDIS_URL')  
    if redis_url:
        print_status("Redis: Configured", "success")
        # Could add connection test here
    else:
        print_status("Redis: Not configured (will use in-memory)", "warning")
    
    return True

def check_core_files():
    """Check that core system files exist"""
    print("\n📄 Checking Core Files...")
    
    core_files = [
        'main.py',
        'requirements.txt', 
        'agents/__init__.py',
        'agents/orchestrator.py',
        'agents/nlu_parser.py',
        'adapters/google_calendar.py'
    ]
    
    all_exist = True
    for file_path in core_files:
        if os.path.exists(file_path):
            print_status(f"{file_path}: Found", "success")
        else:
            print_status(f"{file_path}: Missing", "error")
            all_exist = False
    
    return all_exist

def main():
    """Main setup checker"""
    print("🔍 MULTI-AGENT CALENDAR SYSTEM - SETUP CHECKER")
    print("=" * 60)
    
    checks = [
        ("Python Environment", check_python_environment),
        ("Dependencies", check_dependencies),
        ("Environment File", check_environment_file), 
        ("Google Credentials", check_google_credentials),
        ("Directories", check_directories),
        ("Core Files", check_core_files),
        ("Optional Services", check_optional_services)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_status(f"Error checking {name}: {e}", "error")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SETUP SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "success" if result else "error"
        print_status(f"{name}: {'✓' if result else '✗'}", status)
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print_status("\n🎉 All checks passed! System ready to run.", "success")
        print("\nNext steps:")
        print("1. python test_openai.py")
        print("2. python test_google_auth.py") 
        print("3. streamlit run main.py")
    else:
        print_status(f"\n⚠️ {total - passed} issues need attention", "warning")
        print("\nRecommended actions:")
        
        if not results[1][1]:  # Dependencies
            print("• Run: pip install -r requirements.txt")
        if not results[2][1]:  # Environment file
            print("• Edit .env with your OpenAI API key")
        if not results[3][1]:  # Google credentials
            print("• Download credentials.json from Google Cloud Console")
        
        print("\nRun this script again after fixing issues.")

if __name__ == '__main__':
    main() 