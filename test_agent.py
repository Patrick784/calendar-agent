import os
import datetime
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import openai
import sys
import json
import locale
import traceback

# Force ASCII encoding for HTTP headers
os.environ['PYTHONIOENCODING'] = 'ascii'
os.environ['LC_ALL'] = 'C'

# Force UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Set locale to C to avoid Unicode issues
try:
    locale.setlocale(locale.LC_ALL, 'C')
except:
    pass

# Load env vars
load_dotenv()
SCOPES = ['https://www.googleapis.com/auth/calendar']
CLIENT_SECRETS = 'credentials.json'
TOKEN_PATH = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://openrouter.ai/api/v1')  # Default to OpenRouter

class AdaptiveErrorHandler:
    """Learns from errors and provides specific fixes"""
    def __init__(self):
        self.error_patterns = {
            'invalid_api_key': {
                'pattern': 'invalid_api_key',
                'fix': 'Update OPENAI_API_KEY in .env with real API key from https://platform.openai.com/account/api-keys',
                'severity': 'high'
            },
            'credentials_not_found': {
                'pattern': 'credentials.json',
                'fix': 'Download credentials.json from Google Cloud Console',
                'severity': 'high'
            },
            'unicode_encoding': {
                'pattern': 'UnicodeEncodeError',
                'fix': 'Unicode characters detected - applied safe_string() fixes',
                'severity': 'medium'
            },
            'token_expired': {
                'pattern': 'token_expired',
                'fix': 'Delete token.json to force re-authentication',
                'severity': 'medium'
            }
        }
        self.fixes_applied = []
    
    def analyze_error(self, error_msg):
        """Analyze error and return recommended fix"""
        error_str = str(error_msg).lower()
        for error_type, info in self.error_patterns.items():
            if info['pattern'].lower() in error_str:
                return {
                    'type': error_type,
                    'fix': info['fix'],
                    'severity': info['severity'],
                    'auto_fixable': error_type in ['unicode_encoding']
                }
        return {
            'type': 'unknown',
            'fix': 'Unknown error - manual investigation needed',
            'severity': 'high',
            'auto_fixable': False
        }
    
    def apply_fix(self, error_analysis):
        """Apply automatic fixes where possible"""
        if error_analysis['auto_fixable']:
            self.fixes_applied.append(error_analysis['type'])
            return True
        return False

error_handler = AdaptiveErrorHandler()

def safe_string(s):
    """Safely handle any string, converting to UTF-8 if needed"""
    if s is None:
        return ""
    if isinstance(s, bytes):
        return s.decode('utf-8', errors='replace')
    # Convert to string and handle Unicode
    s = str(s)
    # Replace problematic Unicode characters
    replacements = {
        '\u2026': '...',    # ellipsis
        '\u2019': "'",      # right single quote
        '\u2018': "'",      # left single quote
        '\u201c': '"',      # left double quote
        '\u201d': '"',      # right double quote
        '\u2013': '-',      # en dash
        '\u2014': '--',     # em dash
        '\u00a0': ' ',      # non-breaking space
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    # Encode to ASCII, ignoring any remaining problematic characters
    s = s.encode('ascii', errors='ignore').decode('ascii')
    return s

def test_agent():
    print("=== Testing Calendar Agent ===")
    
    # Test 1: OpenAI API
    print("\n1. Testing OpenAI API...")
    try:
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE
        )
        test_query = "what birthday do i have coming up"
        prompt = f"""
You are a smart calendar assistant. Given the user's request, extract the intent as one of: [add_event, check_availability, retrieve_past_events, other].
If the intent is retrieve_past_events, extract a time range (start and end date in ISO format) and any keyword (e.g., 'birthday', 'dentist').
If the intent is add_event, extract title, start, and end datetime (ISO format if possible).
If the intent is ambiguous or missing info, reply with a clarifying question.
Return a JSON object with 'intent' and a 'details' dict. Example:
{{"intent": "retrieve_past_events", "details": {{"time_min": "2024-06-01T00:00:00Z", "time_max": "2024-07-01T00:00:00Z", "keyword": "birthday"}}}}
User: {safe_string(test_query)}
"""
        response = client.chat.completions.create(
            model="openai/gpt-4o",  # Updated for OpenRouter compatibility
            messages=[{"role": "system", "content": "You are a helpful calendar assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        reply = safe_string(response.choices[0].message.content)
        print(f"OpenAI Response: {reply}")
        
        # Test 2: Parse JSON
        print("\n2. Testing JSON parsing...")
        parsed = json.loads(reply)
        intent = parsed.get('intent')
        details = parsed.get('details', {})
        print(f"Intent: {intent}")
        print(f"Details: {details}")
        
        # Test 3: Google Calendar API
        print("\n3. Testing Google Calendar API...")
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        service = build('calendar', 'v3', credentials=creds)
        
        time_min = details.get('time_min')
        time_max = details.get('time_max')
        keyword = details.get('keyword')
        
        if not time_min or not time_max:
            now = datetime.datetime.utcnow()
            time_min = (now - datetime.timedelta(days=7)).isoformat() + 'Z'
            time_max = now.isoformat() + 'Z'
        
        print(f"Searching from {time_min} to {time_max} for keyword: {keyword}")
        
        past_events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        past_events = past_events_result.get('items', [])
        print(f"Found {len(past_events)} events")
        
        if keyword:
            matches = [ev for ev in past_events if keyword.lower() in safe_string(ev.get('summary','')).lower()]
            print(f"Found {len(matches)} events matching '{keyword}'")
            for ev in matches:
                start = safe_string(ev['start'].get('dateTime', ev['start'].get('date')))
                summary = safe_string(ev.get('summary','No Title'))
                print(f"- {start}: {summary}")
        
        print("\n=== Test completed successfully! ===")
        
    except Exception as e:
        print(f"ERROR: {e}")
        
        # Adaptive error handling
        error_analysis = error_handler.analyze_error(str(e))
        print(f"\n🔍 Error Analysis:")
        print(f"   Type: {error_analysis['type']}")
        print(f"   Severity: {error_analysis['severity']}")
        print(f"   Fix: {error_analysis['fix']}")
        print(f"   Auto-fixable: {error_analysis['auto_fixable']}")
        
        if error_analysis['auto_fixable']:
            print("   ✅ Applying automatic fix...")
            error_handler.apply_fix(error_analysis)
        
        print(f"\n📊 Fixes Applied This Session: {len(error_handler.fixes_applied)}")
        for fix in error_handler.fixes_applied:
            print(f"   - {fix}")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent() 