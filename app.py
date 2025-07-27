import os
import datetime
from dotenv import load_dotenv
import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import openai
import re
import sys
import traceback
import locale

# Force ASCII encoding for HTTP headers
os.environ['PYTHONIOENCODING'] = 'ascii'
os.environ['LC_ALL'] = 'C'

# Force UTF-8 encoding for all operations
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

def safe_display(msg):
    """Safely display a message, handling any encoding issues"""
    try:
        safe_msg = safe_string(msg)
        return safe_msg
    except Exception as e:
        return f"Error displaying message: {str(e)}"

# Helper: get Google creds
def get_creds():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRETS, 
            SCOPES,
            redirect_uri='http://localhost:8501/oauth2callback'
        )
        creds = flow.run_local_server(port=8501, prompt='consent')
        with open(TOKEN_PATH, 'w', encoding='utf-8') as token:
            token.write(creds.to_json())
    return creds

# Build Calendar service
creds = get_creds()
service = build('calendar', 'v3', credentials=creds)

st.title("Calendar Agent 🚀")
st.markdown("---")

# Inject animated rainbow border CSS for the chat input
st.markdown("""
    <style>
    @keyframes rainbow {
        0% { border-image-source: linear-gradient(270deg, #ff0080, #7928ca, #007cf0, #00dfd8, #ff0080); }
        25% { border-image-source: linear-gradient(270deg, #7928ca, #007cf0, #00dfd8, #ff0080, #ff0080); }
        50% { border-image-source: linear-gradient(270deg, #007cf0, #00dfd8, #ff0080, #7928ca, #007cf0); }
        75% { border-image-source: linear-gradient(270deg, #00dfd8, #ff0080, #7928ca, #007cf0, #00dfd8); }
        100% { border-image-source: linear-gradient(270deg, #ff0080, #7928ca, #007cf0, #00dfd8, #ff0080); }
    }
    div[data-testid="stTextInput"] > div > input {
        border: 3px solid;
        border-image: linear-gradient(270deg, #ff0080, #7928ca, #007cf0, #00dfd8, #ff0080) 1;
        animation: rainbow 3s linear infinite;
        box-shadow: 0 0 8px 2px #007cf033;
    }
    </style>
""", unsafe_allow_html=True)

# Unified chat box
st.subheader("Chat with your calendar agent:")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Type your request (add, check, retrieve, etc.):", "")
if st.button("Send") and user_input:
    try:
        # Intent detection with OpenAI (v1.x SDK with OpenRouter support)
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE
        )
        prompt = f"""
You are a smart calendar assistant. Given the user's request, extract the intent as one of: [add_event, check_availability, retrieve_past_events, other].
If the intent is retrieve_past_events, extract a time range (start and end date in ISO format) and any keyword (e.g., 'birthday', 'dentist').
If the intent is add_event, extract title, start, and end datetime (ISO format if possible).
If the intent is ambiguous or missing info, reply with a clarifying question.
Return a JSON object with 'intent' and a 'details' dict. Example:
{{"intent": "retrieve_past_events", "details": {{"time_min": "2024-06-01T00:00:00Z", "time_max": "2024-07-01T00:00:00Z", "keyword": "birthday"}}}}
User: {safe_string(user_input)}
"""
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o",  # Updated for OpenRouter compatibility
                messages=[{"role": "system", "content": "You are a helpful calendar assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300
            )
            reply = safe_string(response.choices[0].message.content)
        except Exception as openai_error:
            st.session_state.chat_history.append(("assistant", f"OpenAI Error: {safe_string(str(openai_error))}"))
            reply = None
        
        if reply:
            import json
            try:
                parsed = json.loads(reply)
                intent = parsed.get('intent')
                details = parsed.get('details', {})
                
                if intent == 'retrieve_past_events':
                    time_min = details.get('time_min')
                    time_max = details.get('time_max')
                    keyword = details.get('keyword')
                    
                    if not time_min or not time_max:
                        now = datetime.datetime.utcnow()
                        time_min = (now - datetime.timedelta(days=7)).isoformat() + 'Z'
                        time_max = now.isoformat() + 'Z'
                    
                    try:
                        past_events_result = service.events().list(
                            calendarId='primary',
                            timeMin=time_min,
                            timeMax=time_max,
                            singleEvents=True,
                            orderBy='startTime'
                        ).execute()
                        
                        past_events = past_events_result.get('items', [])
                        
                        if keyword:
                            matches = [ev for ev in past_events if keyword.lower() in safe_string(ev.get('summary','')).lower()]
                            if matches:
                                summary_lines = []
                                for ev in matches:
                                    start = safe_string(ev['start'].get('dateTime', ev['start'].get('date')))
                                    summary = safe_string(ev.get('summary','No Title'))
                                    summary_lines.append(f"- {start}: {summary}")
                                summary = '\n'.join(summary_lines)
                                agent_reply = f"Here are your '{keyword}' events in the selected range:\n{summary}"
                            else:
                                agent_reply = f"No events found for '{keyword}' in the selected range."
                        else:
                            if not past_events:
                                agent_reply = "No events found in the selected range."
                            else:
                                summary_lines = []
                                for ev in past_events:
                                    start = safe_string(ev['start'].get('dateTime', ev['start'].get('date')))
                                    summary = safe_string(ev.get('summary','No Title'))
                                    summary_lines.append(f"- {start}: {summary}")
                                summary = '\n'.join(summary_lines)
                                agent_reply = f"Here are your events in the selected range:\n{summary}"
                        
                        st.session_state.chat_history.append(("assistant", agent_reply))
                    except Exception as calendar_error:
                        st.session_state.chat_history.append(("assistant", f"Calendar Error: {safe_string(str(calendar_error))}"))
                        
                elif intent == 'add_event':
                    title = safe_string(details.get('title', 'Untitled Event'))
                    start = details.get('start')
                    end = details.get('end')
                    if not start or not end:
                        st.session_state.chat_history.append(("assistant", "Please specify the start and end time for the event."))
                    else:
                        try:
                            event = {
                                'summary': title,
                                'start': {'dateTime': start},
                                'end': {'dateTime': end}
                            }
                            created = service.events().insert(calendarId='primary', body=event).execute()
                            link = safe_string(created.get('htmlLink', ''))
                            st.session_state.chat_history.append(("assistant", f"Created event: {link}"))
                        except Exception as add_error:
                            st.session_state.chat_history.append(("assistant", f"Error creating event: {safe_string(str(add_error))}"))
                            
                elif intent == 'check_availability':
                    st.session_state.chat_history.append(("assistant", "Checking your availability is not yet implemented."))
                else:
                    st.session_state.chat_history.append(("assistant", "I'm not sure what you want. Can you clarify?"))
                    
            except Exception as json_error:
                # If not JSON, treat as clarifying question
                st.session_state.chat_history.append(("assistant", safe_display(reply)))
                
    except Exception as e:
        error_msg = safe_string(str(e))
        st.session_state.chat_history.append(("assistant", f"Error: {error_msg}"))

# Only display agent replies
for role, msg in st.session_state.chat_history:
    if role == "assistant":
        safe_msg = safe_display(msg)
        st.markdown(f"**Agent:** {safe_msg}")

