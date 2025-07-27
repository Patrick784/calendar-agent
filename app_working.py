import os
import datetime
import streamlit as st
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load env vars
load_dotenv()
SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_PATH = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')

# Helper: get Google creds
def get_creds():
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    return creds

# Build Calendar service
creds = get_creds()
service = build('calendar', 'v3', credentials=creds)

def extract_keywords_and_intent(user_input):
    """Smart keyword extraction with priority system"""
    user_input = user_input.lower()
    
    # Define intent patterns with priority (higher number = higher priority)
    intent_patterns = {
        'flight': (['flight', 'flights', 'airplane', 'travel', 'trip', 'departure', 'arrival', 'dl ', 'delta', 'american', 'united'], 10),
        'birthday': (['birthday', 'birthdays', 'born', 'birth'], 9),
        'appointment': (['appointment', 'doctor', 'dentist', 'meeting', 'visit'], 8),
        'work': (['work', 'job', 'office', 'conference', 'presentation'], 7),
        'health': (['doctor', 'dentist', 'hospital', 'checkup', 'medical'], 8),
        'personal': (['dinner', 'lunch', 'party', 'date', 'hangout'], 6),
        'general': (['event', 'events', 'schedule', 'calendar', 'what', 'show'], 1)
    }
    
    # Month mapping
    months = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
        'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
        'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
    }
    
    # Find the highest priority intent
    best_keywords = []
    highest_priority = 0
    
    for intent, (keywords, priority) in intent_patterns.items():
        matched_keywords = [kw for kw in keywords if kw in user_input]
        if matched_keywords and priority > highest_priority:
            best_keywords = matched_keywords
            highest_priority = priority
    
    # Find month
    target_month = None
    for month_name, month_num in months.items():
        if month_name in user_input:
            target_month = month_num
            break
    
    # Determine time context
    is_future = any(word in user_input for word in ['coming', 'upcoming', 'future', 'next'])
    is_past = any(word in user_input for word in ['past', 'previous', 'last', 'had'])
    
    return best_keywords, target_month, is_future, is_past

def get_date_range(target_month, is_future, is_past):
    """Calculate precise date range based on context"""
    now = datetime.datetime.now(datetime.timezone.utc)
    current_year = now.year
    
    if target_month:
        # Specific month query
        if is_future or (not is_past and target_month >= now.month):
            year = current_year if target_month >= now.month else current_year + 1
        else:
            year = current_year if target_month <= now.month else current_year - 1
        
        start_date = datetime.datetime(year, target_month, 1, tzinfo=datetime.timezone.utc)
        if target_month == 12:
            end_date = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc) - datetime.timedelta(seconds=1)
        else:
            end_date = datetime.datetime(year, target_month + 1, 1, tzinfo=datetime.timezone.utc) - datetime.timedelta(seconds=1)
        
        return start_date.isoformat(), end_date.isoformat(), f"{start_date.strftime('%B %Y')}"
    
    elif is_past:
        start_date = now - datetime.timedelta(days=30)
        return start_date.isoformat(), now.isoformat(), "past 30 days"
    else:
        end_date = now + datetime.timedelta(days=30)
        return now.isoformat(), end_date.isoformat(), "next 30 days"

def filter_events_by_keywords(events, keywords):
    """Filter events using keyword matching"""
    if not keywords or not events:
        return events
    
    relevant_events = []
    for event in events:
        event_text = (event.get('summary', '') + ' ' + event.get('description', '')).lower()
        if any(keyword.lower() in event_text for keyword in keywords):
            relevant_events.append(event)
    
    return relevant_events

st.title("🧠 Smart Calendar Agent")
st.markdown("*Intelligent keyword matching & conversation memory*")
st.markdown("---")

# Initialize conversation state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat interface
st.subheader("Chat with your calendar agent:")

user_input = st.text_input("Type your request:", key="user_input")
if st.button("Send") and user_input:
    # Clear previous responses for new conversation
    st.session_state.chat_history = []
    
    try:
        keywords, target_month, is_future, is_past = extract_keywords_and_intent(user_input)
        time_min, time_max, time_desc = get_date_range(target_month, is_future, is_past)
        
        # Get events from calendar
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        # Smart filtering based on keywords
        if keywords:
            filtered_events = filter_events_by_keywords(events, keywords)
            keyword_desc = ', '.join(keywords)
        else:
            filtered_events = events
            keyword_desc = "events"
        
        # Format response
        if filtered_events:
            summary_lines = []
            for ev in filtered_events:
                start = ev['start'].get('dateTime', ev['start'].get('date'))
                if 'T' in start:
                    try:
                        event_date = datetime.datetime.fromisoformat(start.replace('Z', '+00:00'))
                        formatted_date = event_date.strftime('%Y-%m-%d %H:%M')
                    except:
                        formatted_date = start
                else:
                    formatted_date = start
                summary = ev.get('summary', 'No Title')
                summary_lines.append(f"• {formatted_date}: {summary}")
            
            summary = '\n'.join(summary_lines)
            agent_reply = f"Here are your {keyword_desc} in {time_desc}:\n\n{summary}"
        else:
            agent_reply = f"No {keyword_desc} found in {time_desc}."
        
        st.session_state.chat_history.append(("assistant", agent_reply))
        
    except Exception as e:
        st.session_state.chat_history.append(("assistant", f"Error: {str(e)}"))

# Display only the current conversation
for role, msg in st.session_state.chat_history:
    if role == "assistant":
        st.markdown(f"**Agent:** {msg}")

# Add a clear button
if st.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()