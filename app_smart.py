import os
import datetime
import streamlit as st
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load env vars
load_dotenv()
SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_PATH = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')

# Initialize sentence transformer for better intent understanding
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Helper: get Google creds
def get_creds():
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    return creds

# Build Calendar service
creds = get_creds()
service = build('calendar', 'v3', credentials=creds)

def extract_keywords_and_intent(user_input):
    """Use ML to understand what the user is specifically looking for"""
    user_input = user_input.lower()
    
    # Define intent patterns with examples
    intent_patterns = {
        'flight': ['flight', 'flights', 'airplane', 'travel', 'trip', 'departure', 'arrival'],
        'birthday': ['birthday', 'birthdays', 'born', 'birth'],
        'appointment': ['appointment', 'doctor', 'dentist', 'meeting', 'visit'],
        'work': ['work', 'job', 'office', 'conference', 'presentation'],
        'personal': ['dinner', 'lunch', 'party', 'date', 'hangout'],
        'health': ['doctor', 'dentist', 'hospital', 'checkup', 'medical'],
        'general': ['event', 'events', 'schedule', 'calendar', 'what', 'show']
    }
    
    # Month mapping
    months = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
        'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
        'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
    }
    
    # Extract specific keywords and month
    detected_keywords = []
    target_month = None
    
    # Find the most specific intent
    for intent, keywords in intent_patterns.items():
        for keyword in keywords:
            if keyword in user_input:
                detected_keywords.append(keyword)
                if intent != 'general':  # Prioritize specific intents
                    detected_keywords = [keyword]  # Use most specific
                    break
        if detected_keywords and intent != 'general':
            break
    
    # Find month
    for month_name, month_num in months.items():
        if month_name in user_input:
            target_month = month_num
            break
    
    # Determine time context
    is_future = any(word in user_input for word in ['coming', 'upcoming', 'future', 'next'])
    is_past = any(word in user_input for word in ['past', 'previous', 'last', 'had'])
    
    return detected_keywords, target_month, is_future, is_past

def get_date_range(target_month, is_future, is_past):
    """Calculate precise date range based on context"""
    now = datetime.datetime.utcnow()
    current_year = now.year
    
    if target_month:
        # Specific month query
        if is_future or (not is_past and target_month >= now.month):
            year = current_year if target_month >= now.month else current_year + 1
        else:
            year = current_year if target_month <= now.month else current_year - 1
        
        start_date = datetime.datetime(year, target_month, 1)
        if target_month == 12:
            end_date = datetime.datetime(year + 1, 1, 1) - datetime.timedelta(seconds=1)
        else:
            end_date = datetime.datetime(year, target_month + 1, 1) - datetime.timedelta(seconds=1)
        
        return start_date.isoformat() + 'Z', end_date.isoformat() + 'Z', f"{start_date.strftime('%B %Y')}"
    
    elif is_past:
        # Default past range
        start_date = now - datetime.timedelta(days=30)
        return start_date.isoformat() + 'Z', now.isoformat() + 'Z', "past 30 days"
    
    else:
        # Default future range
        end_date = now + datetime.timedelta(days=30)
        return now.isoformat() + 'Z', end_date.isoformat() + 'Z', "next 30 days"

def filter_events_smartly(events, keywords):
    """Use ML similarity to find relevant events"""
    if not keywords or not events:
        return events
    
    # Create query embedding
    query_text = ' '.join(keywords)
    query_embedding = model.encode([query_text])
    
    relevant_events = []
    for event in events:
        event_text = event.get('summary', '') + ' ' + event.get('description', '')
        event_embedding = model.encode([event_text])
        
        # Calculate similarity
        similarity = cosine_similarity(query_embedding, event_embedding)[0][0]
        
        # Also check for direct keyword matches (more reliable for specific terms)
        direct_match = any(keyword in event_text.lower() for keyword in keywords)
        
        if similarity > 0.3 or direct_match:  # Threshold for relevance
            relevant_events.append(event)
    
    return relevant_events

st.title("🧠 Smart Calendar Agent")
st.markdown("*Powered by ML for better understanding*")
st.markdown("---")

# Initialize conversation state
if 'conversation_active' not in st.session_state:
    st.session_state.conversation_active = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat interface
st.subheader("Chat with your calendar agent:")

user_input = st.text_input("Type your request:", "")
if st.button("Send") and user_input:
    # Clear previous responses for new conversation
    if st.session_state.conversation_active:
        st.session_state.chat_history = []
    
    st.session_state.conversation_active = True
    
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
            filtered_events = filter_events_smartly(events, keywords)
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
                    event_date = datetime.datetime.fromisoformat(start.replace('Z', ''))
                    formatted_date = event_date.strftime('%Y-%m-%d %H:%M')
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
    st.session_state.conversation_active = False
    st.experimental_rerun() 