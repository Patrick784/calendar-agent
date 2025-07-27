import os
import datetime
import streamlit as st
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv
import re
import openai
from openai import OpenAI  # Updated import for v1.x SDK
import pytz
from google.auth.transport.requests import Request

# Load environment variables
load_dotenv()

# Initialize OpenAI client for v1.x SDK with OpenRouter support
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")  # Default to OpenRouter

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")
if not OPENAI_API_BASE:
    raise RuntimeError("Missing OPENAI_API_BASE in .env")

# Initialize v1.x OpenAI client for OpenRouter
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE
)

# Configuration
DEMO_MODE = not os.path.exists('credentials.json')  # Use demo mode if no credentials
SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_PATH = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')

# OpenRouter-compatible model ID (updated for OpenRouter)
OPENAI_MODEL = "openai/gpt-3.5-turbo"  # OpenRouter format for GPT-3.5-turbo

if DEMO_MODE:
    st.sidebar.warning("🧪 **DEMO MODE** - No Google credentials found. Using mock calendar data.")
    st.sidebar.info("To use real Google Calendar, add `credentials.json` file and restart.")
    service = None  # Mock service
    
    # Mock calendar functions for demo mode
    def mock_create_event(event_data):
        """Simulate creating a calendar event"""
        return {
            'id': f"mock_event_{datetime.datetime.now().timestamp()}",
            'htmlLink': 'https://calendar.google.com/calendar/mock-event',
            'summary': event_data.get('summary', 'Mock Event'),
            'start': event_data.get('start'),
            'end': event_data.get('end')
        }
    
    def mock_list_events(time_min, time_max):
        """Simulate listing calendar events within the specified time range"""
        from datetime import datetime, timedelta
        import random
        
        # Parse the time range
        start_time = datetime.fromisoformat(time_min.replace('Z', ''))
        end_time = datetime.fromisoformat(time_max.replace('Z', ''))
        
        # Generate realistic mock events within the time range
        mock_events = []
        
        # Event templates
        event_templates = [
            {'summary': 'Team Meeting', 'duration': 1},
            {'summary': 'Doctor Appointment', 'duration': 1},
            {'summary': 'Birthday Party', 'duration': 3},
            {'summary': 'Conference Call', 'duration': 0.5},
            {'summary': 'Lunch Meeting', 'duration': 1.5},
            {'summary': 'Project Review', 'duration': 2},
            {'summary': 'Dentist Appointment', 'duration': 1},
            {'summary': 'Client Presentation', 'duration': 2},
            {'summary': 'Training Session', 'duration': 4},
            {'summary': 'Weekly Standup', 'duration': 0.5}
        ]
        
        # Generate 2-5 events within the time range
        num_events = random.randint(2, min(5, max(2, int((end_time - start_time).days))))
        
        for i in range(num_events):
            # Pick a random day within the range
            total_days = (end_time - start_time).days
            if total_days <= 0:
                random_day = start_time
            else:
                random_day_offset = random.randint(0, total_days)
                random_day = start_time + timedelta(days=random_day_offset)
            
            # Pick a random time of day (9 AM to 5 PM)
            random_hour = random.randint(9, 17)
            random_minute = random.choice([0, 15, 30, 45])
            
            # Set the event start time
            event_start = random_day.replace(hour=random_hour, minute=random_minute, second=0, microsecond=0)
            
            # Skip if event is outside our range
            if event_start < start_time or event_start > end_time:
                continue
            
            # Pick a random event template
            template = random.choice(event_templates)
            duration_hours = template['duration']
            event_end = event_start + timedelta(hours=duration_hours)
            
            mock_events.append({
                'summary': template['summary'],
                'start': {'dateTime': event_start.isoformat() + 'Z'},
                'end': {'dateTime': event_end.isoformat() + 'Z'}
            })
        
        # Sort events by start time
        mock_events.sort(key=lambda x: x['start']['dateTime'])
        
        return {'items': mock_events}
        
else:
    # Real Google Calendar setup
    # Helper: get Google creds
    def get_creds():
        creds = None
        if os.path.exists(TOKEN_PATH):
            try:
                creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
            except Exception as e:
                st.warning(f"⚠️ Existing token.json is invalid: {str(e)}")
                # Delete the bad token file
                os.remove(TOKEN_PATH)
                creds = None
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                st.info("🔄 Refreshing expired token...")
                creds.refresh(Request())
            else:
                st.info("🌐 Starting OAuth flow...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', 
                    SCOPES,
                    redirect_uri='http://localhost:8501/oauth2callback'
                )
                
                try:
                    st.info("🔗 Opening browser for authorization...")
                    creds = flow.run_local_server(port=8501, open_browser=True, 
                                                prompt='consent')
                    success = True
                except Exception as auth_error:
                    st.error(f"❌ Authorization failed: {str(auth_error)}")
                    success = False
                
                if not success:
                    st.error("❌ **OAuth Authorization Failed!**")
                    st.markdown(f"""
                    **Error**: {str(auth_error)}
                    
                    ### 🔧 **CRITICAL FIX NEEDED - Redirect URI Configuration:**
                    
                    **The issue is your Google Cloud Console redirect URI setup. Here's the exact fix:**
                    
                    1. **Go to Google Cloud Console**: https://console.cloud.google.com/
                    2. **Navigate to your project**: `calendar-agent-467017`
                    3. **Go to "APIs & Services" → "Credentials"**
                    4. **Click on your OAuth 2.0 Client ID**
                    5. **In "Authorized redirect URIs", ADD these entries:**
                       - `http://localhost:8080`
                       - `http://localhost:8000`  
                       - `http://localhost`
                       - `http://localhost:8501`
                    6. **Click "Save"**
                    7. **Wait 5 minutes** for changes to propagate
                    8. **Restart this app**
                    
                    ### 🛠️ **Alternative Quick Fix:**
                    
                    **If you want to try manual OAuth:**
                    1. Visit this URL in your browser: https://console.cloud.google.com/apis/credentials/consent
                    2. Add your email as a test user
                    3. Make sure the app status is "Testing" or "Production"
                    
                    ### 📞 **Current OAuth Configuration:**
                    - **Client ID**: `911727645033-i85j7uqtmen6esl47judlr05apecnh5s.apps.googleusercontent.com`
                    - **Configured Redirect**: `http://localhost` (needs more options)
                    
                    ### 🆘 **EMERGENCY MANUAL OAUTH (Try this now!):**
                    
                    If you can't fix the redirect URIs right now, try this manual method:
                    1. **Click this link**: [Manual OAuth Authorization](https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=911727645033-i85j7uqtmen6esl47judlr05apecnh5s.apps.googleusercontent.com&redirect_uri=urn:ietf:wg:oauth:2.0:oob&scope=https://www.googleapis.com/auth/calendar&access_type=offline)
                    2. **Sign in** and click "Allow"
                    3. **Copy the authorization code** you receive
                    4. **Restart the app** and look for "Manual OAuth" option
                    """)
                    
                    # Add manual OAuth option
                    st.markdown("---")
                    st.markdown("### 🔧 **Try Manual OAuth Now:**")
                    
                    manual_code = st.text_input("Paste your authorization code here:", 
                                              help="Get code from the manual OAuth link above")
                    if st.button("Use Manual Authorization Code") and manual_code:
                        try:
                            # Manual OAuth flow
                            from google_auth_oauthlib.flow import Flow
                            flow = Flow.from_client_secrets_file(
                                'credentials.json', 
                                scopes=SCOPES,
                                redirect_uri='urn:ietf:wg:oauth:2.0:oob'
                            )
                            flow.fetch_token(code=manual_code)
                            creds = flow.credentials
                            
                            # Save credentials
                            with open(TOKEN_PATH, 'w') as token:
                                token.write(creds.to_json())
                            
                            st.success("✅ **Manual OAuth Successful!**")
                            st.success("🎉 **Reloading app with Google Calendar access...**")
                            st.experimental_rerun()
                            
                        except Exception as manual_error:
                            st.error(f"Manual OAuth failed: {str(manual_error)}")
                    
                    st.stop()
            
            # Save the credentials for the next run
            try:
                with open(TOKEN_PATH, 'w') as token:
                    token.write(creds.to_json())
            except Exception as e:
                st.warning(f"⚠️ Could not save token: {str(e)}")
        
        return creds

    # Build Calendar service only
    creds = get_creds()
    service = build('calendar', 'v3', credentials=creds)

def get_contact_birthday(person_name):
    """Search for a person's birthday in Google Contacts"""
    try:
        # Get all contacts with birthday information
        results = people_service.people().connections().list(
            resourceName='people/me',
            personFields='names,birthdays',
            sources=['CONTACT']
        ).execute()
        
        connections = results.get('connections', [])
        person_name_lower = person_name.lower()
        
        for person in connections:
            # Check if person has names and birthdays
            names = person.get('names', [])
            birthdays = person.get('birthdays', [])
            
            if not names or not birthdays:
                continue
            
            # Check if any name matches
            for name in names:
                full_name = name.get('displayName', '')
                given_name = name.get('givenName', '')
                family_name = name.get('familyName', '')
                
                if (person_name_lower in full_name.lower() or 
                    person_name_lower in given_name.lower() or
                    person_name_lower in family_name.lower()):
                    
                    # Get the birthday
                    for birthday in birthdays:
                        date = birthday.get('date', {})
                        if date:
                            month = date.get('month')
                            day = date.get('day')
                            year = date.get('year')
                            
                            if month and day:
                                birthday_str = f"{month:02d}-{day:02d}"
                                if year:
                                    birthday_str = f"{year}-{birthday_str}"
                                    age = datetime.datetime.now().year - year
                                    return {
                                        'name': full_name,
                                        'birthday': birthday_str,
                                        'month': month,
                                        'day': day,
                                        'year': year,
                                        'age': age
                                    }
                                else:
                                    return {
                                        'name': full_name,
                                        'birthday': birthday_str,
                                        'month': month,
                                        'day': day,
                                        'year': None,
                                        'age': None
                                    }
        
        return None
        
    except Exception as e:
        print(f"Error fetching birthday: {str(e)}")
        return None

def get_apple_contacts_birthday(person_name):
    """Future: Search for birthdays in Apple Contacts via CardDAV or local access"""
    # TODO: Implement Apple Contacts access
    # Options:
    # 1. CardDAV protocol for iCloud contacts
    # 2. macOS Contacts.app integration (if on Mac)
    # 3. Contact export/import workflow
    return None

def parse_user_request(user_input):
    """Enhanced keyword-based parser with date and context understanding"""
    original_input = user_input  # Keep original for event creation
    user_input = user_input.lower()
    
    # TODO: Re-enable when Google Contacts is properly configured
    # # Check for birthday queries about specific people
    # birthday_patterns = [
    #     r"when is (.+?)['s]* birthday",
    #     r"(.+?)['s]* birthday", 
    #     r"birthday of (.+)",
    #     r"what is (.+?)['s]* birthday",
    #     r"tell me (.+?)['s]* birthday"
    # ]
    # 
    # for pattern in birthday_patterns:
    #     match = re.search(pattern, user_input)
    #     if match:
    #         person_name = match.group(1).strip()
    #         # Remove common words that might interfere
    #         person_name = re.sub(r'\b(is|the|a|an|when|what|tell|me)\b', '', person_name).strip()
    #         if person_name:
    #             return 'get_birthday', {'person_name': person_name}
    
    # Check for add/create intent first
    if any(word in user_input for word in ['add', 'create', 'schedule', 'book', 'set up']):
        return 'add_event', {'raw_input': original_input}
    
    # Month mapping
    months = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12
    }
    
    # Check for specific month
    target_month = None
    for month_name, month_num in months.items():
        if month_name in user_input:
            target_month = month_num
            break
    
    # Determine if asking about birthdays
    is_birthday_query = any(word in user_input for word in ['birthday', 'birthdays'])
    
    # Determine time context
    is_future = any(word in user_input for word in ['coming', 'upcoming', 'future', 'next'])
    is_past = any(word in user_input for word in ['past', 'previous', 'last', 'had'])
    
    now = datetime.datetime.now(datetime.timezone.utc)
    current_year = now.year
    
    if target_month:
        # Specific month query
        if is_future or (not is_past and target_month >= now.month):
            # Future month or current year if month hasn't passed
            year = current_year if target_month >= now.month else current_year + 1
        else:
            # Past month
            year = current_year if target_month <= now.month else current_year - 1
        
        # Create date range for the entire month
        start_date = datetime.datetime(year, target_month, 1)
        if target_month == 12:
            end_date = datetime.datetime(year + 1, 1, 1) - datetime.timedelta(seconds=1)
        else:
            end_date = datetime.datetime(year, target_month + 1, 1) - datetime.timedelta(seconds=1)
        
        time_min = start_date.isoformat() + 'Z'
        time_max = end_date.isoformat() + 'Z'
        
        intent = 'retrieve_future_events' if (year > current_year or (year == current_year and target_month >= now.month)) else 'retrieve_past_events'
        
        if is_birthday_query:
            return intent, {'keyword': 'birthday', 'time_min': time_min, 'time_max': time_max}
        else:
            return intent, {'time_min': time_min, 'time_max': time_max}
    
    elif is_birthday_query:
        # General birthday query
        if is_future or any(word in user_input for word in ['coming', 'upcoming']):
            # Future birthdays
            time_min = now.isoformat() + 'Z'
            time_max = (now + datetime.timedelta(days=365)).isoformat() + 'Z'
            return 'retrieve_future_events', {'keyword': 'birthday', 'time_min': time_min, 'time_max': time_max}
        else:
            # Past birthdays
            time_min = (now - datetime.timedelta(days=365)).isoformat() + 'Z'
            time_max = now.isoformat() + 'Z'
            return 'retrieve_past_events', {'keyword': 'birthday', 'time_min': time_min, 'time_max': time_max}
    
    elif any(word in user_input for word in ['last', 'past', 'previous']):
        # Past events
        if 'week' in user_input:
            days = 7
        elif 'month' in user_input:
            days = 30
        else:
            days = 7
        
        time_min = (now - datetime.timedelta(days=days)).isoformat() + 'Z'
        time_max = now.isoformat() + 'Z'
        return 'retrieve_past_events', {'time_min': time_min, 'time_max': time_max}
    
    else:
        # Default to upcoming events
        time_min = now.isoformat() + 'Z'
        time_max = (now + datetime.timedelta(days=30)).isoformat() + 'Z'
        return 'retrieve_future_events', {'time_min': time_min, 'time_max': time_max}

def create_event_from_text(raw_input):
    """Simple parser to extract event details from natural language"""
    import re
    from datetime import datetime, timedelta
    import pytz
    
    # Extract title (everything after 'add' until time/date keywords)
    title_match = re.search(r'(?:add|create|schedule)\s+(.+?)(?:\s+(?:next|on|at|\d))', raw_input.lower())
    title = title_match.group(1).strip() if title_match else "New Event"
    
    # Use local timezone (Eastern Time)
    local_tz = pytz.timezone('America/New_York')
    now = datetime.now(local_tz)
    start_time = None
    
    if "next tuesday" in raw_input.lower() or "tuesday" in raw_input.lower():
        # Find next Tuesday (Tuesday = 1 in weekday())
        days_ahead = 1 - now.weekday()  # Tuesday is 1
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        target_date = now + timedelta(days=days_ahead)
        
        # Look for time like "2 PM", "2:00 PM", "14:00"
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)?', raw_input)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            am_pm = time_match.group(3)
            
            if am_pm and am_pm.lower() == 'pm' and hour != 12:
                hour += 12
            elif am_pm and am_pm.lower() == 'am' and hour == 12:
                hour = 0
                
            start_time = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    # Default to 1 hour duration, look for duration hints
    duration = 60  # minutes
    duration_match = re.search(r'(\d+)\s*(?:hour|hr|h)', raw_input.lower())
    if duration_match:
        duration = int(duration_match.group(1)) * 60
    
    # If no specific time found, default to 2 PM today
    if not start_time:
        start_time = now.replace(hour=14, minute=0, second=0, microsecond=0)
        if start_time <= now:  # If 2 PM today has passed, make it tomorrow
            start_time += timedelta(days=1)
    
    end_time = start_time + timedelta(minutes=duration)
    
    return {
        'title': title.title(),
        'start_time': start_time,
        'end_time': end_time
    }

st.title("Calendar Agent 🚀")
st.markdown("---")

# Chat interface
st.subheader("Chat with your calendar agent:")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Type your request:", "")
if st.button("Send") and user_input:
    try:
        intent, details = parse_user_request(user_input)
        
        if intent in ['retrieve_past_events', 'retrieve_future_events']:
            time_min = details.get('time_min')
            time_max = details.get('time_max')
            keyword = details.get('keyword')
            
            # Use mock or real calendar service based on mode
            if DEMO_MODE:
                events_result = mock_list_events(time_min, time_max)
            else:
                events_result = service.events().list(
                    calendarId='primary',
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute()
            
            events = events_result.get('items', [])
            
            # Determine the time period for better response
            start_date = datetime.datetime.fromisoformat(time_min.replace('Z', ''))
            end_date = datetime.datetime.fromisoformat(time_max.replace('Z', ''))
            
            # Format time period description
            if start_date.year == end_date.year and start_date.month == end_date.month:
                time_desc = f"{start_date.strftime('%B %Y')}"
            elif start_date.year == end_date.year:
                time_desc = f"{start_date.strftime('%B')} to {end_date.strftime('%B %Y')}"
            else:
                time_desc = f"{start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}"
            
            if keyword:
                matches = [ev for ev in events if keyword.lower() in ev.get('summary','').lower()]
                if matches:
                    summary_lines = []
                    for ev in matches:
                        start = ev['start'].get('dateTime', ev['start'].get('date'))
                        # Format date nicely
                        if 'T' in start:
                            event_date = datetime.datetime.fromisoformat(start.replace('Z', ''))
                            formatted_date = event_date.strftime('%Y-%m-%d')
                        else:
                            formatted_date = start
                        summary = ev.get('summary','No Title')
                        summary_lines.append(f"• {formatted_date}: {summary}")
                    summary = '\n'.join(summary_lines)
                    agent_reply = f"Here are your '{keyword}' events in {time_desc}:\n\n{summary}"
                else:
                    agent_reply = f"No '{keyword}' events found in {time_desc}."
            else:
                if not events:
                    agent_reply = f"No events found in {time_desc}."
                else:
                    summary_lines = []
                    for ev in events:
                        start = ev['start'].get('dateTime', ev['start'].get('date'))
                        # Format date nicely
                        if 'T' in start:
                            event_date = datetime.datetime.fromisoformat(start.replace('Z', ''))
                            formatted_date = event_date.strftime('%Y-%m-%d')
                        else:
                            formatted_date = start
                        summary = ev.get('summary','No Title')
                        summary_lines.append(f"• {formatted_date}: {summary}")
                    summary = '\n'.join(summary_lines)
                    agent_reply = f"Here are your events in {time_desc}:\n\n{summary}"
            
            st.session_state.chat_history.append(("assistant", agent_reply))
        elif intent == 'add_event':
            raw_input = details.get('raw_input')
            event_details = create_event_from_text(raw_input)
            
            # Debug info
            st.write(f"**Debug Info:**")
            st.write(f"- Title: {event_details['title']}")
            st.write(f"- Start: {event_details['start_time']}")
            st.write(f"- End: {event_details['end_time']}")
            
            event = {
                'summary': event_details['title'],
                'start': {
                    'dateTime': event_details['start_time'].isoformat(),
                    'timeZone': 'America/New_York'
                },
                'end': {
                    'dateTime': event_details['end_time'].isoformat(),
                    'timeZone': 'America/New_York'
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 60},
                        {'method': 'popup', 'minutes': 10}
                    ]
                }
            }
            
            try:
                # Use mock or real calendar service based on mode
                if DEMO_MODE:
                    created_event = mock_create_event(event)
                else:
                    created_event = service.events().insert(calendarId='primary', body=event).execute()
                    
                event_link = created_event.get('htmlLink', 'No link available')
                agent_reply = f"✅ Event '{event_details['title']}' created successfully!\n\n📅 **When:** {event_details['start_time'].strftime('%A, %B %d, %Y at %I:%M %p')}\n🔗 **Link:** {event_link}"
                st.session_state.chat_history.append(("assistant", agent_reply))
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                agent_reply = f"❌ Error creating event: {str(e)}\n\n**Debug Details:**\n{error_details}"
                st.session_state.chat_history.append(("assistant", agent_reply))
        # TODO: Re-enable when Google Contacts is properly configured
        # elif intent == 'get_birthday':
        #     person_name = details.get('person_name')
        #     st.write(f"**Searching for:** {person_name}")
        #     
        #     birthday_info = get_contact_birthday(person_name)
        #     
        #     if birthday_info:
        #         name = birthday_info['name']
        #         month = birthday_info['month']
        #         day = birthday_info['day']
        #         year = birthday_info['year']
        #         age = birthday_info['age']
        #         
        #         # Format the birthday nicely
        #         months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
        #                  'July', 'August', 'September', 'October', 'November', 'December']
        #         month_name = months[month] if month <= 12 else 'Unknown'
        #         
        #         if year and age:
        #             agent_reply = f"🎂 **{name}'s Birthday:**\n\n📅 **Date:** {month_name} {day}, {year}\n🎈 **Age:** {age} years old\n\n*Next birthday: {month_name} {day}, {datetime.datetime.now().year + 1}*"
        #         else:
        #             agent_reply = f"🎂 **{name}'s Birthday:**\n\n📅 **Date:** {month_name} {day}\n\n*Next birthday: {month_name} {day}, {datetime.datetime.now().year}*"
        #         
        #         st.session_state.chat_history.append(("assistant", agent_reply))
        #     else:
        #         agent_reply = f"❌ Sorry, I couldn't find birthday information for '{person_name}' in your Google Contacts.\n\n**📱 Have Apple Contacts?**\n• Sync your Apple contacts with Google to access them here\n• Go to iPhone Settings → Mail → Accounts → Add Google Account → Enable Contacts\n\n**💡 Other Tips:**\n• Make sure the person is in your contacts\n• Check that their birthday is saved in their contact info\n• Try using their full name or nickname\n• Verify Google Contacts sync is working"
        #         st.session_state.chat_history.append(("assistant", agent_reply))
        else:
            st.session_state.chat_history.append(("assistant", "I'm not sure what you want. Try asking about birthdays or upcoming events."))
            
    except Exception as e:
        st.session_state.chat_history.append(("assistant", f"Error: {str(e)}"))

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "assistant":
        st.markdown(f"**Agent:** {msg}") 