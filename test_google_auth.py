import os
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

CLIENT_SECRETS = os.getenv("GOOGLE_CREDENTIALS_FILE")
TOKEN_FILE = os.getenv("GOOGLE_TOKEN_FILE", "token.json")
SCOPES = ['https://www.googleapis.com/auth/calendar']

def test_google_auth():
    print("\n🧪 GOOGLE CALENDAR AUTHENTICATION TEST\n" + "="*60)
    
    if not os.path.exists(CLIENT_SECRETS):
        print("❌ Missing credentials.json file")
        return

    print("🔐 Found credentials.json")

    try:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, SCOPES)
        creds = flow.run_local_server(port=8501)
        
        # Save token
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
        
        service = build('calendar', 'v3', credentials=creds)
        profile = service.calendarList().list().execute()
        print("✅ Google Calendar access successful")
        print("📅 Calendars:", [item['summary'] for item in profile['items']])
    
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    test_google_auth()
