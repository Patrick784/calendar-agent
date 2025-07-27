"""
Google Calendar Authentication Test

Run this script to verify your Google Calendar API setup is working correctly.
This will help debug any authentication issues before running the main system.
"""

import os
import sys
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/calendar']

def test_auth():
    """Test Google Calendar API authentication"""
    
    print("🔐 Testing Google Calendar Authentication...")
    
    # Check if credentials.json exists
    if not os.path.exists('credentials.json'):
        print("❌ ERROR: credentials.json not found!")
        print("   Please download your Google Calendar API credentials from:")
        print("   https://console.cloud.google.com/")
        print("   And save as 'credentials.json' in the project root.")
        return False
    
    print("✅ Found credentials.json")
    
    try:
        creds = None
        
        # Load existing token if available
        if os.path.exists('token.json'):
            print("📄 Loading existing token.json...")
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
        # Refresh or create new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("🔄 Refreshing expired credentials...")
                creds.refresh(Request())
            else:
                print("🌐 Starting OAuth flow...")
                print("   A browser window will open for authentication.")
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', 
                    SCOPES
                )
                creds = flow.run_local_server(port=8765, prompt='consent')
            
            # Save credentials for next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
            print("💾 Saved authentication token")
        
        print("✅ Authentication successful!")
        
        # Test API connection
        print("🔍 Testing Calendar API connection...")
        service = build('calendar', 'v3', credentials=creds)
        
        # Get calendar list
        calendars_result = service.calendarList().list().execute()
        calendars = calendars_result.get('items', [])
        
        print(f"✅ Successfully connected to Google Calendar!")
        print(f"📅 Found {len(calendars)} calendars:")
        
        for calendar in calendars[:3]:  # Show first 3 calendars
            name = calendar.get('summary', 'Unnamed Calendar')
            is_primary = calendar.get('primary', False)
            primary_text = " (PRIMARY)" if is_primary else ""
            print(f"   • {name}{primary_text}")
        
        if len(calendars) > 3:
            print(f"   ... and {len(calendars) - 3} more calendars")
        
        print("\n🎉 Google Calendar setup is working correctly!")
        return True
        
    except HttpError as error:
        print(f"❌ Google Calendar API Error: {error}")
        return False
    except Exception as error:
        print(f"❌ Unexpected error: {error}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 GOOGLE CALENDAR AUTHENTICATION TEST")
    print("=" * 60)
    
    success = test_auth()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED - Google Calendar is ready!")
        print("   You can now run: streamlit run main.py")
    else:
        print("❌ TESTS FAILED - Please fix the issues above")
        print("   Check the Google Calendar API setup instructions")
    print("=" * 60)
    
    return success

if __name__ == '__main__':
    main() 
    