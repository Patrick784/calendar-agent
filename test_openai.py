"""
OpenRouter API Connection Test (v1.x syntax)

Verifies your OpenRouter (OpenAI-compatible) setup using the new OpenAI Python SDK.
"""

import os
from dotenv import load_dotenv
import openai
from openai import (
    OpenAI,
    OpenAIError,
    AuthenticationError,
    RateLimitError,
    APIConnectionError
)

def test_openai_connection():
    print("🤖 Testing OpenRouter API Connection...\n")

    # Load .env vars
    load_dotenv()
    api_key  = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    model_id = "openai/gpt-3.5-turbo"  # valid OpenRouter model ID

    # Debug outputs
    print(f"• openai library version: {openai.__version__}")
    print(f"• Loaded API key:    {api_key[:6]}… (length {len(api_key) if api_key else 0})")
    print(f"• Loaded API base:   {api_base}")
    print(f"• Using model_id:    {model_id}\n")

    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not set in .env")
        return False
    if not api_base:
        print("❌ ERROR: OPENAI_API_BASE not set in .env")
        return False

    # Initialize v1.x client
    client = OpenAI(api_key=api_key, base_url=api_base)

    try:
        # 1) Basic “Hello” test
        print("🔍 Testing basic model response...")
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": "Say 'Hello, calendar agent!' in exactly those words."}
            ],
            max_tokens=20,
            temperature=0
        )
        text = resp.choices[0].message.content.strip()
        print(f"✅ API Response: '{text}'\n")

        # 2) Calendar intent test
        print("🧠 Testing calendar-specific functionality...")
        cal = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a calendar assistant. Extract intent from the user's request."},
                {"role": "user",   "content": "Schedule a meeting tomorrow at 2pm"}
            ],
            max_tokens=50,
            temperature=0.2
        )
        cal_text = cal.choices[0].message.content.strip()
        print(f"✅ Calendar AI Response: '{cal_text[:100]}...'\n")

        print("🎉 OpenRouter API is working correctly!")
        return True

    except AuthenticationError:
        print("❌ Authentication Error: Invalid API key or permissions.")
        return False
    except RateLimitError:
        print("❌ Rate Limit Error: Throughput or quota hit on OpenRouter.")
        return False
    except APIConnectionError:
        print("❌ Connection Error: Could not reach OpenRouter endpoint.")
        return False
    except OpenAIError as e:
        print(f"❌ OpenAIError: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    print("=" * 60)
    print("🧪 OPENROUTER API CONNECTION TEST")
    print("=" * 60)

    success = test_openai_connection()

    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED - You’re good to go!")
    else:
        print("❌ TESTS FAILED - Please review the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
