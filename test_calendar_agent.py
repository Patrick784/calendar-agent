"""
Test script for the Calendar Agent

Demonstrates the calendar agent's ability to parse natural language queries
and generate structured output for downstream agents.
"""

import asyncio
import json
import openai
from datetime import datetime
from agents.calendar_agent import CalendarAgent, CalendarQueryResult
from agents.base import AgentMessage

# Test queries covering different scenarios
TEST_QUERIES = [
    # Memory-based queries
    "When last did I go to the barber?",
    "When did I last visit the dentist?",
    "How long ago did I have lunch with Sarah?",
    "When was my last soccer game?",
    
    # Event creation queries
    "Add meeting with Alex next Thursday at 1pm",
    "Schedule dentist appointment for tomorrow",
    "Book a haircut next week",
    "Create lunch meeting with John on Friday",
    
    # Event reading queries
    "Show my dentist appointments",
    "What meetings do I have this week?",
    "Find my soccer games",
    "List my appointments for tomorrow",
    
    # Event deletion queries
    "Cancel the soccer game",
    "Delete my dentist appointment",
    "Remove the meeting with Alex",
    
    # Ambiguous queries
    "Meeting",
    "When?",
    "Cancel it",
    "Show me something",
]

async def test_calendar_agent():
    """Test the calendar agent with various queries"""
    
    # Initialize OpenAI client (you'll need to set your API key)
    try:
        client = openai.OpenAI(
            api_key="your-api-key-here",  # Replace with your actual API key
            base_url="https://openrouter.ai/api/v1"  # Using OpenRouter
        )
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Please set your API key to test with LLM parsing")
        return
    
    # Initialize calendar agent
    agent = CalendarAgent(client, {
        "model": "openai/gpt-4o",
        "max_tokens": 500,
        "temperature": 0.1
    })
    
    print("🧠 Calendar Agent Test")
    print("=" * 50)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 30)
        
        # Create message
        message = AgentMessage(
            recipient="calendar_agent",
            sender="test_user",
            body={"text": query},
            context={"user_timezone": "UTC"}
        )
        
        try:
            # Process the query
            response = await agent.process_message(message)
            
            if response.success:
                result = response.data
                print(f"✅ Intent: {result.intent}")
                print(f"🎯 Target Info: {result.target_info}")
                print(f"📅 Timeframe: {result.timeframe}")
                print(f"🧩 Ambiguities: {result.ambiguities}")
                print(f"🧪 Test Passed: {result.test_passed}")
                print(f"📊 Confidence: {response.confidence:.2f}")
            else:
                print(f"❌ Error: {response.error}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

def test_regex_fallback():
    """Test regex fallback parsing without LLM"""
    
    print("\n🔧 Testing Regex Fallback (without LLM)")
    print("=" * 50)
    
    # Create a mock client that will fail
    class MockClient:
        async def chat(self):
            raise Exception("Mock client - testing regex fallback")
    
    # Initialize agent with mock client
    agent = CalendarAgent(MockClient(), {
        "model": "mock",
        "max_tokens": 500,
        "temperature": 0.1
    })
    
    # Test a few queries with regex fallback
    test_queries = [
        "When last did I go to the barber?",
        "Add meeting with Alex next Thursday",
        "Show my dentist appointments",
        "Cancel the soccer game",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 30)
        
        message = AgentMessage(
            recipient="calendar_agent",
            sender="test_user",
            body={"text": query},
            context={"user_timezone": "UTC"}
        )
        
        try:
            response = asyncio.run(agent.process_message(message))
            
            if response.success:
                result = response.data
                print(f"✅ Intent: {result.intent}")
                print(f"🎯 Target Info: {result.target_info}")
                print(f"📅 Timeframe: {result.timeframe}")
                print(f"🧩 Ambiguities: {result.ambiguities}")
                print(f"🧪 Test Passed: {result.test_passed}")
                print(f"📊 Confidence: {response.confidence:.2f}")
            else:
                print(f"❌ Error: {response.error}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")

def demonstrate_structured_output():
    """Demonstrate the structured output format"""
    
    print("\n📋 Structured Output Format")
    print("=" * 50)
    
    # Example structured output
    example_result = CalendarQueryResult(
        query="When last did I go to the barber?",
        intent="query_memory",
        target_info="barber",
        timeframe="past",
        ambiguities=[],
        test_passed=True
    )
    
    print("Example JSON output for downstream agents:")
    print(json.dumps({
        "query": example_result.query,
        "intent": example_result.intent,
        "target_info": example_result.target_info,
        "timeframe": example_result.timeframe,
        "ambiguities": example_result.ambiguities,
        "test_passed": example_result.test_passed
    }, indent=2))
    
    print("\nThis structured output can be passed to:")
    print("- CalendarReaderAgent (for read_events)")
    print("- MemoryLookupAgent (for query_memory)")
    print("- SchedulerAgent (for create_event)")
    print("- CalendarManagerAgent (for delete_event)")

if __name__ == "__main__":
    print("Calendar Agent Test Suite")
    print("=" * 50)
    
    # Test regex fallback first (works without API key)
    test_regex_fallback()
    
    # Demonstrate structured output
    demonstrate_structured_output()
    
    # Test with LLM (requires API key)
    print("\n" + "=" * 50)
    print("To test with LLM parsing, set your API key and run:")
    print("asyncio.run(test_calendar_agent())")
    
    # Uncomment the line below to test with LLM (requires API key)
    # asyncio.run(test_calendar_agent()) 