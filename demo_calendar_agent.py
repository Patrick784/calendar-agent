"""
Calendar Agent Demo

Demonstrates the specialized calendar agent processing the exact example
from the requirements: "When last did I go to the barber?"
"""

import asyncio
import json
from agents.calendar_agent import CalendarAgent, CalendarQueryResult
from agents.base import AgentMessage

class MockOpenAIClient:
    """Mock client for demonstration"""
    def __init__(self):
        self.chat = None

async def demo_calendar_agent():
    """Demonstrate the calendar agent with the example query"""
    
    print("🧠 Calendar Agent Demo")
    print("=" * 50)
    
    # Initialize calendar agent with mock client (will use regex fallback)
    agent = CalendarAgent(MockOpenAIClient(), {
        "model": "mock",
        "max_tokens": 500,
        "temperature": 0.1
    })
    
    # The exact example from the requirements
    example_query = "When last did I go to the barber?"
    
    print(f"📝 Processing query: '{example_query}'")
    print("-" * 50)
    
    # Create message
    message = AgentMessage(
        recipient="calendar_agent",
        sender="demo_user",
        body={"text": example_query},
        context={"user_timezone": "UTC"}
    )
    
    # Process the query
    response = await agent.process_message(message)
    
    if response.success:
        result = response.data
        
        print("✅ Parsing Results:")
        print(f"   📅 Intent: {result.intent}")
        print(f"   🎯 Target Info: {result.target_info}")
        print(f"   📆 Timeframe: {result.timeframe}")
        print(f"   🧩 Ambiguities: {result.ambiguities}")
        print(f"   🧪 Test Passed: {result.test_passed}")
        print(f"   📊 Confidence: {response.confidence:.2f}")
        
        print("\n📋 Structured JSON Output:")
        print(json.dumps({
            "query": result.query,
            "intent": result.intent,
            "target_info": result.target_info,
            "timeframe": result.timeframe,
            "ambiguities": result.ambiguities,
            "test_passed": result.test_passed
        }, indent=2))
        
        print("\n🔄 Downstream Agent Routing:")
        if result.intent == "query_memory":
            print("   → Route to MemoryLookupAgent")
            print("   → Search past events for 'barber'")
        elif result.intent == "create_event":
            print("   → Route to SchedulerAgent")
            print("   → Create new calendar event")
        elif result.intent == "read_events":
            print("   → Route to CalendarReaderAgent")
            print("   → Search existing events")
        elif result.intent == "delete_event":
            print("   → Route to CalendarManagerAgent")
            print("   → Remove calendar event")
        else:
            print("   → Route to ClarificationAgent")
            print("   → Ask user for more details")
            
    else:
        print(f"❌ Error: {response.error}")

def show_agent_capabilities():
    """Show what the calendar agent can do"""
    
    print("\n🎯 Calendar Agent Capabilities")
    print("=" * 50)
    
    capabilities = {
        "create_event": [
            "Add meeting with Alex next Thursday at 1pm",
            "Schedule dentist appointment for tomorrow",
            "Book a haircut next week"
        ],
        "read_events": [
            "Show my dentist appointments",
            "What meetings do I have this week?",
            "Find my soccer games"
        ],
        "query_memory": [
            "When last did I go to the barber?",
            "When did I last visit the dentist?",
            "How long ago did I have lunch with Sarah?"
        ],
        "delete_event": [
            "Cancel the soccer game",
            "Delete my dentist appointment",
            "Remove the meeting with Alex"
        ]
    }
    
    for intent, examples in capabilities.items():
        print(f"\n📅 {intent.upper()}:")
        for example in examples:
            print(f"   • {example}")

if __name__ == "__main__":
    print("Calendar Agent Specialized Demo")
    print("=" * 50)
    
    # Show capabilities
    show_agent_capabilities()
    
    # Run the demo
    asyncio.run(demo_calendar_agent())
    
    print("\n" + "=" * 50)
    print("✅ Demo completed!")
    print("\nThe calendar agent successfully:")
    print("• Parsed natural language queries")
    print("• Extracted structured data")
    print("• Identified intent and target information")
    print("• Generated output for downstream agents") 