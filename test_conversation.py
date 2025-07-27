#!/usr/bin/env python3
"""
Test script to verify the Calendar Agent's conversational capabilities
"""

import asyncio
import sys
import traceback
from main import system

# Test multiple conversational queries
test_queries = [
    'What do I have coming up next week?',
    'Find my doctor appointments',
    'When is my next meeting?',
    'Schedule a call with John tomorrow at 3pm',
    'Show me my calendar for today',
    'I need to schedule a meeting',
    'What meetings do I have on Friday?'
]

async def test_conversation():
    """Test the conversational agent with various queries"""
    
    print('🧪 Testing Conversational Agent with Multiple Queries...\n')
    
    try:
        # Initialize system
        print('Initializing system...')
        await system.initialize('debug')
        print('✅ System initialized successfully\n')
        
        success_count = 0
        total_queries = len(test_queries)
        
        for i, query in enumerate(test_queries, 1):
            print(f'{i}. Query: "{query}"')
            
            try:
                response = await system.process_user_request(query, {'ui_mode': 'debug'})
                
                if response['success']:
                    print(f'   ✅ Response: {response["data"]}')
                    print(f'   📊 Confidence: {int(response["confidence"] * 100)}%')
                    success_count += 1
                else:
                    print(f'   ❌ Error: {response["error"]}')
                
                if response.get('suggestions'):
                    print(f'   💡 Suggestions: {response["suggestions"]}')
                    
            except Exception as e:
                print(f'   ❌ Exception: {str(e)}')
                
            print()
        
        print(f'🎉 Conversational testing complete!')
        print(f'📊 Success rate: {success_count}/{total_queries} ({int(success_count/total_queries*100)}%)')
        
        if success_count == total_queries:
            print('🚀 All tests passed! The agent is fully conversational.')
        elif success_count > total_queries * 0.7:
            print('✅ Most tests passed! The agent is working well.')
        else:
            print('⚠️  Some issues detected. Check the errors above.')
            
    except Exception as e:
        print(f'❌ Critical error during testing:')
        print(f'Error type: {type(e).__name__}')
        print(f'Error message: {str(e)}')
        traceback.print_exc()
        return False
    
    return success_count > 0

if __name__ == "__main__":
    success = asyncio.run(test_conversation())
    sys.exit(0 if success else 1) 