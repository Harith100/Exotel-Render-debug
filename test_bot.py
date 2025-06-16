#!/usr/bin/env python3
"""
Test script for Malayalam Voicebot
Run this to verify your setup before deployment
"""

import asyncio
import websockets
import json
import base64
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoicebotTester:
    def __init__(self, websocket_url):
        self.websocket_url = websocket_url
        self.sequence_number = 1
    
    async def test_connection(self):
        """Test WebSocket connection to the voicebot"""
        try:
            logger.info(f"Connecting to {self.websocket_url}")
            
            async with websockets.connect(self.websocket_url) as websocket:
                logger.info("Connected successfully!")
                
                # Send connected message
                await self.send_connected(websocket)
                await asyncio.sleep(1)
                
                # Send start message
                await self.send_start(websocket)
                await asyncio.sleep(2)
                
                # Send test audio (silence)
                await self.send_test_audio(websocket)
                await asyncio.sleep(3)
                
                # Send stop message
                await self.send_stop(websocket)
                
                # Listen for responses
                await self.listen_for_responses(websocket)
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
        
        return True
    
    async def send_connected(self, websocket):
        """Send connected event"""
        message = {
            "event": "connected"
        }
        await websocket.send(json.dumps(message))
        logger.info("Sent connected message")
    
    async def send_start(self, websocket):
        """Send start event"""
        message = {
            "event": "start",
            "sequence_number": self.sequence_number,
            "stream_sid": "test_stream_123",
            "start": {
                "stream_sid": "test_stream_123",
                "call_sid": "test_call_456", 
                "account_sid": "test_account_789",
                "from": "+919876543210",
                "to": "+911234567890",
                "custom_parameters": {
                    "language": "malayalam",
                    "test_mode": "true"
                },
                "media_format": {
                    "encoding": "linear16",
                    "sample_rate": "8000",
                    "bit_rate": "128000"
                }
            }
        }
        
        await websocket.send(json.dumps(message))
        self.sequence_number += 1
        logger.info("Sent start message")
    
    async def send_test_audio(self, websocket):
        """Send test audio data (silence)"""
        # Generate 1 second of silence (8000 samples * 2 bytes = 16000 bytes)
        silence = b'\x00' * 16000
        
        # Split into chunks of 3200 bytes (100ms each)
        chunk_size = 3200
        timestamp = 0
        
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i:i + chunk_size]
            
            message = {
                "event": "media",
                "sequence_number": self.sequence_number,
                "stream_sid": "test_stream_123",
                "media": {
                    "chunk": i // chunk_size,
                    "timestamp": str(timestamp),
                    "payload": base64.b64encode(chunk).decode('utf-8')
                }
            }
            
            await websocket.send(json.dumps(message))
            self.sequence_number += 1
            timestamp += 100  # 100ms per chunk
            
            await asyncio.sleep(0.1)  # 100ms delay
        
        logger.info("Sent test audio data")
    
    async def send_stop(self, websocket):
        """Send stop event"""
        message = {
            "event": "stop",
            "sequence_number": self.sequence_number,
            "stream_sid": "test_stream_123",
            "stop": {
                "call_sid": "test_call_456",
                "account_sid": "test_account_789", 
                "reason": "test_completed"
            }
        }
        
        await websocket.send(json.dumps(message))
        self.sequence_number += 1
        logger.info("Sent stop message")
    
    async def listen_for_responses(self, websocket):
        """Listen for responses from the bot"""
        try:
            timeout = 5  # 5 seconds timeout
            while timeout > 0:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    logger.info(f"Received: {data.get('event', 'unknown')} event")
                    
                    if data.get('event') == 'media':
                        logger.info("Bot is sending audio response!")
                    
                    timeout -= 1
                    
                except asyncio.TimeoutError:
                    timeout -= 1
                    continue
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by server")

async def test_http_endpoint(base_url):
    """Test the HTTP endpoint that returns WebSocket URL"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"HTTP endpoint response: {data}")
                    return data.get('websocket_url')
                else:
                    logger.error(f"HTTP endpoint failed: {response.status}")
                    return None
                    
    except Exception as e:
        logger.error(f"HTTP endpoint test failed: {e}")
        return None

async def main():
    """Main test function"""
    if len(sys.argv) < 2:
        print("Usage: python test_bot.py <base_url>")
        print("Example: python test_bot.py http://localhost:5000")
        print("Example: python test_bot.py https://your-app.onrender.com")
        sys.exit(1)
    
    base_url = sys.argv[1]
    
    logger.info("Starting Malayalam Voicebot Test")
    logger.info("=" * 50)
    
    # Test 1: HTTP endpoint
    logger.info("Test 1: Testing HTTP endpoint...")
    websocket_url = await test_http_endpoint(base_url)
    
    if not websocket_url:
        logger.error("HTTP endpoint test failed")
        return
    
    logger.info(f"✅ HTTP endpoint working. WebSocket URL: {websocket_url}")
    
    # Test 2: WebSocket connection
    logger.info("Test 2: Testing WebSocket connection...")
    tester = VoicebotTester(websocket_url)
    
    success = await tester.test_connection()
    
    if success:
        logger.info("✅ WebSocket connection test passed!")
        logger.info("✅ All tests completed successfully!")
        logger.info("Your Malayalam Voicebot is ready for deployment!")
    else:
        logger.error("❌ WebSocket connection test failed")
        logger.error("Please check your configuration and try again")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())