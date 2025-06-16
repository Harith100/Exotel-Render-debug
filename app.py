import asyncio
import websockets
import json
import base64
import logging
import os
from flask import Flask, jsonify, request
from threading import Thread
import google.cloud.speech as speech
import google.cloud.texttospeech as tts
import google.generativeai as genai
from google.oauth2 import service_account
import io
import struct
import time
from collections import deque
import math
from config import get_config

# Get configuration
config = get_config()

# Setup logging
config.setup_logging()
logger = logging.getLogger(__name__)

# Flask app for dynamic endpoint
app = Flask(__name__)

class MalayalamVoiceBot:
    def __init__(self):
        # Validate configuration
        config.validate_config()
        
        self.setup_google_cloud()
        self.setup_gemini()
        
        # Audio configuration from config
        self.SAMPLE_RATE = config.SAMPLE_RATE
        self.CHANNELS = config.CHANNELS
        self.BYTES_PER_SAMPLE = config.BYTES_PER_SAMPLE
        self.CHUNK_SIZE = config.CHUNK_SIZE
        self.MIN_CHUNK_SIZE = config.MIN_CHUNK_SIZE
        self.MAX_CHUNK_SIZE = config.MAX_CHUNK_SIZE
        
        # Streaming configuration
        self.audio_buffer = deque()
        self.is_speaking = False
        self.conversation_context = []
        
    def setup_google_cloud(self):
        """Setup Google Cloud STT and TTS clients"""
        try:
            # Initialize clients - credentials should be set via environment
            self.speech_client = speech.SpeechClient()
            self.tts_client = tts.TextToSpeechClient()
            
            # Configure speech recognition
            self.speech_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.SAMPLE_RATE,
                language_code=config.SPEECH_LANGUAGE_CODE,
                enable_automatic_punctuation=True,
            )
            
            # Configure TTS voice
            self.voice = tts.VoiceSelectionParams(
                language_code=config.SPEECH_LANGUAGE_CODE,
                name=config.TTS_VOICE_NAME,
                ssml_gender=getattr(tts.SsmlVoiceGender, config.TTS_VOICE_GENDER)
            )
            
            self.audio_config = tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.SAMPLE_RATE
            )
            
            logger.info("Google Cloud STT and TTS initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud services: {e}")
            raise
    
    def setup_gemini(self):
        """Setup Gemini API"""
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
            
            # System prompt from config
            self.system_prompt = config.SYSTEM_PROMPT
            
            logger.info("Gemini API initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise
    
    def validate_chunk_size(self, size):
        """Validate chunk size according to Exotel requirements"""
        if size < self.MIN_CHUNK_SIZE:
            logger.warning(f"Chunk size {size} is below minimum {self.MIN_CHUNK_SIZE}")
            return False
        if size > self.MAX_CHUNK_SIZE:
            logger.warning(f"Chunk size {size} exceeds maximum {self.MAX_CHUNK_SIZE}")
            return False
        if size % 320 != 0:
            logger.warning(f"Chunk size {size} is not a multiple of {config.CHUNK_ALIGNMENT} bytes")
            return False
        return True
    
    async def transcribe_audio(self, audio_data):
        """Transcribe audio using Google Cloud STT"""
        try:
            audio = speech.RecognitionAudio(content=audio_data)
            response = self.speech_client.recognize(
                config=self.speech_config, 
                audio=audio
            )
            
            if response.results:
                transcript = response.results[0].alternatives[0].transcript
                logger.info(f"Transcribed: {transcript}")
                return transcript
            return None
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    async def generate_response(self, user_input):
        """Generate response using Gemini API"""
        try:
            # Add context to the conversation
            self.conversation_context.append(f"User: {user_input}")
            
            # Create prompt with context
            context = "\n".join(self.conversation_context[-config.MAX_CONVERSATION_HISTORY:])
            prompt = f"{self.system_prompt}\n\nConversation context:\n{context}\n\nPlease respond to the user's last message:"
            
            response = self.model.generate_content(prompt)
            ai_response = response.text
            
            self.conversation_context.append(f"Assistant: {ai_response}")
            logger.info(f"Generated response: {ai_response}")
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "Sorry, I couldn't understand that. Could you please repeat?"
    
    async def synthesize_speech(self, text):
        """Convert text to speech using Google Cloud TTS"""
        try:
            synthesis_input = tts.SynthesisInput(text=text)
            
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )
            
            logger.info(f"Synthesized {len(response.audio_content)} bytes of audio")
            return response.audio_content
            
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            return None
    
    def chunk_audio(self, audio_data):
        """Split audio into valid chunks for Exotel"""
        chunks = []
        for i in range(0, len(audio_data), self.CHUNK_SIZE):
            chunk = audio_data[i:i + self.CHUNK_SIZE]
            
            # Pad the last chunk if necessary to maintain 320-byte alignment
            if len(chunk) < self.CHUNK_SIZE and len(chunk) % 320 != 0:
                padding_needed = 320 - (len(chunk) % 320)
                chunk += b'\x00' * padding_needed
            
            if self.validate_chunk_size(len(chunk)):
                chunks.append(chunk)
            else:
                logger.warning(f"Skipping invalid chunk of size {len(chunk)}")
        
        return chunks
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connection from Exotel"""
        logger.info(f"New WebSocket connection: {websocket.remote_address}")
        
        try:
            stream_sid = None
            audio_buffer = b''
            sequence_number = 1
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    event = data.get('event')
                    
                    logger.info(f"Received event: {event}")
                    
                    if event == 'connected':
                        logger.info("WebSocket connected")
                        
                    elif event == 'start':
                        stream_sid = data['stream_sid']
                        call_info = data['start']
                        logger.info(f"Stream started: {stream_sid}")
                        logger.info(f"Call from {call_info.get('from')} to {call_info.get('to')}")
                        
                        # Send initial greeting
                        greeting = config.GREETING_MESSAGE
                        await self.send_audio_response(websocket, stream_sid, greeting, sequence_number)
                        sequence_number += 1
                        
                    elif event == 'media':
                        media_data = data['media']
                        payload = base64.b64decode(media_data['payload'])
                        
                        # Accumulate audio data
                        audio_buffer += payload
                        
                        # Process when we have enough data
                        if len(audio_buffer) >= config.AUDIO_BUFFER_SIZE:
                            transcript = await self.transcribe_audio(audio_buffer)
                            
                            if transcript and transcript.strip():
                                logger.info(f"Processing transcript: {transcript}")
                                
                                # Generate and send response
                                response_text = await self.generate_response(transcript)
                                await self.send_audio_response(websocket, stream_sid, response_text, sequence_number)
                                sequence_number += 1
                            
                            # Reset buffer
                            audio_buffer = b''
                    
                    elif event == 'dtmf':
                        dtmf_data = data['dtmf']
                        logger.info(f"DTMF pressed: {dtmf_data['digit']}")
                        
                    elif event == 'stop':
                        stop_data = data['stop']
                        logger.info(f"Stream stopped: {stop_data.get('reason')}")
                        break
                        
                    elif event == 'mark':
                        mark_data = data['mark']
                        logger.info(f"Mark received: {mark_data.get('name')}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    async def send_audio_response(self, websocket, stream_sid, text, sequence_number):
        """Convert text to speech and send to Exotel"""
        try:
            # Synthesize speech
            audio_data = await self.synthesize_speech(text)
            if not audio_data:
                return
            
            # Send mark to indicate start of audio
            mark_message = {
                "event": "mark",
                "sequence_number": sequence_number,
                "stream_sid": stream_sid,
                "mark": {
                    "name": f"start_audio_{sequence_number}"
                }
            }
            await websocket.send(json.dumps(mark_message))
            
            # Split audio into chunks and send
            chunks = self.chunk_audio(audio_data)
            timestamp = 0
            
            for i, chunk in enumerate(chunks):
                media_message = {
                    "event": "media",
                    "sequence_number": sequence_number + i + 1,
                    "stream_sid": stream_sid,
                    "media": {
                        "chunk": i,
                        "timestamp": str(timestamp),
                        "payload": base64.b64encode(chunk).decode('utf-8')
                    }
                }
                
                await websocket.send(json.dumps(media_message))
                
                # Calculate timestamp for next chunk (in milliseconds)
                chunk_duration_ms = (len(chunk) // self.BYTES_PER_SAMPLE) * 1000 // self.SAMPLE_RATE
                timestamp += chunk_duration_ms
                
                # Small delay to avoid overwhelming the connection
                await asyncio.sleep(0.01)
            
            # Send mark to indicate end of audio
            end_mark_message = {
                "event": "mark",
                "sequence_number": sequence_number + len(chunks) + 1,
                "stream_sid": stream_sid,
                "mark": {
                    "name": f"end_audio_{sequence_number}"
                }
            }
            await websocket.send(json.dumps(end_mark_message))
            
            logger.info(f"Sent audio response with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error sending audio response: {e}")

# Initialize the bot
bot = MalayalamVoiceBot()

# Flask routes for dynamic endpoint
@app.route('/', methods=['GET', 'POST'])
def get_websocket_url():
    """Return WebSocket URL for Exotel to connect to"""
    try:
        # Get WebSocket URL using config
        websocket_url = config.get_websocket_url(request.url_root)
        
        logger.info(f"Returning WebSocket URL: {websocket_url}")
        
        return jsonify({
            "websocket_url": websocket_url,
            "status": "ready"
        })
        
    except Exception as e:
        logger.error(f"Error in get_websocket_url: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "malayalam-voicebot"})

# WebSocket server
async def start_websocket_server():
    """Start the WebSocket server"""
    port = config.WEBSOCKET_PORT
    logger.info(f"Starting WebSocket server on port {port}")
    
    server = await websockets.serve(
        bot.handle_websocket,
        "0.0.0.0",
        port,
        ping_interval=config.WEBSOCKET_PING_INTERVAL,
        ping_timeout=config.WEBSOCKET_PING_TIMEOUT,
        max_size=config.WEBSOCKET_MAX_SIZE
    )
    
    logger.info(f"WebSocket server started on ws://0.0.0.0:{port}")
    return server

def run_flask():
    """Run Flask app in a separate thread"""
    port = config.FLASK_PORT
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=config.DEBUG if hasattr(config, 'DEBUG') else False)

async def main():
    """Main function to start both servers"""
    # Start Flask in a separate thread
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Start WebSocket server
    server = await start_websocket_server()
    
    logger.info("Both servers started successfully")
    
    # Keep the servers running
    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        logger.info("Shutting down servers...")
        server.close()
        await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())