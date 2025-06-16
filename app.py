import asyncio
import json
import base64
import logging
import os
import websockets
from websockets.server import serve
from flask import Flask, request, jsonify
from flask_sock import Sock
import threading
from google.cloud import speech
from google.cloud import texttospeech
import google.generativeai as genai
from datetime import datetime
import wave
import io
import struct
from typing import Optional, Dict, Any
import time
from google.oauth2 import service_account


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

# Flask app for HTTP endpoints
app = Flask(__name__)

sock = Sock(app)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Google Cloud credentials should be set via environment variable
    # GOOGLE_APPLICATION_CREDENTIALS should point to your service account JSON file
    
    # Gemini API configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Server configuration
    HTTP_PORT = int(os.getenv('PORT', 5000))
    WS_PORT = int(os.getenv('WS_PORT', 8765))
    
    # Audio configuration - matching Exotel specs
    SAMPLE_RATE = 8000  # 8kHz as per Exotel
    CHANNELS = 1        # mono
    SAMPLE_WIDTH = 2    # 16-bit
    
    # Chunk size configuration (as per Exotel requirements)
    MIN_CHUNK_SIZE = 3200   # 100ms data (3.2k)
    MAX_CHUNK_SIZE = 100000 # 100k
    CHUNK_MULTIPLE = 320    # Must be multiple of 320 bytes

# Initialize Google Cloud clients
# Initialize Google Cloud clients with credentials from env variable
try:
    credentials_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not credentials_json:
        raise ValueError("GOOGLE_CREDENTIALS_JSON environment variable is missing")

    service_account_info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    speech_client = speech.SpeechClient(credentials=credentials)
    tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
    
    logger.info("✅ Google Cloud clients initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Google Cloud clients: {e}")
    speech_client = None
    tts_client = None


# Initialize Gemini
try:
    genai.configure(api_key=Config.GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
    logger.info("Gemini API initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {e}")
    gemini_model = None

class AudioProcessor:
    """Handle audio processing for speech recognition and synthesis"""
    
    @staticmethod
    def decode_audio(base64_payload: str) -> bytes:
        """Decode base64 audio payload from Exotel"""
        try:
            audio_data = base64.b64decode(base64_payload)
            logger.debug(f"Decoded audio chunk: {len(audio_data)} bytes")
            return audio_data
        except Exception as e:
            logger.error(f"Failed to decode audio: {e}")
            return b''
    
    @staticmethod
    def encode_audio(audio_data: bytes) -> str:
        """Encode audio data to base64 for Exotel"""
        try:
            encoded = base64.b64encode(audio_data).decode('utf-8')
            logger.debug(f"Encoded audio chunk: {len(audio_data)} bytes -> {len(encoded)} chars")
            return encoded
        except Exception as e:
            logger.error(f"Failed to encode audio: {e}")
            return ''
    
    @staticmethod
    def ensure_chunk_size(audio_data: bytes) -> bytes:
        """Ensure audio chunk size meets Exotel requirements"""
        chunk_size = len(audio_data)
        
        if chunk_size < Config.MIN_CHUNK_SIZE:
            # Pad with silence if too small
            padding_needed = Config.MIN_CHUNK_SIZE - chunk_size
            silence = b'\x00' * padding_needed
            audio_data += silence
            logger.debug(f"Padded audio chunk from {chunk_size} to {len(audio_data)} bytes")
        
        elif chunk_size > Config.MAX_CHUNK_SIZE:
            # Truncate if too large
            audio_data = audio_data[:Config.MAX_CHUNK_SIZE]
            logger.debug(f"Truncated audio chunk from {chunk_size} to {len(audio_data)} bytes")
        
        # Ensure it's a multiple of 320 bytes
        remainder = len(audio_data) % Config.CHUNK_MULTIPLE
        if remainder != 0:
            padding_needed = Config.CHUNK_MULTIPLE - remainder
            audio_data += b'\x00' * padding_needed
            logger.debug(f"Added {padding_needed} bytes padding for 320-byte alignment")
        
        return audio_data

class SpeechProcessor:
    """Handle speech-to-text and text-to-speech operations"""
    
    def __init__(self):
        self.audio_buffer = bytearray()
        self.last_speech_time = time.time()
        self.speech_timeout = 2.0  # 2 seconds of silence before processing
    
    async def process_speech(self, audio_data: bytes) -> Optional[str]:
        """Process audio data and return recognized Malayalam text"""
        if not speech_client:
            logger.error("Speech client not initialized")
            return None
        
        try:
            self.audio_buffer.extend(audio_data)
            self.last_speech_time = time.time()
            
            # Check if we have enough audio data and silence timeout
            if len(self.audio_buffer) < Config.MIN_CHUNK_SIZE * 3:  # Wait for at least 300ms
                return None
            
            if time.time() - self.last_speech_time < self.speech_timeout:
                return None
            
            # Convert audio buffer to proper format
            audio_content = bytes(self.audio_buffer)
            
            # Configure recognition
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=Config.SAMPLE_RATE,
                language_code="ml-IN",  # Malayalam
                enable_automatic_punctuation=True,
                model="latest_long"
            )
            
            audio = speech.RecognitionAudio(content=audio_content)
            
            # Perform speech recognition
            response = speech_client.recognize(config=config, audio=audio)
            
            # Clear buffer after processing
            self.audio_buffer.clear()
            
            if response.results:
                transcript = response.results[0].alternatives[0].transcript
                logger.info(f"Speech recognized: {transcript}")
                return transcript
            else:
                logger.debug("No speech recognized")
                return None
                
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            self.audio_buffer.clear()
            return None
    
    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Convert Malayalam text to speech"""
        if not tts_client:
            logger.error("TTS client not initialized")
            return None
        
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code="ml-IN",  # Malayalam
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=Config.SAMPLE_RATE
            )
            
            response = tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            logger.info(f"TTS synthesized for text: {text[:50]}...")
            return response.audio_content
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return None

class GeminiProcessor:
    """Handle Gemini API interactions"""
    
    def __init__(self):
        self.conversation_history = []
    
    async def get_response(self, user_input: str) -> str:
        """Get response from Gemini for Malayalam input"""
        if not gemini_model:
            logger.error("Gemini model not initialized")
            return "ക്ഷമിക്കണം, എനിക്ക് ഇപ്പോൾ മറുപടി നൽകാൻ കഴിയുന്നില്ല."  # Sorry, I can't respond right now
        
        try:
            # Add context for Malayalam conversation
            system_prompt = """You are a helpful AI assistant that responds in Malayalam. 
            Keep responses concise and natural for voice conversation. 
            Respond only in Malayalam language."""
            
            # Build conversation context
            context = system_prompt + "\n\n"
            for msg in self.conversation_history[-5:]:  # Keep last 5 exchanges
                context += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n"
            context += f"User: {user_input}\nAssistant: "
            
            response = gemini_model.generate_content(context)
            bot_response = response.text.strip()

            

            
            # Store conversation
            self.conversation_history.append({
                'user': user_input,
                'assistant': bot_response,
                'timestamp': datetime.now().isoformat()
            })
            
            if not bot_response:
                logger.warning("Empty Gemini response")
                return "ക്ഷമിക്കണം, മറുപടി ലഭിച്ചില്ല."

            logger.info(f"Gemini response: {bot_response[:50]}...")
            return bot_response
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "ക്ഷമിക്കണം, എനിക്ക് ഇപ്പോൾ മറുപടി നൽകാൻ കഴിയുന്നില്ല."

class ExotelWebSocketHandler:
    """Handle WebSocket communication with Exotel"""
    
    def __init__(self, websocket, path):
        self.websocket = websocket
        self.path = path
        self.stream_sid = None
        self.call_sid = None
        self.sequence_number = 0
        self.speech_processor = SpeechProcessor()
        self.gemini_processor = GeminiProcessor()
        logger.info(f"New WebSocket connection: {path}")
    
    async def handle_connection(self):
        """Main WebSocket message handler"""
        try:
            async for message in self.websocket:
                await self.process_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {self.stream_sid}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    async def process_message(self, message: str):
        """Process incoming WebSocket message from Exotel"""
        try:
            data = json.loads(message)
            event = data.get('event')
            
            logger.debug(f"Received event: {event}")
            
            if event == 'connected':
                await self.handle_connected(data)
            elif event == 'start':
                await self.handle_start(data)
            elif event == 'media':
                await self.handle_media(data)
            elif event == 'dtmf':
                await self.handle_dtmf(data)
            elif event == 'stop':
                await self.handle_stop(data)
            elif event == 'mark':
                await self.handle_mark(data)
            else:
                logger.warning(f"Unknown event type: {event}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def handle_connected(self, data):
        """Handle connected event"""
        logger.info("WebSocket connected to Exotel")
    
    async def handle_start(self, data):
        """Handle start event"""
        start_data = data.get('start', {})
        self.stream_sid = start_data.get('stream_sid')
        self.call_sid = start_data.get('call_sid')
        
        logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}")
        
        # Send welcome message
        welcome_text = "നമസ്കാരം! ഞാൻ നിങ്ങളുടെ സഹായിയാണ്. എന്തെങ്കിലും ചോദിക്കാനുണ്ടോ?"  # Hello! I'm your assistant. Do you have any questions?
        await self.send_tts_response(welcome_text)
    
    async def handle_media(self, data):
        """Handle media event (incoming audio)"""
        media_data = data.get('media', {})
        payload = media_data.get('payload', '')
        
        if payload:
            # Decode audio
            audio_data = AudioProcessor.decode_audio(payload)
            
            # Process speech
            transcript = await self.speech_processor.process_speech(audio_data)
            
            if transcript:
                logger.info(f"User said: {transcript}")
                
                # Get response from Gemini
                bot_response = await self.gemini_processor.get_response(transcript)
                
                # Convert response to speech and send
                await self.send_tts_response(bot_response)
    
    async def handle_dtmf(self, data):
        """Handle DTMF event"""
        dtmf_data = data.get('dtmf', {})
        digit = dtmf_data.get('digit')
        logger.info(f"DTMF received: {digit}")
    
    async def handle_stop(self, data):
        """Handle stop event"""
        logger.info(f"Stream stopped: {self.stream_sid}")
    
    async def handle_mark(self, data):
        """Handle mark event"""
        mark_data = data.get('mark', {})
        name = mark_data.get('name')
        logger.debug(f"Mark received: {name}")
    
    async def send_tts_response(self, text: str):
        """Convert text to speech and send to Exotel"""
        try:
            # Synthesize speech
            audio_data = await self.speech_processor.synthesize_speech(text)
            
            if not audio_data:
                logger.error("Failed to synthesize speech")
                return
            
            # Split audio into chunks and send
            await self.send_audio_chunks(audio_data)
            
        except Exception as e:
            logger.error(f"Error sending TTS response: {e}")
    
    async def send_audio_chunks(self, audio_data: bytes):
        """Send audio data in chunks to Exotel"""
        try:
            chunk_size = Config.MIN_CHUNK_SIZE * 4  # Use 4x minimum for better performance
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                
                # Ensure chunk meets requirements
                chunk = AudioProcessor.ensure_chunk_size(chunk)
                
                # Encode chunk
                encoded_chunk = AudioProcessor.encode_audio(chunk)
                
                # Create media message
                media_message = {
                    "event": "media",
                    "sequence_number": self.sequence_number,
                    "stream_sid": self.stream_sid,
                    "media": {
                        "chunk": i // chunk_size,
                        "timestamp": str(int(time.time() * 1000)),
                        "payload": encoded_chunk
                    }
                }
                
                # Send message
                await self.websocket.send(json.dumps(media_message))
                self.sequence_number += 1
                
                logger.debug(f"Sent audio chunk {i // chunk_size}")
            
            # Send mark to track completion
            mark_message = {
                "event": "mark",
                "sequence_number": self.sequence_number,
                "stream_sid": self.stream_sid,
                "mark": {
                    "name": f"audio_complete_{int(time.time())}"
                }
            }
            
            await self.websocket.send(json.dumps(mark_message))
            self.sequence_number += 1
            
        except Exception as e:
            logger.error(f"Error sending audio chunks: {e}")

# WebSocket server handler
async def websocket_handler(websocket, path):
    """WebSocket connection handler"""
    handler = ExotelWebSocketHandler(websocket, path)
    await handler.handle_connection()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'speech_client': speech_client is not None,
            'tts_client': tts_client is not None,
            'gemini_model': gemini_model is not None
        }
    })

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    """Dynamic WebSocket endpoint for Exotel"""
    try:
        if request.method == 'POST':
            data = request.get_json(silent=True) or request.form.to_dict()
        else:  # GET method (used by Exotel for initial webhook ping or setup)
            data = request.args.to_dict()

        logger.info(f"🔁 Webhook triggered. Method: {request.method}")
        logger.info(f"📨 Data: {data}")

        # Construct WebSocket URL dynamically
        ws_url = f"wss://{request.host}/media"

        return jsonify({
            'websocket_url': ws_url,
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'Exotel Malayalam Voice Bot',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'webhook': '/webhook',
            'websocket': 'wss://<host>/media'
        }
    })

@sock.route('/media')
def media(ws):
    logger.info("📡 WebSocket /media connected.")
    while True:
        data = ws.receive()
        if data is None:
            break
        logger.info(f"🎧 Received chunk: {len(data)} bytes")


def run_flask_app():
    """Run Flask app in a separate thread"""
    app.run(host='0.0.0.0', port=Config.HTTP_PORT, debug=False)

async def main():
    """Main function to start both HTTP and WebSocket servers"""
    logger.info("Starting Exotel Malayalam Voice Bot")
    
    # Validate configuration
    if not Config.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set")
        return
    
    # Start Flask app in background thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()
    
    logger.info(f"HTTP server started on port {Config.HTTP_PORT}")
    
    # Start WebSocket server
    logger.info(f"Starting WebSocket server on port {Config.WS_PORT}")
    
    async with serve(websocket_handler, "0.0.0.0", Config.WS_PORT):
        logger.info("WebSocket server started successfully")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())