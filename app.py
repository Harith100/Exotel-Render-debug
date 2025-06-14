import io
import re
import sys
import argparse
import time
import os
import json
import base64
import signal
import logging
import threading
import tempfile
from datetime import datetime

from flask import Flask, request
from flask_sockets import Sockets
from six.moves import queue
from threading import Thread
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

# Google Cloud imports
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
import google.generativeai as genai
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voicebot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

app = Flask(__name__)
sockets = Sockets(app)
app.logger.setLevel(logging.INFO)

# Environment variables
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
GOOGLE_CLOUD_CREDENTIALS_JSON = os.getenv('GOOGLE_CLOUD_CREDENTIALS_JSON')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def setup_google_credentials():
    """Setup Google Cloud credentials from environment variables"""
    credentials = None
    
    if GOOGLE_CLOUD_CREDENTIALS_JSON:
        # If JSON content is provided directly
        try:
            credentials_info = json.loads(GOOGLE_CLOUD_CREDENTIALS_JSON)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            app.logger.info("Using Google Cloud credentials from JSON content")
            return credentials
        except json.JSONDecodeError as e:
            app.logger.error(f"Invalid JSON in GOOGLE_CLOUD_CREDENTIALS_JSON: {e}")
    
    if GOOGLE_APPLICATION_CREDENTIALS:
        # Check if it's JSON content or file path
        if GOOGLE_APPLICATION_CREDENTIALS.startswith('{'):
            # It's JSON content, create a temporary file
            try:
                credentials_info = json.loads(GOOGLE_APPLICATION_CREDENTIALS)
                
                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json.dump(credentials_info, temp_file)
                temp_file.close()
                
                # Set the environment variable to the temp file path
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name
                app.logger.info(f"Created temporary credentials file: {temp_file.name}")
                
                # Load credentials from the temp file
                credentials = service_account.Credentials.from_service_account_file(temp_file.name)
                return credentials
                
            except json.JSONDecodeError as e:
                app.logger.error(f"Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS: {e}")
        else:
            # It's a file path
            if os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
                credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
                app.logger.info(f"Using Google Cloud credentials from file: {GOOGLE_APPLICATION_CREDENTIALS}")
                return credentials
            else:
                app.logger.error(f"Credentials file not found: {GOOGLE_APPLICATION_CREDENTIALS}")
    
    app.logger.error("No valid Google Cloud credentials found")
    return None

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
else:
    app.logger.error("GEMINI_API_KEY not found in environment variables")

class MalayalamVoiceBot:
    def __init__(self):
        # Setup Google Cloud credentials
        self.credentials = setup_google_credentials()
        if not self.credentials:
            raise Exception("Failed to setup Google Cloud credentials")
        
        # Initialize Google Cloud clients with credentials
        self.speech_client = speech.SpeechClient(credentials=self.credentials)
        self.tts_client = texttospeech.TextToSpeechClient(credentials=self.credentials)
        self.conversation_history = []
        
        # Malayalam TTS voice configuration
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="ml-IN",
            name="ml-IN-Wavenet-A",  # Female voice
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        # Audio configuration for TTS
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000  # Match Exotel's expected rate
        )
        
        app.logger.info("MalayalamVoiceBot initialized")
    
    def get_gemini_response(self, user_text):
        """Get response from Gemini API for Malayalam conversation"""
        try:
            # Add context for Malayalam conversation
            prompt = f"""You are a helpful assistant that can understand and respond in Malayalam. 
            The user said: "{user_text}"
            
            Please respond in Malayalam in a natural, conversational way. Keep responses concise (under 100 words) 
            as this is for a voice conversation. If the user speaks in English, you can respond in English too.
            
            Previous conversation context: {self.conversation_history[-3:] if self.conversation_history else 'None'}
            """
            
            response = model.generate_content(prompt)
            bot_response = response.text.strip()
            
            # Update conversation history
            self.conversation_history.append({
                'user': user_text,
                'bot': bot_response,
                'timestamp': datetime.now().isoformat()
            })
            
            app.logger.info(f"User: {user_text}")
            app.logger.info(f"Bot: {bot_response}")
            
            return bot_response
            
        except Exception as e:
            app.logger.error(f"Error getting Gemini response: {str(e)}")
            return "മാപ്പ്, എനിക്ക് ഇപ്പോൾ മറുപടി നൽകാൻ കഴിയുന്നില്ല." # "Sorry, I can't respond right now."
    
    def text_to_speech(self, text):
        """Convert text to speech using Google Cloud TTS"""
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )
            
            return response.audio_content
        except Exception as e:
            app.logger.error(f"Error in TTS: {str(e)}")
            return None

def signal_handler(sig, frame):
    app.logger.info("Shutting down gracefully...")
    sys.exit(0)

class AudioStream(object):
    """Manages audio streaming for speech recognition"""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self.buff = queue.Queue()
        self.closed = True
        self.transcript_buffer = ""

    def __enter__(self):
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self.closed = True
        self.buff.put(None)

    def fill_buffer(self, in_data):
        """Add audio data to buffer"""
        if not self.closed:
            self.buff.put(in_data)
        return self

    def generator(self):
        while not self.closed:
            chunk = self.buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self.buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

class TranscriptionHandler:
    def __init__(self, voicebot, websocket, stream_sid):
        self.voicebot = voicebot
        self.websocket = websocket
        self.stream_sid = stream_sid
        self.last_transcript = ""
        self.silence_start = None
        self.processing_response = False

    def process_transcript(self, responses):
        """Process transcription responses and generate bot responses"""
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript.strip()
            
            if result.is_final and transcript:
                app.logger.info(f"Final transcript: {transcript}")
                
                # Avoid processing the same transcript multiple times
                if transcript != self.last_transcript and not self.processing_response:
                    self.last_transcript = transcript
                    self.processing_response = True
                    
                    # Get bot response
                    bot_response = self.voicebot.get_gemini_response(transcript)
                    
                    # Convert to speech and send back
                    self.send_audio_response(bot_response)
                    
                    self.processing_response = False

    def send_audio_response(self, text):
        """Convert text to speech and send via websocket"""
        try:
            audio_content = self.voicebot.text_to_speech(text)
            
            if audio_content and not self.websocket.closed:
                # Convert audio to base64 and send via websocket
                audio_b64 = base64.b64encode(audio_content).decode('ascii')
                
                # Split audio into chunks for streaming
                chunk_size = 1024  # Adjust based on needs
                for i in range(0, len(audio_b64), chunk_size):
                    chunk = audio_b64[i:i + chunk_size]
                    
                    message = {
                        'event': 'media',
                        'stream_sid': self.stream_sid,
                        'media': {
                            'payload': chunk
                        }
                    }
                    
                    if not self.websocket.closed:
                        self.websocket.send(json.dumps(message))
                        time.sleep(0.05)  # Small delay between chunks
                
                app.logger.info(f"Sent audio response: {len(audio_content)} bytes")
                
        except Exception as e:
            app.logger.error(f"Error sending audio response: {str(e)}")

# Global instances
active_streams = {}

@app.route('/call', methods=['POST'])
def handle_call():
    """Handle incoming call from Exotel and return XML response"""
    try:
        # Log the incoming call data
        call_data = request.get_json() or request.form.to_dict()
        app.logger.info(f"Incoming call: {call_data}")
        
        # Return XML response to connect to WebSocket
        response = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://exotel-render-debug.onrender.com/media"/>
    </Connect>
</Response>'''
        
        return response, 200, {'Content-Type': 'application/xml'}
        
    except Exception as e:
        app.logger.error(f"Error handling call: {str(e)}")
        return '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Sorry, there was an error. Please try again later.</Say>
</Response>''', 500, {'Content-Type': 'application/xml'}


@sockets.route('/media')
def handle_media_stream(ws):
    app.logger.info("New WebSocket connection accepted")
    
    if not ws:
        app.logger.error("Invalid WebSocket connection")
        return
    
    stream_sid = None
    audio_stream = None
    transcription_handler = None
    transcription_thread = None
    
    try:
        while not ws.closed:
            try:
                message = ws.receive()
                if message is None:
                    app.logger.info("Received None message, connection might be closing")
                    break

                # Handle both Exotel and Twilio formats
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as e:
                    app.logger.error(f"Invalid JSON received: {message[:100]}...")
                    continue
                
                # Handle different event types
                event_type = data.get('event', data.get('type', 'unknown'))
                
                if event_type in ["connected", "connection"]:
                    app.logger.info(f"Connection established: {data}")
                    
                elif event_type in ["start", "stream_start"]:
                    app.logger.info(f"Stream started: {data}")
                    # Initialize your audio processing here
                    # ... rest of your existing start logic ...
                    
                elif event_type in ["media", "audio"]:
                    if audio_stream:
                        # Handle audio data - adapt for Exotel format
                        payload = data.get('payload', data.get('audio', ''))
                        if payload:
                            chunk = base64.b64decode(payload)
                            audio_stream.fill_buffer(chunk)
                        
                elif event_type in ["stop", "stream_end"]:
                    app.logger.info(f"Stream stopped: {data}")
                    break
                else:
                    app.logger.info(f"Unknown event type: {event_type}, data: {data}")
                    
            except Exception as e:
                app.logger.error(f"Error processing message: {str(e)}")
                continue

    except Exception as e:
        app.logger.error(f"WebSocket error: {str(e)}")
    
    finally:
        # Cleanup code remains the same
        if stream_sid and stream_sid in active_streams:
            stream_data = active_streams[stream_sid]
            if stream_data['stream']:
                stream_data['stream'].__exit__(None, None, None)
            del active_streams[stream_sid]
        
        app.logger.info("WebSocket connection closed")

# 3. Add a simple HTTP endpoint to test WebSocket accessibility
@app.route('/media', methods=['GET'])
def media_info():
    """Info endpoint for WebSocket - prevents 400 errors from HTTP requests"""
    return {
        'message': 'This is a WebSocket endpoint. Use wss:// to connect.',
        'websocket_url': 'wss://exotel-render-debug.onrender.com/media',
        'status': 'WebSocket endpoint active'
    }



@app.route('/health')
def health_check():
    """Health check endpoint for deployment"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_streams': len(active_streams)
    }

# 4. Enhanced webhook endpoint
@app.route('/webhook', methods=['POST'])
def exotel_webhook():
    """Enhanced webhook handler for Exotel events"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
            
        app.logger.info(f"Webhook received: {data}")
        
        # Process different webhook events
        event_type = data.get('EventType', data.get('event_type', 'unknown'))
        
        if event_type == 'CallStarted':
            app.logger.info(f"Call started: {data.get('CallSid', 'unknown')}")
        elif event_type == 'CallEnded':
            app.logger.info(f"Call ended: {data.get('CallSid', 'unknown')}")
        
        return {'status': 'received', 'event': event_type}
        
    except Exception as e:
        app.logger.error(f"Webhook error: {str(e)}")
        return {'status': 'error', 'message': str(e)}, 500

if __name__ == '__main__':
    # Validate environment variables
    if not GOOGLE_APPLICATION_CREDENTIALS and not GOOGLE_CLOUD_CREDENTIALS_JSON:
        app.logger.error("No Google Cloud credentials found. Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_CREDENTIALS_JSON")
        sys.exit(1)
    
    if not GEMINI_API_KEY:
        app.logger.error("GEMINI_API_KEY not set")
        sys.exit(1)

    # Initialize voicebot (this will test credentials)
    try:
        voicebot = MalayalamVoiceBot()
        app.logger.info("MalayalamVoiceBot initialized successfully")
    except Exception as e:
        app.logger.error(f"Failed to initialize MalayalamVoiceBot: {str(e)}")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Malayalam Exotel VoiceBot')
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', 5000)), 
                       help='Port for the server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host for the server')
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    app.logger.info(f"Starting Malayalam VoiceBot server on {args.host}:{args.port}")
    app.logger.info(f"WebSocket endpoint: ws://{args.host}:{args.port}/media")
    app.logger.info(f"Health check: http://{args.host}:{args.port}/health")

    server = pywsgi.WSGIServer((args.host, args.port), app, handler_class=WebSocketHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        app.logger.info("Server stopped by user")
    except Exception as e:
        app.logger.error(f"Server error: {str(e)}")
        sys.exit(1)