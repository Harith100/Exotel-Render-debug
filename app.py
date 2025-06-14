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
        
        # Audio configuration for TTS - Updated for Exotel format
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,  # Exotel expects LINEAR16
            sample_rate_hertz=8000  # Exotel uses 8kHz
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
        """Convert text to speech and send via websocket - Updated for Exotel format"""
        try:
            audio_content = self.voicebot.text_to_speech(text)
            
            if audio_content and not self.websocket.closed:
                # Convert audio to base64 for Exotel format
                audio_b64 = base64.b64encode(audio_content).decode('ascii')
                
                # Send audio in Exotel format - single message with full audio
                message = {
                    'event': 'media',
                    'stream_sid': self.stream_sid,
                    'media': {
                        'payload': audio_b64
                    }
                }
                
                if not self.websocket.closed:
                    self.websocket.send(json.dumps(message))
                
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
        # Initialize voicebot instance
        voicebot = MalayalamVoiceBot()
        
        while not ws.closed:
            try:
                message = ws.receive()
                if message is None:
                    app.logger.info("Received None message, connection might be closing")
                    break

                # Parse JSON message
                try:
                    data = json.loads(message)
                    app.logger.info(f"Received message: {data}")
                except json.JSONDecodeError as e:
                    app.logger.error(f"Invalid JSON received: {message[:100]}...")
                    continue
                
                # Handle different event types - Updated for Exotel format
                event_type = data.get('event', 'unknown')
                
                if event_type == "connected":
                    app.logger.info(f"WebSocket connected: {data}")
                    
                elif event_type == "start":
                    app.logger.info(f"Stream started: {data}")
                    stream_sid = data.get('stream_sid', data.get('streamSid', 'unknown'))
                    
                    # Initialize audio stream for Exotel (8kHz, LINEAR16)
                    audio_stream = AudioStream(8000, 1024)
                    audio_stream.__enter__()
                    
                    # Initialize transcription handler
                    transcription_handler = TranscriptionHandler(voicebot, ws, stream_sid)
                    
                    # Configure speech recognition for Exotel format
                    config = speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # Exotel uses LINEAR16
                        sample_rate_hertz=8000,  # Exotel uses 8kHz
                        language_code="ml-IN",
                        enable_automatic_punctuation=True,
                        model="telephony",
                        use_enhanced=True
                    )
                    
                    streaming_config = speech.StreamingRecognitionConfig(
                        config=config,
                        interim_results=True,
                        single_utterance=False
                    )
                    
                    # Start transcription in separate thread
                    audio_generator = audio_stream.generator()
                    requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
                              for chunk in audio_generator)
                    
                    def transcribe():
                        try:
                            responses = voicebot.speech_client.streaming_recognize(
                                streaming_config, requests
                            )
                            transcription_handler.process_transcript(responses)
                        except Exception as e:
                            app.logger.error(f"Transcription error: {str(e)}")
                    
                    transcription_thread = Thread(target=transcribe)
                    transcription_thread.daemon = True
                    transcription_thread.start()
                    
                    # Store stream data
                    active_streams[stream_sid] = {
                        'stream': audio_stream,
                        'handler': transcription_handler,
                        'thread': transcription_thread
                    }
                    
                elif event_type == "media" and audio_stream:
                    # Handle audio data - Updated for Exotel format
                    media_data = data.get('media', {})
                    payload = media_data.get('payload', '')
                    
                    if payload:
                        try:
                            # Decode base64 audio data (Exotel sends LINEAR16 PCM)
                            chunk = base64.b64decode(payload)
                            audio_stream.fill_buffer(chunk)
                        except Exception as e:
                            app.logger.error(f"Error processing audio chunk: {str(e)}")
                        
                elif event_type == "stop":
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
        # Cleanup
        if stream_sid and stream_sid in active_streams:
            stream_data = active_streams[stream_sid]
            if stream_data['stream']:
                stream_data['stream'].__exit__(None, None, None)
            del active_streams[stream_sid]
        
        if audio_stream:
            audio_stream.__exit__(None, None, None)
            
        app.logger.info("WebSocket connection closed")

@app.route('/media', methods=['GET'])
def media_info():
    """Info endpoint for WebSocket - prevents 400 errors from HTTP requests"""
    return {
        'message': 'This is a WebSocket endpoint. Use wss:// to connect.',
        'websocket_url': 'wss://exotel-render-debug.onrender.com/media',
        'status': 'WebSocket endpoint active',
        'active_streams': len(active_streams)
    }

@app.route('/health')
def health_check():
    """Health check endpoint for deployment"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_streams': len(active_streams),
        'credentials_configured': bool(GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_CREDENTIALS_JSON),
        'gemini_configured': bool(GEMINI_API_KEY)
    }

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

@app.route('/test-tts', methods=['GET', 'POST'])
def test_tts():
    """Test endpoint for TTS functionality"""
    try:
        if request.method == 'GET':
            # GET request - use default text
            text = 'ഹലോ, എങ്ങനെയുണ്ട്?'  # "Hello, how are you?" in Malayalam
        else:
            # POST request - get text from JSON or form data
            if request.is_json:
                data = request.get_json() or {}
                text = data.get('text', 'ഹലോ, എങ്ങനെയുണ്ട്?')
            else:
                text = request.form.get('text', 'ഹലോ, എങ്ങനെയുണ്ട്?')
        
        voicebot = MalayalamVoiceBot()
        audio_content = voicebot.text_to_speech(text)
        
        if audio_content:
            audio_b64 = base64.b64encode(audio_content).decode('ascii')
            return {
                'status': 'success',
                'text': text,
                'audio_length': len(audio_content),
                'audio_base64_preview': audio_b64[:100] + '...',  # First 100 chars for testing
                'method': request.method
            }
        else:
            return {'status': 'error', 'message': 'Failed to generate audio'}
            
    except Exception as e:
        app.logger.error(f"TTS test error: {str(e)}")
        return {'status': 'error', 'message': str(e)}, 500

@app.route('/test-gemini', methods=['GET', 'POST'])
def test_gemini():
    """Test endpoint for Gemini functionality"""
    try:
        if request.method == 'GET':
            # GET request - use default text
            text = 'Hello, how are you?'
        else:
            # POST request - get text from JSON or form data
            if request.is_json:
                data = request.get_json() or {}
                text = data.get('text', 'Hello, how are you?')
            else:
                text = request.form.get('text', 'Hello, how are you?')
        
        voicebot = MalayalamVoiceBot()
        response = voicebot.get_gemini_response(text)
        
        return {
            'status': 'success',
            'input_text': text,
            'gemini_response': response,
            'method': request.method
        }
            
    except Exception as e:
        app.logger.error(f"Gemini test error: {str(e)}")
        return {'status': 'error', 'message': str(e)}, 500

@app.route('/test', methods=['GET'])
def test_all():
    """Combined test endpoint for all components"""
    results = {}
    
    # Test environment variables
    results['environment'] = {
        'google_credentials': bool(GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_CREDENTIALS_JSON),
        'gemini_api_key': bool(GEMINI_API_KEY),
        'active_streams': len(active_streams)
    }
    
    # Test Google Cloud credentials
    try:
        credentials = setup_google_credentials()
        results['credentials'] = {
            'status': 'success' if credentials else 'failed',
            'type': type(credentials).__name__ if credentials else None
        }
    except Exception as e:
        results['credentials'] = {
            'status': 'error',
            'message': str(e)
        }
    
    # Test TTS
    try:
        voicebot = MalayalamVoiceBot()
        audio_content = voicebot.text_to_speech('ഹലോ')
        results['tts'] = {
            'status': 'success' if audio_content else 'failed',
            'audio_length': len(audio_content) if audio_content else 0
        }
    except Exception as e:
        results['tts'] = {
            'status': 'error',
            'message': str(e)
        }
    
    # Test Gemini
    try:
        if 'voicebot' not in locals():
            voicebot = MalayalamVoiceBot()
        response = voicebot.get_gemini_response('Hello')
        results['gemini'] = {
            'status': 'success',
            'response_length': len(response) if response else 0
        }
    except Exception as e:
        results['gemini'] = {
            'status': 'error',
            'message': str(e)
        }
    
    return results
    # Validate environment variables
    if not GOOGLE_APPLICATION_CREDENTIALS and not GOOGLE_CLOUD_CREDENTIALS_JSON:
        app.logger.error("No Google Cloud credentials found. Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_CREDENTIALS_JSON")
        sys.exit(1)
    
    if not GEMINI_API_KEY:
        app.logger.error("GEMINI_API_KEY not set")
        sys.exit(1)

    # Test initialization
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
    app.logger.info(f"WebSocket endpoint: wss://{args.host}:{args.port}/media")
    app.logger.info(f"Health check: http://{args.host}:{args.port}/health")
    app.logger.info(f"TTS Test: http://{args.host}:{args.port}/test-tts")

    server = pywsgi.WSGIServer((args.host, args.port), app, handler_class=WebSocketHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        app.logger.info("Server stopped by user")
    except Exception as e:
        app.logger.error(f"Server error: {str(e)}")
        sys.exit(1)