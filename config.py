"""
Configuration file for Malayalam Voicebot
"""
import os
import logging

class Config:
    """Configuration class for the voicebot"""
    
    # Server Configuration
    FLASK_PORT = int(os.getenv('PORT', 5000))
    WEBSOCKET_PORT = int(os.getenv('WS_PORT', 8765))
    BASE_URL = os.getenv('BASE_URL', 'http://localhost:5000')
    
    # Google Cloud Configuration
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
    
    # Gemini API Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Audio Configuration (Exotel Requirements)
    SAMPLE_RATE = 8000
    CHANNELS = 1
    BYTES_PER_SAMPLE = 2  # 16-bit
    CHUNK_SIZE = 3200  # 100ms at 8kHz, 16-bit, mono
    MIN_CHUNK_SIZE = 3200
    MAX_CHUNK_SIZE = 100000
    CHUNK_ALIGNMENT = 320  # Must be multiple of 320 bytes
    
    # Language Configuration
    SPEECH_LANGUAGE_CODE = "ml-IN"  # Malayalam
    TTS_VOICE_NAME = "ml-IN-Wavenet-A"
    TTS_VOICE_GENDER = "FEMALE"
    
    # System Prompts
    SYSTEM_PROMPT = """You are a helpful AI assistant that can speak Malayalam. 
    You should respond in Malayalam script (മലയാളം). Keep responses concise and natural for voice conversation.
    Be friendly, helpful, and culturally aware. If the user speaks in English, you can respond in English too.
    Keep your responses under 2-3 sentences for better voice interaction."""
    
    GREETING_MESSAGE = "നമസ്കാരം! ഞാൻ നിങ്ങളുടെ എഐ സഹായിയാണ്. എനിക്ക് എങ്ങനെ സഹായിക്കാം?"
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # WebSocket Configuration
    WEBSOCKET_PING_INTERVAL = 30
    WEBSOCKET_PING_TIMEOUT = 10
    WEBSOCKET_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Audio Processing Configuration
    AUDIO_BUFFER_SIZE = 16000  # 1 second of audio at 8kHz, 16-bit
    SILENCE_THRESHOLD = 0.01
    VAD_FRAME_DURATION = 30  # Voice Activity Detection frame duration in ms
    
    # Conversation Configuration
    MAX_CONVERSATION_HISTORY = 10
    RESPONSE_TIMEOUT = 30  # seconds
    
    @classmethod
    def validate_config(cls):
        """Validate configuration and environment variables"""
        errors = []
        
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY environment variable is required")
        
        if not cls.GOOGLE_APPLICATION_CREDENTIALS:
            errors.append("GOOGLE_APPLICATION_CREDENTIALS environment variable is required")
        
        if cls.CHUNK_SIZE % cls.CHUNK_ALIGNMENT != 0:
            errors.append(f"CHUNK_SIZE must be a multiple of {cls.CHUNK_ALIGNMENT}")
        
        if cls.CHUNK_SIZE < cls.MIN_CHUNK_SIZE:
            errors.append(f"CHUNK_SIZE must be at least {cls.MIN_CHUNK_SIZE}")
        
        if cls.CHUNK_SIZE > cls.MAX_CHUNK_SIZE:
            errors.append(f"CHUNK_SIZE must not exceed {cls.MAX_CHUNK_SIZE}")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format=cls.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(),
            ]
        )
        
        # Suppress noisy logs from libraries
        logging.getLogger('websockets').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('google').setLevel(logging.WARNING)
    
    @classmethod
    def get_websocket_url(cls, request_host=None):
        """Generate WebSocket URL based on configuration"""
        if request_host:
            # Use the request host for dynamic URL generation
            protocol = 'wss' if request_host.startswith('https') else 'ws'
            base_url = request_host.replace('https://', '').replace('http://', '')
            return f"{protocol}://{base_url}/media"
        else:
            # Use configured base URL
            base_url = cls.BASE_URL
            protocol = 'wss' if base_url.startswith('https') else 'ws'
            base_url = base_url.replace('https://', '').replace('http://', '')
            return f"{protocol}://{base_url}/media"

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'INFO'
    
    # Production-specific settings
    WEBSOCKET_MAX_SIZE = 50 * 1024 * 1024  # 50MB for production
    MAX_CONVERSATION_HISTORY = 20

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    LOG_LEVEL = 'DEBUG'
    CHUNK_SIZE = 1600  # Smaller chunks for testing

# Configuration factory
def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'production').lower()
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    return config_map.get(env, ProductionConfig)