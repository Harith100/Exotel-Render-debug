# Exotel Malayalam Voice Bot

A bidirectional Malayalam voice bot powered by Google Cloud Speech-to-Text, Text-to-Speech, and Gemini AI, designed to work with Exotel's streaming API.

## Features

- **Bidirectional Audio Streaming**: Real-time audio processing with Exotel
- **Malayalam Speech Recognition**: Using Google Cloud Speech-to-Text with Malayalam language support
- **Malayalam Text-to-Speech**: Natural-sounding Malayalam voice responses
- **AI-Powered Conversations**: Intelligent responses using Google's Gemini AI
- **Optimized Audio Processing**: Proper chunk size handling for Exotel requirements
- **Comprehensive Logging**: Debug and monitor all operations
- **Ready for Production**: Deployable on Render.com

## Architecture

```
Exotel Call → WebSocket Connection → Speech-to-Text → Gemini AI → Text-to-Speech → Audio Response
```

## Requirements

- Python 3.9+
- Google Cloud Project with Speech and Text-to-Speech APIs enabled
- Google Cloud Service Account with appropriate permissions
- Gemini API key
- Render.com account (for deployment)

## Setup Instructions

### 1. Google Cloud Setup

1. Create a Google Cloud Project
2. Enable the following APIs:
   - Cloud Speech-to-Text API
   - Cloud Text-to-Speech API
3. Create a Service Account with the following roles:
   - Cloud Speech Client
   - Cloud Text-to-Speech Client
4. Download the service account JSON key file

### 2. Gemini API Setup

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Save the API key securely

### 3. Local Development

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account.json"
   export PORT=5000
   export WS_PORT=8765
   ```

4. Run the application:
   ```bash
   python app.py
   ```

### 4. Deployment on Render

1. Fork this repository to your GitHub account

2. Create a new Web Service on Render.com:
   - Connect your GitHub repository
   - Use the following settings:
     - **Environment**: Python
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `python app.py`

3. Set Environment Variables in Render:
   - `GEMINI_API_KEY`: Your Gemini API key
   - `GOOGLE_APPLICATION_CREDENTIALS`: Upload your service account JSON file and set this to the file path
   - `PORT`: 10000 (Render default)
   - `WS_PORT`: 10001

4. Deploy the service

### 5. Exotel Configuration

1. In your Exotel dashboard, create a new Stream Applet
2. Set the URL to your deployed service:
   - **Dynamic Method**: `https://your-render-app.onrender.com/webhook`
   - **Static Method**: `wss://your-render-app.onrender.com/media`

## API Endpoints

- **GET /**: Service information and status
- **GET /health**: Health check endpoint
- **POST /webhook**: Dynamic WebSocket endpoint for Exotel
- **WebSocket /media**: Main WebSocket endpoint for audio streaming

## Audio Specifications

The application handles audio according to Exotel's requirements:

- **Format**: 16-bit, 8kHz, mono PCM (little-endian)
- **Encoding**: Base64
- **Chunk Size**: 3.2k to 100k bytes, multiples of 320 bytes
- **Minimum Chunk**: 3.2k (100ms data)
- **Maximum Chunk**: 100k

## Logging

The application provides comprehensive logging:

- All WebSocket events and audio processing
- Speech recognition results
- Gemini API interactions
- Error handling and debugging information
- Logs are saved to `bot.log` file

## Conversation Flow

1. **Call Initiated**: User calls the Exotel number
2. **WebSocket Connection**: Exotel establishes WebSocket connection
3. **Welcome Message**: Bot greets user in Malayalam
4. **Speech Processing**: Continuous audio processing and recognition
5. **AI Response**: Gemini generates appropriate Malayalam responses
6. **Voice Synthesis**: Responses converted to speech and played back
7. **Call End**: Clean connection termination

## Error Handling

The application includes robust error handling for:

- Network connectivity issues
- Google Cloud API failures
- Audio processing errors
- WebSocket connection problems
- Invalid audio formats

## Performance Optimization

- Efficient audio chunking
- Conversation history management (last 5 exchanges)
- Asynchronous processing
- Memory-efficient audio buffering
- Proper silence detection

## Troubleshooting

### Common Issues

1. **Audio Quality Problems**:
   - Check chunk size configuration
   - Verify network connectivity
   - Monitor audio buffer sizes

2. **Speech Recognition Failures**:
   - Ensure Malayalam language model is properly configured
   - Check audio format and sample rate
   - Verify Google Cloud credentials

3. **TTS Issues**:
   - Confirm Malayalam voice is available
   - Check audio encoding settings
   - Verify Text-to-Speech API quotas

4. **WebSocket Connection Problems**:
   - Check firewall settings
   - Verify WebSocket URL configuration
   - Monitor connection logs

### Debug Mode

Enable detailed logging by setting log level to DEBUG:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs for error messages
3. Create an issue in the GitHub repository

## Changelog

### v1.0.0
- Initial release
- Bidirectional Malayalam voice bot
- Google Cloud integration
- Gemini AI responses
- Render.com deployment ready