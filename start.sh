#!/bin/bash

# Malayalam Voicebot Startup Script
# This script starts the voicebot application

echo "Starting Malayalam Voicebot..."
echo "================================"

# Check if required environment variables are set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY environment variable is not set"
    exit 1
fi

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Error: GOOGLE_APPLICATION_CREDENTIALS environment variable is not set"
    exit 1
fi

# Set default values if not provided
export PORT=${PORT:-5000}
export WS_PORT=${WS_PORT:-8765}
export FLASK_ENV=${FLASK_ENV:-production}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "Configuration:"
echo "- Flask Port: $PORT"
echo "- WebSocket Port: $WS_PORT"
echo "- Environment: $FLASK_ENV"
echo "- Log Level: $LOG_LEVEL"
echo "================================"

# Start the application
python main.py