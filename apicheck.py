from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

gapp=os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not gapp:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. Please set it to the path of your Google credentials JSON file.")

gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to your Gemini API key.")

gcloud_project = os.getenv('GOOGLE_CLOUD_PROJECT')
if not gcloud_project:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set. Please set it to your Google Cloud project ID.")

print(gapp)
print(gemini_api_key)
print(gcloud_project)