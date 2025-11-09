from typing import List
import json
import random
import string
from datetime import datetime, timedelta

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

# Google API Imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import os
from dotenv import load_dotenv

load_dotenv()

import logging
import json

# Set up logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_PATH = 'token.json'
CREDENTIALS_PATH = 'credentials.json' # This is the path that was downloaded from Google Cloud Console


# Function to get Google Calendar service
def get_calendar_service():
    """
    Authenticates with the Google Calendar API and returns a service object.
    Handles the OAuth 2.0 flow and token refreshing.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except Exception as e:
            logger.warning(f"Could not load credentials from {TOKEN_PATH}: {e}")
            
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Error refreshing token: {e}")
                creds = None # Force re-authentication
        
        if not creds:
            if not os.path.exists(CREDENTIALS_PATH):
                logger.error(f"Missing credentials file: {CREDENTIALS_PATH}")
                logger.error("Please download it from the Google Cloud Console and save it in this directory.")
                return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                logger.error(f"Error during authentication flow: {e}")
                return None
                
        # Save the credentials for the next run. This is important to avoid re-authentication.
        try:
            with open(TOKEN_PATH, 'w') as token_file:
                token_file.write(creds.to_json())
            logger.info(f"Credentials saved to {TOKEN_PATH}")
        except Exception as e:
            logger.error(f"Error saving token to {TOKEN_PATH}: {e}")

    try:
        service = build('calendar', 'v3', credentials=creds)
        return service
    except Exception as e:
        logger.error(f"Error building Google Calendar service: {e}")
        return None
    
# Google Calendar Tools
@tool
def create_google_calendar_event(
    summary: str,
    start_time: str,
    end_time: str,
    description: str = None,
    location: str = None,
    timezone: str = "Asia/Singapore"
) -> str:
    """
    Creates an event on the user's primary Google Calendar.

    Args:
        summary: The title or summary of the event.
        start_time: The start time in ISO format (e.g., '2025-11-03T10:00:00').
        end_time: The end time in ISO format (e.g., '2025-11-03T11:00:00').
        description: Optional description for the event.
        location: Optional location for the event.
        timezone: The timezone for the event, defaults to 'Asia/Singapore'.

    Returns:
        A JSON string with the event link or an error message.
    """
    service = get_calendar_service()
    if not service:
        return json.dumps({"error": "Failed to get Google Calendar service. Check authentication."})

    event = {
        'summary': summary,
        'location': location,
        'description': description,
        'start': {
            'dateTime': start_time,
            'timeZone': timezone,
        },
        'end': {
            'dateTime': end_time,
            'timeZone': timezone,
        },
    }

    try:
        # 'primary' refers to the user's main calendar
        created_event = service.events().insert(calendarId='primary', body=event).execute()
        logger.info(f"Event created: {created_event.get('summary')}")
        return json.dumps({
            "status": "success",
            "summary": created_event.get("summary"),
            "htmlLink": created_event.get('htmlLink')
        })

    except HttpError as error:
        logger.error(f'An error occurred: {error}')
        return json.dumps({"error": str(error)})
    except Exception as e:
        logger.error(f'An unexpected error occurred: {e}')
        return json.dumps({"error": f"An unexpected error occurred: {e}"})

TOOLS = [create_google_calendar_event]

API_KEY = os.getenv("API_KEY")

llm = ChatOpenAI(
    # model="aisingapore/Llama-SEA-LION-v3.5-70B-R",
    # model="aisingapore/Gemma-SEA-LION-v4-27B-IT", 

    # Chosen Model
    model="aisingapore/Qwen-SEA-LION-v4-32B-IT", 
    temperature=0,
    openai_api_base="https://api.sea-lion.ai/v1", 
    api_key=API_KEY
)

SYSTEM_MESSAGE = (
    "You are a helpful personal assistant that can create events on Google Calendar."
)

agent = create_agent(llm, TOOLS, system_prompt=SYSTEM_MESSAGE)

def run_agent(user_input: str, history: List[BaseMessage]) -> AIMessage:
    """Single-turn agent runner with automatic tool execution via LangGraph."""
    try:
        result = agent.invoke(
            {"messages": history + [HumanMessage(content=user_input)]},
            config={"recursion_limit": 50}
        )
        # Return the last AI message
        return result["messages"][-1]
    except Exception as e:
        # Return error as an AI message so the conversation can continue
        return AIMessage(content=f"Error: {str(e)}\n\nPlease try rephrasing your request or provide more specific details.")


if __name__ == "__main__":
    print("=" * 60)
    print(r"      [0_0]      YOUR PERSONAL")
    print(r"     /|___|\     CALENDAR AI")
    print(r"    / [___] \    -----------")
    print(r"      |___|      READY TO SCHEDULE.")
    print("")
    print("    Google Calendar x LangChain Agent")
    print("=" * 60)

    print("Initializing Google Calendar service...")
    print("If this is your first time, a browser window will open for authentication.")
    if get_calendar_service():
        print("Authentication successful! Calendar service is ready.")
    else:
        print("FATAL ERROR: Could not authenticate with Google Calendar.")
        print("Please check your credentials.json file and network connection.")
        exit(1)

    print("Create Google Calendar events.")
    print()
    print("Some prompt examples that you can use below:")
    print("  - Create an event titled 'Gym' for 2025-11-05 at 6 PM")
    print("  - Create an event with description 'Discuss project updates' on 2025-11-05 from 3 PM to 4 PM")
    print("  - Create an event at location 'Office' on 2025-11-06 from 10 AM to 11 AM")
    print()
    print("Commands: 'quit' or 'exit' to end")
    print("=" * 60)

    history: List[BaseMessage] = []

    while True:
        user_input = input("You: ").strip()

        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'q', ""]:
            print("Goodbye!")
            break

        print("Agent: ", end="", flush=True)
        response = run_agent(user_input, history)
        print(response.content)
        print()

        # Update conversation history
        history += [HumanMessage(content=user_input), response]