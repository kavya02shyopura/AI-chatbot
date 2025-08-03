import streamlit as st
import requests
import os
import json
import yfinance as yf
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, END
from langchain.tools import Tool
from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime
import uuid

# Define the state schema for LangGraph
class ChatState(TypedDict):
    user_input: str
    service_decision: Optional[str]
    response: Optional[str]
    chat_history: List[Dict[str, str]]
    error: Optional[str]

# Load environment variables from .env file
import pathlib

# Get the directory where the script is located
script_dir = pathlib.Path(__file__).parent.absolute()
env_path = script_dir / '.env'

# Load .env file from the script directory
load_dotenv(env_path)

# Chat History Management Functions
def get_chat_history_dir():
    """Get the directory for storing chat history files."""
    history_dir = script_dir / "chat_history"
    history_dir.mkdir(exist_ok=True)
    return history_dir

def save_chat_session(session_id: str, messages: List[Dict[str, str]], title: str = None):
    """Save a chat session to a JSON file."""
    history_dir = get_chat_history_dir()
    
    # Generate title if not provided
    if not title:
        if messages:
            # Use first user message as title (truncated)
            first_user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "Untitled Chat")
            title = first_user_msg[:50] + "..." if len(first_user_msg) > 50 else first_user_msg
        else:
            title = "Untitled Chat"
    
    chat_data = {
        "session_id": session_id,
        "title": title,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "message_count": len(messages),
        "messages": messages
    }
    
    file_path = history_dir / f"{session_id}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)
    
    return file_path

def load_chat_session(session_id: str) -> Dict[str, Any]:
    """Load a chat session from a JSON file."""
    history_dir = get_chat_history_dir()
    file_path = history_dir / f"{session_id}.json"
    
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_all_chat_sessions() -> List[Dict[str, Any]]:
    """Get all saved chat sessions."""
    history_dir = get_chat_history_dir()
    sessions = []
    
    for file_path in history_dir.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                sessions.append(session_data)
        except Exception as e:
            print(f"Error loading session {file_path}: {e}")
    
    # Sort by last updated (newest first)
    sessions.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
    return sessions

def delete_chat_session(session_id: str) -> bool:
    """Delete a chat session file."""
    history_dir = get_chat_history_dir()
    file_path = history_dir / f"{session_id}.json"
    
    if file_path.exists():
        file_path.unlink()
        return True
    return False

def update_chat_session(session_id: str, messages: List[Dict[str, str]]):
    """Update an existing chat session with new messages."""
    session_data = load_chat_session(session_id)
    if session_data:
        session_data["messages"] = messages
        session_data["last_updated"] = datetime.now().isoformat()
        session_data["message_count"] = len(messages)
        
        history_dir = get_chat_history_dir()
        file_path = history_dir / f"{session_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

# Debug: Print environment loading info
print(f"Script directory: {script_dir}")
print(f"Looking for .env at: {env_path}")
print(f".env file exists: {env_path.exists()}")

# Set API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = None
if GROQ_API_KEY not in [None, "", "your_groq_api_key_here"]:
    llm = ChatOpenAI(
        openai_api_key=GROQ_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
        model_name="llama3-8b-8192",
        temperature=0
    )

st.set_page_config(
    page_title="AI Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)


# Add CSS to make input box width adjust to content length
st.markdown(
    """
    <style>
    /* Make the input text box width fit the content */
    div.stTextInput > div > div > input {
        width: 100% !important;
        min-width: 200px;
        max-width: 900px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Modern dark theme CSS
st.markdown(
    """
    <style>
    /* Modern dark theme styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }
    
    /* Clean header without box */
    .header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 0;
    }
    
    .header h1 {
        color: #1a1a1a;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header p {
        color: #666;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Chat container with dark theme */
    .chat-container {
        background: #f8f9fa;
        border: none;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Message styling with modern design */
    .message {
        margin-bottom: 1rem;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 15%;
        text-align: right;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .ai-message {
        background: white;
        color: #2c3e50;
        margin-right: 15%;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .error-message {
        background: #fee;
        color: #c53030;
        border: 1px solid #fed7d7;
        margin-right: 15%;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        background: #f8f9fa;
        color: #2c3e50;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
        background: white;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #6c757d;
        opacity: 0.8;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Secondary button */
    .secondary-btn {
        background: #6c757d !important;
        color: white !important;
        border-radius: 12px !important;
    }
    
    .secondary-btn:hover {
        background: #545b62 !important;
        transform: translateY(-1px);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .sidebar-title {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    /* Status indicators */
    .status-success {
        color: #38a169;
        font-weight: 500;
    }
    
    .status-warning {
        color: #d69e2e;
        font-weight: 500;
    }
    
    .status-error {
        color: #e53e3e;
        font-weight: 500;
    }
    
    /* Hide Streamlit defaults except MainMenu to keep sidebar toggle */
    #MainMenu {visibility: visible;}
    footer {visibility: hidden;}
    /*header {visibility: hidden;}*/
    
    /* Responsive design */
    @media (max-width: 768px) {
        .user-message, .ai-message {
            margin-left: 0;
            margin-right: 0;
        }
        
        .header h1 {
            font-size: 2.2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Clean header
st.markdown(
    """
    <div class="header">
        <h1>AI Assistant</h1>
        <p>Your intelligent companion for information and assistance</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Check if API key is properly set
if GEMINI_API_KEY in [None, "your_gemini_api_key_here", ""]:
    st.error("⚠️ Please set your Gemini API key! Add it to your environment variables as GEMINI_API_KEY or update the code.")
    st.info("Get your API key from your Gemini API provider")
    st.stop()

# Add checks for other API keys
if SERPAPI_KEY in [None, "", "your_serpapi_key_here"]:
    st.warning("SerpAPI key is not set. News features will not work.")
if WEATHER_API_KEY in [None, "", "your_weather_api_key_here"]:
    st.warning("Weather API key is not set. Weather features will not work.")
if ALPHA_VANTAGE_API_KEY in [None, "", "your_alpha_vantage_api_key_here"]:
    st.warning("Alpha Vantage API key is not set. Stock features will not work.")
if GROQ_API_KEY in [None, "", "your_groq_api_key_here"]:
    st.warning("Groq API key is not set. LangGraph agent will not work.")

# API test section will be added after function definitions

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "current_session_id" not in st.session_state:
    st.session_state["current_session_id"] = str(uuid.uuid4())

if "chat_sessions" not in st.session_state:
    st.session_state["chat_sessions"] = get_all_chat_sessions()

if "show_save_dialog" not in st.session_state:
    st.session_state["show_save_dialog"] = False

# Display chat messages directly without container
for msg in st.session_state["messages"]:
    role = msg.get("role", "")
    content = msg.get("content", "")
    
    if role == "user":
        st.markdown(
            f'<div class="message user-message">{content}</div>',
            unsafe_allow_html=True
        )
    else:
        css_class = "ai-message"
        if isinstance(content, str) and content.startswith("Error:"):
            css_class = "error-message"
        
        st.markdown(
            f'<div class="message {css_class}">{content}</div>',
            unsafe_allow_html=True
        )

def extract_stock_symbols(user_input):
    """
    Extract multiple stock symbols from user input using improved heuristics and dictionary.
    """
    # Extended dictionary for common stocks (including Indian stocks)
    stock_dict = {
        # Indian Stocks
        "tata": "TATAMOTORS.NS",
        "tata motors": "TATAMOTORS.NS",
        "tatamotors": "TATAMOTORS.NS",
        "tata steel": "TATASTEEL.NS",
        "tatasteel": "TATASTEEL.NS",
        "reliance": "RELIANCE.NS",
        "reliance industries": "RELIANCE.NS",
        "infosys": "INFY.NS",
        "tcs": "TCS.NS",
        "wipro": "WIPRO.NS",
        "hcl": "HCLTECH.NS",
        "hcl tech": "HCLTECH.NS",
        "tech mahindra": "TECHM.NS",
        "lt": "LT.NS",
        "bharti airtel": "BHARTIARTL.NS",
        "airtel": "BHARTIARTL.NS",
        "axis bank": "AXISBANK.NS",
        "hdfc": "HDFCBANK.NS",
        "hdfc bank": "HDFCBANK.NS",
        "icici": "ICICIBANK.NS",
        "icici bank": "ICICIBANK.NS",
        "indusind": "INDUSINDBK.NS",
        "sbi": "SBIN.NS",
        "state bank": "SBIN.NS",
        "itc": "ITC.NS",
        "maruti": "MARUTI.NS",
        "maruti suzuki": "MARUTI.NS",
        "hindalco": "HINDALCO.NS",
        "jsw steel": "JSWSTEEL.NS",
        "sun pharma": "SUNPHARMA.NS",
        "dr reddy": "DRREDDY.NS",
        "dr reddys": "DRREDDY.NS",
        "bajaj auto": "BAJAJ-AUTO.NS",
        "bajaj": "BAJAJ-AUTO.NS",
        "hero moto": "HEROMOTOCO.NS",
        "hero": "HEROMOTOCO.NS",
        "eicher": "EICHERMOT.NS",
        "eicher motors": "EICHERMOT.NS",
        "ashok leyland": "ASHOKLEY.NS",
        "ashok": "ASHOKLEY.NS",
        "mahindra": "M&M.NS",
        "mahindra and mahindra": "M&M.NS",
        "cipla": "CIPLA.NS",
        "divis": "DIVISLAB.NS",
        "divis labs": "DIVISLAB.NS",
        "nestle": "NESTLEIND.NS",
        "nestle india": "NESTLEIND.NS",
        "britannia": "BRITANNIA.NS",
        "hul": "HINDUNILVR.NS",
        "hindustan unilever": "HINDUNILVR.NS",
        "unilever": "HINDUNILVR.NS",
        "asian paints": "ASIANPAINT.NS",
        "paints": "ASIANPAINT.NS",
        "ultratech": "ULTRACEMCO.NS",
        "ultratech cement": "ULTRACEMCO.NS",
        "cement": "ULTRACEMCO.NS",
        "grasim": "GRASIM.NS",
        "adani": "ADANIENT.NS",
        "adani enterprises": "ADANIENT.NS",
        "adani ports": "ADANIPORTS.NS",
        "adani power": "ADANIPOWER.NS",
        "power grid": "POWERGRID.NS",
        "ongc": "ONGC.NS",
        "oil and gas": "ONGC.NS",
        "coal india": "COALINDIA.NS",
        "coal": "COALINDIA.NS",
        "ntpc": "NTPC.NS",
        "bajaj finserv": "BAJFINANCE.NS",
        "bajaj finance": "BAJFINANCE.NS",
        "kotak": "KOTAKBANK.NS",
        "kotak bank": "KOTAKBANK.NS",
        "yes bank": "YESBANK.NS",
        "yes": "YESBANK.NS",
        
        # US Stocks
        "apple": "AAPL",
        "google": "GOOGL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "facebook": "META",
        "tesla": "TSLA",
        "netflix": "NFLX",
        "ibm": "IBM",
        "intel": "INTC",
        "nvidia": "NVDA",
        "amd": "AMD",
        "oracle": "ORCL",
        "salesforce": "CRM",
        "adobe": "ADBE",
        "paypal": "PYPL",
        "visa": "V",
        "mastercard": "MA",
        "coca cola": "KO",
        "coke": "KO",
        "mcdonalds": "MCD",
        "walmart": "WMT",
        "disney": "DIS",
        "nike": "NKE",
        "starbucks": "SBUX"
    }
    
    user_input_lower = user_input.lower()
    symbols_found = set()
    
    # Blacklist of common non-stock words to exclude
    blacklist = {"stock", "price", "quote", "shares", "share", "market", "of", "the", "and", "for", "in", "on", "at", "to", "with", "a", "an"}
    
    # Check for exact matches in dictionary
    for key in stock_dict:
        if key in user_input_lower:
            symbols_found.add(stock_dict[key])
    
    # Look for common stock-related patterns
    import re
    
    # Pattern for stock symbols (3-5 letters, often in caps)
    symbol_pattern = r'\b[A-Z]{3,5}\b'
    symbols = re.findall(symbol_pattern, user_input.upper())
    for sym in symbols:
        if sym.lower() not in blacklist:
            symbols_found.add(sym)
    
    # Look for words that might be company names
    words = user_input_lower.split()
    for word in words:
        if word in stock_dict:
            symbols_found.add(stock_dict[word])
        elif word not in blacklist:
            # Avoid adding blacklisted words as symbols
            pass
    
    # If no symbols found, try to extract Indian company names
    if not symbols_found:
        # Look for common Indian company patterns
        indian_companies = [
            "tata", "reliance", "infosys", "tcs", "wipro", "hcl", "tech mahindra", "lt", 
            "bharti", "airtel", "axis", "hdfc", "icici", "indusind", "sbi", "state bank",
            "itc", "maruti", "hindalco", "jsw", "sun pharma", "dr reddy", "bajaj", "hero",
            "eicher", "ashok", "mahindra", "cipla", "divis", "nestle", "britannia", "hul",
            "asian paints", "ultratech", "grasim", "adani", "power grid", "ongc", "coal",
            "ntpc", "kotak", "yes bank"
        ]
        
        for company in indian_companies:
            if company in user_input_lower:
                # Try to find the most specific match
                for key in stock_dict:
                    if company in key and key in stock_dict:
                        symbols_found.add(stock_dict[key])
                        break
        
        # If still no symbols found, return default
        if not symbols_found:
            return ["AAPL"]
    
    # Remove duplicates and prioritize .NS symbols
    final_symbols = []
    ns_symbols = [s for s in symbols_found if s.endswith('.NS')]
    other_symbols = [s for s in symbols_found if not s.endswith('.NS')]
    
    # Add .NS symbols first, then others
    final_symbols.extend(ns_symbols)
    final_symbols.extend(other_symbols)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_symbols = []
    for symbol in final_symbols:
        if symbol not in seen:
            seen.add(symbol)
            unique_symbols.append(symbol)
    
    return unique_symbols

def fetch_alpha_vantage_data(query):
    """
    Fetch stock data from Alpha Vantage API.
    """
    if ALPHA_VANTAGE_API_KEY in [None, "", "your_alpha_vantage_api_key_here"]:
        return "Error: Alpha Vantage API key is not set."
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": query,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        # print(f"Alpha Vantage API response for symbol '{query}': {data}")  # Debug log
        return data
    except Exception as e:
        return f"Error fetching Alpha Vantage data: {str(e)}"

def fetch_serpapi_data(query):
    if SERPAPI_KEY in [None, "", "your_serpapi_key_here"]:
        return "Error: SerpAPI key is not set."
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google_news",
        "num": 3,
        "hl": "en"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        return f"Error fetching SerpAPI data: {str(e)}"

def fetch_weather_data(location):
    if WEATHER_API_KEY in [None, "", "your_weather_api_key_here"]:
        return "Error: Weather API key is not set."
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

def extract_location_from_weather_query(user_input):
    """
    Extract location from weather-related queries.
    """
    user_input_lower = user_input.lower()
    
    # Common weather keywords to remove (including contractions)
    weather_keywords = ["weather", "temperature", "forecast", "rain", "sunny", "cloudy", "hot", "cold", "today", "tomorrow", "now", "current", "what's", "what", "is", "the", "in", "of", "for", "at", "how", "like", "how's", "what's", "it's", "that's", "there's"]
    
    # Remove weather keywords to get location
    words = user_input_lower.split()
    location_words = [word for word in words if word not in weather_keywords and len(word) > 2]
    
    print(f"Debug - Input: '{user_input}' -> Words: {words} -> Location words: {location_words}")  # Debug log
    
    # If we have location words, join them
    if location_words:
        location = " ".join(location_words)
        # Clean up any remaining common words
        location = location.replace("today's", "").replace("tomorrow's", "").strip()
        if location:
            print(f"Debug - Final location: '{location}'")  # Debug log
            return location
    
    # Default locations if no specific location mentioned
    print(f"Debug - Using default location: 'New York'")  # Debug log
    return "New York"  # Default fallback

def select_service(user_input):
    """
    Improved keyword-based selection logic to decide which service to call.
    """
    user_input_lower = user_input.lower()
    
    # Stock-related keywords (expanded)
    stock_keywords = [
        "stock", "price", "market", "finance", "share", "quote", "trading", "invest", "portfolio",
        "stock price", "share price", "market price", "current price", "trading price",
        "stock quote", "share quote", "market quote", "price quote", "stock value",
        "share value", "market value", "stock worth", "share worth", "market worth"
    ]
    if any(keyword in user_input_lower for keyword in stock_keywords):
        return "alpha_vantage"
    
    # Check for common company names that are likely stock queries, including Indian companies
    company_names = [
        "apple", "google", "microsoft", "amazon", "facebook", "tesla", "netflix", "ibm", "intel",
        "nvidia", "amd", "oracle", "salesforce", "adobe", "paypal", "visa", "mastercard",
        "coca cola", "coke", "mcdonalds", "walmart", "disney", "nike", "starbucks",
        "tata", "reliance", "infosys", "tcs", "wipro", "hcl", "tech mahindra", "lt", "bharti airtel", "axis bank", "hdfc", "icici", "indusind", "sbi"
    ]
    if any(company in user_input_lower for company in company_names):
        return "alpha_vantage"
    
    # Check for stock symbols (3-5 letter patterns) - but exclude common words
    import re
    stock_symbol_pattern = r'\b[A-Z]{3,5}\b'
    common_words = ['AI', 'API', 'CEO', 'CFO', 'USA', 'UK', 'EU', 'UN', 'WHO', 'WTO', 'IMF', 'NASA', 'FBI', 'CIA']
    matches = re.findall(stock_symbol_pattern, user_input.upper())
    # Only consider it a stock symbol if it's not a common word
    if matches and not any(word in common_words for word in matches):
        return "alpha_vantage"
    
    # News-related keywords
    news_keywords = ["news", "headline", "breaking", "update", "latest", "current events", "what's happening"]
    if any(keyword in user_input_lower for keyword in news_keywords):
        return "newsapi"
    
    # Weather-related keywords
    weather_keywords = ["weather", "whether", "temperature", "forecast", "rain", "sunny", "cloudy", "hot", "cold", "humidity"]
    if any(keyword in user_input_lower for keyword in weather_keywords):
        return "weather"
    
    # Time-related keywords
    time_keywords = ["date", "time", "day", "today", "current time", "what time", "when"]
    if any(keyword in user_input_lower for keyword in time_keywords):
        return "time"
    
    # Default to LLMs
    return "llm"

def query_llm(user_input):
    """
    Query Gemini API directly to get LLM response, using persona if set.
    """
    # Check if Gemini API key is available
    if GEMINI_API_KEY in [None, "", "your_gemini_api_key_here"]:
        return "Error: Gemini API key is not set."
    
    persona = st.session_state.get("persona", "Friendly")
    prompt = f"You are a {persona} AI assistant. Respond accordingly.\nUser: {user_input}"
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            ai_response = ""
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate:
                    content = candidate["content"]
                    if isinstance(content, dict) and "parts" in content:
                        parts = content["parts"]
                        if isinstance(parts, list):
                            for part in parts:
                                if "text" in part:
                                    ai_response += part["text"]
                    elif isinstance(content, str):
                        ai_response = content
                    else:
                        ai_response = str(content)
                else:
                    ai_response = str(candidate)
            if not ai_response:
                ai_response = "Error: No response content received from Gemini API."
            return ai_response
        else:
            return f"Error: API returned status code {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error querying Gemini API: {str(e)}"

def fetch_current_time(user_input=""):
    """
    Fetch current date and time from WorldTimeAPI. Fallback to local system time if API call fails.
    Can also return just the day of the week if specifically requested.
    """
    from datetime import datetime
    
    # Check if user is asking specifically for the day of the week or date
    user_input_lower = user_input.lower() if user_input else ""
    day_keywords = ["day", "what day", "which day", "day of the week", "day today", "today's day"]
    date_keywords = ["date", "what date", "which date", "date today", "today's date", "current date"]
    
    is_asking_for_day = any(keyword in user_input_lower for keyword in day_keywords)
    is_asking_for_date = any(keyword in user_input_lower for keyword in date_keywords)
    
    try:
        response = requests.get("http://worldtimeapi.org/api/ip", timeout=5)
        response.raise_for_status()
        data = response.json()
        datetime_str = data.get("datetime", "")
        timezone = data.get("timezone", "Unknown timezone")
        
        # Parse the datetime string
        try:
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except:
            # Fallback to local time if parsing fails
            dt = datetime.now()
        
        # Format the response based on what user is asking for
        if is_asking_for_day or is_asking_for_date:
            day_name = dt.strftime("%A")  # Gets full day name (Monday, Tuesday, etc.)
            date_str = dt.strftime("%B %d, %Y")  # Gets date like "January 15, 2024"
            return f"Today is {day_name}, {date_str}."
        else:
            return f"Current date and time: {datetime_str} ({timezone})"
    except Exception as e:
        # Use local system time as fallback without showing error to user
        local_time = datetime.now()
        if is_asking_for_day or is_asking_for_date:
            day_name = local_time.strftime("%A")
            date_str = local_time.strftime("%B %d, %Y")
            return f"Today is {day_name}, {date_str}."
        else:
            local_time_str = local_time.strftime("%Y-%m-%d %H:%M:%S")
            return f"Current date and time: {local_time_str} (Local time)"

def fetch_yfinance_data(symbol: str):
    """
    Fetch stock data using yfinance for Indian stocks.
    """
    try:
        print(f"[DEBUG] fetch_yfinance_data called with symbol: {symbol}")
        
        # Check if yfinance is available
        if not yf:
            return "Error: yfinance library not available"
        
        ticker = yf.Ticker(symbol)
        print(f"[DEBUG] Created ticker object for: {symbol}")
        
        # Try to get basic info first
        try:
            info = ticker.info
            print(f"[DEBUG] yfinance info keys: {list(info.keys()) if info else 'No info'}")
        except Exception as info_error:
            print(f"[DEBUG] Error getting ticker info: {info_error}")
            info = {}
        
        # Try alternative method to get current price
        try:
            hist = ticker.history(period="1d")
            print(f"[DEBUG] History data shape: {hist.shape if hasattr(hist, 'shape') else 'No history'}")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                print(f"[DEBUG] From history - Current: {current_price}, Previous: {previous_close}")
            else:
                current_price = info.get("regularMarketPrice")
                previous_close = info.get("previousClose")
                print(f"[DEBUG] From info - Current: {current_price}, Previous: {previous_close}")
        except Exception as hist_error:
            print(f"[DEBUG] Error getting history: {hist_error}")
            current_price = info.get("regularMarketPrice")
            previous_close = info.get("previousClose")
        
        change_percent = None
        if current_price is not None and previous_close is not None and previous_close != 0:
            change_percent = ((current_price - previous_close) / previous_close) * 100
        
        print(f"[DEBUG] Final values - Current: {current_price}, Previous: {previous_close}, Change: {change_percent}")
        
        if current_price is None:
            return f"No stock data found for {symbol}. This might be due to market hours or data availability."
        
        # Format currency symbol based on stock type and exchange
        if symbol.endswith(".NS"):
            currency_symbol = "₹"  # Indian Rupee
        elif symbol.endswith(".L"):
            currency_symbol = "£"  # British Pound
        elif symbol.endswith(".TO") or symbol.endswith(".V"):
            currency_symbol = "C$"  # Canadian Dollar
        elif symbol.endswith(".AX"):
            currency_symbol = "A$"  # Australian Dollar
        else:
            currency_symbol = "$"  # US Dollar (default)
        
        change_percent_str = f"{change_percent:.2f}%" if change_percent is not None else "N/A"
        
        # Get company name if available
        company_name = info.get("longName", symbol)
        
        result = f"Stock Quote:\n{company_name} ({symbol}): {currency_symbol}{current_price} ({change_percent_str})"
        print(f"[DEBUG] Returning result: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Error fetching stock data from yfinance: {str(e)}"
        print(f"[DEBUG] Exception in fetch_yfinance_data: {error_msg}")
        return error_msg

def query_real_time_service(service, user_input):
    if service == "alpha_vantage":
        symbol = extract_stock_symbol(user_input)
        print(f"Extracted stock symbol: {symbol}")  # Debug log
        # Use yfinance for Indian stocks ending with .NS
        if symbol.endswith(".NS"):
            data = fetch_yfinance_data(symbol)
            return data
        data = fetch_alpha_vantage_data(symbol)
        if isinstance(data, str) and data.startswith("Error"):
            return data
        # Format Alpha Vantage data for user
        try:
            if "Global Quote" not in data or not data["Global Quote"]:
                return "No stock data found for your query."
            quote = data["Global Quote"]
            symbol = quote.get("01. symbol", "N/A")
            price = quote.get("05. price", "N/A")
            change_percent = quote.get("10. change percent", "N/A")
            result_str = f"Stock Quote:\n{symbol}: ${price} ({change_percent})"
            return result_str
        except Exception as e:
            return f"Error processing Alpha Vantage data: {str(e)}"
    elif service == "serpapi":
        data = fetch_serpapi_data(user_input)
        if isinstance(data, str) and data.startswith("Error"):
            return data
        try:
            news_results = data.get("news_results", [])
            if not news_results:
                return "No news articles found for your query."
            result_str = "Top News Articles:\n"
            for article in news_results[:3]:
                title = article.get("title", "N/A")
                source = article.get("source", "N/A")
                if isinstance(source, dict):
                    source_name = source.get("name", "N/A")
                else:
                    source_name = str(source)
                link = article.get("link", "")
                result_str += f"\n- {title} ({source_name})\n  {link}\n"
            return result_str
        except Exception as e:
            return f"Error processing SerpAPI data: {str(e)}"
    elif service == "weather":
        location = extract_location_from_weather_query(user_input)
        print(f"Extracted weather location: '{location}' from input: '{user_input}'")  # Debug log
        data = fetch_weather_data(location)
        if isinstance(data, str) and data.startswith("Error"):
            return data
        try:
            weather_desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            city = data.get("name", "N/A")
            country = data.get("sys", {}).get("country", "N/A")
            result_str = f"Weather in {city}, {country}:\n{weather_desc.capitalize()}, {temp}°C"
            return result_str
        except Exception as e:
            return f"Error processing weather data: {str(e)}"
    elif service == "time":
        return fetch_current_time(user_input)
    else:
        return "Service not recognized."

# Tool functions for LangGraph/LangChain
def weather_tool(location: str) -> str:
    """Get the weather for a location."""
    # Extract location from the input if it's a full question
    extracted_location = extract_location_from_weather_query(location)
    data = fetch_weather_data(extracted_location)
    if isinstance(data, str) and data.startswith("Error"):
        return data
    try:
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        city = data.get("name", "N/A")
        country = data.get("sys", {}).get("country", "N/A")
        return f"Weather in {city}, {country}: {weather_desc.capitalize()}, {temp}°C"
    except Exception as e:
        return f"Error processing weather data: {str(e)}"

def stock_tool(symbol: str) -> str:
    """Get the stock price for one or more symbols."""
    print(f"[DEBUG] stock_tool called with input: {symbol}")
    
    # Extract multiple stock symbols from the input
    extracted_symbols = extract_stock_symbols(symbol)
    print(f"[DEBUG] extracted symbols: {extracted_symbols}")
    
    results = []
    processed_symbols = set()  # Track processed symbols to avoid duplicates
    
    for extracted_symbol in extracted_symbols:
        # Skip if we've already processed this symbol
        if extracted_symbol in processed_symbols:
            continue
            
        # Skip invalid or general question-like inputs
        if len(extracted_symbol) > 10 or extracted_symbol.lower() in ['ai', 'artificial intelligence', 'what', 'how', 'why', 'when', 'where']:
            results.append(f"I don't have stock data for '{extracted_symbol}'. This appears to be a general question. Let me help you with that instead.")
            processed_symbols.add(extracted_symbol)
            continue
        
        # Handle Indian stocks (.NS)
        if extracted_symbol.endswith(".NS"):
            data = fetch_yfinance_data(extracted_symbol)
            print(f"[DEBUG] stock_tool using yfinance data: {data}")
            if data and not data.startswith("Error") and not data.startswith("No stock data"):
                results.append(data)
            else:
                results.append(f"Unable to fetch data for {extracted_symbol}. This might be due to market hours or data availability.")
            processed_symbols.add(extracted_symbol)
            continue
        
        # Handle US stocks using yfinance (more reliable than Alpha Vantage)
        # Check if there's a corresponding .NS version that we should prioritize
        base_symbol = extracted_symbol.replace('.NS', '')
        if base_symbol + '.NS' in extracted_symbols:
            # Skip this US symbol since we have the Indian version
            processed_symbols.add(extracted_symbol)
            continue
            
        # Use yfinance for US stocks too (more reliable)
        data = fetch_yfinance_data(extracted_symbol)
        print(f"[DEBUG] stock_tool using yfinance for US stock: {data}")
        if data and not data.startswith("Error") and not data.startswith("No stock data"):
            results.append(data)
        else:
            results.append(f"Unable to fetch data for {extracted_symbol}. This might be due to market hours or data availability.")
        processed_symbols.add(extracted_symbol)
    
    return "\n".join(results)

def news_tool(query: str) -> str:
    """Get news articles for a query."""
    data = fetch_serpapi_data(query)
    if isinstance(data, str) and data.startswith("Error"):
        return data
    try:
        news_results = data.get("news_results", [])
        if not news_results:
            return "No news articles found for your query."
        result = "Top News Articles:\n"
        for article in news_results[:3]:
            title = article.get("title", "N/A")
            source = article.get("source", "N/A")
            if isinstance(source, dict):
                source_name = source.get("name", "N/A")
            else:
                source_name = str(source)
            link = article.get("link", "")
            result += f"\n- {title} ({source_name})\n  {link}\n"
        return result
    except Exception as e:
        return f"Error processing SerpAPI data: {str(e)}"

def gemini_tool(user_input: str) -> str:
    """Get a response from Gemini LLM."""
    return query_llm(user_input)

def time_tool(user_input: str) -> str:
    """Get the current date and time, day of the week, or date based on the user's question."""
    return fetch_current_time(user_input)

# LangGraph routing functions
def route_query(state: ChatState) -> ChatState:
    """Use LLM to decide which service should handle the query."""
    user_input = state["user_input"]
    
    # Create a prompt for the LLM to determine the service
    routing_prompt = f"""You are a smart routing assistant. Based on the user's question, determine which service should handle it.

Available services:
1. stock_tool - for stock prices, market data, company stock information, share prices, trading data
2. weather_tool - for weather, temperature, climate information, forecasts
3. news_tool - for current news, headlines, recent events, breaking news
4. time_tool - for current time, date, day of the week, what day is it
5. gemini_tool - for general questions, explanations, definitions, educational content, conversations

User question: "{user_input}"

Respond with ONLY the service name (stock_tool, weather_tool, news_tool, time_tool, or gemini_tool)."""

    try:
        # Use Gemini to determine the service
        service_response = query_llm(routing_prompt)
        service = service_response.strip().lower()
        
        # Clean up the response to get just the service name
        if "stock" in service:
            service = "stock_tool"
        elif "weather" in service:
            service = "weather_tool"
        elif "news" in service:
            service = "news_tool"
        elif "time" in service:
            service = "time_tool"
        else:
            service = "gemini_tool"
        
        state["service_decision"] = service
        return state
    except Exception as e:
        state["error"] = f"Error in routing: {str(e)}"
        return state

def execute_stock_service(state: ChatState) -> ChatState:
    """Execute stock service."""
    try:
        user_input = state["user_input"]
        response = stock_tool(user_input)
        state["response"] = response
        return state
    except Exception as e:
        state["error"] = f"Error in stock service: {str(e)}"
        return state

def execute_weather_service(state: ChatState) -> ChatState:
    """Execute weather service."""
    try:
        user_input = state["user_input"]
        response = weather_tool(user_input)
        state["response"] = response
        return state
    except Exception as e:
        state["error"] = f"Error in weather service: {str(e)}"
        return state

def execute_news_service(state: ChatState) -> ChatState:
    """Execute news service."""
    try:
        user_input = state["user_input"]
        response = news_tool(user_input)
        state["response"] = response
        return state
    except Exception as e:
        state["error"] = f"Error in news service: {str(e)}"
        return state

def execute_time_service(state: ChatState) -> ChatState:
    """Execute time service."""
    try:
        user_input = state["user_input"]
        response = time_tool(user_input)
        state["response"] = response
        return state
    except Exception as e:
        state["error"] = f"Error in time service: {str(e)}"
        return state

def execute_gemini_service(state: ChatState) -> ChatState:
    """Execute Gemini service."""
    try:
        user_input = state["user_input"]
        response = gemini_tool(user_input)
        state["response"] = response
        return state
    except Exception as e:
        state["error"] = f"Error in Gemini service: {str(e)}"
        return state

def handle_error(state: ChatState) -> ChatState:
    """Handle any errors that occurred."""
    if state.get("error"):
        state["response"] = f"An error occurred: {state['error']}"
    return state

# Create the LangGraph workflow
def create_langgraph_workflow():
    """Create the LangGraph workflow for intelligent routing."""
    
    # Create the state graph
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("route_query", route_query)
    workflow.add_node("execute_stock_service", execute_stock_service)
    workflow.add_node("execute_weather_service", execute_weather_service)
    workflow.add_node("execute_news_service", execute_news_service)
    workflow.add_node("execute_time_service", execute_time_service)
    workflow.add_node("execute_gemini_service", execute_gemini_service)
    workflow.add_node("handle_error", handle_error)
    
    # Set entry point
    workflow.set_entry_point("route_query")
    
    # Add conditional edges based on service decision
    def route_to_service(state: ChatState) -> str:
        service = state.get("service_decision", "gemini_tool")
        return f"execute_{service.replace('_tool', '_service')}"
    
    workflow.add_conditional_edges(
        "route_query",
        route_to_service,
        {
            "execute_stock_service": "execute_stock_service",
            "execute_weather_service": "execute_weather_service", 
            "execute_news_service": "execute_news_service",
            "execute_time_service": "execute_time_service",
            "execute_gemini_service": "execute_gemini_service"
        }
    )
    
    # Add edges from service nodes to end
    workflow.add_edge("execute_stock_service", END)
    workflow.add_edge("execute_weather_service", END)
    workflow.add_edge("execute_news_service", END)
    workflow.add_edge("execute_time_service", END)
    workflow.add_edge("execute_gemini_service", END)
    
    # Compile the workflow
    return workflow.compile()

# Initialize LangGraph workflow
langgraph_workflow = create_langgraph_workflow()

def run_langgraph_workflow(user_input: str, chat_history: List[Dict[str, str]]) -> str:
    """Run the LangGraph workflow to get a response."""
    try:
        # Prepare initial state
        initial_state = {
            "user_input": user_input,
            "service_decision": None,
            "response": None,
            "chat_history": chat_history,
            "error": None
        }
        
        # Run the workflow
        result = langgraph_workflow.invoke(initial_state)
        
        # Return the clean response without routing information
        if result.get("error"):
            return f"Error: {result['error']}"
        elif result.get("response"):
            return result["response"]
        else:
            return "No response generated."
            
    except Exception as e:
        return f"Error running LangGraph workflow: {str(e)}"

# Set up Groq LLM for agent routing (requires GROQ_API_KEY in .env)
llm = None
if GROQ_API_KEY not in [None, "", "your_groq_api_key_here"]:
    llm = ChatOpenAI(
        openai_api_key=GROQ_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
        model_name="llama3-8b-8192",
        temperature=0
    )

# Wrap tools for agent
weather_tool_lc = Tool.from_function(
    func=weather_tool,
    name="weather_tool",
    description="Use this tool when the user asks about weather, temperature, climate, or wants to know the weather conditions in a specific location. Pass the full user question as the location parameter."
)
stock_tool_lc = Tool.from_function(
    func=stock_tool,
    name="stock_tool", 
    description="Use this tool when the user asks about stock prices, market data, company stock information, or wants to know the current price of a specific stock. Pass the full user question as the symbol parameter."
)
news_tool_lc = Tool.from_function(
    func=news_tool,
    name="news_tool",
    description="Use this tool when the user asks about current news, recent events, headlines, or wants to know what's happening in the world. Pass the full user question as the query parameter."
)
gemini_tool_lc = Tool.from_function(
    func=gemini_tool,
    name="gemini_tool",
    description="Use this tool for general questions, explanations, definitions, educational content, or conversational queries that don't require real-time data from other APIs. Pass the full user question as the user_input parameter."
)
time_tool_lc = Tool.from_function(
    func=time_tool,
    name="time_tool",
    description="Use this tool when the user asks about current time, date, day of the week, or wants to know what day it is today. Pass the full user question as the user_input parameter."
)

agent_tools = [weather_tool_lc, stock_tool_lc, news_tool_lc, gemini_tool_lc, time_tool_lc]

# Create the agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to multiple tools. Use the appropriate tool based on the user's question:

1. Use stock_tool ONLY for questions about stock prices, market data, or specific company stock information. Examples: "What's Apple's stock price?", "Show me Tesla stock", "What's the price of AAPL?"
2. Use weather_tool for questions about weather, temperature, or climate. Examples: "What's the weather in London?", "How hot is it in New York?"
3. Use news_tool for questions about current news, headlines, or recent events. Examples: "What's the latest news about technology?", "Show me breaking news"
4. Use time_tool for questions about current time, date, or day of the week. Examples: "What time is it?", "What day is today?"
5. Use gemini_tool for general questions, explanations, definitions, educational content, or conversational queries. Examples: "What is AI?", "How does photosynthesis work?", "Tell me a joke"

IMPORTANT: If the user asks a general question like "What is AI?" or "How does something work?", use gemini_tool, NOT stock_tool. Only use stock_tool when the user specifically asks about stock prices or market data.

For stock queries, extract the company name or symbol and use stock_tool. For weather queries, extract the location and use weather_tool. For news queries, use the topic as the query for news_tool.

Always use the most appropriate tool for the user's question."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = None
agent_executor = None
if llm is not None:
    agent = create_openai_functions_agent(
        llm=llm,
        tools=agent_tools,
        prompt=prompt
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=agent_tools,
        verbose=True,
        handle_parsing_errors=True
    )

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def run_langgraph_agent(user_input):
    if agent_executor is None:
        return "Error: Groq API key is not set. Agent cannot run."
    try:
        # Prepare chat history from memory
        chat_history = memory.load_memory_variables({}).get("chat_history", [])
        # Invoke agent_executor with input and chat_history
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        # Extract output from result dictionary
        response = result.get("output", "No response from agent.")
        # Save the output to memory
        memory.save_context({"input": user_input}, {"output": response})
        return response
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg or "429" in error_msg or "quota" in error_msg.lower():
            return "Error: Groq API quota exceeded. Please check your plan and billing details."
        elif "invalid_api_key" in error_msg or "401" in error_msg:
            return "Error: Invalid Groq API key. Falling back to Gemini routing."
        else:
            return f"Error running agent_executor: {error_msg}"

def smart_route_with_gemini(user_input):
    """
    Use Gemini to intelligently route queries to appropriate services.
    """
    # Create a prompt for Gemini to determine which service to use
    routing_prompt = f"""You are a smart routing assistant. Based on the user's question, determine which service should handle it.

Available services:
1. stock_tool - for stock prices, market data, company stock information
2. weather_tool - for weather, temperature, climate information
3. news_tool - for current news, headlines, recent events
4. time_tool - for current time, date, day of the week
5. gemini_tool - for general questions, explanations, definitions, educational content

User question: "{user_input}"

Respond with ONLY the service name (stock_tool, weather_tool, news_tool, time_tool, or gemini_tool)."""

    try:
        # Use Gemini to determine the service
        service_response = query_llm(routing_prompt)
        service = service_response.strip().lower()
        
        # Clean up the response to get just the service name
        if "stock" in service:
            service = "alpha_vantage"
        elif "weather" in service:
            service = "weather"
        elif "news" in service:
            service = "newsapi"
        elif "time" in service:
            service = "time"
        else:
            service = "llm"
        
        # Route to the appropriate service
        if service in ["alpha_vantage", "weather", "newsapi", "time"]:
            return query_real_time_service(service, user_input)
        else:
            return query_llm(user_input)
            
    except Exception as e:
        # Fallback to keyword-based routing
        return fallback_keyword_routing(user_input)

def fallback_keyword_routing(user_input):
    """
    Simple keyword-based routing as fallback when Gemini routing fails.
    """
    user_input_lower = user_input.lower()
    
    # Stock-related keywords (more specific)
    stock_keywords = ["stock price", "share price", "market price", "stock quote", "trading price"]
    if any(keyword in user_input_lower for keyword in stock_keywords):
        return query_real_time_service("alpha_vantage", user_input)
    
    # Weather-related keywords
    weather_keywords = ["weather", "temperature", "forecast", "rain", "sunny", "cloudy", "hot", "cold"]
    if any(keyword in user_input_lower for keyword in weather_keywords):
        return query_real_time_service("weather", user_input)
    
    # News-related keywords
    news_keywords = ["news", "headline", "breaking", "latest news", "current events"]
    if any(keyword in user_input_lower for keyword in news_keywords):
        return query_real_time_service("serpapi", user_input)
    
    # Time-related keywords
    time_keywords = ["what time", "current time", "what day", "today's date", "current date"]
    if any(keyword in user_input_lower for keyword in time_keywords):
        return query_real_time_service("time", user_input)
    
    # Default to Gemini for everything else
    return query_llm(user_input)

# Input section
# Removed the "Send a Message" label as per user request
# st.markdown("### Send a Message")

# Initialize input counter in session state if not exists
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

user_input = st.text_input(
    "Type your message...",
    key=f"user_input_{st.session_state.input_counter}",
    placeholder="Ask about weather, stocks, news, or anything else"
)

# Action buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    submitted = st.button("Send", use_container_width=True)

with col2:
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()

with col3:
    if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "assistant":
        if st.button("Regenerate"):
            # Remove last AI response and rerun the last user message
            last_user_msg = None
            for msg in reversed(st.session_state["messages"][:-1]):
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break
            if last_user_msg:
                st.session_state["messages"].pop()  # Remove last assistant message
                user_input = last_user_msg
                ai_response = None
                try:
                    with st.spinner("Processing your request... 🤖✨"):
                        # Use LangGraph workflow for intelligent routing
                        chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state["messages"][:-1]]
                        ai_response = run_langgraph_workflow(user_input, chat_history)
                except Exception as e:
                    ai_response = f"Error: {str(e)}"
                st.session_state["messages"].append({"role": "assistant", "content": ai_response})
                
                # Auto-save chat history after regeneration
                if len(st.session_state["messages"]) > 0:
                    update_chat_session(st.session_state["current_session_id"], st.session_state["messages"])
                
                st.rerun()

with col4:
    if st.session_state["messages"]:
        # Create enhanced chat export with metadata
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_text = f"Chat History Export\n"
        chat_text += f"Generated: {current_time}\n"
        chat_text += f"Session ID: {st.session_state['current_session_id']}\n"
        chat_text += f"Total Messages: {len(st.session_state['messages'])}\n"
        chat_text += "=" * 50 + "\n\n"
        
        for i, msg in enumerate(st.session_state["messages"], 1):
            role = "You" if msg['role'] == 'user' else "AI"
            chat_text += f"[{i}] {role}: {msg['content']}\n\n"
        
        filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        st.download_button("📥 Download Chat", chat_text, file_name=filename)

if submitted and user_input.strip():
    st.session_state["messages"].append({"role": "user", "content": user_input})
    ai_response = None
    try:
        with st.spinner("Processing your request... 🤖✨"):
            # Use LangGraph workflow for intelligent routing
            chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state["messages"][:-1]]
            ai_response = run_langgraph_workflow(user_input, chat_history)
    except Exception as e:
        ai_response = f"Error: {str(e)}"
    
    st.session_state["messages"].append({"role": "assistant", "content": ai_response})
    
    # Auto-save chat history after each message
    if len(st.session_state["messages"]) > 0:
        update_chat_session(st.session_state["current_session_id"], st.session_state["messages"])
    
    # Increment counter to clear input field
    st.session_state.input_counter += 1
    st.rerun()

# Enhanced Sidebar with Chat History
with st.sidebar:
    st.header("🤖 AI Assistant Settings")
    
    # Make chat history more prominent
    st.markdown("---")
    st.markdown("## 💬 **CHAT HISTORY**")
    
    # Current session info
    st.markdown(f"**📱 Current Session:** {len(st.session_state['messages'])} messages")
    
    # Save current chat - make it more prominent
    if st.session_state["messages"]:
        st.markdown("### 💾 Save Chat")
        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            if st.button("💾 SAVE CURRENT CHAT", use_container_width=True, type="primary"):
                st.session_state["show_save_dialog"] = True
        
        if st.session_state.get("show_save_dialog", False):
            title = st.text_input("📝 Chat Title (optional):", value="", key="save_title")
            if st.button("✅ CONFIRM SAVE", key="confirm_save", type="primary"):
                save_chat_session(
                    st.session_state["current_session_id"],
                    st.session_state["messages"],
                    title if title else None
                )
                st.session_state["chat_sessions"] = get_all_chat_sessions()
                st.session_state["show_save_dialog"] = False
                st.success("✅ Chat saved successfully!")
                st.rerun()
            if st.button("❌ Cancel", key="cancel_save"):
                st.session_state["show_save_dialog"] = False
                st.rerun()
    
    st.markdown("---")
    
    # Load previous chats - make it more visible
    st.markdown("### 📚 Previous Chats")
    
    # Force refresh button
    if st.button("🔄 Refresh Chat List", use_container_width=True):
        st.session_state["chat_sessions"] = get_all_chat_sessions()
        st.rerun()
    
    if st.session_state["chat_sessions"]:
        st.success(f"Found {len(st.session_state['chat_sessions'])} saved chats!")
        
        for i, session in enumerate(st.session_state["chat_sessions"][:5]):  # Show last 5 sessions
            session_id = session["session_id"]
            title = session["title"]
            message_count = session["message_count"]
            last_updated = datetime.fromisoformat(session["last_updated"]).strftime("%m/%d %H:%M")
            
            st.markdown(f"**{i+1}. {title}**")
            st.markdown(f"📊 {message_count} messages • 🕒 {last_updated}")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"📄 Load Chat", key=f"load_{session_id}"):
                    loaded_session = load_chat_session(session_id)
                    if loaded_session:
                        st.session_state["messages"] = loaded_session["messages"]
                        st.session_state["current_session_id"] = session_id
                        st.success(f"✅ Loaded: {title}")
                        st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{session_id}"):
                    delete_chat_session(session_id)
                    st.session_state["chat_sessions"] = get_all_chat_sessions()
                    st.success("✅ Chat deleted!")
                    st.rerun()
            st.markdown("---")
    else:
        st.info("📭 No saved chats yet. Start a conversation and save it to see it here!")
    
    # Start new chat - make it more prominent
    st.markdown("### 🆕 New Chat")
    if st.button("🆕 START NEW CHAT", use_container_width=True, type="secondary"):
        st.session_state["messages"] = []
        st.session_state["current_session_id"] = str(uuid.uuid4())
        st.success("✅ New chat started!")
        st.rerun()
    
    # Chat Statistics
    if st.session_state["messages"]:
        st.markdown("---")
        st.subheader("📊 Chat Statistics")
        
        user_messages = len([m for m in st.session_state["messages"] if m["role"] == "user"])
        ai_messages = len([m for m in st.session_state["messages"] if m["role"] == "assistant"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Your Messages", user_messages)
        with col2:
            st.metric("AI Responses", ai_messages)
        
        # Calculate average message length
        if user_messages > 0:
            avg_user_length = sum(len(m["content"]) for m in st.session_state["messages"] if m["role"] == "user") / user_messages
            st.metric("Avg. Message Length", f"{avg_user_length:.1f} chars")
    
    st.markdown("---")
    
    # Search Chat History
    if st.session_state["messages"]:
        st.subheader("🔍 Search Chat")
        search_term = st.text_input("Search in current chat:", placeholder="Type to search...", key="chat_search")
        
        if search_term:
            matching_messages = []
            for i, msg in enumerate(st.session_state["messages"]):
                if search_term.lower() in msg["content"].lower():
                    matching_messages.append((i, msg))
            
            if matching_messages:
                st.markdown(f"**Found {len(matching_messages)} matches:**")
                for idx, msg in matching_messages[:3]:  # Show first 3 matches
                    role = "You" if msg["role"] == "user" else "AI"
                    content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                    st.markdown(f"**{role} (msg {idx+1}):** {content}")
            else:
                st.info("No matches found.")
    
    st.markdown("---")
    
    # Persona Selection
    st.subheader("AI Persona")
    persona = st.selectbox(
        "Choose personality:",
        ["Friendly", "Professional", "Creative", "Concise"],
        index=0,
        key="persona_selector"
    )
    st.session_state["persona"] = persona
    
    st.markdown("---")
    
    # API Status
    st.subheader("API Status")
    
    # Weather API
    if WEATHER_API_KEY not in [None, "", "your_weather_api_key_here"]:
        st.markdown('<span class="status-success">✓ Weather API</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warning">⚠ Weather API</span>', unsafe_allow_html=True)
    
    # Stock API
    if ALPHA_VANTAGE_API_KEY not in [None, "", "your_alpha_vantage_api_key_here"]:
        st.markdown('<span class="status-success">✓ Stock API</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warning">⚠ Stock API</span>', unsafe_allow_html=True)
    
    # News API
    if SERPAPI_KEY not in [None, "", "your_serpapi_key_here"]:
        st.markdown('<span class="status-success">✓ News API</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warning">⚠ News API</span>', unsafe_allow_html=True)
    
    # Groq API
    if GROQ_API_KEY not in [None, "", "your_groq_api_key_here"]:
        st.markdown('<span class="status-success">✓ Groq API</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warning">⚠ Groq API</span>', unsafe_allow_html=True)
    
    # Test APIs Button
    if st.button("Test APIs", use_container_width=True):
        with st.spinner("Testing APIs..."):
            # Test Weather API
            if WEATHER_API_KEY not in [None, "", "your_weather_api_key_here"]:
                try:
                    weather_test = fetch_weather_data("London")
                    if isinstance(weather_test, str) and weather_test.startswith("Error"):
                        st.error("Weather API: " + weather_test)
                    else:
                        st.success("Weather API: Working")
                except Exception as e:
                    st.error(f"Weather API: {str(e)}")
            else:
                st.warning("Weather API: No key set")
            
            # Test Stock API
            if ALPHA_VANTAGE_API_KEY not in [None, "", "your_alpha_vantage_api_key_here"]:
                try:
                    stock_test = fetch_alpha_vantage_data("AAPL")
                    if isinstance(stock_test, str) and stock_test.startswith("Error"):
                        st.error("Stock API: " + stock_test)
                    else:
                        st.success("Stock API: Working")
                except Exception as e:
                    st.error(f"Stock API: {str(e)}")
            else:
                st.warning("Stock API: No key set")
            
            # Test SerpAPI
            if SERPAPI_KEY not in [None, "", "your_serpapi_key_here"]:
                try:
                    news_test = fetch_serpapi_data("technology")
                    if isinstance(news_test, str) and news_test.startswith("Error"):
                        st.error("SerpAPI: " + news_test)
                    else:
                        st.success("SerpAPI: Working")
                except Exception as e:
                    st.error(f"SerpAPI: {str(e)}")
            else:
                st.warning("SerpAPI: No key set")
    
    st.markdown("---")
    
    # Features
    st.subheader("Features")
    st.markdown("""
    - **Weather**: Real-time weather data
    - **Stocks**: Stock prices & market data
    - **News**: Latest headlines & breaking news
    - **Time**: Current time & date information
    - **AI Chat**: Intelligent conversations
    """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("**Powered by Gemini API & LangGraph**")
    st.markdown("*Made by Kavya Shyopura*")

# Add some spacing at the bottom
st.markdown("<br>", unsafe_allow_html=True)
