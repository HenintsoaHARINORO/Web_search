import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

# Ollama Configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434")

# Data Paths
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
PORTFOLIO_FILE = DATA_DIR / os.getenv("PORTFOLIO_FILE", "portfolio_entreprises.csv")
VECTORSTORE_DIR = DATA_DIR / os.getenv("VECTORSTORE_DIR", "vectorstore")
VECTOR_STORE_PATH = BASE_DIR / os.getenv("VECTOR_STORE_PATH", "portfolio_vectorstore")

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

# Scraping & Request Settings
SCRAPE_MAX_CHARS = int(os.getenv("SCRAPE_MAX_CHARS", 5000))
SCRAPE_TIMEOUT = int(os.getenv("SCRAPE_TIMEOUT", 15000))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 10))