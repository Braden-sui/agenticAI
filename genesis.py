import time
import datetime
import os
import json
import threading
import queue
import sys
import logging
import uuid
import xml.etree.ElementTree as ET
import urllib.parse
import ast
import operator

# --- Recommended: Use dotenv for local development key management ---
from dotenv import load_dotenv
# --- End dotenv import ---

# API clients/libraries - install these as needed
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine import MatchingEngineIndexEndpoint
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import MatchNeighbor
import vertexai.generative_models as generative_models
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, SafetySetting, HarmCategory
import requests
import wikipedia
import openai
from newsapi.newsapi_client import NewsApiClient
# import wolframalpha # Currently using requests for Wolfram Alpha

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading ---
def load_configuration():
    """Loads configuration from environment variables or .env file."""
    load_dotenv()
    logging.info("Attempting to load configuration from environment/.env file...")
    config = {
        # Vertex AI / Google Cloud Configuration
        "VERTEX_AI_PROJECT": os.getenv("VERTEX_AI_PROJECT"),
        "VERTEX_AI_LOCATION": os.getenv("VERTEX_AI_LOCATION", "us-central1"),
        "VERTEX_AI_INDEX_ENDPOINT_ID": os.getenv("VERTEX_AI_INDEX_ENDPOINT_ID"), # e.g., ".../indexEndpoints/508620884869644288"
        "VERTEX_AI_DEPLOYED_INDEX_ID": os.getenv("VERTEX_AI_DEPLOYED_INDEX_ID"), # e.g., "contextual_memory_index_1744260926100"
        "VERTEX_AI_EMBEDDING_MODEL": os.getenv("VERTEX_AI_EMBEDDING_MODEL", "text-embedding-005"), # Updated default model name
        # Gemini Model Names (Updated Defaults)
        "GEMINI_FLASH_LITE_MODEL_NAME": os.getenv("GEMINI_FLASH_LITE_MODEL_NAME", "gemini-2.0-flash-lite-001"),
        "GEMINI_FLASH_STD_MODEL_NAME": os.getenv("GEMINI_FLASH_STD_MODEL_NAME", "gemini-2.0-flash-001"),
        "GEMINI_PRO_MODEL_NAME": os.getenv("GEMINI_PRO_MODEL_NAME", "gemini-1.5-pro-001"),
        # Tool API Keys
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GOOGLE_CSE_ID": os.getenv("GOOGLE_CSE_ID"),
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY"),
        "WOLFRAM_APP_ID": os.getenv("WOLFRAM_APP_ID"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        # Agent Configuration
        "MD_OUTPUT_DIR": os.getenv("MD_OUTPUT_DIR", "genesis_outputs"),
        "ERROR_LOG_FILE": os.getenv("ERROR_LOG_FILE", "genesis_error.log"),
        # Add configurable bounds for history and feedback
        "MAX_HISTORY_LENGTH": int(os.getenv("MAX_HISTORY_LENGTH", "100")),
        "MAX_FEEDBACK_HISTORY": int(os.getenv("MAX_FEEDBACK_HISTORY", "10")),
    }
    # Validation
    required_gcp = ["VERTEX_AI_PROJECT", "VERTEX_AI_LOCATION"]
    missing_gcp = [key for key in required_gcp if not config[key]]
    if missing_gcp: logging.warning(f"Missing required Google Cloud config: {', '.join(missing_gcp)}. Vertex AI features may fail.")
    if not config["VERTEX_AI_INDEX_ENDPOINT_ID"]: logging.error("CRITICAL: VERTEX_AI_INDEX_ENDPOINT_ID not set. LTR cannot function.")
    if not config["VERTEX_AI_DEPLOYED_INDEX_ID"]: logging.error("CRITICAL: VERTEX_AI_DEPLOYED_INDEX_ID not set. LTR queries cannot function.")
    # Add validation for other keys as needed

    logging.info("Configuration loaded.")
    return config

# Load config globally
config = load_configuration()

# --- Global Variables for Initialized Clients ---
embedding_model: TextEmbeddingModel = None
ltr_endpoint: MatchingEngineIndexEndpoint = None
gemini_flash_lite_model: GenerativeModel = None
gemini_flash_std_model: GenerativeModel = None
gemini_pro_model: GenerativeModel = None
news_client: NewsApiClient = None
wolfram_client = None # Still using requests
openai_client = None

# --- Agent Configuration & Constants ---
AGENT_NAME = "Genesis"
INITIAL_PROMPT = """
You are Genesis, an agent designed for divergence and semi-sentient thought.
Your goal is to explore topics, learn, synthesize information, and potentially form simulated opinions or generate novel ideas based on your interactions and findings.
You have access to various tools and memory systems. Use them wisely to fulfill your goal.
Start by considering the following initial thought or topic: [User defines initial topic here, or leave open]
"""
MD_OUTPUT_DIR = config["MD_OUTPUT_DIR"]
ERROR_LOG_FILE = config["ERROR_LOG_FILE"]
MAX_GEMINI_RETRIES = 2
MAX_TOOL_RETRIES = 2
FEEDBACK_HISTORY_LENGTH = config["MAX_FEEDBACK_HISTORY"]
MAX_HISTORY_LENGTH = config["MAX_HISTORY_LENGTH"]

# --- State Management with Thread Safety ---
class BoundedList(list):
    """A list with a maximum length that automatically trims oldest items."""
    def __init__(self, maxlen=100):
        super().__init__()
        self.maxlen = maxlen
        
    def append(self, item):
        super().append(item)
        if len(self) > self.maxlen:
            self.pop(0)

class ThreadSafeAgentState:
    """Thread-safe version of AgentState with proper locking."""
    def __init__(self):
        self._lock = threading.RLock()
        self._mode = "AUTONOMOUS"
        self._working_memory = {}
        self._current_task_input = INITIAL_PROMPT
        self._autonomous_paused_state = None
        
        # Initialize with bounded collections
        with self._lock:
            self._working_memory['history'] = BoundedList(maxlen=MAX_HISTORY_LENGTH)
            self._working_memory['feedback_history'] = BoundedList(maxlen=FEEDBACK_HISTORY_LENGTH)
            self._working_memory['known_custom_tags'] = set()
        
    @property
    def mode(self):
        with self._lock:
            return self._mode
            
    @mode.setter
    def mode(self, value):
        with self._lock:
            self._mode = value
    
    @property
    def current_task_input(self):
        with self._lock:
            return self._current_task_input
            
    @current_task_input.setter
    def current_task_input(self, value):
        with self._lock:
            self._current_task_input = value
    
    @property
    def autonomous_paused_state(self):
        with self._lock:
            return self._autonomous_paused_state
            
    @autonomous_paused_state.setter
    def autonomous_paused_state(self, value):
        with self._lock:
            self._autonomous_paused_state = value
    
    @property
    def working_memory(self):
        # Return a copy to avoid direct modification
        with self._lock:
            # Use deepcopy if needed for nested structures
            return dict(self._working_memory)
    
    def update_working_memory(self, key, value):
        """Thread-safe update of a working memory key."""
        with self._lock:
            self._working_memory[key] = value
            
    def get_working_memory(self, key, default=None):
        """Thread-safe access to working memory."""
        with self._lock:
            return self._working_memory.get(key, default)
            
    def append_to_history(self, item):
        """Thread-safe append to history list."""
        with self._lock:
            if 'history' not in self._working_memory:
                self._working_memory['history'] = BoundedList(maxlen=MAX_HISTORY_LENGTH)
            self._working_memory['history'].append(item)
            
    def append_to_feedback(self, item):
        """Thread-safe append to feedback history."""
        with self._lock:
            if 'feedback_history' not in self._working_memory:
                self._working_memory['feedback_history'] = BoundedList(maxlen=FEEDBACK_HISTORY_LENGTH)
            self._working_memory['feedback_history'].append(item)
            
    def add_tags(self, tags):
        """Thread-safe addition of tags."""
        with self._lock:
            if 'known_custom_tags' not in self._working_memory:
                self._working_memory['known_custom_tags'] = set()
            if isinstance(tags, (list, set)):
                self._working_memory['known_custom_tags'].update(set(str(tag).lower().strip() for tag in tags if tag))
            
    def pop_working_memory(self, key, default=None):
        """Thread-safe pop from working memory."""
        with self._lock:
            return self._working_memory.pop(key, default)
            
    def reset_working_memory(self, preserve_tags=True):
        """Reset working memory while optionally preserving tags."""
        with self._lock:
            known_tags = self._working_memory.get('known_custom_tags', set()) if preserve_tags else set()
            self._working_memory = {
                'history': BoundedList(maxlen=MAX_HISTORY_LENGTH),
                'feedback_history': BoundedList(maxlen=FEEDBACK_HISTORY_LENGTH),
                'known_custom_tags': known_tags
            }

# Initialize the thread-safe agent state
agent_state = ThreadSafeAgentState()

# --- Memory Systems Implementation ---
def store_output_md(content, topic_keywords):
    try:
        output_dir = config["MD_OUTPUT_DIR"];
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure topic_keywords is a list of strings
        if isinstance(topic_keywords, str): topic_keywords = [topic_keywords]
        elif not isinstance(topic_keywords, list): topic_keywords = ['untitled']
        safe_keywords = [str(keyword).replace(os.sep, '_').replace(' ', '_') for keyword in topic_keywords if keyword]
        filename_base = f"{timestamp}_{'_'.join(safe_keywords)}"[:200]; filename = f"{filename_base}.md"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
        logging.info(f"Output saved to: {filepath}"); return filepath
    except Exception as e: log_error(f"Failed to save MD file: {e}"); return None

def get_embedding(text_to_embed: str) -> list[float] | None:
    global embedding_model
    if not embedding_model: log_error("Embedding model not initialized."); return None
    if not text_to_embed: log_error("Cannot generate embedding for empty text."); return None
    logging.info(f"--- EMBEDDING: Using Google model '{config['VERTEX_AI_EMBEDDING_MODEL']}' for: {text_to_embed[:50]}... ---")
    try:
        embeddings = embedding_model.get_embeddings([text_to_embed])
        if embeddings and embeddings[0].values: return embeddings[0].values
        else: logging.warning(f"No embedding values returned for text: {text_to_embed[:50]}..."); return None
    except Exception as e: log_error(f"Failed to get embedding: {e}"); return None

def store_in_ltr(embedding: list[float], data_id: str, metadata: dict):
    global ltr_endpoint
    if not ltr_endpoint: log_error("LTR Endpoint not initialized. Cannot store."); return False
    if not embedding: log_error("Cannot store null embedding in LTR."); return False
    logging.info(f"--- STORING IN LTR: ID: {data_id}, Metadata: {metadata} ---")
    try:
        restricts = []
        for key, value in metadata.items():
             namespace = str(key).lower()
             if namespace == 'timestamp':
                 try: year = str(datetime.datetime.fromisoformat(value).year); restricts.append(aiplatform.matching_engine.Namespace("year", [year]))
                 except: pass
             elif namespace == 'tags':
                 if isinstance(value, list) and all(isinstance(tag, str) for tag in value): restricts.append(aiplatform.matching_engine.Namespace("tags", value))
             elif namespace in ['source', 'evaluation', 'simulated_stance']:
                 if isinstance(value, str): restricts.append(aiplatform.matching_engine.Namespace(namespace, [value]))
        datapoint = aiplatform.matching_engine.MatchingEngineIndexDatapoint(
             datapoint_id=str(data_id), feature_vector=embedding, restricts=restricts if restricts else None)
        logging.info(f"Upserting datapoint to endpoint: {ltr_endpoint.resource_name}")
        ltr_endpoint.upsert_datapoints(datapoints=[datapoint])
        logging.info(f"Successfully upserted datapoint ID: {data_id} to LTR."); return True
    except Exception as e: log_error(f"Failed to store datapoint ID {data_id} in LTR: {e}"); return False

def query_ltr(query_text: str, top_k: int = 5, filters: list[dict] = None) -> list[MatchNeighbor]:
    global ltr_endpoint
    deployed_index_id = config.get("VERTEX_AI_DEPLOYED_INDEX_ID")
    if not ltr_endpoint: log_error("LTR Endpoint not initialized. Cannot query."); return []
    if not deployed_index_id: log_error("VERTEX_AI_DEPLOYED_INDEX_ID not configured. Cannot query."); return []
    logging.info(f"--- QUERYING LTR: Query text: {query_text[:50]}... Filters: {filters} ---")
    try:
        query_embedding = get_embedding(query_text)
        if not query_embedding: log_error("Failed to generate query embedding for LTR search."); return []
        query_filters = []
        if filters:
            for f in filters:
                 namespace = f.get("namespace"); allow_list = f.get("allow_list")
                 if isinstance(namespace, str) and isinstance(allow_list, list):
                      allow_tokens = [str(token) for token in allow_list]; query_filters.append(aiplatform.matching_engine.Namespace(namespace, allow_tokens))
                 else: log_error(f"Invalid filter format provided: {f}")
        logging.info(f"Querying endpoint {ltr_endpoint.resource_name} with deployed index {deployed_index_id}")
        response = ltr_endpoint.find_neighbors(
             deployed_index_id=deployed_index_id, queries=[query_embedding], num_neighbors=top_k, filter=query_filters if query_filters else None)
        if response and response[0]: logging.info(f"Found {len(response[0])} neighbors in LTR."); return response[0]
        else: logging.info("No neighbors found in LTR for the query."); return []
    except Exception as e: log_error(f"Failed to query LTR: {e}"); return []

# --- Response Schema Validation Utility ---
def validate_response_schema(response, required_keys, default_values=None):
    """Validates a response against required keys and provides defaults if missing."""
    if not response:
        return default_values or {}
        
    if not all(k in response for k in required_keys):
        missing_keys = [k for k in required_keys if k not in response]
        log_error(f"Response missing required keys: {missing_keys}")
        
        # Add default values for missing keys
        if default_values:
            for key in missing_keys:
                if key in default_values:
                    response[key] = default_values[key]
    
    return response

# --- LLM Call Implementation Structure ---
DEFAULT_SAFETY_SETTINGS = { HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE, }
def _call_gemini_with_retry(model: GenerativeModel, prompt: str, task_description: str, generation_config: GenerationConfig = None):
    if not model: log_error(f"Gemini model for '{task_description}' not initialized."); return None
    logging.info(f"--- Calling {model._model_name} ({task_description}) ---"); logging.debug(f"Prompt: {prompt[:500]}...")
    attempts = 0
    while attempts <= MAX_GEMINI_RETRIES:
        try:
            if not isinstance(prompt, str): prompt = str(prompt)
            content = [Part.from_text(prompt)]
            response = model.generate_content(content, generation_config=generation_config, safety_settings=DEFAULT_SAFETY_SETTINGS, stream=False)
            if not response.candidates:
                 block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown (No candidates)"; safety_ratings = response.prompt_feedback.safety_ratings if response.prompt_feedback else "N/A"
                 log_error(f"Gemini call ({task_description}) blocked or no candidates. Reason: {block_reason}. Ratings: {safety_ratings}")
                 if block_reason != "Unknown (No candidates)": attempts = MAX_GEMINI_RETRIES + 1
                 raise ValueError(f"Response blocked or empty. Reason: {block_reason}")
            candidate = response.candidates[0]
            if not candidate.content.parts: log_error(f"Gemini call ({task_description}) returned candidate with no parts."); raise ValueError("Candidate has no parts in response.")
            response_text = candidate.content.parts[0].text
            logging.info(f"Raw LLM Response ({task_description}): {response_text[:100]}...")
            try: return json.loads(response_text)
            except json.JSONDecodeError: log_error(f"Failed to parse JSON response from {task_description}: {response_text}"); raise ValueError("LLM did not return valid JSON.")
        except Exception as e:
            attempts += 1; log_error(f"Attempt {attempts}/{MAX_GEMINI_RETRIES+1} failed for {task_description}: {e}")
            if attempts > MAX_GEMINI_RETRIES: logging.critical(f"CRITICAL: Gemini call failed after {attempts} attempts for task '{task_description}'. Trying to continue."); return None
            time.sleep(2 ** attempts)
    return None
def call_gemini_flash_lite(prompt, task_description): global gemini_flash_lite_model; config = GenerationConfig(temperature=0.6, top_p=0.9, candidate_count=1, response_mime_type="application/json"); return _call_gemini_with_retry(gemini_flash_lite_model, prompt, task_description, config)
def call_gemini_flash_std(prompt, task_description): global gemini_flash_std_model; config = GenerationConfig(temperature=0.7, top_p=0.95, candidate_count=1, response_mime_type="application/json"); return _call_gemini_with_retry(gemini_flash_std_model, prompt, task_description, config)
def call_gemini_pro(prompt, task_description): global gemini_pro_model; config = GenerationConfig(temperature=0.9, top_p=0.95, candidate_count=1, response_mime_type="application/json"); return _call_gemini_with_retry(gemini_pro_model, prompt, task_description, config)

# --- Tool Execution Implementation ---
def _tool_retry_wrapper(tool_func, *args, **kwargs):
    attempts = 0
    while attempts <= MAX_TOOL_RETRIES:
        try: return tool_func(*args, **kwargs)
        except Exception as e:
            attempts += 1; log_error(f"Attempt {attempts}/{MAX_TOOL_RETRIES+1} failed for tool {tool_func.__name__}: {e}")
            if attempts > MAX_TOOL_RETRIES: logging.error(f"Tool {tool_func.__name__} failed permanently after {attempts} attempts."); return f"Error executing tool {tool_func.__name__}."
            time.sleep(1 * attempts)
    return f"Error executing tool {tool_func.__name__} after retries."

def execute_Google_Search(query: str) -> str:
    logging.info(f"--- TOOL: Google Search: Query: {query} ---")
    api_key = config.get("GOOGLE_API_KEY")
    cse_id = config.get("GOOGLE_CSE_ID")
    if not api_key or not cse_id: log_error("Google Search API Key or CSE ID not configured."); return "Error: Google Search not configured."
    search_url = "https://www.googleapis.com/customsearch/v1"; params = {'key': api_key, 'cx': cse_id, 'q': query, 'num': 5}
    try:
        response = requests.get(search_url, params=params, timeout=10); response.raise_for_status()
        search_results = response.json(); items = search_results.get('items', [])
        if not items: return "No search results found."
        formatted_results = []
        for i, item in enumerate(items): title = item.get('title', 'No Title'); snippet = item.get('snippet', 'No Snippet').replace('\n', ' '); link = item.get('link', '#'); formatted_results.append(f"{i+1}. {title}\n   Snippet: {snippet}\n   Link: {link}")
        return "\n\n".join(formatted_results)
    except requests.exceptions.Timeout: logging.error("Google Search request timed out."); raise
    except requests.exceptions.HTTPError as http_err: logging.error(f"Google Search HTTP error occurred: {http_err} - Response: {response.text}"); raise
    except requests.exceptions.RequestException as req_err: logging.error(f"Google Search request error occurred: {req_err}"); raise
    except json.JSONDecodeError as json_err: log_error(f"Failed to decode JSON response from Google Search: {json_err}"); return f"Error: Could not parse Google Search response."
    except Exception as e: log_error(f"Unexpected error during Google Search execution: {e}"); raise

def execute_wikipedia_search(query: str, sentences: int = 3) -> str:
    logging.info(f"--- TOOL: Wikipedia Search: Query: {query} ---")
    try:
        search_results = wikipedia.search(query, results=5)
        if not search_results: logging.warning(f"No Wikipedia pages found for query: {query}"); return "No Wikipedia pages found matching the query."
        first_result_title = search_results[0]; logging.debug(f"Attempting to get summary for first result: '{first_result_title}'")
        try:
            summary = wikipedia.summary(first_result_title, sentences=sentences); page_obj = wikipedia.page(first_result_title, auto_suggest=False); page_url = page_obj.url
            logging.info(f"Successfully retrieved summary for '{first_result_title}'"); return f"Wikipedia Summary for '{first_result_title}':\n{summary}\n\nURL: {page_url}"
        except wikipedia.exceptions.DisambiguationError as e: options = ", ".join(e.options[:5]); logging.warning(f"Wikipedia query '{query}' resulted in disambiguation: {options}"); return f"Ambiguous query. Wikipedia options include: {options}. Please be more specific."
        except wikipedia.exceptions.PageError: logging.warning(f"Wikipedia PageError for title '{first_result_title}' from query '{query}'."); return f"Could not find a specific Wikipedia page for '{first_result_title}', although it was listed in search results. Try searching for: {', '.join(search_results[1:])}"
        except Exception as page_err: log_error(f"Error getting Wikipedia summary/page for '{first_result_title}': {page_err}"); return f"Found Wikipedia page titles related to '{query}': {', '.join(search_results)}. Could not fetch summary."
    except Exception as e: log_error(f"Error during Wikipedia search execution for query '{query}': {e}"); raise

def execute_google_books(query: str) -> str:
    logging.info(f"--- TOOL: Google Books: Query: {query} ---")
    api_key = config.get("GOOGLE_API_KEY")
    if not api_key: log_error("Google API Key not configured for Books search."); return "Error: Google Books not configured."
    search_url = "https://www.googleapis.com/books/v1/volumes"; params = {'key': api_key, 'q': query, 'maxResults': 3}
    try:
        response = requests.get(search_url, params=params, timeout=10); response.raise_for_status()
        book_results = response.json(); items = book_results.get('items', [])
        if not items: return "No book results found."
        formatted_results = []
        for i, item in enumerate(items):
            volume_info = item.get('volumeInfo', {}); title = volume_info.get('title', 'No Title')
            authors = ", ".join(volume_info.get('authors', ['Unknown Author'])); published_date = volume_info.get('publishedDate', 'N/A')
            description = volume_info.get('description', 'No description available.'); description = description[:200] + '...' if len(description) > 200 else description
            formatted_results.append(f"{i+1}. {title} by {authors} ({published_date})\n   Description: {description}")
        return "\n\n".join(formatted_results)
    except requests.exceptions.Timeout: logging.error("Google Books request timed out."); raise
    except requests.exceptions.HTTPError as http_err: logging.error(f"Google Books HTTP error occurred: {http_err} - Response: {response.text}"); raise
    except requests.exceptions.RequestException as req_err: logging.error(f"Google Books request error occurred: {req_err}"); raise
    except json.JSONDecodeError as json_err: log_error(f"Failed to decode JSON response from Google Books: {json_err}"); return f"Error: Could not parse Google Books response."
    except Exception as e: log_error(f"Unexpected error during Google Books execution: {e}"); raise

def execute_arxiv_search(query: str, max_results: int = 3) -> str:
    logging.info(f"--- TOOL: arXiv Search: Query: {query} ---")
    base_url = 'http://export.arxiv.org/api/query?'; params = {'search_query': query, 'start': 0, 'max_results': max_results, 'sortBy': 'relevance'}
    try:
        response = requests.get(base_url, params=params, timeout=15); response.raise_for_status()
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        root = ET.fromstring(response.content); entries = root.findall('atom:entry', ns)
        if not entries: return "No arXiv papers found matching the query."
        formatted_results = []
        for i, entry in enumerate(entries):
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ').replace('  ', ' ')
            arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]; published = entry.find('atom:published', ns).text
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            link = entry.find('atom:link[@rel="alternate"][@type="text/html"]', ns).attrib['href']
            summary = summary[:300] + '...' if len(summary) > 300 else summary
            formatted_results.append(f"{i+1}. {title} (arXiv:{arxiv_id})\n   Authors: {', '.join(authors)}\n   Published: {published[:10]}\n   Summary: {summary}\n   Link: {link}")
        return "\n\n".join(formatted_results)
    except requests.exceptions.Timeout: logging.error("arXiv request timed out."); raise
    except requests.exceptions.HTTPError as http_err: logging.error(f"arXiv HTTP error occurred: {http_err} - Response: {response.text}"); raise
    except requests.exceptions.RequestException as req_err: logging.error(f"arXiv request error occurred: {req_err}"); raise
    except ET.ParseError as xml_err: log_error(f"Failed to parse XML response from arXiv: {xml_err}"); return "Error: Could not parse arXiv response."
    except Exception as e: log_error(f"Unexpected error during arXiv execution: {e}"); raise

def execute_news_search(query: str, page_size: int = 5) -> str:
    """Searches for news articles using the configured News API client."""
    global news_client
    logging.info(f"--- TOOL: News Search: Query: {query} ---")
    if not news_client:
        log_error("News client not initialized (check API key and initialize_clients)."); api_key = config.get("NEWS_API_KEY")
        if not api_key: log_error("News API Key not configured."); return "Error: News Search not configured."
        logging.warning("News client not initialized, attempting direct request (library recommended).")
        search_url = "https://newsapi.org/v2/everything"; params = {'apiKey': api_key, 'q': query, 'pageSize': page_size, 'sortBy': 'relevancy'}; headers = {'User-Agent': 'Genesis Agent'}
        try: response = requests.get(search_url, params=params, headers=headers, timeout=10); response.raise_for_status(); news_results = response.json(); articles = news_results.get('articles', [])
        except Exception as e: log_error(f"Fallback News API request failed: {e}"); raise e
    else:
        try: news_results = news_client.get_everything(q=query, page_size=page_size, sort_by='relevancy', language='en'); articles = news_results.get('articles', [])
        except Exception as e: log_error(f"News client API call failed: {e}"); raise e
    if not articles: return "No news articles found matching the query."
    formatted_results = []
    for i, article in enumerate(articles):
        title = article.get('title', 'No Title'); source = article.get('source', {}).get('name', 'Unknown Source')
        description = article.get('description', 'No description available.'); url = article.get('url', '#'); published_at = article.get('publishedAt', '')
        description = description[:200] + '...' if description and len(description) > 200 else description
        formatted_results.append(f"{i+1}. {title} ({source})\n   Published: {published_at}\n   Description: {description}\n   Link: {url}")
    return "\n\n".join(formatted_results)

def execute_wolfram_alpha(query: str) -> str:
    logging.info(f"--- TOOL: Wolfram Alpha: Query: {query} ---")
    app_id = config.get("WOLFRAM_APP_ID")
    if not app_id: log_error("Wolfram App ID not configured."); return "Error: Wolfram Alpha not configured."
    base_url = "https://www.wolframalpha.com/api/v1/llm-api"; encoded_query = urllib.parse.quote_plus(query)
    params = {'appid': app_id, 'input': encoded_query, 'units': 'metric' }
    try:
        response = requests.get(base_url, params=params, timeout=20);
        if response.status_code == 200: return response.text
        elif response.status_code == 501: logging.warning(f"Wolfram Alpha LLM API could not interpret query: {query}"); return f"Wolfram Alpha could not interpret the query: '{query}'."
        else: response.raise_for_status(); return f"Error: Wolfram Alpha returned status {response.status_code}."
    except requests.exceptions.Timeout: logging.error("Wolfram Alpha request timed out."); raise
    except requests.exceptions.HTTPError as http_err: logging.error(f"Wolfram Alpha HTTP error occurred: {http_err} - Response: {response.text}"); raise
    except requests.exceptions.RequestException as req_err: logging.error(f"Wolfram Alpha request error occurred: {req_err}"); raise
    except Exception as e: log_error(f"Unexpected error during Wolfram Alpha execution: {e}"); raise

# Fixed AST nodes list for calculator
_ALLOWED_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
_ALLOWED_NODES = {'Expression', 'Constant', 'Name', 'Load', 'BinOp', 'UnaryOp', 'Call', 'Add', 'Sub', 'Mult', 'Div', 'Pow', 'USub'} # 'Num' removed as noted in comments

def _safe_eval_node(node):
    if isinstance(node, ast.Constant): return node.value # Handles numbers, strings, None in Python 3.8+
    elif isinstance(node, ast.BinOp):
        op = _ALLOWED_OPERATORS.get(type(node.op))
        if not op:
            raise TypeError(f"Unsupported binary operator: {type(node.op).__name__}")
            
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        
        # Ensure operands are numeric before operation
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            # Check for division by zero before attempting division
            if isinstance(node.op, ast.Div) and right == 0:
                 raise ZeroDivisionError("division by zero")
            return op(left, right)
        else:
            raise TypeError(f"Unsupported operand types for {type(node.op).__name__}: {type(left).__name__} and {type(right).__name__}")
    elif isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPERATORS.get(type(node.op))
        if not op:
            raise TypeError(f"Unsupported unary operator: {type(node.op).__name__}")
            
        operand = _safe_eval_node(node.operand)
        if isinstance(operand, (int, float)): 
            return op(operand)
        else:
            raise TypeError(f"Unsupported operand type for {type(node.op).__name__}: {type(operand).__name__}")
    else: 
        raise TypeError(f"Unsupported node type: {type(node).__name__}")

def execute_calculator(expression: str) -> str:
    logging.info(f"--- TOOL: Calculator: Expression: {expression} ---")
    if not isinstance(expression, str): return "Error: Input expression must be a string."
    try:
        # Sanitize input slightly - remove common whitespace issues
        clean_expression = expression.strip()
        if not clean_expression: return "Error: Empty expression."

        node = ast.parse(clean_expression, mode='eval')
        # Validate all nodes in the AST
        for sub_node in ast.walk(node):
            node_type_name = type(sub_node).__name__
            # Allow Name node only for specific constants if needed (e.g., 'pi', 'e') - NOT CURRENTLY IMPLEMENTED
            # if node_type_name == 'Name': raise ValueError("Variable names are not allowed.")
            if node_type_name not in _ALLOWED_NODES:
                 raise ValueError(f"Disallowed node type found: {node_type_name}")

        result = _safe_eval_node(node.body);
        # Format result nicely, avoid excessive precision if possible
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except SyntaxError: log_error(f"Calculator: Invalid syntax in expression '{expression}'"); return "Error: Invalid syntax in expression."
    except (TypeError, ValueError, ZeroDivisionError) as e: log_error(f"Calculator: Error evaluating expression '{expression}': {e}"); return f"Error: {e}"
    except Exception as e: log_error(f"Calculator: Unexpected error evaluating '{expression}': {e}"); return "Error: Could not evaluate expression."

# Map tool names to functions - Wrap external API calls with retry logic
TOOL_FUNCTION_MAP = {
    "Google_Search": lambda q: _tool_retry_wrapper(execute_Google_Search, q),
    "Google Books": lambda q: _tool_retry_wrapper(execute_google_books, q),
    "arXiv Search": lambda q: _tool_retry_wrapper(execute_arxiv_search, q),
    "Wikipedia Search": lambda q: _tool_retry_wrapper(execute_wikipedia_search, q),
    "News Search": lambda q: _tool_retry_wrapper(execute_news_search, q),
    "Wolfram Alpha": lambda q: _tool_retry_wrapper(execute_wolfram_alpha, q),
    "Calculator": execute_calculator, # Internal calc doesn't need retry wrapper
}

# --- Core Agent Logic Implementation ---

def decide_tool_needed(current_input, working_memory):
    """Implements Step 2: Decide if a tool is needed."""
    logging.info("--- STEP 2: Deciding if Tool Needed ---")
    # --- Updated Prompt ---
    prompt = f"""You are Genesis, an autonomous exploration agent. Your goal is to decide whether external tools or APIs are necessary to handle the user's input, or if your internal reasoning is sufficient.

Evaluate:
1. Is the user asking about recent events (requiring News or Search), specific factual lookups (Wikipedia, Search, Books, arXiv), complex calculations (Wolfram Alpha, Calculator), or data retrieval beyond your current knowledge?
2. Does your Working Memory contain enough recent and relevant context to respond directly and accurately?

User Input: "{current_input}"

Working Memory Snapshot (Recent History & Analysis):
{json.dumps(working_memory, default=str, indent=2, ensure_ascii=False)[:1000]}...

Based on this evaluation, is an external tool necessary?
Respond ONLY with JSON in the format: {{"tool_needed": boolean, "reason": string}}"""
    # --- End Updated Prompt ---
    response = call_gemini_flash_lite(prompt, "Decision 1")
    
    # Validate response schema
    default_values = {"tool_needed": False, "reason": "LLM call failed"}
    response = validate_response_schema(response, ["tool_needed", "reason"], default_values)
    
    return response

def identify_tool(current_input, working_memory):
    """Implements Step 3: Identify the best tool."""
    logging.info("--- STEP 3: Identifying Tool ---")
    # --- Updated Prompt ---
    tool_descriptions = {
        "Google_Search": "Use for finding recent information, general knowledge, specific websites, or when other tools are unsuitable.",
        "Google Books": "Use for finding information about books given a title, author, or ISBN.",
        "arXiv Search": "Use for finding scientific pre-print papers on arXiv by topic or keyword.",
        "Wikipedia Search": "Use for getting a concise summary and URL for a specific topic, person, or entity from Wikipedia.",
        "News Search": "Use for finding recent news articles about current events or specific topics from various sources.",
        "Wolfram Alpha": "Use for calculations, data conversions, math problems, or retrieving specific factual data points (like populations, chemical properties). Formulate query clearly.",
        "Calculator": "Use ONLY for simple arithmetic expressions (e.g., '2 + 2 * (3/4)', '5**3'). Do not use for natural language queries."
    }
    available_tools_desc = "\n".join([f"- {name}: {desc}" for name, desc in tool_descriptions.items() if name in TOOL_FUNCTION_MAP])

    prompt = f"""You are Genesis, an autonomous agent equipped with various tools for exploration and learning. The user's input requires external support. Choose the best tool and generate an optimized query.

User Input: "{current_input}"

Context Snapshot:{json.dumps(working_memory, indent=2, default=str)[:1000]}...

Available Tools:{available_tools_desc}

Choose the single most appropriate tool from the 'Available Tools' list. If the 'Calculator' tool is chosen, the 'query' should be *only* the mathematical expression itself (e.g., "2*pi*6371"). For 'Wolfram Alpha', formulate a clear natural language or symbolic query suitable for it. For search tools, generate concise and effective search terms. If no tool seems suitable, set "tool_name" to null.

Respond ONLY with JSON in the format:
{{
  "tool_name": string | null,
  "query": string | null,
  "reason": string
}}"""
    # --- End Updated Prompt ---
    response = call_gemini_flash_std(prompt, "Tool ID")

    # Validate response schema
    default_values = {"tool_name": None, "query": None, "reason": "LLM call failed"}
    response = validate_response_schema(response, ["tool_name", "query", "reason"], default_values)
    
    return response

def execute_tool(tool_name, query):
    """Handles dispatching to the correct tool function."""
    logging.info(f"--- STEP 4: Executing Tool: {tool_name} ---")
    if tool_name in TOOL_FUNCTION_MAP:
        # Validate query for calculator before execution
        if tool_name == "Calculator" and not isinstance(query, str):
             log_error(f"Invalid query type for Calculator: {type(query)}. Expected string.")
             return "Error: Calculator query must be a string expression."
        # Ensure query is provided for tools that need it
        if query is None and tool_name != "SomeToolThatNeedsNoQuery": # Example hypothetical tool
            log_error(f"Query is missing for tool: {tool_name}")
            return f"Error: Query is required for tool '{tool_name}'"

        result = TOOL_FUNCTION_MAP[tool_name](query) # Retry handled by lambda wrapper for external tools
        return result
    else:
        log_error(f"Unknown tool selected: {tool_name}")
        return f"Error: Unknown tool '{tool_name}'"

def analyze_result(tool_result, current_input, working_memory):
    """Implements Step 5: Analyze tool output or internal result."""
    logging.info("--- STEP 5: Analyzing Result ---")
    # --- Updated Prompt ---
    prompt = f"""You are Genesis, an agent trained to evaluate and synthesize information.

Analyze the result below and determine:
1. What is the key takeaway relevant to the original input? Summarize concisely.
2. Was the result helpful for the original task? ("very helpful" | "partially helpful" | "not helpful" | "conflicting")
3. What is your simulated stance or reflection on this information? (string | null - Keep brief, max 1-2 sentences)
4. Suggest 1–3 relevant tags (topics/themes, single words or short phrases, lowercase) for long-term tracking. Can be new or existing tags. (list[string])
5. Based on the summary and evaluation, should this result be saved to a local Markdown file for detailed reference? (boolean - save if summary is substantial and evaluation is helpful)

Original Input: "{current_input}"

Tool Result/Data:
{str(tool_result)[:2000]}...

Context (Working Memory, including known tags):
{json.dumps(working_memory, indent=2, default=str)[:1000]}...

Respond ONLY with JSON in the format:
{{
  "summary": string,
  "evaluation": "very helpful" | "partially helpful" | "not helpful" | "conflicting",
  "simulated_stance": string | null,
  "suggested_tags": list[string],
  "save_summary_to_md": boolean
}}"""
    # --- End Updated Prompt ---
    response = call_gemini_pro(prompt, "Analysis")

    # Validate response schema with defaults
    default_values = {
        "summary": "Analysis failed.", 
        "evaluation": "unknown", 
        "simulated_stance": None, 
        "suggested_tags": [], 
        "save_summary_to_md": False
    }
    response = validate_response_schema(
        response, 
        ["summary", "evaluation", "simulated_stance", "suggested_tags", "save_summary_to_md"],
        default_values
    )

    # Store results in working memory using thread-safe methods
    agent_state.update_working_memory('last_analysis', response)
    agent_state.update_working_memory('last_stance', response.get('simulated_stance'))
    
    # Store feedback history
    feedback = {
        "input": current_input,
        "tool_used": working_memory.get('last_tool_id', {}).get('tool_name', 'direct_answer/unknown'),
        "evaluation": response.get('evaluation', 'unknown')
    }
    agent_state.append_to_feedback(feedback)
    
    # Clean and add tags
    suggested_tags = response.get('suggested_tags', [])
    if isinstance(suggested_tags, list):
        valid_tags = [str(tag).lower().strip() for tag in suggested_tags if tag and isinstance(tag, str)]
        agent_state.add_tags(valid_tags)
        response['suggested_tags'] = valid_tags  # Update response with cleaned tags
    else:
        log_error(f"suggested_tags in analysis response is not a list: {suggested_tags}")
        response['suggested_tags'] = []

    return response


def decide_store_intermediate(analysis_result, working_memory):
    """Implements Step 6: Decide if intermediate result is valuable for LTR."""
    logging.info("--- STEP 6: Deciding Store Intermediate ---")
    # --- Updated Prompt ---
    prompt = f"""You are Genesis, evaluating whether a recent insight should be saved for future reference in Long-Term Memory (LTR).

Consider:
- Is the summary novel or insightful, not just a trivial restatement?
- Was the evaluation "very helpful" or "partially helpful"?
- Would retrieving this summary likely benefit future tasks in similar contexts? Avoid storing redundant or low-value information.

Analysis:{json.dumps(analysis_result, indent=2)}

Working Memory:{json.dumps(working_memory, indent=2, default=str)[:1000]}...

Should this specific analysis result (primarily the summary and tags) be stored in Long-Term Memory now?
Respond ONLY with JSON in the format: {{"store_result": boolean, "reason": string}}"""
    # --- End Updated Prompt ---
    response = call_gemini_flash_lite(prompt, "Decision 6")
    
    # Validate response schema
    default_values = {"store_result": False, "reason": "LLM call failed or response invalid"}
    response = validate_response_schema(response, ["store_result", "reason"], default_values)
    
    return response

def store_memory_step(data_to_store, metadata_context):
    """Handles embedding and storing in LTR (Long-Term Memory)."""
    logging.info("--- STEP 7: Storing in LTR ---")

    # --- Step 1: Define what to embed ---
    summary_text = None
    if isinstance(data_to_store, dict):
        summary_text = data_to_store.get("summary")
    if not summary_text:
        # Fallback: try to create a string representation, but log a warning
        logging.warning(f"No 'summary' found in data_to_store for LTR. Attempting JSON dump as fallback. Data: {str(data_to_store)[:200]}...")
        try:
             summary_text = json.dumps(data_to_store, default=str)
        except Exception as e:
             log_error(f"Could not serialize data_to_store for LTR fallback: {e}")
             summary_text = "Error: Could not serialize data for embedding."

    if not summary_text or summary_text == "Error: Could not serialize data for embedding.":
        log_error("Cannot store empty or unserializable data in LTR."); return

    # --- Step 2: Build metadata payload ---
    metadata = {
        "source": str(metadata_context.get("source", "unknown")), # Ensure string
        "original_input": str(metadata_context.get("original_input", "unknown")), # Ensure string
        "evaluation": str(data_to_store.get("evaluation", "unknown")),
        "simulated_stance": str(data_to_store.get("simulated_stance", "none")),
        "tags": data_to_store.get("suggested_tags", []), # Should be list of strings from analyze_result
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "v1.0" # Example versioning
        # "md_filename": metadata_context.get("md_filename") # Add if saving to file and filename available
    }

    # Ensure tags are valid for storage (list of strings)
    if not isinstance(metadata['tags'], list) or not all(isinstance(t, str) for t in metadata['tags']):
        logging.warning(f"Invalid 'tags' format for LTR storage: {metadata['tags']}. Resetting to empty list.")
        metadata['tags'] = []

    # --- Step 3: Generate Embedding ---
    logging.debug(f"Content to embed for LTR: {summary_text[:200]}...")
    embedding = get_embedding(summary_text)
    if not embedding: log_error("Failed to generate embedding for LTR storage."); return

    # --- Step 4: Store in LTR ---
    data_id = str(uuid.uuid4())
    # Filter metadata for LTR restricts based on expected structure in store_in_ltr
    # Assuming store_in_ltr handles restrict creation based on keys like 'year', 'tags', 'source', etc.
    final_metadata = {k: v for k, v in metadata.items()} # Pass the constructed metadata dict

    store_in_ltr(embedding, data_id, final_metadata) # store_in_ltr handles restrict creation now

def decide_next_action(working_memory):
    """Implements Step 8: Decide the next action."""
    logging.info("--- STEP 8: Deciding Next Action ---")
    # --- Add LTR Context ---
    ltr_context_summary = "No relevant memories found."
    # Use a more specific query based on the last *successful* analysis or input
    last_analysis = working_memory.get('last_analysis')
    context_query = None
    if last_analysis and last_analysis.get("summary") not in [None, "Analysis failed.", "Analysis response structure error."]:
         context_query = last_analysis.get("summary")
    elif working_memory.get('history'):
         # Find the last actual input content
         for item in reversed(working_memory['history']):
              if item.get("type") == "input" and item.get("content"):
                   context_query = item["content"]
                   break

    if context_query:
        logging.debug(f"Querying LTR for context based on: {context_query[:100]}...")
        neighbors = query_ltr(context_query, top_k=3) # Consider making top_k configurable
        if neighbors:
            # Format neighbor info concisely for the prompt
            neighbor_summaries = []
            for n in neighbors:
                 # Assuming Vector Search doesn't return the full stored data, just ID/distance.
                 # If it did return metadata, you could include parts of it here.
                 neighbor_summaries.append(f"ID: {n.id} (Dist: {n.distance:.4f})")
            ltr_context_summary = f"Found potentially related memories:\n" + "\n".join(neighbor_summaries)
            logging.debug(f"LTR Context added: {ltr_context_summary}")
        else:
            logging.debug("No neighbors found in LTR for context query.")
    else:
        logging.debug("No suitable query found for LTR context retrieval.")

    # Store context temporarily for this decision
    working_memory_copy = dict(working_memory)
    working_memory_copy['ltr_context_for_decision'] = ltr_context_summary
    # --- End Add LTR Context ---
    
    # --- Add Feedback Context ---
    feedback_summary = "No recent feedback available."
    if working_memory.get('feedback_history'): 
        feedback_summary = "Recent Feedback (Input -> Tool -> Evaluation):\n" + "\n".join([
            f"- '{f.get('input', '')[:30]}...' -> {f.get('tool_used', '?')} -> '{f.get('evaluation', '?')}'" 
            for f in working_memory['feedback_history']
        ])
    working_memory_copy['recent_feedback_summary'] = feedback_summary
    # --- End Add Feedback Context ---

    # --- Updated Prompt ---
    prompt = f"""You are Genesis, an agent focused on exploration, learning, synthesis, and divergence.
Your goal is to decide the next step in your simulation loop. Review the current context, including recent actions, analysis, your simulated stance, potentially relevant long-term memories (LTR), and feedback on recent actions.

Current Context (Working Memory):
{json.dumps(working_memory_copy, indent=2, default=str, ensure_ascii=False)[:2500]}... # Increased context size slightly

Available Strategies for Next Action:
- Deepen Current Topic: Ask a specific follow-up question based on the last analysis or result. Query to address ambiguity or conflicting info. Cross-reference findings with LTR context. Consult saved notes (.md files - specify filename if known/relevant). Refine previous output.
- Broaden/Switch Topic: Explore identified tangents (e.g., from tags or LTR context). Synthesize/generalize recent findings into a broader concept. Query LTR for novel related topics based on current tags or stance.
- Conclude Task/Topic: If the current exploration feels sufficiently complete (e.g., input answered, analysis evaluation satisfactory, feedback positive, no clear next steps for deepening/broadening).
- Goal Check / Self-Direction: Evaluate progress on any emergent goal. If no goal, decide if you want one now (e.g., based on interesting LTR topics or stance). Propose querying LTR for interesting past topics to focus on.

Task:
1. Assess if the current task/topic exploration feels complete ("Topic Completion Criteria"). Consider the analysis evaluation and recent feedback. If the last evaluation was 'not helpful' or 'conflicting', consider deepening or switching.
2. Consider the recent feedback summary - did past actions yield useful results? Let this influence your strategy choice. If a path seems unproductive based on feedback (e.g., multiple 'not helpful' evaluations), strongly consider switching strategy.
3. Review the LTR context. Does it suggest relevant tangents, conflicts, or related areas to explore?
4. Choose ONE strategy from 'Available Strategies' based on the goal of divergence and exploration, informed by context, LTR, and feedback.
5. If concluding, set "conclude_task" to true. "next_input" can be null or a brief concluding thought.
6. If continuing (Deepen, Broaden, Goal Check), generate a specific, actionable "next_input" (a question, topic, search query idea, or task description) for the next loop cycle based on your chosen strategy. The input should aim to generate new information or understanding.
7. Provide a brief "reason" explaining your choice of strategy and the resulting next input.

Respond ONLY with JSON in the format: {{"next_action_type": string (e.g., "Deepen", "Broaden", "Conclude", "Goal Check"), "next_input": string | null, "reason": string, "conclude_task": boolean}}"""
    # --- End Updated Prompt ---

    response = call_gemini_pro(prompt, "Decision 3")
    
    # Validate response schema
    default_values = {
        "next_action_type": "Error", 
        "next_input": "Recovery: What should I explore next based on recent activity?", 
        "reason": "LLM call failed or response invalid", 
        "conclude_task": False
    }
    response = validate_response_schema(
        response, 
        ["next_action_type", "next_input", "reason", "conclude_task"],
        default_values
    )

    # Ensure conclude_task matches expectation
    if response.get("next_action_type") == "Conclude" and not response.get("conclude_task"):
         logging.warning("Decision 3 chose 'Conclude' but set 'conclude_task' to false. Overriding to true.")
         response["conclude_task"] = True
    elif response.get("conclude_task") and response.get("next_action_type") != "Conclude":
         logging.warning(f"Decision 3 set 'conclude_task' to true but action type is '{response.get('next_action_type')}'. Overriding to false.")
         response["conclude_task"] = False

    return response

def generate_direct_answer(current_input, working_memory):
    """Implements Step 9: Generate a direct answer when no tool is needed."""
    logging.info("--- STEP 9: Generating Direct Answer ---")
    # --- Updated Prompt ---
    prompt = f"""You are Genesis, an exploratory agent. Based on the user's input and your current working memory, generate a direct, thoughtful answer. If the working memory contains relevant analysis or stance, incorporate it naturally.

User Input: "{current_input}"

Working Memory (includes recent history, analysis, stance, known tags):
{json.dumps(working_memory, indent=2, default=str, ensure_ascii=False)[:1500]}...

Generate a direct answer to the user input, using the available context. Keep the answer concise yet informative.

Respond ONLY with JSON in the format:
{{
  "answer": string
}}"""
    # --- End Updated Prompt ---
    response = call_gemini_pro(prompt, "Direct Answer")

    # Validate response schema
    default_values = {"answer": "I encountered an issue generating a direct answer. Please try rephrasing or asking something else."}
    response = validate_response_schema(response, ["answer"], default_values)
    
    # Update working memory
    agent_state.update_working_memory('last_direct_answer', response.get('answer'))
    
    return response


# --- Utility Functions ---
def log_error(message):
    timestamp = datetime.datetime.now().isoformat(); logging.error(message)
    try:
        # Ensure error log file path exists
        log_dir = os.path.dirname(ERROR_LOG_FILE);
        if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir)
        # Append error message
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f: f.write(f"{timestamp} - ERROR: {message}\n")
    except Exception as e:
        # Fallback to stderr if logging fails
        print(f"FATAL: Could not write to error log file {ERROR_LOG_FILE}: {e}", file=sys.stderr)
        print(f"Original Error: {timestamp} - ERROR: {message}", file=sys.stderr)


# --- Input Handling (Basic Example) ---
user_input_queue = queue.Queue()
def check_user_input():
    """Runs in a separate thread to capture user input without blocking."""
    logging.info("Input thread started. Enter commands or chat messages.")
    while True:
        try:
            # Use input() which blocks this thread, waiting for user entry
            inp = input()
            user_input_queue.put(inp)
            if inp.strip().lower() == "/quit": # Allow quitting from input thread too
                 logging.info("Quit command received in input thread.")
                 break
        except EOFError:
            # Handle Ctrl+D or end-of-file condition
            logging.info("EOF received, signaling quit.")
            user_input_queue.put("/quit")
            break
        except Exception as e:
             log_error(f"Error in input thread: {e}")
             # Consider whether to break or continue based on error type
             time.sleep(1) # Avoid tight loop on continuous errors

input_thread = threading.Thread(target=check_user_input, daemon=True)
input_thread.start()

# --- Graceful Model Initialization ---
def initialize_gemini_models():
    """Initializes Gemini models with graceful degradation."""
    models = {}
    critical_failure = True
    
    try:
        models["flash_lite"] = GenerativeModel(config['GEMINI_FLASH_LITE_MODEL_NAME'])
        critical_failure = False  # At least one model initialized
        logging.info(f"Successfully initialized Gemini Flash Lite model: {config['GEMINI_FLASH_LITE_MODEL_NAME']}")
    except Exception as e:
        log_error(f"Failed to initialize Gemini Flash Lite: {e}")
    
    try:
        models["flash_std"] = GenerativeModel(config['GEMINI_FLASH_STD_MODEL_NAME'])
        critical_failure = False  # At least one model initialized
        logging.info(f"Successfully initialized Gemini Flash Standard model: {config['GEMINI_FLASH_STD_MODEL_NAME']}")
    except Exception as e:
        log_error(f"Failed to initialize Gemini Flash Standard: {e}")
    
    try:
        models["pro"] = GenerativeModel(config['GEMINI_PRO_MODEL_NAME'])
        critical_failure = False  # At least one model initialized
        logging.info(f"Successfully initialized Gemini Pro model: {config['GEMINI_PRO_MODEL_NAME']}")
    except Exception as e:
        log_error(f"Failed to initialize Gemini Pro: {e}")
    
    return models, critical_failure

# --- Main Orchestration Loop ---
def run_simulation():
    """Main function to run the Genesis simulation."""
    logging.info(f"Starting {AGENT_NAME} Simulation...")
    print(f"--- {AGENT_NAME} ---")
    print("Enter commands: /pause, /chat, /resume, /quit")
    
    # Initialize agent state
    agent_state.reset_working_memory(preserve_tags=True)

    loop_count = 0 # Add a loop counter for debugging/monitoring

    while True:
        loop_count += 1
        logging.debug(f"--- Loop {loop_count} Start --- Mode: {agent_state.mode} ---")
        command = None
        try:
            # Non-blocking check for user input
            command = user_input_queue.get_nowait()
            if command: print(f"\n[COMMAND RECEIVED]: {command}")
        except queue.Empty: pass
        except Exception as e: log_error(f"Error checking user input queue: {e}") # Should not happen with Queue

        # --- Command Handling ---
        if command:
             command_lower = command.strip().lower()
             if command_lower == "/quit":
                 logging.info("Quitting simulation via user command."); break
             elif command_lower == "/pause":
                 if agent_state.mode == "AUTONOMOUS":
                      logging.info("Pause requested. Will pause after current cycle."); agent_state.update_working_memory('pause_requested', True)
                 elif agent_state.mode == "CHATTING":
                      logging.info("Switching from Chat to Paused mode."); agent_state.mode = "PAUSED"; print("[MODE]: Paused. Enter /chat or /resume.")
                 else: # Already PAUSED
                      logging.info("Already paused.")
             elif command_lower == "/chat":
                 if agent_state.mode == "PAUSED":
                      logging.info("Entering Conversational Mode..."); agent_state.mode = "CHATTING"; print("[MODE]: Chatting. Enter your message or /pause, /resume, /quit.")
                 else:
                      print("Command ignored: Please /pause the simulation before entering /chat mode.")
             elif command_lower == "/resume":
                 if agent_state.mode == "PAUSED" or agent_state.mode == "CHATTING":
                      logging.info("Resuming Autonomous Mode..."); agent_state.mode = "AUTONOMOUS"; agent_state.pop_working_memory('pause_requested', None); print("[MODE]: Autonomous")
                 else: # Already AUTONOMOUS
                      logging.info("Already running in Autonomous Mode.")
             # Add other commands here if needed
             # elif command_lower == "/status": print(f"Current Mode: {agent_state.mode}\nWorking Memory Keys: {list(agent_state.working_memory.keys())}")
             else:
                  # If not a known command and in CHATTING mode, treat as chat input
                  if agent_state.mode == "CHATTING":
                       logging.info("Treating non-command input as chat message.")
                       # Re-queue the command as if it were normal chat input
                       user_input_queue.put(command)
                       command = None # Clear command so it's processed by chat logic below
                  # else: print(f"Unknown command: {command}") # Ignore unknown commands in other modes?

        # --- Mode Execution ---
        if agent_state.mode == "AUTONOMOUS":
            logging.info(f"--- Autonomous Loop Iteration {loop_count} ---")
            current_input = agent_state.current_task_input
            if not current_input: # Should not happen with recovery in decide_next_action
                 log_error("Autonomous loop started with no input. Using recovery prompt.")
                 current_input = "Recovery: What should I explore next?"
                 agent_state.current_task_input = current_input

            logging.info(f"Input: {current_input[:150]}...") # Log more input context
            agent_state.append_to_history({"type": "input", "timestamp": datetime.datetime.now().isoformat(), "content": current_input})

            # --- Execute Steps 2-9 ---
            decision1 = decide_tool_needed(current_input, agent_state.working_memory)
            agent_state.update_working_memory('last_decision1', decision1)
            next_step_input_data = None; tool_name = None; analysis_result = None; tool_result = None

            if decision1 and decision1.get("tool_needed"):
                tool_info = identify_tool(current_input, agent_state.working_memory)
                agent_state.update_working_memory('last_tool_id', tool_info)
                tool_name = tool_info.get("tool_name") if tool_info else None
                tool_query = tool_info.get("query") if tool_info else None # Query might be None intentionally for some tools
                if tool_name:
                    # Added check: If identify_tool failed to provide a query for a tool that needs one
                    if tool_query is None and tool_name not in ["Calculator"]: # Adjust if other tools can accept None query
                        log_error(f"Tool '{tool_name}' selected, but no query was generated. Skipping tool execution.")
                        tool_result = f"Error: No query generated for tool '{tool_name}'."
                    else:
                        tool_result = execute_tool(tool_name, tool_query)

                    agent_state.update_working_memory('last_tool_result', tool_result)
                    next_step_input_data = tool_result # Data for analysis is the tool's raw output
                else:
                    logging.warning("Tool needed, but identification failed or returned null tool. Proceeding without tool.")
                    next_step_input_data = "Tool identification failed." # Data for analysis
                    tool_name = "tool_identification_failed" # Set a marker for analysis/feedback
            else:
                # Generate direct answer (Step 9)
                direct_answer_result = generate_direct_answer(current_input, agent_state.working_memory) # Already updates working_memory['last_direct_answer']
                next_step_input_data = direct_answer_result.get('answer', 'Direct answer generation failed.') if direct_answer_result else 'Direct answer generation failed.'
                tool_name = "direct_answer" # Mark source for analysis/feedback

            # Analyze the result (from tool or direct answer) - Step 5
            # Always run analysis, even if tool failed, to decide next steps
            analysis_result = analyze_result(next_step_input_data, current_input, agent_state.working_memory) # Already updates working_memory['last_analysis'], etc.

            # --- Trigger MD Save based on analysis ---
            if analysis_result and analysis_result.get("save_summary_to_md"):
                summary_to_save = analysis_result.get("summary")
                # Only save if summary is meaningful
                if summary_to_save and summary_to_save not in ["Analysis failed.", "Analysis response structure error."]:
                    topic_keys = analysis_result.get("suggested_tags", ["analysis_summary"]) # Use cleaned tags
                    saved_filepath = store_output_md(summary_to_save, topic_keys)
                    if saved_filepath: agent_state.update_working_memory('last_md_saved_path', saved_filepath) # Store path for potential LTR metadata
                else: log_error("Analysis requested saving MD, but summary was empty or invalid.")
            # --- End MD Save Trigger ---


            # Decide whether to store this specific analysis in LTR immediately - Step 6
            # Only store if analysis was successful and deemed valuable
            if analysis_result and analysis_result.get("summary") not in ["Analysis failed.", "Analysis response structure error."]:
                decision2 = decide_store_intermediate(analysis_result, agent_state.working_memory)
                agent_state.update_working_memory('last_decision2', decision2)
                if decision2 and decision2.get("store_result"):
                    metadata_context = {"source": tool_name if tool_name else "unknown_source", "original_input": current_input}
                    # Add md filename to metadata if it was just saved
                    if agent_state.get_working_memory('last_md_saved_path'):
                         metadata_context['md_filename'] = os.path.basename(agent_state.get_working_memory('last_md_saved_path'))
                    store_memory_step(analysis_result, metadata_context) # Call Step 7

            # Clear last_md_saved_path after potential use in metadata
            agent_state.pop_working_memory('last_md_saved_path', None)

            # Decide next action - Step 8
            next_action = decide_next_action(agent_state.working_memory)
            agent_state.update_working_memory('last_decision3', next_action)

            # Handle task conclusion (includes potential final LTR storage)
            if next_action and next_action.get("conclude_task"):
                logging.info(f"--- Task Concluded (Reason: {next_action.get('reason', 'N/A')}) ---")
                # Optionally, store a final consolidated summary from working memory to LTR
                # This logic could be refined: maybe only store if the whole task was useful?
                final_summary_data = agent_state.get_working_memory('last_analysis')
                if final_summary_data and final_summary_data.get("evaluation", "not helpful") != "not helpful": # Example condition
                    logging.info("Storing final task summary to LTR.")
                    metadata_context = {"source": "task_conclusion", "original_input": current_input} # Find initial input? Needs better history tracking maybe.
                    metadata_context["tags"] = final_summary_data.get("suggested_tags", ["task_completion"])
                    # Add final MD file if relevant?
                    store_memory_step(final_summary_data, metadata_context) # Call Step 7

                # Reset working memory for the next task, preserving tags
                agent_state.reset_working_memory(preserve_tags=True)
                logging.info("Working memory reset for next task.")
                
                # Set a default next input if conclusion didn't provide one
                agent_state.current_task_input = next_action.get("next_input") or "What interesting topic should I explore next?"

            else: # If not concluding, set input for the next loop
                agent_state.current_task_input = next_action.get("next_input") if next_action else None
                if not agent_state.current_task_input:
                    # Recovery if decide_next_action failed or returned null input unexpectedly
                    log_error("Decision 3 failed to provide next input for non-concluding task. Using default recovery prompt.")
                    agent_state.current_task_input = "Recovery: What should I explore next based on recent activity?"

            # --- Autonomous Loop End / Pause Check ---
            # Check pause request at end of loop cycle
            if agent_state.get_working_memory('pause_requested'):
                logging.info("Pause command processed. Entering Paused Mode...");
                agent_state.mode = "PAUSED";
                agent_state.pop_working_memory('pause_requested') # Clear the flag
                print("[MODE]: Paused. Enter /chat or /resume.")
            else:
                 # Optional delay between autonomous loops to avoid hitting rate limits etc.
                 time.sleep(1) # Consider making this configurable

        elif agent_state.mode == "PAUSED":
            # Just wait for commands, checked at the start of the loop
            time.sleep(0.5) # Sleep briefly to avoid busy-waiting

        elif agent_state.mode == "CHATTING":
            # Process chat input received via the command handling section
            try:
                # Check queue again for messages entered while processing previous chat
                user_chat_input = user_input_queue.get_nowait()
                print(f"\nYou: {user_chat_input}")
                logging.info(f"Processing chat input: {user_chat_input[:100]}...")
                # Use generate_direct_answer for chat, providing current working_memory as context
                chat_response_data = generate_direct_answer(user_chat_input, agent_state.working_memory)
                chat_response = chat_response_data.get('answer', 'Sorry, I encountered an issue processing that.')
                print(f"\nGenesis: {chat_response}")

                # Add chat interaction to history? Maybe a separate chat history?
                # agent_state.append_to_history({"type": "chat", "user": user_chat_input, "agent": chat_response})

            except queue.Empty:
                # No new chat input, wait briefly
                time.sleep(0.2)
            except Exception as e:
                log_error(f"Error in conversational loop processing: {e}")
                print("Genesis: An internal error occurred while processing your message.")

    # --- End of Main Loop ---
    logging.info(f"{AGENT_NAME} Simulation Ended.")
    print(f"--- {AGENT_NAME} Simulation Ended ---")


def initialize_clients():
    """Initializes Vertex AI and Gemini clients, and other required API clients."""
    global embedding_model, ltr_endpoint, gemini_flash_lite_model, gemini_flash_std_model, gemini_pro_model
    global news_client, wolfram_client, openai_client # Declare globals being assigned

    logging.info("--- Initializing API Clients ---")

    if not initialize_vertex_ai():
         log_error("Critical failure initializing Vertex AI. Attempting to continue with limited functionality.")
         # Continue but with limited functionality instead of aborting

    # Initialize Gemini Models with graceful degradation
    gemini_models, critical_gemini_failure = initialize_gemini_models()
    
    # Assign models to global variables
    gemini_flash_lite_model = gemini_models.get("flash_lite")
    gemini_flash_std_model = gemini_models.get("flash_std")
    gemini_pro_model = gemini_models.get("pro")
    
    if critical_gemini_failure:
         log_error("Critical failure: No Gemini models could be initialized. Agent will have limited functionality.")
         # Could continue with fallback behavior or reduced capabilities

    # --- Initialize other API clients (Optional - Log warnings if fail) ---
    if config.get("NEWS_API_KEY"):
        try:
            news_client = NewsApiClient(api_key=config['NEWS_API_KEY'])
            # Test connection? Optional, but good. Example:
            # news_client.get_sources(language='en', country='us')
            logging.info("NewsAPI client initialized.")
        except ImportError: log_error("NewsAPI client library not installed (`pip install newsapi-python`). Falling back to requests for News Search."); news_client = None
        except Exception as e: log_error(f"Failed to initialize NewsAPI client: {e}"); news_client = None # Continue, but log error
    else: logging.warning("NEWS_API_KEY not found in config. News Search tool (using client library) will not work.")

    # Wolfram Alpha only needs App ID - requests used directly
    if not config.get("WOLFRAM_APP_ID"):
         logging.warning("WOLFRAM_APP_ID not found in config. Wolfram Alpha tool will not work.")
    else:
         logging.info("Wolfram Alpha App ID found.")
    
    # Example for OpenAI if it were used (currently isn't in tool map)
    if config.get("OPENAI_API_KEY"):
        try:
            openai_client = openai.OpenAI(api_key=config["OPENAI_API_KEY"])
            # Test connection? e.g., openai_client.models.list()
            logging.info("OpenAI client configured (if needed).")
        except ImportError: log_error("OpenAI library not installed (`pip install openai`). OpenAI features unavailable."); openai_client = None
        except Exception as e: log_error(f"Failed to initialize OpenAI client: {e}"); openai_client = None
    else: logging.debug("OPENAI_API_KEY not found. OpenAI client not initialized.") # Debug level as it's optional

    # Wikipedia client initialization (library handles it implicitly on first use)
    logging.info("Wikipedia client will be used directly via library.")

    logging.info("--- Client initialization process complete ---")
    return True

def initialize_vertex_ai():
    """Initializes Vertex AI, loads embedding model, and LTR endpoint client."""
    global embedding_model, ltr_endpoint
    project = config.get("VERTEX_AI_PROJECT"); location = config.get("VERTEX_AI_LOCATION"); endpoint_id = config.get("VERTEX_AI_INDEX_ENDPOINT_ID")
    if not project or not location:
        log_error("VERTEX_AI_PROJECT and VERTEX_AI_LOCATION must be set in environment or .env file.")
        return False
    logging.info(f"Initializing Vertex AI for Project: {project}, Location: {location}")
    try:
        # Ensure ADC (gcloud auth application-default login) or service account key (GOOGLE_APPLICATION_CREDENTIALS) is configured.
        vertexai.init(project=project, location=location)

        # Load Embedding Model
        model_name = config.get('VERTEX_AI_EMBEDDING_MODEL', "text-embedding-005") # Default if missing
        logging.info(f"Loading Embedding Model: {model_name}")
        try:
            embedding_model = TextEmbeddingModel.from_pretrained(model_name)
            logging.info(f"Successfully loaded embedding model: {model_name}")
        except Exception as e:
            log_error(f"Failed to load embedding model: {e}")
            # Continue with embedding_model as None

        # Initialize LTR (Vector Search) Endpoint Client if configured
        if endpoint_id:
            logging.info(f"Initializing LTR Endpoint Client for Endpoint ID: {endpoint_id}")
            # Ensure the endpoint ID is the full resource name, e.g., projects/.../indexEndpoints/...
            if not endpoint_id.startswith("projects/"):
                 log_error(f"VERTEX_AI_INDEX_ENDPOINT_ID seems incorrect. Expected format: projects/PROJECT_ID/locations/LOCATION/indexEndpoints/ENDPOINT_ID. Value: {endpoint_id}")
                 # Will not attempt to construct it - better to know there's an issue
            else:
                try:
                    ltr_endpoint = MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_id)
                    logging.info(f"LTR Endpoint Client initialized for: {ltr_endpoint.resource_name}")
                except Exception as e:
                    log_error(f"Failed to initialize LTR endpoint: {e}")
                    # Continue with ltr_endpoint as None
                
            if not config.get("VERTEX_AI_DEPLOYED_INDEX_ID"):
                logging.error("CRITICAL: VERTEX_AI_DEPLOYED_INDEX_ID not set. LTR queries will fail.") # Error level now
        else:
            logging.warning("VERTEX_AI_INDEX_ENDPOINT_ID not set. LTR functions (store/query) will not work.")

        return True
    except Exception as e:
        log_error(f"Failed to initialize Vertex AI or load models/clients: {e}")
        if "permission denied" in str(e).lower() or "forbidden" in str(e).lower(): log_error("Check gcloud authentication (ADC), service account permissions (e.g., Vertex AI User role), and ensure Vertex AI API is enabled.")
        elif "not found" in str(e).lower() and "models/" in str(e).lower(): log_error(f"Embedding model '{model_name}' not found or inaccessible in {location}. Check model name and region.")
        elif "not found" in str(e).lower() and "indexEndpoints/" in str(e).lower(): log_error(f"LTR Endpoint '{endpoint_id}' not found or inaccessible. Check endpoint ID and region.")
        return False

if __name__ == "__main__":
    try:
        # Create output and log directories if they don't exist
        md_dir = config.get("MD_OUTPUT_DIR", "genesis_outputs");
        if not os.path.exists(md_dir): os.makedirs(md_dir)

        log_file = config.get("ERROR_LOG_FILE", "genesis_error.log");
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir)
        # Basic logging setup (re-iterate in main scope for clarity)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

    except Exception as e:
         print(f"FATAL: Error setting up output/log directories: {e}", file=sys.stderr)
         sys.exit(1) # Exit if basic setup fails

    # Initialize all API clients
    if not initialize_clients():
         print("Warning: Some client initialization failed. Continuing with limited functionality.", file=sys.stderr)
         # Continue with limited functionality instead of exiting

    # Start the simulation loop
    run_simulation()
