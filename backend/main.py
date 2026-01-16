from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from openai import AsyncAzureOpenAI
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import urllib.robotparser
from urllib.parse import urljoin, urlparse
import os
from datetime import datetime
import uuid
import logging
import json
from pyprojroot import here
from dotenv import dotenv_values
from ddgs import DDGS


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open(here("backend/config.json"), "r") as f:
    config = json.load(f)

# Extract Azure OpenAI config
azure_config = config["azure_openai"]
CHAT_DEPLOYMENT_NAME = azure_config["chat_deployment_name"]
EMBEDDING_DEPLOYMENT_NAME = azure_config["embedding_deployment_name"]
API_VERSION = azure_config["api_version"]

app = FastAPI(title="AI Article Assistant API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
env_vars = dotenv_values(here("backend/.env"))

# Initialize Azure OpenAI client
openai_client = AsyncAzureOpenAI(
    api_key=env_vars.get("AZURE_OPENAI_KEY"),
    api_version=API_VERSION,
    azure_endpoint=env_vars.get("AZURE_OPENAI_ENDPOINT")
)

# Initialize DDGS client
ddgs_client = DDGS()

# ChromaDB Configuration
chroma_client = chromadb.PersistentClient(path=here("backend/chroma_db"))
collection = chroma_client.get_or_create_collection(
    name="articles",
    metadata={"hnsw:space": "cosine"}
)

# Pydantic Models
class CrawlRequest(BaseModel):
    keywords: List[str]
    max_articles: int = 10
    region: Optional[str] = "uk-en"

class ChatMessage(BaseModel):
    role: str
    content: str

class QuestionRequest(BaseModel):
    question: str
    messages: Optional[List[ChatMessage]] = []

class Article(BaseModel):
    id: str
    title: str
    url: str
    summary: Optional[str] = None
    date_published: Optional[str] = None
    date_added: str
    public: bool
    source: str
    content_type: Optional[str] = None
    region: Optional[str] = None
    sentiment: Optional[str] = None  # "positive", "neutral", "negative"

class ArticleStatistics(BaseModel):
    """Statistics about the article collection"""
    total_articles: int
    positive_count: int
    neutral_count: int
    negative_count: int
    unclassified_count: int
    full_articles_count: int
    link_only_count: int
    regions: dict  # region code -> count
    sources: dict  # source domain -> count

# Tool definitions for function calling
ARTICLE_STATS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_article_statistics",
        "description": "Get statistics about the article collection including counts by sentiment (positive, neutral, negative), content type (full articles vs link-only), regions, and sources. Use this when the user asks about how many articles there are, sentiment distribution, or any numerical questions about the collection.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

def get_article_statistics() -> ArticleStatistics:
    """Get current statistics about the article collection"""
    try:
        results = collection.get()
        
        if not results['ids']:
            return ArticleStatistics(
                total_articles=0,
                positive_count=0,
                neutral_count=0,
                negative_count=0,
                unclassified_count=0,
                full_articles_count=0,
                link_only_count=0,
                regions={},
                sources={}
            )
        
        total = len(results['ids'])
        positive = 0
        neutral = 0
        negative = 0
        unclassified = 0
        full_articles = 0
        link_only = 0
        regions = {}
        sources = {}
        
        for metadata in results['metadatas']:
            # Sentiment counts
            sentiment = metadata.get('sentiment')
            if sentiment == 'positive':
                positive += 1
            elif sentiment == 'neutral':
                neutral += 1
            elif sentiment == 'negative':
                negative += 1
            else:
                unclassified += 1
            
            # Content type counts
            content_type = metadata.get('content_type', 'full')
            if content_type == 'link_only':
                link_only += 1
            else:
                full_articles += 1
            
            # Region counts
            region = metadata.get('region', 'unknown')
            regions[region] = regions.get(region, 0) + 1
            
            # Source counts
            source = metadata.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        return ArticleStatistics(
            total_articles=total,
            positive_count=positive,
            neutral_count=neutral,
            negative_count=negative,
            unclassified_count=unclassified,
            full_articles_count=full_articles,
            link_only_count=link_only,
            regions=regions,
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error getting article statistics: {e}")
        return ArticleStatistics(
            total_articles=0,
            positive_count=0,
            neutral_count=0,
            negative_count=0,
            unclassified_count=0,
            full_articles_count=0,
            link_only_count=0,
            regions={},
            sources={}
        )

# Helper Functions

async def filter_inappropriate_content(content: str, title: str = "") -> dict:
    """
    Use OpenAI to analyze content for inappropriate material
    Returns: {"is_safe": bool, "reason": str}
    """
    try:
        # Prepare content for analysis (truncate if too long)
        analysis_text = content[:3000] + ("..." if len(content) > 3000 else "")
        if title:
            analysis_text = f"Title: {title}\n\nContent: {analysis_text}"
        
        # Define the response schema for structured output
        CONTENT_FILTER_SCHEMA = {
            "type": "json_schema",
            "json_schema": {
                "name": "content_moderation_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_safe": {"type": "boolean"},
                        "reason": {"type": "string"}
                    },
                    "required": ["is_safe", "reason"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        
        # Use OpenAI to analyze content appropriateness
        prompt = f"""Analyze the following content for appropriateness in a professional work environment. 

Consider the content inappropriate if it contains:
- Explicit sexual content or imagery descriptions
- Adult/pornographic material
- Graphic violence or disturbing content
- Hate speech or discriminatory content
- Content clearly intended for adult entertainment
- Gambling or illegal activities promotion

Content to analyze:
{analysis_text}"""
        
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=CHAT_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a content moderation assistant. Analyze content for workplace appropriateness."},
                    {"role": "user", "content": prompt}
                ],
                response_format=CONTENT_FILTER_SCHEMA,
                max_tokens=100,
                temperature=0.1
            ),
            timeout=15.0
        )
        
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        
        return result
        
    except Exception as e:
        logger.error(f"Content filtering error: {e}")
        # Default to safe if filtering fails to avoid blocking legitimate content
        return {"is_safe": True, "reason": f"Filter error: {str(e)}"}

async def classify_sentiment(content: str, title: str = "") -> str:
    """
    Classify article sentiment as positive, neutral, or negative.
    Returns: "positive", "neutral", or "negative"
    """
    try:
        # Prepare content for analysis (truncate if too long)
        analysis_text = content[:2000] + ("..." if len(content) > 2000 else "")
        if title:
            analysis_text = f"Title: {title}\n\nContent: {analysis_text}"
        
        # Define the response schema for structured output
        SENTIMENT_SCHEMA = {
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "neutral", "negative"],
                            "description": "The overall sentiment of the article"
                        }
                    },
                    "required": ["sentiment"],
                    "additionalProperties": False
                }
            }
        }
        
        prompt = f"""Analyze the sentiment of this news article. Classify it as:
- "positive": Good news, achievements, breakthroughs, solutions, optimistic outlook
- "neutral": Factual reporting, balanced coverage, informational content
- "negative": Bad news, problems, failures, warnings, pessimistic outlook

Article:
{analysis_text}"""
        
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=CHAT_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis assistant. Classify news articles by their overall tone and sentiment."},
                    {"role": "user", "content": prompt}
                ],
                response_format=SENTIMENT_SCHEMA,
                max_tokens=50,
                temperature=0.1
            ),
            timeout=15.0
        )
        
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        
        return result.get("sentiment", "neutral")
        
    except Exception as e:
        logger.error(f"Sentiment classification error: {e}")
        # Default to neutral if classification fails
        return "neutral"

async def get_embedding(text: str) -> List[float]:
    """Generate embedding using Azure OpenAI"""
    try:
        response = await asyncio.wait_for(
            openai_client.embeddings.create(
                model=EMBEDDING_DEPLOYMENT_NAME,
                input=text
            ),
            timeout=30.0  # 30 second timeout
        )
        return response.data[0].embedding
    except asyncio.TimeoutError:
        logger.error("Embedding generation timed out")
        raise HTTPException(status_code=500, detail="Embedding generation timed out")
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")

async def generate_summary(text: str) -> str:
    """Generate summary using Azure OpenAI"""
    try:
        # Add timeout and better error handling
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=CHAT_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries of articles. Summarize the following article in 3-5 sentences, focusing on the main points and key insights."},
                    {"role": "user", "content": f"Article text: {text[:4000]}"}  # Limit text length
                ],
                max_tokens=200,
                temperature=0.3
            ),
            timeout=30.0  # 30 second timeout
        )
        return response.choices[0].message.content.strip()
    except asyncio.TimeoutError:
        logger.error("Summary generation timed out")
        return "Summary generation timed out."
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "Summary generation failed."

def can_fetch(url: str) -> bool:
    """Check if URL can be fetched according to robots.txt"""
    try:
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        
        return rp.can_fetch("*", url)
    except:
        return True  # If robots.txt can't be checked, assume it's okay

async def scrape_article(session: aiohttp.ClientSession, url: str) -> Optional[dict]:
    """Scrape article content from URL"""
    try:
        if not can_fetch(url):
            logger.info(f"Robots.txt disallows fetching {url}")
            return None
            
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
            if response.status != 200:
                return None
                
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "No title"
            
            # Extract main content (simple approach)
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Try to find main content areas
            content_selectors = [
                'article', 'main', '.content', '.post-content', 
                '.entry-content', '.article-body', 'div[role="main"]'
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text()
                    break
            
            if not content:
                # Fallback to body text
                body = soup.find('body')
                content = body.get_text() if body else ""
            
            # Clean up content
            content = ' '.join(content.split())
            
            # Check if content seems to be behind paywall
            paywall_indicators = [
                "subscribe to read", "premium content", "sign up to continue",
                "login to view", "paywall", "subscription required"
            ]
            
            is_public = not any(indicator in content.lower() for indicator in paywall_indicators)
            
            # Extract publication date (basic attempt)
            date_published = None
            date_selectors = ['time[datetime]', '.date', '.published', 'meta[property="article:published_time"]']
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    date_published = date_elem.get('datetime') or date_elem.get('content') or date_elem.get_text()
                    break
            
            return {
                'title': title,
                'content': content[:5000],  # Limit content length
                'url': url,
                'date_published': date_published,
                'public': is_public,
                'source': urlparse(url).netloc
            }
            
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return None

async def search_web(keywords: List[str], max_results: int = 10, region: str = "uk-en") -> List[dict]:
    """Search web using DuckDuckGo"""
    try:
        # Construct search query
        query = " ".join(keywords)
        
        # Perform search using ddgs
        # Use asyncio.to_thread to run the synchronous ddgs.text() in an async context
        results = await asyncio.to_thread(
            ddgs_client.text,
            query,
            region=region,
            safesearch="moderate",
            timelimit="m",  # Recent results from last month
            max_results=max_results,
            backend="auto"
        )
        
        # Extract search result data including titles
        search_results = []
        for result in results:
            if 'href' in result:
                search_results.append({
                    'url': result['href'],
                    'title': result.get('title', 'Untitled'),
                    'body': result.get('body', ''),
                })
        
        logger.info(f"DuckDuckGo search returned {len(search_results)} results for query: {query}")
        return search_results[:max_results]
        
    except Exception as e:
        logger.error(f"Error in DuckDuckGo search: {e}")
        return []

# API Endpoints
@app.post("/crawl")
async def crawl_articles(request: CrawlRequest):
    """Crawl web for articles based on keywords"""
    try:
        # Search for articles
        search_results = await search_web(request.keywords, request.max_articles, request.region)
        
        articles_added = 0
        articles_filtered = 0
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for search_result in search_results:
                url = search_result['url']
                search_title = search_result['title']
                search_body = search_result['body']
                try:
                    # Check if article already exists
                    try:
                        existing = collection.get(where={"url": url})
                        if existing['ids']:
                            logger.info(f"Article already exists, skipping: {url}")
                            continue
                    except Exception as e:
                        logger.warning(f"Error checking existing article for {url}: {e}")
                    
                    # Try to scrape article content
                    article_data = await scrape_article(session, url)
                    
                    if article_data:
                        # Full article scraped successfully - now filter for inappropriate content
                        try:
                            filter_result = await filter_inappropriate_content(
                                article_data['content'], 
                                article_data['title']
                            )
                            
                            if not filter_result['is_safe']:
                                logger.warning(f"Content filtered out: {url} - {filter_result['reason']}")
                                articles_filtered += 1
                                continue  # Skip this article entirely
                            
                        except Exception as e:
                            logger.error(f"Content filtering error for {url}: {e}")
                            # Continue with article if filter fails to avoid blocking legitimate content
                        
                        try:
                            summary = await generate_summary(article_data['content'])
                            if not summary or summary.startswith("Summary generation"):
                                logger.warning(f"Summary generation failed for {url}, using truncated content")
                                summary = article_data['content'][:500] + "..."
                        except Exception as e:
                            logger.error(f"Summary generation error for {url}: {e}")
                            summary = article_data['content'][:500] + "..."
                        
                        # Classify sentiment
                        try:
                            sentiment = await classify_sentiment(article_data['content'], article_data['title'])
                        except Exception as e:
                            logger.error(f"Sentiment classification error for {url}: {e}")
                            sentiment = "neutral"
                        
                        # Generate embedding
                        try:
                            embedding = await get_embedding(summary)
                        except Exception as e:
                            logger.error(f"Embedding generation failed for {url}: {e}")
                            continue
                        
                        # Store full article with content
                        try:
                            article_id = str(uuid.uuid4())
                            collection.add(
                                ids=[article_id],
                                embeddings=[embedding],
                                documents=[summary],
                                metadatas=[{
                                    "title": article_data['title'] or "Untitled",
                                    "url": article_data['url'],
                                    "date_published": article_data['date_published'] or "",
                                    "date_added": datetime.now().isoformat(),
                                    "public": bool(article_data['public']),
                                    "source": article_data['source'] or "Unknown",
                                    "content_type": "full",
                                    "region": request.region,
                                    "sentiment": sentiment
                                }]
                            )
                            
                            articles_added += 1
                            logger.info(f"Added full article: {article_data['title']}")
                        except Exception as e:
                            logger.error(f"ChromaDB storage failed for {url}: {e}")
                            continue
                    else:
                        # Scraping failed, store minimal article with search result data
                        logger.info(f"Scraping failed for {url}, storing as link-only article: {search_title}")
                        
                        # Filter search result content for appropriateness
                        search_content_to_check = f"{search_title}\n{search_body}" if search_body else search_title
                        try:
                            filter_result = await filter_inappropriate_content(search_content_to_check, search_title)
                            
                            if not filter_result['is_safe']:
                                logger.warning(f"Search result filtered out: {url} - {filter_result['reason']}")
                                articles_filtered += 1
                                continue  # Skip this search result entirely
                                
                        except Exception as e:
                            logger.error(f"Content filtering error for search result {url}: {e}")
                            # Continue with search result if filter fails
                        
                        # Use search result body as summary if available, otherwise use title
                        summary = search_body if search_body else f"External article: {search_title}"
                        
                        # Classify sentiment for link-only articles
                        try:
                            sentiment = await classify_sentiment(search_content_to_check, search_title)
                        except Exception as e:
                            logger.error(f"Sentiment classification error for search result {url}: {e}")
                            sentiment = "neutral"
                        
                        # Generate embedding for search result data
                        try:
                            embedding = await get_embedding(summary)
                        except Exception as e:
                            logger.error(f"Embedding generation failed for search result {url}: {e}")
                            continue
                        
                        # Store link-only article
                        try:
                            article_id = str(uuid.uuid4())
                            collection.add(
                                ids=[article_id],
                                embeddings=[embedding],
                                documents=[summary],
                                metadatas=[{
                                    "title": search_title,
                                    "url": url,
                                    "date_published": "",
                                    "date_added": datetime.now().isoformat(),
                                    "public": True,  # Default to public for link-only articles
                                    "source": urlparse(url).netloc,
                                    "content_type": "link_only",
                                    "region": request.region,
                                    "sentiment": sentiment
                                }]
                            )
                            
                            articles_added += 1
                            logger.info(f"Added link-only article: {search_title}")
                        except Exception as e:
                            logger.error(f"ChromaDB storage failed for link-only article {url}: {e}")
                            continue
                    
                except Exception as e:
                    logger.error(f"Unexpected error processing {url}: {e}")
                    continue
        
        return {
            "message": "Crawl completed", 
            "articles_added": articles_added,
            "articles_filtered": articles_filtered
        }
        
    except Exception as e:
        logger.error(f"Crawl error: {e}")
        raise HTTPException(status_code=500, detail="Crawl failed")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Answer question using RAG"""
    try:
        # Generate embedding for question
        question_embedding = await get_embedding(request.question)
        
        # Search for relevant articles
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=5
        )
        
        if not results['documents'] or not results['documents'][0]:
            return {"answer": "I don't have any relevant articles to answer your question. Please run a crawl first to gather some articles."}
        
        # Prepare context from retrieved articles
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            # Handle both full articles and link-only articles
            content_type = metadata.get('content_type', 'full')
            if content_type == 'link_only':
                article_info = f"Article {i+1}: {metadata.get('title', 'Untitled')}\nType: External link only\nSource: {metadata.get('url', 'Unknown')}\n"
            else:
                article_info = f"Article {i+1}: {metadata.get('title', 'Untitled')}\nSummary: {doc}\nSource: {metadata.get('url', 'Unknown')}\n"
            context_parts.append(article_info)
        
        context = "\n".join(context_parts)
        
        # Build conversation messages with context
        conversation_messages = [
            {
                "role": "system", 
                "content": f"You are an AI assistant that answers questions based on a collection of articles. Some articles have full summaries, while others are external links with titles only. Use only the information provided to answer questions. For external link articles, acknowledge that you only have the title and suggest the user click the link to read more. Always be helpful and cite which articles you're referencing.\n\nAvailable Articles:\n{context}"
            }
        ]
        
        # Add conversation history if provided
        if request.messages:
            for msg in request.messages:
                conversation_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add the current question
        conversation_messages.append({
            "role": "user", 
            "content": request.question
        })
        
        # Generate answer using Azure OpenAI
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=CHAT_DEPLOYMENT_NAME,
                messages=conversation_messages,
                max_tokens=500,
                temperature=0.3
            ),
            timeout=30.0
        )
        
        answer = response.choices[0].message.content.strip()
        
        return {"answer": answer}
        
    except asyncio.TimeoutError:
        logger.error("Question answering timed out")
        raise HTTPException(status_code=500, detail="Question answering timed out")
    except Exception as e:
        logger.error(f"Question answering error: {e}")
        raise HTTPException(status_code=500, detail="Failed to answer question")

@app.post("/ask/stream")
async def ask_question_stream(request: QuestionRequest):
    """Answer question using RAG with streaming response and tool support"""
    
    async def generate_stream():
        try:
            # Generate embedding for question
            question_embedding = await get_embedding(request.question)
            
            # Search for relevant articles
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=5
            )
            
            if not results['documents'] or not results['documents'][0]:
                no_articles_msg = "I don't have any relevant articles to answer your question. Please run a crawl first to gather some articles."
                yield f"data: {json.dumps({'content': no_articles_msg, 'done': True})}\n\n"
                return
            
            # Prepare context from retrieved articles
            context_parts = []
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                # Handle both full articles and link-only articles
                content_type = metadata.get('content_type', 'full')
                sentiment = metadata.get('sentiment', 'unclassified')
                if content_type == 'link_only':
                    article_info = f"Article {i+1}: {metadata.get('title', 'Untitled')}\nType: External link only\nSentiment: {sentiment}\nSource: {metadata.get('url', 'Unknown')}\n"
                else:
                    article_info = f"Article {i+1}: {metadata.get('title', 'Untitled')}\nSentiment: {sentiment}\nSummary: {doc}\nSource: {metadata.get('url', 'Unknown')}\n"
                context_parts.append(article_info)
            
            context = "\n".join(context_parts)
            
            # Build conversation messages with context
            system_prompt = f"""You are an AI assistant that answers questions based on a collection of articles. 

You have access to a tool called 'get_article_statistics' that provides accurate counts of articles by sentiment, content type, region, and source. ALWAYS use this tool when the user asks about:
- How many articles there are (total or by category)
- Sentiment distribution (positive, neutral, negative counts)
- Statistics or numbers about the collection
- Breakdown by region or source

Some articles have full summaries, while others are external links with titles only. Use only the information provided to answer questions. For external link articles, acknowledge that you only have the title and suggest the user click the link to read more. Always be helpful and cite which articles you're referencing.

Available Articles (sample for content questions):
{context}"""

            conversation_messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history if provided
            if request.messages:
                for msg in request.messages:
                    conversation_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Add the current question
            conversation_messages.append({
                "role": "user", 
                "content": request.question
            })
            
            # First, make a non-streaming call to check for tool use
            initial_response = await asyncio.wait_for(
                openai_client.chat.completions.create(
                    model=CHAT_DEPLOYMENT_NAME,
                    messages=conversation_messages,
                    tools=[ARTICLE_STATS_TOOL],
                    tool_choice="auto",
                    max_tokens=500,
                    temperature=0.3
                ),
                timeout=30.0
            )
            
            response_message = initial_response.choices[0].message
            
            # Check if the model wants to use a tool
            if response_message.tool_calls:
                # Process tool calls
                tool_results = []
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "get_article_statistics":
                        # Execute the tool
                        stats = get_article_statistics()
                        tool_result = {
                            "total_articles": stats.total_articles,
                            "sentiment_breakdown": {
                                "positive": stats.positive_count,
                                "neutral": stats.neutral_count,
                                "negative": stats.negative_count,
                                "unclassified": stats.unclassified_count
                            },
                            "content_type_breakdown": {
                                "full_articles": stats.full_articles_count,
                                "link_only": stats.link_only_count
                            },
                            "by_region": stats.regions,
                            "by_source": stats.sources
                        }
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": json.dumps(tool_result)
                        })
                
                # Add the assistant message with tool calls
                conversation_messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in response_message.tool_calls
                    ]
                })
                
                # Add tool results
                for tr in tool_results:
                    conversation_messages.append(tr)
                
                # Now stream the final response with tool results
                stream = await asyncio.wait_for(
                    openai_client.chat.completions.create(
                        model=CHAT_DEPLOYMENT_NAME,
                        messages=conversation_messages,
                        max_tokens=500,
                        temperature=0.3,
                        stream=True
                    ),
                    timeout=30.0
                )
                
                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        yield f"data: {json.dumps({'content': content, 'done': False})}\n\n"
            else:
                # No tool call, just stream the response content
                if response_message.content:
                    # Since we already have the full response, send it
                    yield f"data: {json.dumps({'content': response_message.content, 'done': False})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
            
        except asyncio.TimeoutError:
            logger.error("Streaming response timed out")
            yield f"data: {json.dumps({'content': 'Error: Response generation timed out.', 'done': True})}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'content': 'Error: Failed to generate response.', 'done': True})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.get("/articles", response_model=List[Article])
async def get_articles():
    """Get all stored articles"""
    try:
        results = collection.get()
        
        articles = []
        for i, (doc_id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
            articles.append(Article(
                id=doc_id,
                title=metadata['title'],
                url=metadata['url'],
                summary=results['documents'][i],
                date_published=metadata.get('date_published'),
                date_added=metadata['date_added'],
                public=metadata['public'],
                source=metadata['source'],
                content_type=metadata.get('content_type', 'full'),
                region=metadata.get('region'),
                sentiment=metadata.get('sentiment')
            ))
        
        # Sort by date_added (newest first)
        articles.sort(key=lambda x: x.date_added, reverse=True)
        
        return articles
        
    except Exception as e:
        logger.error(f"Error fetching articles: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch articles")

@app.delete("/articles/{article_id}")
async def delete_article(article_id: str):
    """Delete an article"""
    try:
        collection.delete(ids=[article_id])
        return {"message": "Article deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting article: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete article")

@app.post("/articles/classify-sentiment")
async def classify_unclassified_articles():
    """Classify sentiment for all articles that don't have sentiment yet"""
    try:
        results = collection.get()
        
        classified_count = 0
        already_classified = 0
        errors = 0
        
        for i, (doc_id, metadata, document) in enumerate(zip(results['ids'], results['metadatas'], results['documents'])):
            # Check if already has sentiment
            if metadata.get('sentiment'):
                already_classified += 1
                continue
            
            # Get content for classification
            title = metadata.get('title', '')
            content = document or title  # Use document (summary) or title
            
            try:
                # Classify sentiment
                sentiment = await classify_sentiment(content, title)
                
                # Update the article metadata
                # ChromaDB requires updating by deleting and re-adding
                embedding = results['embeddings'][i] if results.get('embeddings') else None
                
                # Get the embedding if not included
                if embedding is None:
                    embedding = await get_embedding(document)
                
                # Update metadata with sentiment
                new_metadata = {**metadata, "sentiment": sentiment}
                
                # Delete and re-add with new metadata
                collection.delete(ids=[doc_id])
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[document],
                    metadatas=[new_metadata]
                )
                
                classified_count += 1
                logger.info(f"Classified article '{title[:50]}...' as {sentiment}")
                
            except Exception as e:
                logger.error(f"Error classifying article {doc_id}: {e}")
                errors += 1
                continue
        
        return {
            "message": "Sentiment classification completed",
            "classified": classified_count,
            "already_classified": already_classified,
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment classification: {e}")
        raise HTTPException(status_code=500, detail="Failed to classify articles")

@app.get("/articles/stats", response_model=ArticleStatistics)
async def get_stats():
    """Get article collection statistics"""
    return get_article_statistics()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
