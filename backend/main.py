from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from openai import AsyncAzureOpenAI
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import urllib.robotparser
from urllib.parse import urlparse
import os
from datetime import datetime
import uuid
import logging
import json
from pathlib import Path
from dotenv import dotenv_values
from ddgs import DDGS
import asyncpg
from pgvector.asyncpg import register_vector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this file lives (backend/)
BACKEND_DIR = Path(__file__).parent

# Load configuration
with open(BACKEND_DIR / "config.json", "r") as f:
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from .env file (for local development)
env_path = BACKEND_DIR / ".env"
env_vars = dotenv_values(env_path) if env_path.exists() else {}

# Get POSTGRES_URL from environment or .env
POSTGRES_URL = os.environ.get("POSTGRES_URL") or env_vars.get("POSTGRES_URL")

# Initialize Azure OpenAI client
openai_client = AsyncAzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_KEY") or env_vars.get("AZURE_OPENAI_KEY"),
    api_version=API_VERSION,
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT") or env_vars.get("AZURE_OPENAI_ENDPOINT")
)

# Initialize DDGS client
ddgs_client = DDGS()

# Database connection pool
db_pool: Optional[asyncpg.Pool] = None

async def init_db():
    """Initialize database connection pool"""
    global db_pool
    if db_pool is None and POSTGRES_URL:
        db_pool = await asyncpg.create_pool(
            POSTGRES_URL,
            min_size=1,
            max_size=10,
            init=register_vector
        )
        logger.info("Database connection pool initialized")

async def get_db():
    """Get database connection from pool"""
    if db_pool is None:
        await init_db()
    return db_pool

@app.on_event("startup")
async def startup():
    await init_db()

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()

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
    sentiment: Optional[str] = None

class ArticleStatistics(BaseModel):
    """Statistics about the article collection"""
    total_articles: int
    positive_count: int
    neutral_count: int
    negative_count: int
    unclassified_count: int
    full_articles_count: int
    link_only_count: int
    regions: dict
    sources: dict

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

async def get_article_statistics() -> ArticleStatistics:
    """Get current statistics about the article collection"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            # Get total count
            total = await conn.fetchval("SELECT COUNT(*) FROM articles")
            
            if total == 0:
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
            
            # Get sentiment counts
            sentiment_counts = await conn.fetch("""
                SELECT sentiment, COUNT(*) as count 
                FROM articles 
                GROUP BY sentiment
            """)
            
            positive = 0
            neutral = 0
            negative = 0
            unclassified = 0
            for row in sentiment_counts:
                if row['sentiment'] == 'positive':
                    positive = row['count']
                elif row['sentiment'] == 'neutral':
                    neutral = row['count']
                elif row['sentiment'] == 'negative':
                    negative = row['count']
                else:
                    unclassified = row['count']
            
            # Get content type counts
            content_counts = await conn.fetch("""
                SELECT content_type, COUNT(*) as count 
                FROM articles 
                GROUP BY content_type
            """)
            
            full_articles = 0
            link_only = 0
            for row in content_counts:
                if row['content_type'] == 'link_only':
                    link_only = row['count']
                else:
                    full_articles = row['count']
            
            # Get region counts
            region_rows = await conn.fetch("""
                SELECT region, COUNT(*) as count 
                FROM articles 
                WHERE region IS NOT NULL 
                GROUP BY region
            """)
            regions = {row['region']: row['count'] for row in region_rows}
            
            # Get source counts
            source_rows = await conn.fetch("""
                SELECT source, COUNT(*) as count 
                FROM articles 
                WHERE source IS NOT NULL 
                GROUP BY source
            """)
            sources = {row['source']: row['count'] for row in source_rows}
            
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
    """Use OpenAI to analyze content for inappropriate material"""
    try:
        analysis_text = content[:3000] + ("..." if len(content) > 3000 else "")
        if title:
            analysis_text = f"Title: {title}\n\nContent: {analysis_text}"
        
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
        return json.loads(result_text)
        
    except Exception as e:
        logger.error(f"Content filtering error: {e}")
        return {"is_safe": True, "reason": f"Filter error: {str(e)}"}

async def classify_sentiment(content: str, title: str = "") -> str:
    """Classify article sentiment as positive, neutral, or negative."""
    try:
        analysis_text = content[:2000] + ("..." if len(content) > 2000 else "")
        if title:
            analysis_text = f"Title: {title}\n\nContent: {analysis_text}"
        
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
        return "neutral"

async def get_embedding(text: str) -> List[float]:
    """Generate embedding using Azure OpenAI"""
    try:
        response = await asyncio.wait_for(
            openai_client.embeddings.create(
                model=EMBEDDING_DEPLOYMENT_NAME,
                input=text
            ),
            timeout=30.0
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
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=CHAT_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries of articles. Summarize the following article in 3-5 sentences, focusing on the main points and key insights."},
                    {"role": "user", "content": f"Article text: {text[:4000]}"}
                ],
                max_tokens=200,
                temperature=0.3
            ),
            timeout=30.0
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
        return True

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
            
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "No title"
            
            for script in soup(["script", "style"]):
                script.decompose()
                
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
                body = soup.find('body')
                content = body.get_text() if body else ""
            
            content = ' '.join(content.split())
            
            paywall_indicators = [
                "subscribe to read", "premium content", "sign up to continue",
                "login to view", "paywall", "subscription required"
            ]
            
            is_public = not any(indicator in content.lower() for indicator in paywall_indicators)
            
            date_published = None
            date_selectors = ['time[datetime]', '.date', '.published', 'meta[property="article:published_time"]']
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    date_published = date_elem.get('datetime') or date_elem.get('content') or date_elem.get_text()
                    break
            
            return {
                'title': title,
                'content': content[:5000],
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
        query = " ".join(keywords)
        
        results = await asyncio.to_thread(
            ddgs_client.text,
            query,
            region=region,
            safesearch="moderate",
            timelimit="m",
            max_results=max_results,
            backend="auto"
        )
        
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

async def article_exists(url: str) -> bool:
    """Check if an article with this URL already exists"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM articles WHERE url = $1)",
                url
            )
            return result
    except Exception as e:
        logger.error(f"Error checking if article exists: {e}")
        return False

async def store_article(
    title: str,
    url: str,
    summary: str,
    embedding: List[float],
    date_published: Optional[str],
    is_public: bool,
    source: str,
    content_type: str,
    region: str,
    sentiment: str
) -> str:
    """Store an article in the database"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            article_id = await conn.fetchval(
                """
                INSERT INTO articles (title, url, summary, embedding, date_published, is_public, source, content_type, region, sentiment)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
                """,
                title, url, summary, embedding, date_published, is_public, source, content_type, region, sentiment
            )
            return str(article_id)
    except Exception as e:
        logger.error(f"Error storing article: {e}")
        raise

async def search_similar_articles(query_embedding: List[float], limit: int = 5) -> List[dict]:
    """Search for similar articles using vector similarity"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, url, summary, date_published, date_added, is_public, source, content_type, region, sentiment,
                       1 - (embedding <=> $1) as similarity
                FROM articles
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                query_embedding, limit
            )
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error searching similar articles: {e}")
        return []

# API Endpoints
@app.post("/crawl")
async def crawl_articles(request: CrawlRequest):
    """Crawl web for articles based on keywords"""
    try:
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
                    if await article_exists(url):
                        logger.info(f"Article already exists, skipping: {url}")
                        continue
                    
                    # Try to scrape article content
                    article_data = await scrape_article(session, url)
                    
                    if article_data:
                        # Full article scraped successfully
                        try:
                            filter_result = await filter_inappropriate_content(
                                article_data['content'], 
                                article_data['title']
                            )
                            
                            if not filter_result['is_safe']:
                                logger.warning(f"Content filtered out: {url} - {filter_result['reason']}")
                                articles_filtered += 1
                                continue
                        except Exception as e:
                            logger.error(f"Content filtering error for {url}: {e}")
                        
                        try:
                            summary = await generate_summary(article_data['content'])
                            if not summary or summary.startswith("Summary generation"):
                                summary = article_data['content'][:500] + "..."
                        except Exception as e:
                            logger.error(f"Summary generation error for {url}: {e}")
                            summary = article_data['content'][:500] + "..."
                        
                        try:
                            sentiment = await classify_sentiment(article_data['content'], article_data['title'])
                        except Exception as e:
                            logger.error(f"Sentiment classification error for {url}: {e}")
                            sentiment = "neutral"
                        
                        try:
                            embedding = await get_embedding(summary)
                        except Exception as e:
                            logger.error(f"Embedding generation failed for {url}: {e}")
                            continue
                        
                        try:
                            await store_article(
                                title=article_data['title'] or "Untitled",
                                url=article_data['url'],
                                summary=summary,
                                embedding=embedding,
                                date_published=article_data['date_published'],
                                is_public=bool(article_data['public']),
                                source=article_data['source'] or "Unknown",
                                content_type="full",
                                region=request.region,
                                sentiment=sentiment
                            )
                            articles_added += 1
                            logger.info(f"Added full article: {article_data['title']}")
                        except Exception as e:
                            logger.error(f"Database storage failed for {url}: {e}")
                            continue
                    else:
                        # Scraping failed, store link-only article
                        logger.info(f"Scraping failed for {url}, storing as link-only: {search_title}")
                        
                        search_content = f"{search_title}\n{search_body}" if search_body else search_title
                        
                        try:
                            filter_result = await filter_inappropriate_content(search_content, search_title)
                            if not filter_result['is_safe']:
                                logger.warning(f"Search result filtered out: {url} - {filter_result['reason']}")
                                articles_filtered += 1
                                continue
                        except Exception as e:
                            logger.error(f"Content filtering error for search result {url}: {e}")
                        
                        summary = search_body if search_body else f"External article: {search_title}"
                        
                        try:
                            sentiment = await classify_sentiment(search_content, search_title)
                        except Exception as e:
                            logger.error(f"Sentiment classification error for search result {url}: {e}")
                            sentiment = "neutral"
                        
                        try:
                            embedding = await get_embedding(summary)
                        except Exception as e:
                            logger.error(f"Embedding generation failed for search result {url}: {e}")
                            continue
                        
                        try:
                            await store_article(
                                title=search_title,
                                url=url,
                                summary=summary,
                                embedding=embedding,
                                date_published=None,
                                is_public=True,
                                source=urlparse(url).netloc,
                                content_type="link_only",
                                region=request.region,
                                sentiment=sentiment
                            )
                            articles_added += 1
                            logger.info(f"Added link-only article: {search_title}")
                        except Exception as e:
                            logger.error(f"Database storage failed for link-only article {url}: {e}")
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
        question_embedding = await get_embedding(request.question)
        results = await search_similar_articles(question_embedding, limit=5)
        
        if not results:
            return {"answer": "I don't have any relevant articles to answer your question. Please run a crawl first to gather some articles."}
        
        context_parts = []
        for i, row in enumerate(results):
            content_type = row.get('content_type', 'full')
            if content_type == 'link_only':
                article_info = f"Article {i+1}: {row['title']}\nType: External link only\nSource: {row['url']}\n"
            else:
                article_info = f"Article {i+1}: {row['title']}\nSummary: {row['summary']}\nSource: {row['url']}\n"
            context_parts.append(article_info)
        
        context = "\n".join(context_parts)
        
        conversation_messages = [
            {
                "role": "system", 
                "content": f"You are an AI assistant that answers questions based on a collection of articles. Use only the information provided to answer questions. Always be helpful and cite which articles you're referencing.\n\nAvailable Articles:\n{context}"
            }
        ]
        
        if request.messages:
            for msg in request.messages:
                conversation_messages.append({"role": msg.role, "content": msg.content})
        
        conversation_messages.append({"role": "user", "content": request.question})
        
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=CHAT_DEPLOYMENT_NAME,
                messages=conversation_messages,
                max_tokens=500,
                temperature=0.3
            ),
            timeout=30.0
        )
        
        return {"answer": response.choices[0].message.content.strip()}
        
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
            question_embedding = await get_embedding(request.question)
            results = await search_similar_articles(question_embedding, limit=5)
            
            if not results:
                no_articles_msg = "I don't have any relevant articles to answer your question. Please run a crawl first to gather some articles."
                yield f"data: {json.dumps({'content': no_articles_msg, 'done': True})}\n\n"
                return
            
            context_parts = []
            for i, row in enumerate(results):
                content_type = row.get('content_type', 'full')
                sentiment = row.get('sentiment', 'unclassified')
                if content_type == 'link_only':
                    article_info = f"Article {i+1}: {row['title']}\nType: External link only\nSentiment: {sentiment}\nSource: {row['url']}\n"
                else:
                    article_info = f"Article {i+1}: {row['title']}\nSentiment: {sentiment}\nSummary: {row['summary']}\nSource: {row['url']}\n"
                context_parts.append(article_info)
            
            context = "\n".join(context_parts)
            
            system_prompt = f"""You are an AI assistant that answers questions based on a collection of articles. 

You have access to a tool called 'get_article_statistics' that provides accurate counts of articles by sentiment, content type, region, and source. ALWAYS use this tool when the user asks about:
- How many articles there are (total or by category)
- Sentiment distribution (positive, neutral, negative counts)
- Statistics or numbers about the collection
- Breakdown by region or source

Available Articles (sample for content questions):
{context}"""

            conversation_messages = [{"role": "system", "content": system_prompt}]
            
            if request.messages:
                for msg in request.messages:
                    conversation_messages.append({"role": msg.role, "content": msg.content})
            
            conversation_messages.append({"role": "user", "content": request.question})
            
            # Check for tool use
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
            
            if response_message.tool_calls:
                tool_results = []
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "get_article_statistics":
                        stats = await get_article_statistics()
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
                
                for tr in tool_results:
                    conversation_messages.append(tr)
                
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
                if response_message.content:
                    yield f"data: {json.dumps({'content': response_message.content, 'done': False})}\n\n"
            
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
        pool = await get_db()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, url, summary, date_published, date_added, is_public, source, content_type, region, sentiment
                FROM articles
                ORDER BY date_added DESC
                """
            )
            
            articles = []
            for row in rows:
                articles.append(Article(
                    id=str(row['id']),
                    title=row['title'],
                    url=row['url'],
                    summary=row['summary'],
                    date_published=row['date_published'],
                    date_added=row['date_added'].isoformat() if row['date_added'] else "",
                    public=row['is_public'],
                    source=row['source'],
                    content_type=row['content_type'],
                    region=row['region'],
                    sentiment=row['sentiment']
                ))
            
            return articles
        
    except Exception as e:
        logger.error(f"Error fetching articles: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch articles")

@app.delete("/articles/{article_id}")
async def delete_article(article_id: str):
    """Delete an article"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM articles WHERE id = $1", uuid.UUID(article_id))
        return {"message": "Article deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting article: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete article")

@app.post("/articles/classify-sentiment")
async def classify_unclassified_articles():
    """Classify sentiment for all articles that don't have sentiment yet"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, title, summary FROM articles WHERE sentiment IS NULL"
            )
            
            classified_count = 0
            errors = 0
            
            for row in rows:
                try:
                    sentiment = await classify_sentiment(row['summary'] or row['title'], row['title'])
                    await conn.execute(
                        "UPDATE articles SET sentiment = $1 WHERE id = $2",
                        sentiment, row['id']
                    )
                    classified_count += 1
                    logger.info(f"Classified article '{row['title'][:50]}...' as {sentiment}")
                except Exception as e:
                    logger.error(f"Error classifying article {row['id']}: {e}")
                    errors += 1
            
            already_classified = await conn.fetchval(
                "SELECT COUNT(*) FROM articles WHERE sentiment IS NOT NULL"
            ) - classified_count
            
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
    return await get_article_statistics()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
