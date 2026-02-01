from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
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

# Vertex AI imports
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from vertexai.language_models import TextEmbeddingModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this file lives (backend/)
BACKEND_DIR = Path(__file__).parent

# Load configuration
with open(BACKEND_DIR / "config.json", "r") as f:
    config = json.load(f)

# Extract Vertex AI config
vertex_config = config["vertex_ai"]
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or vertex_config.get("project_id")
LOCATION = vertex_config.get("location", "europe-west2")
CHAT_MODEL_NAME = vertex_config.get("chat_model", "gemini-2.5-flash")
EMBEDDING_MODEL_NAME = vertex_config.get("embedding_model", "text-embedding-005")

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

# Initialize Vertex AI
# For service account auth, set GOOGLE_APPLICATION_CREDENTIALS env var to path of JSON key file
# Or the credentials will be auto-detected in GCP environments
def init_vertex_ai():
    """Initialize Vertex AI with project and location"""
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or env_vars.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    logger.info(f"Vertex AI initialized with project={PROJECT_ID}, location={LOCATION}")

init_vertex_ai()

# Initialize Gemini model for chat
chat_model = GenerativeModel(CHAT_MODEL_NAME)

# Initialize embedding model
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

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
    """Use Gemini to analyze content for inappropriate material"""
    try:
        analysis_text = content[:3000] + ("..." if len(content) > 3000 else "")
        if title:
            analysis_text = f"Title: {title}\n\nContent: {analysis_text}"
        
        prompt = f"""You are a content moderation assistant. Analyze content for workplace appropriateness.

Analyze the following content for appropriateness in a professional work environment. 

Consider the content inappropriate if it contains:
- Explicit sexual content or imagery descriptions
- Adult/pornographic material
- Graphic violence or disturbing content
- Hate speech or discriminatory content
- Content clearly intended for adult entertainment
- Gambling or illegal activities promotion

Content to analyze:
{analysis_text}

Respond with a JSON object containing:
- "is_safe": true or false
- "reason": brief explanation

Return ONLY the JSON object, no other text."""
        
        generation_config = GenerationConfig(
            temperature=0.1,
        )
        
        response = await asyncio.wait_for(
            asyncio.to_thread(
                chat_model.generate_content,
                prompt,
                generation_config=generation_config
            ),
            timeout=15.0
        )
        
        result_text = response.text.strip()
        logger.info(f"Content filter raw response: '{result_text[:100]}...'")
        
        # Clean up potential markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()
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
        
        prompt = f"""Analyze the sentiment of this news article. Respond with exactly one word: positive, neutral, or negative.

- positive: Good news, achievements, breakthroughs, solutions, optimistic outlook
- neutral: Factual reporting, balanced coverage, informational content  
- negative: Bad news, problems, failures, warnings, pessimistic outlook

Article:
{analysis_text}

Sentiment (one word only):"""
        
        generation_config = GenerationConfig(
            temperature=0.1,
        )
        
        response = await asyncio.wait_for(
            asyncio.to_thread(
                chat_model.generate_content,
                prompt,
                generation_config=generation_config
            ),
            timeout=15.0
        )
        
        result_text = response.text.strip().lower()
        logger.info(f"Sentiment raw response: '{result_text}'")
        
        # Extract sentiment from response
        if "positive" in result_text:
            return "positive"
        elif "negative" in result_text:
            return "negative"
        elif "neutral" in result_text:
            return "neutral"
        else:
            logger.warning(f"Unexpected sentiment response: '{result_text}', defaulting to neutral")
            return "neutral"
        
    except Exception as e:
        logger.error(f"Sentiment classification error: {e}")
        return "neutral"

async def get_embedding(text: str) -> List[float]:
    """Generate embedding using Vertex AI"""
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                embedding_model.get_embeddings,
                [text]
            ),
            timeout=30.0
        )
        return response[0].values
    except asyncio.TimeoutError:
        logger.error("Embedding generation timed out")
        raise HTTPException(status_code=500, detail="Embedding generation timed out")
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")

async def generate_summary(text: str) -> str:
    """Generate summary using Vertex AI Gemini"""
    try:
        prompt = f"""You are a helpful assistant that creates concise summaries of articles. 
Summarize the following article in 3-5 sentences, focusing on the main points and key insights.

Article text: {text[:4000]}"""
        
        generation_config = GenerationConfig(
            temperature=0.3,
        )
        
        response = await asyncio.wait_for(
            asyncio.to_thread(
                chat_model.generate_content,
                prompt,
                generation_config=generation_config
            ),
            timeout=30.0
        )
        return response.text.strip()
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
        
        # Build conversation history for Gemini
        history_text = ""
        if request.messages:
            for msg in request.messages:
                role_label = "User" if msg.role == "user" else "Assistant"
                history_text += f"{role_label}: {msg.content}\n\n"
        
        prompt = f"""You are an AI assistant that answers questions based on a collection of articles. Use only the information provided to answer questions. Always be helpful and cite which articles you're referencing.

Available Articles:
{context}

{history_text}User: {request.question}

Please provide a helpful answer based on the articles above."""
        
        generation_config = GenerationConfig(
            temperature=0.3,
        )
        
        response = await asyncio.wait_for(
            asyncio.to_thread(
                chat_model.generate_content,
                prompt,
                generation_config=generation_config
            ),
            timeout=30.0
        )
        
        return {"answer": response.text.strip()}
        
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
            
            # Check if the question is about statistics
            stats_keywords = ["how many", "count", "total", "statistics", "breakdown", "distribution", "sentiment"]
            question_lower = request.question.lower()
            needs_stats = any(keyword in question_lower for keyword in stats_keywords)
            
            stats_context = ""
            if needs_stats:
                stats = await get_article_statistics()
                stats_context = f"""

Article Statistics:
- Total articles: {stats.total_articles}
- Sentiment breakdown:
  - Positive: {stats.positive_count}
  - Neutral: {stats.neutral_count}
  - Negative: {stats.negative_count}
  - Unclassified: {stats.unclassified_count}
- Content types:
  - Full articles: {stats.full_articles_count}
  - Link-only: {stats.link_only_count}
- By region: {json.dumps(stats.regions)}
- By source: {json.dumps(stats.sources)}
"""
            
            # Build conversation history for Gemini
            history_text = ""
            if request.messages:
                for msg in request.messages:
                    role_label = "User" if msg.role == "user" else "Assistant"
                    history_text += f"{role_label}: {msg.content}\n\n"
            
            prompt = f"""You are an AI assistant that answers questions based on a collection of articles. Use only the information provided to answer questions. Always be helpful and cite which articles you're referencing.

Available Articles (sample for content questions):
{context}
{stats_context}
{history_text}User: {request.question}

Please provide a helpful answer based on the information above."""
            
            generation_config = GenerationConfig(
                max_output_tokens=1024,  # Gemini 2.5 uses ~200-300 thinking tokens
                temperature=0.3,
            )
            
            # Gemini doesn't support async streaming directly, so we generate and stream chunks
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    chat_model.generate_content,
                    prompt,
                    generation_config=generation_config
                ),
                timeout=30.0
            )
            
            # Stream the response in chunks for a better UX
            full_text = response.text
            chunk_size = 50  # Characters per chunk
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i + chunk_size]
                yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
                await asyncio.sleep(0.02)  # Small delay for streaming effect
            
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

# Vercel serverless function handler
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    # Fallback for local development
    handler = app
