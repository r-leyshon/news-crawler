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

# Helper Functions
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

async def search_web(keywords: List[str], max_results: int = 10) -> List[dict]:
    """Search web using DuckDuckGo"""
    try:
        # Construct search query
        query = " ".join(keywords)
        
        # Perform search using ddgs
        # Use asyncio.to_thread to run the synchronous ddgs.text() in an async context
        results = await asyncio.to_thread(
            ddgs_client.text,
            query,
            region="uk-en",
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
        search_results = await search_web(request.keywords, request.max_articles)
        
        articles_added = 0
        
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
                        # Full article scraped successfully
                        try:
                            summary = await generate_summary(article_data['content'])
                            if not summary or summary.startswith("Summary generation"):
                                logger.warning(f"Summary generation failed for {url}, using truncated content")
                                summary = article_data['content'][:500] + "..."
                        except Exception as e:
                            logger.error(f"Summary generation error for {url}: {e}")
                            summary = article_data['content'][:500] + "..."
                        
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
                                    "content_type": "full"
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
                        
                        # Use search result body as summary if available, otherwise use title
                        summary = search_body if search_body else f"External article: {search_title}"
                        
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
                                    "content_type": "link_only"
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
        
        return {"message": "Crawl completed", "articles_added": articles_added}
        
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
    """Answer question using RAG with streaming response"""
    
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
            
            # Generate streaming answer using Azure OpenAI
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
                content_type=metadata.get('content_type', 'full')
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
