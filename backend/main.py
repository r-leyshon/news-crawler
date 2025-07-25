from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from openai import AzureOpenAI
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
openai_client = AzureOpenAI(
    api_key=env_vars.get("AZURE_OPENAI_KEY"),
    api_version=API_VERSION,
    azure_endpoint=env_vars.get("AZURE_OPENAI_ENDPOINT")
)

# ChromaDB Configuration
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="articles",
    metadata={"hnsw:space": "cosine"}
)

# Pydantic Models
class CrawlRequest(BaseModel):
    keywords: List[str]
    max_articles: int = 10

class QuestionRequest(BaseModel):
    question: str

class Article(BaseModel):
    id: str
    title: str
    url: str
    summary: str
    date_published: Optional[str] = None
    date_added: str
    public: bool
    source: str

# Helper Functions
async def get_embedding(text: str) -> List[float]:
    """Generate embedding using Azure OpenAI"""
    try:
        response = await openai_client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT_NAME,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")

async def generate_summary(text: str) -> str:
    """Generate summary using Azure OpenAI"""
    try:
        response = await openai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries of articles. Summarize the following article in 3-5 sentences, focusing on the main points and key insights."},
                {"role": "user", "content": f"Article text: {text[:4000]}"}  # Limit text length
            ],
            max_tokens=200,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
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
            
        async with session.get(url, timeout=10) as response:
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

async def search_web(keywords: List[str], max_results: int = 10) -> List[str]:
    """Mock web search - replace with actual search API"""
    # This is a placeholder. In a real implementation, you would use:
    # - Bing Web Search API
    # - Google Custom Search JSON API
    # - SerpAPI
    # - Or other search services
    
    # For demo purposes, return some sample URLs
    sample_urls = [
        "https://example.com/ai-news-1",
        "https://example.com/ml-breakthrough-2",
        "https://example.com/tech-trends-3",
        "https://example.com/ai-research-4",
        "https://example.com/future-tech-5"
    ]
    
    return sample_urls[:max_results]

# API Endpoints
@app.post("/crawl")
async def crawl_articles(request: CrawlRequest):
    """Crawl web for articles based on keywords"""
    try:
        # Search for URLs
        urls = await search_web(request.keywords, request.max_articles)
        
        articles_added = 0
        
        async with aiohttp.ClientSession() as session:
            for url in urls:
                try:
                    # Check if article already exists
                    existing = collection.get(where={"url": url})
                    if existing['ids']:
                        continue
                    
                    # Scrape article
                    article_data = await scrape_article(session, url)
                    if not article_data:
                        continue
                    
                    # Generate summary
                    summary = await generate_summary(article_data['content'])
                    
                    # Generate embedding
                    embedding = await get_embedding(summary)
                    
                    # Store in ChromaDB
                    article_id = str(uuid.uuid4())
                    collection.add(
                        ids=[article_id],
                        embeddings=[embedding],
                        documents=[summary],
                        metadatas=[{
                            "title": article_data['title'],
                            "url": article_data['url'],
                            "date_published": article_data['date_published'],
                            "date_added": datetime.now().isoformat(),
                            "public": article_data['public'],
                            "source": article_data['source']
                        }]
                    )
                    
                    articles_added += 1
                    logger.info(f"Added article: {article_data['title']}")
                    
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
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
        
        if not results['documents'][0]:
            return {"answer": "I don't have any relevant articles to answer your question. Please run a crawl first to gather some articles."}
        
        # Prepare context from retrieved articles
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            context_parts.append(f"Article {i+1}: {metadata['title']}\nSummary: {doc}\nSource: {metadata['url']}\n")
        
        context = "\n".join(context_parts)
        
        # Generate answer using Azure OpenAI
        response = await openai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an AI assistant that answers questions based on a collection of article summaries. Use only the information provided in the articles to answer questions. If the answer is not in the articles, say so. Always be helpful and cite which articles you're referencing when possible."
                },
                {
                    "role": "user", 
                    "content": f"Articles:\n{context}\n\nQuestion: {request.question}\n\nAnswer:"
                }
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content.strip()
        
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"Question answering error: {e}")
        raise HTTPException(status_code=500, detail="Failed to answer question")

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
                source=metadata['source']
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
