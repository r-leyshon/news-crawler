#!/usr/bin/env python3
"""
Standalone article crawl script for use with GitHub Actions.
Reads keywords from keywords.txt and crawls UK AI articles.
"""

import asyncio
import aiohttp
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
import urllib.robotparser

from bs4 import BeautifulSoup
import chromadb
from openai import AsyncAzureOpenAI
from ddgs import DDGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Find project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_config():
    """Load configuration from config.json"""
    config_path = SCRIPT_DIR / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def load_keywords():
    """Load search keywords from keywords.txt"""
    keywords_path = PROJECT_ROOT / "keywords.txt"
    if not keywords_path.exists():
        logger.error(f"Keywords file not found: {keywords_path}")
        sys.exit(1)
    
    with open(keywords_path, "r") as f:
        keywords = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(keywords)} keywords from {keywords_path}")
    return keywords


def init_openai_client(azure_config: dict) -> AsyncAzureOpenAI:
    """Initialize the Azure OpenAI client"""
    # Get credentials from environment variables (for GitHub Actions)
    # or from .env file (for local development)
    api_key = os.environ.get("AZURE_OPENAI_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    # Try loading from .env file if not in environment
    if not api_key or not endpoint:
        try:
            from dotenv import dotenv_values
            env_vars = dotenv_values(SCRIPT_DIR / ".env")
            api_key = api_key or env_vars.get("AZURE_OPENAI_KEY")
            endpoint = endpoint or env_vars.get("AZURE_OPENAI_ENDPOINT")
        except Exception:
            pass
    
    if not api_key or not endpoint:
        logger.error("Missing Azure OpenAI credentials. Set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT.")
        sys.exit(1)
    
    return AsyncAzureOpenAI(
        api_key=api_key,
        api_version=azure_config["api_version"],
        azure_endpoint=endpoint
    )


def init_chromadb():
    """Initialize ChromaDB client"""
    chroma_path = SCRIPT_DIR / "chroma_db"
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name="articles",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


async def get_embedding(client: AsyncAzureOpenAI, deployment_name: str, text: str) -> List[float]:
    """Generate embedding using Azure OpenAI"""
    try:
        response = await asyncio.wait_for(
            client.embeddings.create(
                model=deployment_name,
                input=text
            ),
            timeout=30.0
        )
        return response.data[0].embedding
    except asyncio.TimeoutError:
        logger.error("Embedding generation timed out")
        raise
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise


async def generate_summary(client: AsyncAzureOpenAI, deployment_name: str, text: str) -> str:
    """Generate summary using Azure OpenAI"""
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=deployment_name,
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


async def classify_sentiment(client: AsyncAzureOpenAI, deployment_name: str, content: str, title: str = "") -> str:
    """Classify article sentiment as positive, neutral, or negative"""
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
        
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis assistant. Analyze the sentiment of the provided article and classify it as positive, neutral, or negative."},
                    {"role": "user", "content": analysis_text}
                ],
                response_format=SENTIMENT_SCHEMA,
                max_tokens=50,
                temperature=0.1
            ),
            timeout=15.0
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result.get("sentiment", "neutral")
    except Exception as e:
        logger.error(f"Sentiment classification error: {e}")
        return "neutral"


async def filter_inappropriate_content(client: AsyncAzureOpenAI, deployment_name: str, content: str, title: str = "") -> dict:
    """Filter inappropriate content"""
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
            client.chat.completions.create(
                model=deployment_name,
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
        
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        logger.error(f"Content filtering error: {e}")
        return {"is_safe": True, "reason": f"Filter error: {str(e)}"}


def can_fetch(url: str) -> bool:
    """Check if URL can be fetched according to robots.txt"""
    try:
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        
        return rp.can_fetch("*", url)
    except Exception:
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
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "No title"
            
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
                body = soup.find('body')
                content = body.get_text() if body else ""
            
            content = ' '.join(content.split())
            
            # Check for paywall
            paywall_indicators = [
                "subscribe to read", "premium content", "sign up to continue",
                "login to view", "paywall", "subscription required"
            ]
            is_public = not any(indicator in content.lower() for indicator in paywall_indicators)
            
            # Extract publication date
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


async def search_web(ddgs_client: DDGS, query: str, max_results: int = 10, region: str = "uk-en") -> List[dict]:
    """Search web using DuckDuckGo"""
    try:
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


async def crawl_keyword(
    keyword: str,
    collection,
    openai_client: AsyncAzureOpenAI,
    chat_deployment: str,
    embedding_deployment: str,
    ddgs_client: DDGS,
    max_articles: int = 5,
    region: str = "uk-en"
) -> tuple[int, int]:
    """Crawl articles for a single keyword"""
    articles_added = 0
    articles_filtered = 0
    
    search_results = await search_web(ddgs_client, keyword, max_articles, region)
    
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
                        filter_result = await filter_inappropriate_content(
                            openai_client, chat_deployment,
                            article_data['content'], article_data['title']
                        )
                        
                        if not filter_result['is_safe']:
                            logger.warning(f"Content filtered out: {url} - {filter_result['reason']}")
                            articles_filtered += 1
                            continue
                    except Exception as e:
                        logger.error(f"Content filtering error for {url}: {e}")
                    
                    try:
                        summary = await generate_summary(openai_client, chat_deployment, article_data['content'])
                        if not summary or summary.startswith("Summary generation"):
                            summary = article_data['content'][:500] + "..."
                    except Exception as e:
                        logger.error(f"Summary generation error for {url}: {e}")
                        summary = article_data['content'][:500] + "..."
                    
                    try:
                        sentiment = await classify_sentiment(
                            openai_client, chat_deployment,
                            article_data['content'], article_data['title']
                        )
                    except Exception as e:
                        logger.error(f"Sentiment classification error for {url}: {e}")
                        sentiment = "neutral"
                    
                    try:
                        embedding = await get_embedding(openai_client, embedding_deployment, summary)
                    except Exception as e:
                        logger.error(f"Embedding generation failed for {url}: {e}")
                        continue
                    
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
                                "region": region,
                                "sentiment": sentiment
                            }]
                        )
                        articles_added += 1
                        logger.info(f"Added full article: {article_data['title']}")
                    except Exception as e:
                        logger.error(f"ChromaDB storage failed for {url}: {e}")
                        continue
                else:
                    # Scraping failed, store link-only article
                    logger.info(f"Scraping failed for {url}, storing as link-only: {search_title}")
                    
                    search_content = f"{search_title}\n{search_body}" if search_body else search_title
                    
                    try:
                        filter_result = await filter_inappropriate_content(
                            openai_client, chat_deployment, search_content, search_title
                        )
                        
                        if not filter_result['is_safe']:
                            logger.warning(f"Search result filtered out: {url} - {filter_result['reason']}")
                            articles_filtered += 1
                            continue
                    except Exception as e:
                        logger.error(f"Content filtering error for search result {url}: {e}")
                    
                    summary = search_body if search_body else f"External article: {search_title}"
                    
                    try:
                        sentiment = await classify_sentiment(
                            openai_client, chat_deployment, search_content, search_title
                        )
                    except Exception as e:
                        logger.error(f"Sentiment classification error for search result {url}: {e}")
                        sentiment = "neutral"
                    
                    try:
                        embedding = await get_embedding(openai_client, embedding_deployment, summary)
                    except Exception as e:
                        logger.error(f"Embedding generation failed for search result {url}: {e}")
                        continue
                    
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
                                "public": True,
                                "source": urlparse(url).netloc,
                                "content_type": "link_only",
                                "region": region,
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
    
    return articles_added, articles_filtered


async def main():
    """Main entry point for the crawl script"""
    logger.info("=" * 60)
    logger.info("Starting article crawl")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    azure_config = config["azure_openai"]
    
    # Load keywords
    keywords = load_keywords()
    
    # Initialize clients
    openai_client = init_openai_client(azure_config)
    collection = init_chromadb()
    ddgs_client = DDGS()
    
    # Get deployment names from config
    chat_deployment = azure_config["chat_deployment_name"]
    embedding_deployment = azure_config["embedding_deployment_name"]
    
    # Crawl each keyword
    total_added = 0
    total_filtered = 0
    
    for keyword in keywords:
        logger.info(f"Crawling keyword: {keyword}")
        try:
            added, filtered = await crawl_keyword(
                keyword=keyword,
                collection=collection,
                openai_client=openai_client,
                chat_deployment=chat_deployment,
                embedding_deployment=embedding_deployment,
                ddgs_client=ddgs_client,
                max_articles=5,  # 5 articles per keyword
                region="uk-en"
            )
            total_added += added
            total_filtered += filtered
            logger.info(f"Keyword '{keyword}': {added} added, {filtered} filtered")
        except Exception as e:
            logger.error(f"Error crawling keyword '{keyword}': {e}")
            continue
        
        # Small delay between keywords to be polite
        await asyncio.sleep(2)
    
    logger.info("=" * 60)
    logger.info(f"Crawl complete: {total_added} articles added, {total_filtered} filtered")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
