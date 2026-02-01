#!/usr/bin/env python3
"""Reset the articles table with new embedding dimension for Vertex AI."""

import asyncio
import os
from pathlib import Path
from dotenv import dotenv_values

import asyncpg

SCRIPT_DIR = Path(__file__).parent

async def reset_database():
    # Load env vars
    env_path = SCRIPT_DIR / ".env"
    env_vars = dotenv_values(env_path) if env_path.exists() else {}
    postgres_url = os.environ.get("POSTGRES_URL") or env_vars.get("POSTGRES_URL")
    
    if not postgres_url:
        print("Error: POSTGRES_URL not found in environment or .env file")
        return
    
    print(f"Connecting to database...")
    conn = await asyncpg.connect(postgres_url)
    
    try:
        # Enable pgvector extension if not exists
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("✓ pgvector extension enabled")
        
        # Drop existing table
        await conn.execute("DROP TABLE IF EXISTS articles;")
        print("✓ Dropped existing articles table")
        
        # Create new table with 768-dim embeddings
        await conn.execute("""
            CREATE TABLE articles (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                title TEXT NOT NULL,
                url TEXT UNIQUE NOT NULL,
                summary TEXT,
                embedding vector(768),
                date_published TEXT,
                date_added TIMESTAMP DEFAULT NOW(),
                is_public BOOLEAN DEFAULT TRUE,
                source TEXT,
                content_type TEXT,
                region TEXT,
                sentiment TEXT
            );
        """)
        print("✓ Created articles table with vector(768) embedding column")
        
        # Create vector index
        await conn.execute("""
            CREATE INDEX ON articles USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        """)
        print("✓ Created vector similarity index")
        
        print("\n✅ Database reset complete! Ready for Vertex AI embeddings.")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(reset_database())
