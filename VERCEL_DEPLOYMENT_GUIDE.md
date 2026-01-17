# Vercel Deployment Guide for News Crawler

This guide covers deploying a **two-project architecture** (separate frontend and backend) to Vercel, with lessons learned from real deployment challenges.

## Architecture Overview

```
news-crawler/
├── app/                    # Next.js frontend (App Router)
├── backend/                # Python FastAPI backend (separate Vercel project)
│   ├── api/
│   │   └── index.py        # Vercel entry point
│   ├── main.py             # FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── vercel.json         # Backend Vercel config
├── components/             # React components
├── lib/                    # Utility functions
└── vercel.json             # Frontend Vercel config (minimal)
```

### Two Separate Vercel Projects

| Project | Type | URL Pattern |
|---------|------|-------------|
| Frontend | Next.js | `news-crawler-xxx.vercel.app` |
| Backend | Python/FastAPI | `news-crawler-backend-xxx.vercel.app` |

---

## Lessons Learned: Common Pitfalls

### 🚨 Pitfall 1: SSR vs Client-Side URL Detection

**Problem:** Next.js renders pages on the server first, where `window` is undefined.

```typescript
// ❌ BROKEN - window is undefined during SSR
const apiBase = window.location.hostname !== 'localhost' 
  ? 'https://backend.vercel.app' 
  : 'http://localhost:8000'
```

**Solution:** Use `useEffect` to detect URLs only on the client side:

```typescript
// ✅ CORRECT - Only runs on client
const [apiBase, setApiBase] = useState('')
const [isApiReady, setIsApiReady] = useState(false)

useEffect(() => {
  if (typeof window !== 'undefined') {
    const url = process.env.NEXT_PUBLIC_API_URL || 
      (window.location.hostname === 'localhost' 
        ? 'http://localhost:8000' 
        : 'https://news-crawler-backend-xxx.vercel.app')
    setApiBase(url)
    setIsApiReady(true)
  }
}, [])

// Wait for API to be ready before fetching
useEffect(() => {
  if (isApiReady) {
    fetchArticles()
  }
}, [isApiReady])
```

**Why this matters:** Without this fix, the initial server render uses the wrong URL, causing hydration mismatches and failed API calls.

---

### 🚨 Pitfall 2: Environment Variables on the WRONG Project

**Problem:** When transitioning from monorepo to separate projects, it's easy to set environment variables on the wrong Vercel project.

| Variable | Correct Project | Wrong Project |
|----------|-----------------|---------------|
| `POSTGRES_URL` | ✅ Backend | ❌ Frontend |
| `AZURE_OPENAI_KEY` | ✅ Backend | ❌ Frontend |
| `AZURE_OPENAI_ENDPOINT` | ✅ Backend | ❌ Frontend |
| `NEXT_PUBLIC_API_URL` | ✅ Frontend | ❌ Backend |
| `NEXTAUTH_SECRET` | ✅ Frontend | ❌ Backend |
| `GITHUB_ID` / `GITHUB_SECRET` | ✅ Frontend | ❌ Backend |

**Critical:** If `NEXT_PUBLIC_API_URL` points to the frontend domain instead of the backend domain, all API calls will fail with 404/405 errors!

```
# ❌ WRONG - Points to frontend
NEXT_PUBLIC_API_URL=https://news-crawler-ochre.vercel.app/api

# ✅ CORRECT - Points to backend
NEXT_PUBLIC_API_URL=https://news-crawler-backend-xxx.vercel.app
```

---

### 🚨 Pitfall 3: Mixed Deployment Configuration

**Problem:** Frontend `vercel.json` contains Python build configuration meant for a monorepo setup.

```json
// ❌ WRONG - Frontend trying to build Python
{
  "builds": [
    { "src": "api/index.py", "use": "@vercel/python" },
    { "src": "package.json", "use": "@vercel/next" }
  ],
  "routes": [
    { "src": "/ask/stream", "dest": "/api/index.py" }
  ]
}
```

**Solution:** Keep frontend `vercel.json` minimal:

```json
// ✅ CORRECT - Pure Next.js frontend
{
  "version": 2,
  "framework": "nextjs"
}
```

**Also:** Remove any `/api/*.py` files from the frontend repo - the backend has its own.

---

## Environment Variables Reference

### Backend Project (`news-crawler-backend`)

| Variable | Description | Required |
|----------|-------------|----------|
| `POSTGRES_URL` | Vercel Postgres connection string | ✅ |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key | ✅ |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | ✅ |

### Frontend Project (`news-crawler`)

| Variable | Description | Required |
|----------|-------------|----------|
| `NEXTAUTH_SECRET` | Random secret for session encryption | ✅ |
| `NEXTAUTH_URL` | Frontend URL (e.g., `https://news-crawler.vercel.app`) | ✅ |
| `GITHUB_ID` | GitHub OAuth App Client ID | ✅ |
| `GITHUB_SECRET` | GitHub OAuth App Client Secret | ✅ |
| `NEXT_PUBLIC_API_URL` | Backend URL (optional - has fallback) | ⚪ |

---

## Deployment Steps

### 1. Deploy Backend First

```bash
cd backend
vercel --prod
```

Note the deployed URL (e.g., `https://news-crawler-backend-xxx.vercel.app`)

### 2. Set Backend Environment Variables

Via Vercel Dashboard or CLI:
```bash
vercel env add POSTGRES_URL
vercel env add AZURE_OPENAI_KEY
vercel env add AZURE_OPENAI_ENDPOINT
```

### 3. Verify Backend Health

```bash
curl https://news-crawler-backend-xxx.vercel.app/health
# Should return: {"status":"healthy","timestamp":"..."}
```

### 4. Deploy Frontend

The frontend can be deployed via:
- **Git integration** (push to main triggers deploy)
- **Vercel CLI**: `vercel --prod`

### 5. Set Frontend Environment Variables

```bash
vercel env add NEXTAUTH_SECRET
vercel env add NEXTAUTH_URL
vercel env add GITHUB_ID
vercel env add GITHUB_SECRET
```

### 6. Verify Full Stack

1. Visit frontend URL
2. Check that articles load
3. Test the chat functionality
4. Verify OAuth login works

---

## Configuration Files

### Frontend `vercel.json`

```json
{
  "version": 2,
  "framework": "nextjs"
}
```

### Backend `backend/vercel.json`

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ]
}
```

### Backend `backend/api/index.py`

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

# Export the FastAPI app as 'app' for Vercel's ASGI support
```

---

## Troubleshooting

### "No articles" / Empty Database

1. Check backend health: `curl https://backend-url/health`
2. Check backend has `POSTGRES_URL` set
3. Verify database has articles: `curl https://backend-url/articles`
4. Check frontend console for API errors

### 404 / 405 Errors on API Calls

1. Check `NEXT_PUBLIC_API_URL` points to **backend** domain
2. Verify backend is deployed and routes are working
3. Check browser Network tab for actual URL being called

### CORS Errors

Backend `main.py` must allow frontend origin:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://news-crawler-xxx.vercel.app", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Chat/Streaming Not Working

1. Verify `/ask/stream` endpoint works: 
   ```bash
   curl -X POST https://backend-url/ask/stream \
     -H "Content-Type: application/json" \
     -d '{"question":"test","articles":[]}'
   ```
2. Check Azure OpenAI credentials are set on backend

---

## Quick Debugging Commands

```bash
# Test backend health
curl https://news-crawler-backend-xxx.vercel.app/health

# Get articles from backend
curl https://news-crawler-backend-xxx.vercel.app/articles

# Check Vercel logs
vercel logs https://news-crawler-backend-xxx.vercel.app

# List environment variables
vercel env ls
```

---

## Key Takeaways

1. **Use `useEffect` for client-side URL detection** - Never access `window` during SSR
2. **Environment variables go on the correct project** - Backend vars on backend, frontend vars on frontend
3. **Keep frontend config minimal** - No Python builds in frontend `vercel.json`
4. **Test backend independently** - Use curl to verify before debugging frontend
5. **Check the actual URL being called** - Browser DevTools Network tab is your friend
