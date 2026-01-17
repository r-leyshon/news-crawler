"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { ArrowLeft, ExternalLink, Search, Calendar, Filter } from "lucide-react"
import { availableRegions, getRegionInfo } from "@/lib/regions"

interface Article {
  id: string
  title: string
  url: string
  summary?: string
  date_published?: string
  date_added: string
  public: boolean
  source: string
  content_type?: string
  region?: string
}

export default function ArticlesPage() {
  const [articles, setArticles] = useState<Article[]>([])
  const [filteredArticles, setFilteredArticles] = useState<Article[]>([])
  const [searchTerm, setSearchTerm] = useState("")
  const [filterType, setFilterType] = useState<"all" | "full" | "link_only">("all")
  const [regionFilter, setRegionFilter] = useState<"all" | string>("all")
  const [isLoading, setIsLoading] = useState(true)

  // Backend API URL - separate Vercel deployment for backend
  const [apiBase, setApiBase] = useState('')
  const [isApiReady, setIsApiReady] = useState(false)
  
  useEffect(() => {
    // Set API base URL on client side
    if (typeof window !== 'undefined') {
      // Use environment variable, or detect: localhost for dev, deployed backend for prod
      const url = process.env.NEXT_PUBLIC_API_URL || 
        (window.location.hostname === 'localhost' 
          ? 'http://localhost:8000' 
          : 'https://news-crawler-backend-r-leyshons-projects.vercel.app')
      setApiBase(url)
      setIsApiReady(true)
    }
  }, [])

  // Get unique regions from articles for filter dropdown
  const uniqueRegions = Array.from(new Set(articles.map(article => article.region).filter((region): region is string => Boolean(region))))

  useEffect(() => {
    if (isApiReady) {
      fetchArticles()
    }
  }, [isApiReady])

  useEffect(() => {
    // Filter and search articles
    let filtered = articles

    // Filter by type
    if (filterType !== "all") {
      filtered = filtered.filter(article => article.content_type === filterType)
    }

    // Filter by region
    if (regionFilter !== "all") {
      filtered = filtered.filter(article => article.region === regionFilter)
    }

    // Search by title, source, or summary
    if (searchTerm) {
      const search = searchTerm.toLowerCase()
      filtered = filtered.filter(article => 
        article.title.toLowerCase().includes(search) ||
        article.source.toLowerCase().includes(search) ||
        (article.summary && article.summary.toLowerCase().includes(search))
      )
    }

    setFilteredArticles(filtered)
  }, [articles, searchTerm, filterType, regionFilter])

  const fetchArticles = async () => {
    try {
      const response = await fetch(`${apiBase}/articles`)
      if (response.ok) {
        const data = await response.json()
        setArticles(data)
        setFilteredArticles(data)
      }
    } catch (error) {
      console.error("Failed to fetch articles:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDeleteArticle = async (articleId: string) => {
    try {
      const response = await fetch(`${apiBase}/articles/${articleId}`, {
        method: "DELETE"
      })
      if (response.ok) {
        setArticles(prev => prev.filter(article => article.id !== articleId))
      }
    } catch (error) {
      console.error("Failed to delete article:", error)
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mt-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-2 text-gray-600">Loading articles...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center gap-4 mb-4">
            <Button 
              variant="outline" 
              onClick={() => window.close()}
              className="flex items-center gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Chat
            </Button>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">All Articles</h1>
              <p className="text-gray-600">Browse and manage your article collection</p>
            </div>
          </div>

          {/* Search and Filter */}
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search articles by title, source, or content..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-gray-500" />
              <select 
                value={filterType}
                onChange={(e) => setFilterType(e.target.value as "all" | "full" | "link_only")}
                className="px-3 py-2 border border-gray-300 rounded-md bg-white text-sm"
              >
                <option value="all">All Types</option>
                <option value="full">Full Articles</option>
                <option value="link_only">External Links</option>
              </select>
              <select 
                value={regionFilter}
                onChange={(e) => setRegionFilter(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md bg-white text-sm"
              >
                <option value="all">All Regions</option>
                {uniqueRegions.map((region) => {
                  const regionInfo = getRegionInfo(region)
                  return (
                    <option key={region} value={region}>
                      {regionInfo ? `${regionInfo.flag} ${regionInfo.name}` : region}
                    </option>
                  )
                })}
              </select>
            </div>
          </div>

          {/* Stats */}
          <div className="mt-4 text-sm text-gray-600">
            Showing {filteredArticles.length} of {articles.length} articles
          </div>
        </div>

        {/* Articles Grid */}
        {filteredArticles.length === 0 ? (
          <div className="text-center py-12">
            <Calendar className="h-12 w-12 mx-auto text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              {searchTerm || filterType !== "all" || regionFilter !== "all" ? "No articles found" : "No articles yet"}
            </h3>
            <p className="text-gray-600">
              {searchTerm || filterType !== "all" || regionFilter !== "all"
                ? "Try adjusting your search or filter criteria." 
                : "Run a web search to discover and add articles to your collection."
              }
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredArticles.map((article) => (
              <Card key={article.id} className="flex flex-col">
                <CardHeader className="flex-shrink-0">
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <Badge 
                      variant={article.content_type === "link_only" ? "outline" : (article.public ? "default" : "secondary")} 
                      className="text-xs"
                    >
                      {article.content_type === "link_only" 
                        ? "External Link" 
                        : (article.public ? "Full Article" : "Limited Access")
                      }
                    </Badge>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteArticle(article.id)}
                      className="text-xs text-red-600 hover:text-red-800 p-1"
                    >
                      Delete
                    </Button>
                  </div>
                  <CardTitle className="text-base mb-2 overflow-hidden" style={{
                    display: '-webkit-box',
                    WebkitLineClamp: 3,
                    WebkitBoxOrient: 'vertical'
                  }}>
                    {article.title}
                  </CardTitle>
                  <div className="text-xs text-gray-500 space-y-1">
                    <div className="flex items-center gap-1">
                      <Calendar className="h-3 w-3" />
                      {new Date(article.date_added).toLocaleDateString()}
                    </div>
                    <div>Source: {article.source}</div>
                    {article.region && (
                      <div className="flex items-center gap-1">
                        Region: {getRegionInfo(article.region)?.flag} {getRegionInfo(article.region)?.name}
                      </div>
                    )}
                    {article.date_published && (
                      <div>Published: {new Date(article.date_published).toLocaleDateString()}</div>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col">
                  {article.content_type === "link_only" ? (
                    <p className="text-sm text-gray-600 italic mb-4 flex-1">
                      External link only - click below to read the full article
                    </p>
                  ) : (
                    <p className="text-sm text-gray-600 mb-4 flex-1 overflow-hidden" style={{
                      display: '-webkit-box',
                      WebkitLineClamp: 4,
                      WebkitBoxOrient: 'vertical'
                    }}>
                      {article.summary || "No summary available"}
                    </p>
                  )}
                  <a
                    href={article.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-2 mt-auto"
                  >
                    <ExternalLink className="h-3 w-3" />
                    {article.content_type === "link_only" ? "Read article" : "Read original"}
                  </a>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  )
} 