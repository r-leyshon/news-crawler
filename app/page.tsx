"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Loader2, Play, MessageCircle, Globe, Calendar, ExternalLink } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

interface Article {
  id: string
  title: string
  url: string
  summary: string
  date_published?: string
  date_added: string
  public: boolean
  source: string
}

export default function ArticleAssistant() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isCrawling, setIsCrawling] = useState(false)
  const [articles, setArticles] = useState<Article[]>([])
  const [crawlStatus, setCrawlStatus] = useState("")
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  useEffect(() => {
    fetchArticles()
  }, [])

  const fetchArticles = async () => {
    try {
      const response = await fetch(`${API_BASE}/articles`)
      if (response.ok) {
        const data = await response.json()
        setArticles(data)
      }
    } catch (error) {
      console.error("Failed to fetch articles:", error)
    }
  }

  const handleCrawl = async () => {
    setIsCrawling(true)
    setCrawlStatus("Starting crawl...")

    try {
      const response = await fetch(`${API_BASE}/crawl`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          keywords: ["artificial intelligence", "machine learning", "technology news"],
          max_articles: 10,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setCrawlStatus(`Crawl completed! Found ${data.articles_added} new articles.`)
        toast({
          title: "Crawl Completed",
          description: `Successfully added ${data.articles_added} new articles.`,
        })
        fetchArticles()
      } else {
        throw new Error("Crawl failed")
      }
    } catch (error) {
      setCrawlStatus("Crawl failed. Please try again.")
      toast({
        title: "Crawl Failed",
        description: "There was an error during the crawling process.",
        variant: "destructive",
      })
    } finally {
      setIsCrawling(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: input }),
      })

      if (response.ok) {
        const data = await response.json()
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: data.answer,
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, assistantMessage])
      } else {
        throw new Error("Failed to get response")
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to get response from the assistant.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">AI Article Discovery Assistant</h1>
          <p className="text-gray-600">Discover, summarize, and chat with your curated article collection</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Control Panel */}
          <div className="lg:col-span-1">
            <Card className="mb-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Globe className="h-5 w-5" />
                  Crawl Control
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Button onClick={handleCrawl} disabled={isCrawling} className="w-full mb-4">
                  {isCrawling ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Crawling...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Run Crawl
                    </>
                  )}
                </Button>
                {crawlStatus && <p className="text-sm text-gray-600">{crawlStatus}</p>}
              </CardContent>
            </Card>

            {/* Articles List */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Calendar className="h-5 w-5" />
                  Recent Articles ({articles.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="max-h-96 overflow-y-auto">
                {articles.length === 0 ? (
                  <p className="text-gray-500 text-sm">No articles yet. Run a crawl to get started!</p>
                ) : (
                  <div className="space-y-3">
                    {articles.slice(0, 10).map((article) => (
                      <div key={article.id} className="border-b pb-3 last:border-b-0">
                        <h4 className="font-medium text-sm line-clamp-2 mb-1">{article.title}</h4>
                        <div className="flex items-center gap-2 mb-2">
                          <Badge variant={article.public ? "default" : "secondary"} className="text-xs">
                            {article.public ? "Public" : "Limited"}
                          </Badge>
                          <span className="text-xs text-gray-500">{article.source}</span>
                        </div>
                        <p className="text-xs text-gray-600 line-clamp-2 mb-2">{article.summary}</p>
                        <a
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-blue-600 hover:text-blue-800 flex items-center gap-1"
                        >
                          <ExternalLink className="h-3 w-3" />
                          Read original
                        </a>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Chat Interface */}
          <div className="lg:col-span-2">
            <Card className="h-[600px] flex flex-col">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageCircle className="h-5 w-5" />
                  Chat with Your Articles
                </CardTitle>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                {/* Messages */}
                <div className="flex-1 overflow-y-auto mb-4 space-y-4">
                  {messages.length === 0 ? (
                    <div className="text-center text-gray-500 mt-8">
                      <MessageCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Start a conversation about your articles!</p>
                      <p className="text-sm mt-2">Try asking: "What are the main topics in recent articles?"</p>
                    </div>
                  ) : (
                    messages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`max-w-[80%] p-3 rounded-lg ${
                            message.role === "user" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-900"
                          }`}
                        >
                          <p className="whitespace-pre-wrap">{message.content}</p>
                          <p className={`text-xs mt-1 ${message.role === "user" ? "text-blue-100" : "text-gray-500"}`}>
                            {message.timestamp.toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                    ))
                  )}
                  {isLoading && (
                    <div className="flex justify-start">
                      <div className="bg-gray-100 p-3 rounded-lg">
                        <Loader2 className="h-4 w-4 animate-spin" />
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>

                {/* Input Form */}
                <form onSubmit={handleSubmit} className="flex gap-2">
                  <Input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask a question about your articles..."
                    disabled={isLoading}
                    className="flex-1"
                  />
                  <Button type="submit" disabled={isLoading || !input.trim()}>
                    {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Send"}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
