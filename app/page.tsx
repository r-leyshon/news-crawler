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
  summary?: string
  date_published?: string
  date_added: string
  public: boolean
  source: string
  content_type?: string
}

export default function ArticleAssistant() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [searchKeywords, setSearchKeywords] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isCrawling, setIsCrawling] = useState(false)
  const [articles, setArticles] = useState<Article[]>([])
  const [crawlStatus, setCrawlStatus] = useState("")
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

  // Function to render text with clickable links and basic markdown
  const renderTextWithLinks = (text: string, isUser: boolean = false) => {
    const urlRegex = /(https?:\/\/[^\s\]]+)/g
    const parts = text.split(urlRegex)
    
    return parts.map((part, index) => {
      if (urlRegex.test(part)) {
        return (
          <a
            key={index}
            href={part}
            target="_blank"
            rel="noopener noreferrer"
            className={`underline break-all ${
              isUser 
                ? "text-blue-100 hover:text-white" 
                : "text-blue-600 hover:text-blue-800"
            }`}
          >
            {part}
          </a>
        )
      }
      return part
    })
  }

  // Function to render markdown formatting
  const renderMarkdown = (text: string) => {
    const lines = text.split('\n')
    const elements: JSX.Element[] = []
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]
      
      // Handle numbered lists
      if (line.match(/^\d+\.\s/)) {
        const listItems = []
        let j = i
        while (j < lines.length && lines[j].match(/^\d+\.\s/)) {
          const content = lines[j].replace(/^\d+\.\s/, '')
          listItems.push(
            <li key={j} className="mb-1">
              {renderInlineMarkdown(content)}
            </li>
          )
          j++
        }
        elements.push(
          <ol key={i} className="mb-2 pl-4 list-decimal">
            {listItems}
          </ol>
        )
        i = j - 1
        continue
      }
      
      // Handle bullet lists
      if (line.match(/^[\*\-\+]\s/)) {
        const listItems = []
        let j = i
        while (j < lines.length && lines[j].match(/^[\*\-\+]\s/)) {
          const content = lines[j].replace(/^[\*\-\+]\s/, '')
          listItems.push(
            <li key={j} className="mb-1">
              {renderInlineMarkdown(content)}
            </li>
          )
          j++
        }
        elements.push(
          <ul key={i} className="mb-2 pl-4 list-disc">
            {listItems}
          </ul>
        )
        i = j - 1
        continue
      }
      
      // Handle regular paragraphs
      if (line.trim() !== '') {
        elements.push(
          <p key={i} className="mb-2 last:mb-0">
            {renderInlineMarkdown(line)}
          </p>
        )
      }
    }
    
    return elements
  }

  // Function to handle inline markdown (bold, italic, links)
  const renderInlineMarkdown = (text: string) => {
    const parts = []
    let remaining = text
    let key = 0
    
    while (remaining.length > 0) {
      // Handle bold text **text**
      const boldMatch = remaining.match(/\*\*(.*?)\*\*/)
      if (boldMatch) {
        const beforeBold = remaining.substring(0, boldMatch.index)
        if (beforeBold) {
          parts.push(<span key={key++}>{renderTextWithLinks(beforeBold)}</span>)
        }
        parts.push(<strong key={key++} className="font-semibold">{boldMatch[1]}</strong>)
        remaining = remaining.substring(boldMatch.index! + boldMatch[0].length)
        continue
      }
      
      // Handle italic text *text*
      const italicMatch = remaining.match(/\*(.*?)\*/)
      if (italicMatch) {
        const beforeItalic = remaining.substring(0, italicMatch.index)
        if (beforeItalic) {
          parts.push(<span key={key++}>{renderTextWithLinks(beforeItalic)}</span>)
        }
        parts.push(<em key={key++} className="italic">{italicMatch[1]}</em>)
        remaining = remaining.substring(italicMatch.index! + italicMatch[0].length)
        continue
      }
      
      // No more markdown, add the rest
      parts.push(<span key={key++}>{renderTextWithLinks(remaining)}</span>)
      break
    }
    
    return parts
  }

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
    if (!searchKeywords.trim()) {
      toast({
        title: "Search Keywords Required",
        description: "Please enter search keywords before starting a crawl.",
        variant: "destructive",
      })
      return
    }

    setIsCrawling(true)
    setCrawlStatus("Starting crawl...")

    // Split keywords by comma and clean them up
    const keywordList = searchKeywords
      .split(',')
      .map(keyword => keyword.trim())
      .filter(keyword => keyword.length > 0)

    try {
      const response = await fetch(`${API_BASE}/crawl`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          keywords: keywordList,
          max_articles: 10,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setCrawlStatus(`Search & crawl completed! Found ${data.articles_added} new articles using DuckDuckGo.`)
        toast({
          title: "Search & Crawl Completed",
          description: `Successfully found and added ${data.articles_added} new articles.`,
        })
        fetchArticles()
      } else {
        throw new Error("Crawl failed")
      }
    } catch (error) {
      setCrawlStatus("Search & crawl failed. Please try again.")
      toast({
        title: "Search & Crawl Failed",
        description: "There was an error during the search and crawling process.",
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
    const questionInput = input
    setInput("")
    setIsLoading(true)

    // Create assistant message with empty content that we'll update
    const assistantMessageId = (Date.now() + 1).toString()
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, assistantMessage])

    try {
      // Prepare conversation history (exclude the current user message and the empty assistant message just added)
      const conversationHistory = messages.slice(0, -2).map(msg => ({
        role: msg.role,
        content: msg.content
      }))

      const response = await fetch(`${API_BASE}/ask/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          question: questionInput,
          messages: conversationHistory
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to get response")
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (reader) {
        let buffer = ""
        
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ""

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6))
                if (data.content) {
                  setMessages((prev) => 
                    prev.map((msg) => 
                      msg.id === assistantMessageId 
                        ? { ...msg, content: msg.content + data.content }
                        : msg
                    )
                  )
                }
                if (data.done) {
                  setIsLoading(false)
                }
              } catch (parseError) {
                console.error("Error parsing streaming data:", parseError)
              }
            }
          }
        }
      }
    } catch (error) {
      console.error("Streaming error:", error)
      setMessages((prev) => 
        prev.map((msg) => 
          msg.id === assistantMessageId 
            ? { ...msg, content: "Sorry, there was an error generating the response." }
            : msg
        )
      )
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
    <div className="h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4 overflow-hidden">
      <div className="max-w-7xl mx-auto h-full flex flex-col">
        {/* Header */}
        <div className="mb-6 flex-shrink-0">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">AI-Powered News Discovery Assistant</h1>
          <p className="text-gray-600">Search the web, discover articles, and chat with your curated collection using DuckDuckGo</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
          {/* Control Panel */}
          <div className="lg:col-span-1 flex flex-col min-h-0">
            <Card className="mb-6 flex-shrink-0">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Globe className="h-5 w-5" />
                  Web Search & Crawl
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <label htmlFor="search-keywords" className="block text-sm font-medium text-gray-700 mb-2">
                      Search Keywords
                    </label>
                    <Input
                      id="search-keywords"
                      value={searchKeywords}
                      onChange={(e) => setSearchKeywords(e.target.value)}
                      placeholder="e.g., artificial intelligence, machine learning, technology news"
                      disabled={isCrawling}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Separate multiple keywords with commas
                    </p>
                  </div>
                  <Button onClick={handleCrawl} disabled={isCrawling || !searchKeywords.trim()} className="w-full">
                    {isCrawling ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Searching & Crawling...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Search & Crawl Articles
                      </>
                    )}
                  </Button>
                  {crawlStatus && <p className="text-sm text-gray-600 mt-2">{crawlStatus}</p>}
                </div>
              </CardContent>
            </Card>

            {/* Articles List */}
            <Card className="flex-1 flex flex-col min-h-0">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 justify-between">
                  <div className="flex items-center gap-2">
                    <Calendar className="h-5 w-5" />
                    Recent Articles ({Math.min(10, articles.length)} of {articles.length})
                  </div>
                  {articles.length > 10 && (
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => window.open('/articles', '_blank')}
                      className="text-xs"
                    >
                      View All {articles.length}
                    </Button>
                  )}
                </CardTitle>
                <p className="text-xs text-gray-500 mt-1">
                  <span className="font-medium">Full Article:</span> Content scraped & summarized • 
                  <span className="font-medium"> External Link:</span> Title only, click to read
                </p>
              </CardHeader>
              <CardContent className="flex-1 overflow-y-auto min-h-0">
                {articles.length === 0 ? (
                  <p className="text-gray-500 text-sm">No articles yet. Enter search keywords and run a search to get started!</p>
                ) : (
                  <div className="space-y-3">
                    {articles.slice(0, 10).map((article) => (
                      <div key={article.id} className="border-b pb-3 last:border-b-0">
                        <h4 className="font-medium text-sm line-clamp-2 mb-1">{article.title}</h4>
                        <div className="flex items-center gap-2 mb-2">
                          <Badge 
                            variant={article.content_type === "link_only" ? "outline" : (article.public ? "default" : "secondary")} 
                            className="text-xs"
                          >
                            {article.content_type === "link_only" 
                              ? "External Link" 
                              : (article.public ? "Full Article" : "Limited Access")
                            }
                          </Badge>
                          <span className="text-xs text-gray-500">{article.source}</span>
                        </div>
                        {article.content_type === "link_only" ? (
                          <p className="text-xs text-gray-600 mb-2 italic">
                            Title and link only - click below to read the full article
                          </p>
                        ) : (
                          <p className="text-xs text-gray-600 line-clamp-2 mb-2">{article.summary || "No summary available"}</p>
                        )}
                        <a
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-blue-600 hover:text-blue-800 flex items-center gap-1"
                        >
                          <ExternalLink className="h-3 w-3" />
                          {article.content_type === "link_only" ? "Read article" : "Read original"}
                        </a>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Chat Interface */}
          <div className="lg:col-span-2 flex flex-col min-h-0">
            <Card className="flex-1 flex flex-col min-h-0">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageCircle className="h-5 w-5" />
                  Chat with Your Articles
                </CardTitle>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col min-h-0">
                {/* Messages */}
                <div className="flex-1 overflow-y-auto mb-4 space-y-4 min-h-0">
                  {messages.length === 0 ? (
                    <div className="text-center text-gray-500 mt-8">
                      <MessageCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Start a conversation about your articles!</p>
                      <p className="text-sm mt-2">Try asking: "What are the main topics in recent articles?" or "Summarize the latest AI developments"</p>
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
                          {message.role === "user" ? (
                            <p className="whitespace-pre-wrap">{renderTextWithLinks(message.content, true)}</p>
                          ) : (
                            <div className="prose prose-sm max-w-none">
                              {renderMarkdown(message.content)}
                            </div>
                          )}
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
                        <div className="flex items-center gap-1">
                          <span className="text-sm text-gray-500">AI is thinking</span>
                          <div className="flex gap-1">
                            <div className="w-1 h-1 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0ms'}} />
                            <div className="w-1 h-1 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}} />
                            <div className="w-1 h-1 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}} />
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>

                {/* Input Form */}
                <form onSubmit={handleSubmit} className="flex gap-2 flex-shrink-0">
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
