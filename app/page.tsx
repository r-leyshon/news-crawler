"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { useSession, signIn, signOut } from "next-auth/react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Loader2, MessageCircle, Globe, Calendar, ExternalLink, X, PanelRightClose, Search, Filter, Trash2, Sun, Moon, TrendingUp, TrendingDown, Minus, Settings, LogIn, LogOut, Github, Info } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { getRegionInfo } from "@/lib/regions"

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
  region?: string
  sentiment?: string  // "positive", "neutral", "negative"
}

export default function ArticleAssistant() {
  const { data: session, status } = useSession()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [articles, setArticles] = useState<Article[]>([])
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState("")
  const [filterType, setFilterType] = useState<"all" | "full" | "link_only">("all")
  const [regionFilter, setRegionFilter] = useState<"all" | string>("all")
  const [sentimentFilter, setSentimentFilter] = useState<"all" | "positive" | "neutral" | "negative">("all")
  const [isDarkMode, setIsDarkMode] = useState(true)
  const [isClassifying, setIsClassifying] = useState(false)
  const [isInfoOpen, setIsInfoOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  // Check if user is the owner (can delete articles)
  const isOwner = session?.user?.isOwner === true

  // In production, use the separate backend deployment. Locally, fallback to localhost:8000
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' && window.location.hostname !== 'localhost' ? 'https://news-crawler-backend-r-leyshons-projects.vercel.app' : 'http://localhost:8000')

  // Initialize theme from localStorage
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme')
    if (savedTheme) {
      setIsDarkMode(savedTheme === 'dark')
    }
  }, [])

  // Apply theme class to document
  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDarkMode)
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light')
  }, [isDarkMode])

  // Get unique regions from articles for filter dropdown
  const uniqueRegions = Array.from(new Set(articles.map(article => article.region).filter((region): region is string => Boolean(region))))

  // Calculate sentiment counts
  const unclassifiedCount = articles.filter(a => !a.sentiment).length
  const sentimentCounts = {
    positive: articles.filter(a => a.sentiment === "positive").length,
    neutral: articles.filter(a => a.sentiment === "neutral").length,
    negative: articles.filter(a => a.sentiment === "negative").length,
  }

  // Filter and sort articles by date
  const filteredArticles = articles
    .filter(article => {
      // Filter by type
      if (filterType !== "all" && article.content_type !== filterType) {
        return false
      }
      // Filter by region
      if (regionFilter !== "all" && article.region !== regionFilter) {
        return false
      }
      // Filter by sentiment
      if (sentimentFilter !== "all") {
        const articleSentiment = article.sentiment || "neutral"
        if (articleSentiment !== sentimentFilter) {
          return false
        }
      }
      // Search by title, source, or summary
      if (searchTerm) {
        const search = searchTerm.toLowerCase()
        return (
          article.title.toLowerCase().includes(search) ||
          article.source.toLowerCase().includes(search) ||
          (article.summary && article.summary.toLowerCase().includes(search))
        )
      }
      return true
    })
    .sort((a, b) => new Date(b.date_added).getTime() - new Date(a.date_added).getTime())

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
                ? "text-emerald-100 hover:text-white" 
                : "text-emerald-600 hover:text-emerald-800 dark:text-emerald-400 dark:hover:text-emerald-300"
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

  // Format date for display
  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))
    
    if (diffDays === 0) {
      return 'Today'
    } else if (diffDays === 1) {
      return 'Yesterday'
    } else if (diffDays < 7) {
      return `${diffDays} days ago`
    } else {
      return date.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
    }
  }

  // Get sentiment icon and color
  const getSentimentDisplay = (sentiment?: string) => {
    switch (sentiment) {
      case "positive":
        return { icon: TrendingUp, color: "text-green-500", bgColor: "bg-green-500/10", borderColor: "border-green-500/30" }
      case "negative":
        return { icon: TrendingDown, color: "text-red-500", bgColor: "bg-red-500/10", borderColor: "border-red-500/30" }
      default:
        return { icon: Minus, color: "text-slate-400", bgColor: "bg-slate-500/10", borderColor: "border-slate-500/30" }
    }
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

  const handleDeleteArticle = async (articleId: string) => {
    try {
      const response = await fetch(`${API_BASE}/articles/${articleId}`, {
        method: "DELETE"
      })
      if (response.ok) {
        setArticles(prev => prev.filter(article => article.id !== articleId))
        toast({
          title: "Article Deleted",
          description: "The article has been removed from your collection.",
        })
      }
    } catch (error) {
      console.error("Failed to delete article:", error)
      toast({
        title: "Delete Failed",
        description: "Failed to delete the article. Please try again.",
        variant: "destructive",
      })
    }
  }

  const handleClassifySentiment = async () => {
    setIsClassifying(true)
    try {
      const response = await fetch(`${API_BASE}/articles/classify-sentiment`, {
        method: "POST"
      })
      if (response.ok) {
        const data = await response.json()
        toast({
          title: "Sentiment Classification Complete",
          description: `Classified ${data.classified} articles. ${data.already_classified} were already classified.`,
        })
        // Refresh articles to get updated sentiment
        fetchArticles()
      } else {
        throw new Error("Classification failed")
      }
    } catch (error) {
      console.error("Failed to classify sentiment:", error)
      toast({
        title: "Classification Failed",
        description: "Failed to classify article sentiment. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsClassifying(false)
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
    <div className={`h-screen overflow-hidden flex transition-colors duration-200 ${
      isDarkMode 
        ? 'bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900' 
        : 'bg-gradient-to-br from-slate-50 via-white to-slate-100'
    }`}>
      {/* Main Content Area */}
      <div className={`flex-1 flex flex-col p-6 transition-all duration-300 ${isChatOpen ? 'mr-0' : ''}`}>
        {/* Header */}
        <div className="mb-6 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <h1 className={`text-3xl font-bold tracking-tight ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>
                  UK AI News
                </h1>
                <Button
                  onClick={() => setIsInfoOpen(true)}
                  variant="ghost"
                  size="icon"
                  className={`h-8 w-8 rounded-full ${
                    isDarkMode 
                      ? 'text-slate-400 hover:text-white hover:bg-slate-700' 
                      : 'text-slate-500 hover:text-slate-900 hover:bg-slate-200'
                  }`}
                >
                  <Info className="h-4 w-4" />
                </Button>
              </div>
              <p className={`text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                AI and machine learning news from the United Kingdom
              </p>
            </div>
            <div className="flex items-center gap-2">
              {/* Auth Button */}
              {status === "loading" ? (
                <Button variant="outline" size="sm" disabled className={`${
                  isDarkMode ? 'bg-slate-800/50 border-slate-600' : 'bg-white border-slate-300'
                }`}>
                  <Loader2 className="h-4 w-4 animate-spin" />
                </Button>
              ) : session ? (
                <Button
                  onClick={() => signOut()}
                  variant="outline"
                  size="sm"
                  className={`flex items-center gap-2 ${
                    isDarkMode 
                      ? 'bg-slate-800/50 border-slate-600 text-slate-200 hover:bg-slate-700 hover:text-white' 
                      : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-100'
                  }`}
                >
                  <LogOut className="h-4 w-4" />
                  <span className="hidden sm:inline">{session.user?.name || 'Logout'}</span>
                </Button>
              ) : (
                <Button
                  onClick={() => signIn("github")}
                  variant="outline"
                  size="sm"
                  className={`flex items-center gap-2 ${
                    isDarkMode 
                      ? 'bg-slate-800/50 border-slate-600 text-slate-200 hover:bg-slate-700 hover:text-white' 
                      : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-100'
                  }`}
                >
                  <Github className="h-4 w-4" />
                  <span className="hidden sm:inline">Admin Login</span>
                </Button>
              )}

              {/* Theme Toggle */}
              <Button
                onClick={() => setIsDarkMode(!isDarkMode)}
                variant="outline"
                size="icon"
                className={`${
                  isDarkMode 
                    ? 'bg-slate-800/50 border-slate-600 text-slate-200 hover:bg-slate-700 hover:text-white' 
                    : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-100'
                }`}
              >
                {isDarkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </Button>
              
              {/* Chat Toggle */}
              <Button
                onClick={() => setIsChatOpen(!isChatOpen)}
                className={`flex items-center gap-2 ${
                  isChatOpen 
                    ? 'bg-violet-600 hover:bg-violet-700 text-white' 
                    : 'bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white shadow-lg shadow-violet-500/25'
                }`}
              >
                {isChatOpen ? (
                  <>
                    <PanelRightClose className="h-4 w-4" />
                    Hide Chat
                  </>
                ) : (
                  <>
                    <MessageCircle className="h-4 w-4" />
                    Open Chat
                    {messages.length > 0 && (
                      <Badge className="ml-1 bg-white/20 text-white text-xs">{messages.length}</Badge>
                    )}
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* Sentiment Filter Pills */}
        <div className="flex items-center gap-3 mb-4 flex-shrink-0">
          <span className={`text-sm font-medium ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>Sentiment:</span>
          <div className="flex gap-2">
            <button
              onClick={() => setSentimentFilter("all")}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                sentimentFilter === "all"
                  ? isDarkMode 
                    ? 'bg-slate-600 text-white' 
                    : 'bg-slate-800 text-white'
                  : isDarkMode
                    ? 'bg-slate-800/50 text-slate-400 hover:bg-slate-700'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              All ({articles.length})
            </button>
            <button
              onClick={() => setSentimentFilter("positive")}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${
                sentimentFilter === "positive"
                  ? 'bg-green-600 text-white'
                  : isDarkMode
                    ? 'bg-green-500/10 text-green-400 border border-green-500/30 hover:bg-green-500/20'
                    : 'bg-green-50 text-green-700 border border-green-200 hover:bg-green-100'
              }`}
            >
              <TrendingUp className="h-3 w-3" />
              Positive ({sentimentCounts.positive})
            </button>
            <button
              onClick={() => setSentimentFilter("neutral")}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${
                sentimentFilter === "neutral"
                  ? isDarkMode ? 'bg-slate-500 text-white' : 'bg-slate-600 text-white'
                  : isDarkMode
                    ? 'bg-slate-500/10 text-slate-400 border border-slate-500/30 hover:bg-slate-500/20'
                    : 'bg-slate-100 text-slate-600 border border-slate-200 hover:bg-slate-200'
              }`}
            >
              <Minus className="h-3 w-3" />
              Neutral ({sentimentCounts.neutral})
            </button>
            <button
              onClick={() => setSentimentFilter("negative")}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${
                sentimentFilter === "negative"
                  ? 'bg-red-600 text-white'
                  : isDarkMode
                    ? 'bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20'
                    : 'bg-red-50 text-red-700 border border-red-200 hover:bg-red-100'
              }`}
            >
              <TrendingDown className="h-3 w-3" />
              Negative ({sentimentCounts.negative})
            </button>
            
            {/* Classify unclassified articles button - owner only */}
            {isOwner && unclassifiedCount > 0 && (
              <button
                onClick={handleClassifySentiment}
                disabled={isClassifying}
                className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ml-2 ${
                  isClassifying
                    ? 'bg-violet-600/50 text-violet-200 cursor-wait'
                    : 'bg-violet-600 text-white hover:bg-violet-700'
                }`}
              >
                {isClassifying ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Classifying...
                  </>
                ) : (
                  <>
                    <Settings className="h-3 w-3" />
                    Classify {unclassifiedCount} unclassified
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {/* Filter Bar */}
        <div className="flex items-center gap-4 mb-4 flex-shrink-0">
          <div className="flex-1 relative">
            <Search className={`absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`} />
            <Input
              placeholder="Filter articles..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={`pl-10 ${
                isDarkMode 
                  ? 'bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500' 
                  : 'bg-white border-slate-300 text-slate-900 placeholder:text-slate-400'
              }`}
            />
          </div>
          <div className="flex items-center gap-2">
            <Filter className={`h-4 w-4 ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`} />
            <select 
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as "all" | "full" | "link_only")}
              className={`px-3 py-2 text-sm border rounded-md ${
                isDarkMode 
                  ? 'border-slate-700 bg-slate-800 text-slate-200' 
                  : 'border-slate-300 bg-white text-slate-700'
              }`}
            >
              <option value="all">All Types</option>
              <option value="full">Full Articles</option>
              <option value="link_only">External Links</option>
            </select>
            {uniqueRegions.length > 0 && (
              <select 
                value={regionFilter}
                onChange={(e) => setRegionFilter(e.target.value)}
                className={`px-3 py-2 text-sm border rounded-md ${
                  isDarkMode 
                    ? 'border-slate-700 bg-slate-800 text-slate-200' 
                    : 'border-slate-300 bg-white text-slate-700'
                }`}
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
            )}
          </div>
          <div className={`text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>
            {filteredArticles.length} of {articles.length} articles
          </div>
        </div>

        {/* Articles List */}
        <div className="flex-1 overflow-y-auto min-h-0">
          {filteredArticles.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <Globe className={`h-16 w-16 mb-4 ${isDarkMode ? 'text-slate-600' : 'text-slate-300'}`} />
              <h3 className={`text-xl font-medium mb-2 ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
                {searchTerm || filterType !== "all" || regionFilter !== "all" || sentimentFilter !== "all" ? "No matching articles" : "No articles yet"}
              </h3>
              <p className={`max-w-md ${isDarkMode ? 'text-slate-500' : 'text-slate-500'}`}>
                {searchTerm || filterType !== "all" || regionFilter !== "all" || sentimentFilter !== "all"
                  ? "Try adjusting your filters to see more results." 
                  : "Articles are updated weekly via automated crawling."
                }
              </p>
            </div>
          ) : (
            <div className="space-y-2 pb-4">
              {filteredArticles.map((article) => {
                const sentimentDisplay = getSentimentDisplay(article.sentiment)
                const SentimentIcon = sentimentDisplay.icon
                
                return (
                  <div 
                    key={article.id} 
                    className={`flex items-start gap-4 p-4 rounded-lg border transition-colors group ${
                      isDarkMode 
                        ? 'bg-slate-800/30 border-slate-700 hover:border-slate-600 hover:bg-slate-800/50' 
                        : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50 shadow-sm'
                    }`}
                  >
                    {/* Sentiment Indicator */}
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${sentimentDisplay.bgColor} ${sentimentDisplay.borderColor} border`}>
                      <SentimentIcon className={`h-4 w-4 ${sentimentDisplay.color}`} />
                    </div>
                    
                    {/* Date Column */}
                    <div className={`flex-shrink-0 w-24 text-xs ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                      <div className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {formatDate(article.date_added)}
                      </div>
                    </div>
                    
                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start gap-2 mb-1">
                        <Badge 
                          variant="outline"
                          className={`text-xs flex-shrink-0 ${
                            article.content_type === "link_only" 
                              ? 'border-amber-500/50 text-amber-500' 
                              : 'border-emerald-500/50 text-emerald-500'
                          }`}
                        >
                          {article.content_type === "link_only" ? "Link" : "Full"}
                        </Badge>
                        <h3 className={`text-sm font-medium leading-snug ${isDarkMode ? 'text-slate-200' : 'text-slate-800'}`}>
                          {article.title}
                        </h3>
                      </div>
                      
                      <div className={`flex items-center gap-2 text-xs mb-2 ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                        <span>{article.source}</span>
                        {article.region && (
                          <>
                            <span>•</span>
                            <span>{getRegionInfo(article.region)?.flag} {getRegionInfo(article.region)?.name}</span>
                          </>
                        )}
                      </div>
                      
                      {article.content_type !== "link_only" && article.summary && (
                        <p className={`text-xs line-clamp-2 ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                          {article.summary}
                        </p>
                      )}
                    </div>
                    
                    {/* Actions */}
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <a
                        href={article.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-emerald-500 hover:text-emerald-400 p-1"
                      >
                        <ExternalLink className="h-4 w-4" />
                      </a>
                      {isOwner && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteArticle(article.id)}
                          className={`opacity-0 group-hover:opacity-100 transition-opacity h-7 w-7 p-0 ${
                            isDarkMode 
                              ? 'text-slate-500 hover:text-red-400 hover:bg-red-500/10' 
                              : 'text-slate-400 hover:text-red-500 hover:bg-red-50'
                          }`}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Chat Sidebar */}
      <div className={`${isChatOpen ? 'w-96' : 'w-0'} transition-all duration-300 overflow-hidden border-l flex flex-col ${
        isDarkMode 
          ? 'border-slate-700 bg-slate-800/50' 
          : 'border-slate-200 bg-slate-50'
      }`}>
        <div className={`p-4 border-b flex items-center justify-between flex-shrink-0 ${isDarkMode ? 'border-slate-700' : 'border-slate-200'}`}>
          <div className="flex items-center gap-2">
            <MessageCircle className="h-5 w-5 text-violet-500" />
            <h2 className={`font-semibold ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>Chat with Articles</h2>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsChatOpen(false)}
            className={`h-8 w-8 p-0 ${isDarkMode ? 'text-slate-400 hover:text-white hover:bg-slate-700' : 'text-slate-500 hover:text-slate-900 hover:bg-slate-200'}`}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
          {messages.length === 0 ? (
            <div className={`text-center mt-8 ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
              <MessageCircle className="h-10 w-10 mx-auto mb-3 opacity-50" />
              <p className="text-sm">Ask questions about your articles</p>
              <p className={`text-xs mt-2 ${isDarkMode ? 'text-slate-600' : 'text-slate-400'}`}>
                Try: "Summarize the main topics" or "What are the key insights?"
              </p>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[85%] p-3 rounded-lg text-sm ${
                    message.role === "user" 
                      ? "bg-violet-600 text-white" 
                      : isDarkMode 
                        ? "bg-slate-700 text-slate-200" 
                        : "bg-white text-slate-800 border border-slate-200"
                  }`}
                >
                  {message.role === "user" ? (
                    <p className="whitespace-pre-wrap">{renderTextWithLinks(message.content, true)}</p>
                  ) : (
                    <div className="prose prose-sm max-w-none dark:prose-invert">
                      {renderMarkdown(message.content)}
                    </div>
                  )}
                  <p className={`text-xs mt-1 ${
                    message.role === "user" 
                      ? "text-violet-200" 
                      : isDarkMode ? "text-slate-500" : "text-slate-400"
                  }`}>
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))
          )}
          {isLoading && messages[messages.length - 1]?.content === "" && (
            <div className="flex justify-start">
              <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-slate-700' : 'bg-white border border-slate-200'}`}>
                <div className="flex items-center gap-1">
                  <span className={`text-xs ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Thinking</span>
                  <div className="flex gap-1">
                    <div className={`w-1 h-1 rounded-full animate-bounce ${isDarkMode ? 'bg-slate-500' : 'bg-slate-400'}`} style={{animationDelay: '0ms'}} />
                    <div className={`w-1 h-1 rounded-full animate-bounce ${isDarkMode ? 'bg-slate-500' : 'bg-slate-400'}`} style={{animationDelay: '150ms'}} />
                    <div className={`w-1 h-1 rounded-full animate-bounce ${isDarkMode ? 'bg-slate-500' : 'bg-slate-400'}`} style={{animationDelay: '300ms'}} />
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className={`p-4 border-t flex-shrink-0 ${isDarkMode ? 'border-slate-700' : 'border-slate-200'}`}>
          <div className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about your articles..."
              disabled={isLoading}
              className={`flex-1 ${
                isDarkMode 
                  ? 'bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-500' 
                  : 'bg-white border-slate-300 text-slate-900 placeholder:text-slate-400'
              }`}
            />
            <Button 
              type="submit" 
              disabled={isLoading || !input.trim()}
              className="bg-violet-600 hover:bg-violet-700 text-white"
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Send"}
            </Button>
          </div>
        </form>
      </div>

      {/* Info Modal */}
      {isInfoOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <div 
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setIsInfoOpen(false)}
          />
          
          {/* Modal Content */}
          <div className={`relative w-full max-w-2xl mx-4 rounded-2xl shadow-2xl overflow-hidden ${
            isDarkMode 
              ? 'bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700' 
              : 'bg-white border border-slate-200'
          }`}>
            {/* Header */}
            <div className={`flex items-center justify-between p-6 border-b ${
              isDarkMode ? 'border-slate-700' : 'border-slate-200'
            }`}>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600">
                  <Globe className="h-5 w-5 text-white" />
                </div>
                <h2 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>
                  About UK AI News
                </h2>
              </div>
              <Button
                onClick={() => setIsInfoOpen(false)}
                variant="ghost"
                size="icon"
                className={isDarkMode ? 'text-slate-400 hover:text-white hover:bg-slate-700' : 'text-slate-500 hover:text-slate-900'}
              >
                <X className="h-5 w-5" />
              </Button>
            </div>
            
            {/* Body */}
            <div className={`p-6 space-y-4 max-h-[60vh] overflow-y-auto ${
              isDarkMode ? 'text-slate-300' : 'text-slate-600'
            }`}>
              <p className="leading-relaxed">
                <strong className={isDarkMode ? 'text-white' : 'text-slate-900'}>UK AI News</strong> is an automated news aggregator that collects and curates articles about artificial intelligence, machine learning, and related technologies with a focus on the United Kingdom.
              </p>
              
              <div className={`p-4 rounded-xl ${isDarkMode ? 'bg-slate-800/50' : 'bg-slate-50'}`}>
                <h3 className={`font-semibold mb-2 ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>
                  🔍 How It Works
                </h3>
                <ul className="space-y-2 text-sm">
                  <li>• Articles are automatically discovered weekly using targeted search queries</li>
                  <li>• Each article is summarized using AI to extract key insights</li>
                  <li>• Sentiment analysis classifies articles as positive, neutral, or negative</li>
                  <li>• Vector embeddings enable semantic search through the AI chat assistant</li>
                </ul>
              </div>
              
              <div className={`p-4 rounded-xl ${isDarkMode ? 'bg-slate-800/50' : 'bg-slate-50'}`}>
                <h3 className={`font-semibold mb-2 ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>
                  💬 Chat Assistant
                </h3>
                <p className="text-sm">
                  Use the chat feature to ask questions about the collected articles. The AI assistant uses retrieval-augmented generation (RAG) to find relevant articles and provide informed answers based on the content.
                </p>
              </div>
              
              <div className={`p-4 rounded-xl ${isDarkMode ? 'bg-slate-800/50' : 'bg-slate-50'}`}>
                <h3 className={`font-semibold mb-2 ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>
                  🛠️ Technology Stack
                </h3>
                <div className="flex flex-wrap gap-2 text-sm">
                  <span className={`px-2 py-1 rounded-md ${isDarkMode ? 'bg-slate-700 text-slate-300' : 'bg-slate-200 text-slate-700'}`}>Next.js</span>
                  <span className={`px-2 py-1 rounded-md ${isDarkMode ? 'bg-slate-700 text-slate-300' : 'bg-slate-200 text-slate-700'}`}>FastAPI</span>
                  <span className={`px-2 py-1 rounded-md ${isDarkMode ? 'bg-slate-700 text-slate-300' : 'bg-slate-200 text-slate-700'}`}>PostgreSQL</span>
                  <span className={`px-2 py-1 rounded-md ${isDarkMode ? 'bg-slate-700 text-slate-300' : 'bg-slate-200 text-slate-700'}`}>pgvector</span>
                  <span className={`px-2 py-1 rounded-md ${isDarkMode ? 'bg-slate-700 text-slate-300' : 'bg-slate-200 text-slate-700'}`}>Azure OpenAI</span>
                  <span className={`px-2 py-1 rounded-md ${isDarkMode ? 'bg-slate-700 text-slate-300' : 'bg-slate-200 text-slate-700'}`}>Vercel</span>
                </div>
              </div>
              
              <p className={`text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                Built by <a href="https://github.com/r-leyshon" target="_blank" rel="noopener noreferrer" className="text-violet-500 hover:underline">@r-leyshon</a> • Articles are updated weekly via automated workflows.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
