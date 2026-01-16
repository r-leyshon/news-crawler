"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Loader2, Play, MessageCircle, Globe, Calendar, ExternalLink, ChevronDown, ChevronUp, Settings, X, PanelRightOpen, PanelRightClose, Search, Filter, Trash2, Sun, Moon } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { availableRegions, getRegionInfo } from "@/lib/regions"

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
}

export default function ArticleAssistant() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [searchKeywords, setSearchKeywords] = useState("")
  const [showAdvancedSearch, setShowAdvancedSearch] = useState(false)
  const [advancedSearch, setAdvancedSearch] = useState({
    requiredTerms: "",
    excludedTerms: "",
    exactPhrase: "",
    fileType: "",
    includeSite: "",
    excludeSite: "",
    inTitle: "",
    inUrl: "",
    regions: ["uk-en"] // Default to UK
  })
  const [isLoading, setIsLoading] = useState(false)
  const [isCrawling, setIsCrawling] = useState(false)
  const [articles, setArticles] = useState<Article[]>([])
  const [crawlStatus, setCrawlStatus] = useState("")
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState("")
  const [filterType, setFilterType] = useState<"all" | "full" | "link_only">("all")
  const [regionFilter, setRegionFilter] = useState<"all" | string>("all")
  const [isDarkMode, setIsDarkMode] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

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

  // Function to build search query with advanced operators
  const buildSearchQuery = () => {
    const parts = []
    
    // Basic keywords
    if (searchKeywords.trim()) {
      parts.push(searchKeywords.trim())
    }
    
    // Required terms (+ operator)
    if (advancedSearch.requiredTerms.trim()) {
      const required = advancedSearch.requiredTerms.split(',').map(term => `+${term.trim()}`).join(' ')
      parts.push(required)
    }
    
    // Excluded terms (- operator) 
    if (advancedSearch.excludedTerms.trim()) {
      const excluded = advancedSearch.excludedTerms.split(',').map(term => `-${term.trim()}`).join(' ')
      parts.push(excluded)
    }
    
    // Exact phrase (quotes)
    if (advancedSearch.exactPhrase.trim()) {
      parts.push(`"${advancedSearch.exactPhrase.trim()}"`)
    }
    
    // File type
    if (advancedSearch.fileType.trim()) {
      parts.push(`filetype:${advancedSearch.fileType.trim()}`)
    }
    
    // Include site
    if (advancedSearch.includeSite.trim()) {
      parts.push(`site:${advancedSearch.includeSite.trim()}`)
    }
    
    // Exclude site
    if (advancedSearch.excludeSite.trim()) {
      parts.push(`-site:${advancedSearch.excludeSite.trim()}`)
    }
    
    // In title
    if (advancedSearch.inTitle.trim()) {
      parts.push(`intitle:${advancedSearch.inTitle.trim()}`)
    }
    
    // In URL
    if (advancedSearch.inUrl.trim()) {
      parts.push(`inurl:${advancedSearch.inUrl.trim()}`)
    }
    
    return parts.join(' ')
  }

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

  const handleCrawl = async () => {
    const searchQuery = buildSearchQuery()
    
    if (!searchQuery.trim()) {
      toast({
        title: "Search Keywords Required",
        description: "Please enter search keywords or advanced search criteria before discovering articles.",
        variant: "destructive",
      })
      return
    }

    setIsCrawling(true)
    const selectedRegions = advancedSearch.regions
    const regionNames = selectedRegions.map(code => 
      availableRegions.find(r => r.code === code)?.name || code
    ).join(", ")
    
    setCrawlStatus(`Starting search in ${regionNames} with: "${searchQuery}"`)

    try {
      // If multiple regions selected, we'll search each region separately
      let totalArticlesAdded = 0
      let totalArticlesFiltered = 0
      const articlesPerRegion = Math.ceil(10 / selectedRegions.length)
      
      for (const region of selectedRegions) {
        setCrawlStatus(`Searching ${availableRegions.find(r => r.code === region)?.name || region}...`)
        
      const response = await fetch(`${API_BASE}/crawl`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
            keywords: [searchQuery],
            max_articles: articlesPerRegion,
            region: region
        }),
      })

      if (response.ok) {
        const data = await response.json()
          totalArticlesAdded += data.articles_added
          totalArticlesFiltered += (data.articles_filtered || 0)
        }
      }

      // Update status with combined results
      const filterMessage = totalArticlesFiltered > 0 ? ` (${totalArticlesFiltered} filtered for content)` : ""
      setCrawlStatus(`Search completed! Found ${totalArticlesAdded} new articles from ${regionNames}${filterMessage}.`)
      toast({
        title: "Multi-Region Search Completed",
        description: `Successfully found and added ${totalArticlesAdded} new articles from ${selectedRegions.length} region(s).${filterMessage}`,
      })
      fetchArticles()
    } catch (error) {
      setCrawlStatus("Article discovery failed. Please try again.")
      toast({
        title: "Article Discovery Failed",
        description: "There was an error during the article discovery process.",
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
              <h1 className={`text-3xl font-bold mb-1 tracking-tight ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>
                News Discovery
              </h1>
              <p className={`text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                Search, discover, and analyze articles with AI
              </p>
            </div>
            <div className="flex items-center gap-2">
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
                variant="outline"
                className={`flex items-center gap-2 ${
                  isDarkMode 
                    ? 'bg-slate-800/50 border-slate-600 text-slate-200 hover:bg-slate-700 hover:text-white' 
                    : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-100'
                }`}
              >
                {isChatOpen ? (
                  <>
                    <PanelRightClose className="h-4 w-4" />
                    Hide Chat
                  </>
                ) : (
                  <>
                    <PanelRightOpen className="h-4 w-4" />
                    Open Chat
                    {messages.length > 0 && (
                      <Badge className="ml-1 bg-emerald-500 text-white text-xs">{messages.length}</Badge>
                    )}
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* Search Controls */}
        <Card className={`mb-6 flex-shrink-0 ${
          isDarkMode 
            ? 'bg-slate-800/50 border-slate-700' 
            : 'bg-white border-slate-200 shadow-sm'
        }`}>
          <CardContent className="pt-4">
            <div className="flex flex-col gap-4">
              <div className="flex gap-3">
                <div className="flex-1">
                  <Input
                    value={searchKeywords}
                    onChange={(e) => setSearchKeywords(e.target.value)}
                    placeholder="Search keywords (e.g., artificial intelligence, machine learning)"
                    disabled={isCrawling}
                    className={`${
                      isDarkMode 
                        ? 'bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-500' 
                        : 'bg-slate-50 border-slate-300 text-slate-900 placeholder:text-slate-400'
                    }`}
                  />
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => setShowAdvancedSearch(!showAdvancedSearch)}
                  className={`${isDarkMode ? 'text-slate-400 hover:text-white hover:bg-slate-700' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'}`}
                >
                  <Settings className="h-4 w-4 mr-2" />
                  Advanced
                  {showAdvancedSearch ? <ChevronUp className="h-4 w-4 ml-1" /> : <ChevronDown className="h-4 w-4 ml-1" />}
                </Button>
                <Button 
                  onClick={handleCrawl} 
                  disabled={isCrawling || !buildSearchQuery().trim()} 
                  className="bg-emerald-600 hover:bg-emerald-700 text-white"
                >
                  {isCrawling ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Searching...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Discover
                    </>
                  )}
                </Button>
              </div>
              
              {/* Advanced Search Fields */}
              {showAdvancedSearch && (
                <div className={`grid grid-cols-2 md:grid-cols-4 gap-3 pt-3 border-t ${isDarkMode ? 'border-slate-700' : 'border-slate-200'}`}>
                  <div>
                    <label className={`block text-xs font-medium mb-1 ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>Must Include</label>
                    <Input
                      value={advancedSearch.requiredTerms}
                      onChange={(e) => setAdvancedSearch(prev => ({...prev, requiredTerms: e.target.value}))}
                      placeholder="AI, justice"
                      className={`text-xs ${isDarkMode ? 'bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-500' : 'bg-slate-50 border-slate-300 text-slate-900 placeholder:text-slate-400'}`}
                      disabled={isCrawling}
                    />
                  </div>
                  <div>
                    <label className={`block text-xs font-medium mb-1 ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>Exclude</label>
                    <Input
                      value={advancedSearch.excludedTerms}
                      onChange={(e) => setAdvancedSearch(prev => ({...prev, excludedTerms: e.target.value}))}
                      placeholder="sports, ads"
                      className={`text-xs ${isDarkMode ? 'bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-500' : 'bg-slate-50 border-slate-300 text-slate-900 placeholder:text-slate-400'}`}
                      disabled={isCrawling}
                    />
                  </div>
                  <div>
                    <label className={`block text-xs font-medium mb-1 ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>Exact Phrase</label>
                    <Input
                      value={advancedSearch.exactPhrase}
                      onChange={(e) => setAdvancedSearch(prev => ({...prev, exactPhrase: e.target.value}))}
                      placeholder="machine learning"
                      className={`text-xs ${isDarkMode ? 'bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-500' : 'bg-slate-50 border-slate-300 text-slate-900 placeholder:text-slate-400'}`}
                      disabled={isCrawling}
                    />
                  </div>
                  <div>
                    <label className={`block text-xs font-medium mb-1 ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>Site Filter</label>
                    <Input
                      value={advancedSearch.includeSite}
                      onChange={(e) => setAdvancedSearch(prev => ({...prev, includeSite: e.target.value}))}
                      placeholder="gov.uk"
                      className={`text-xs ${isDarkMode ? 'bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-500' : 'bg-slate-50 border-slate-300 text-slate-900 placeholder:text-slate-400'}`}
                      disabled={isCrawling}
                    />
                  </div>
                  <div className="col-span-2 md:col-span-4">
                    <label className={`block text-xs font-medium mb-2 ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>Search Regions</label>
                    <div className="flex flex-wrap gap-2">
                      {availableRegions.map((region) => (
                        <label 
                          key={region.code} 
                          className={`flex items-center gap-1.5 text-xs cursor-pointer px-2 py-1 rounded border transition-colors ${
                            advancedSearch.regions.includes(region.code)
                              ? 'bg-emerald-600/20 border-emerald-500 text-emerald-600 dark:text-emerald-300'
                              : isDarkMode 
                                ? 'bg-slate-900/50 border-slate-600 text-slate-400 hover:border-slate-500'
                                : 'bg-slate-50 border-slate-300 text-slate-600 hover:border-slate-400'
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={advancedSearch.regions.includes(region.code)}
                            onChange={(e) => {
                              const newRegions = e.target.checked
                                ? [...advancedSearch.regions, region.code]
                                : advancedSearch.regions.filter(r => r !== region.code)
                              if (newRegions.length > 0) {
                                setAdvancedSearch(prev => ({...prev, regions: newRegions}))
                              }
                            }}
                            className="sr-only"
                            disabled={isCrawling}
                          />
                          <span>{region.flag}</span>
                          <span>{region.name}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              
              {crawlStatus && (
                <p className="text-sm text-emerald-500">{crawlStatus}</p>
              )}
            </div>
          </CardContent>
        </Card>

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
                {searchTerm || filterType !== "all" || regionFilter !== "all" ? "No matching articles" : "No articles yet"}
              </h3>
              <p className={`max-w-md ${isDarkMode ? 'text-slate-500' : 'text-slate-500'}`}>
                {searchTerm || filterType !== "all" || regionFilter !== "all"
                  ? "Try adjusting your filters to see more results." 
                  : "Enter search keywords above and click 'Discover' to find and save articles."
                }
              </p>
            </div>
          ) : (
            <div className="space-y-2 pb-4">
              {filteredArticles.map((article) => (
                <div 
                  key={article.id} 
                  className={`flex items-start gap-4 p-4 rounded-lg border transition-colors group ${
                    isDarkMode 
                      ? 'bg-slate-800/30 border-slate-700 hover:border-slate-600 hover:bg-slate-800/50' 
                      : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50 shadow-sm'
                  }`}
                >
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
                  </div>
                </div>
              ))}
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
            <MessageCircle className="h-5 w-5 text-emerald-500" />
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
                      ? "bg-emerald-600 text-white" 
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
                      ? "text-emerald-200" 
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
              className="bg-emerald-600 hover:bg-emerald-700 text-white"
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Send"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
