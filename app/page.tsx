"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Loader2, Play, MessageCircle, Globe, Calendar, ExternalLink, ChevronDown, ChevronUp, Settings } from "lucide-react"
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
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"



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
        }
      }

             // Update status with combined results
       setCrawlStatus(`Search completed! Found ${totalArticlesAdded} new articles from ${regionNames} using DuckDuckGo.`)
        toast({
         title: "Multi-Region Search Completed",
         description: `Successfully found and added ${totalArticlesAdded} new articles from ${selectedRegions.length} region(s).`,
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
    <div className="h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4 overflow-hidden">
      <div className="max-w-7xl mx-auto h-full flex flex-col">
        {/* Header */}
        <div className="mb-6 flex-shrink-0">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">AI-Powered News Discovery Assistant</h1>
          <p className="text-gray-600">Search multiple regions with advanced operators, discover articles, and chat with your curated collection using DuckDuckGo</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
          {/* Control Panel */}
          <div className="lg:col-span-1 flex flex-col min-h-0">
            <Card className="flex-1 flex flex-col min-h-0">
              <CardHeader className="flex-shrink-0">
                <CardTitle className="flex items-center gap-2">
                  <Globe className="h-5 w-5" />
                  Article Discovery
                </CardTitle>
              </CardHeader>
              <CardContent className="flex-1 overflow-y-auto min-h-0">
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
                  
                  {/* Advanced Search Toggle */}
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowAdvancedSearch(!showAdvancedSearch)}
                    className="w-full justify-between text-xs"
                  >
                    <span className="flex items-center gap-1">
                      <Settings className="h-3 w-3" />
                      Advanced Search Options
                    </span>
                    {showAdvancedSearch ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                  </Button>
                  
                  {/* Advanced Search Fields */}
                  {showAdvancedSearch && (
                    <div className="space-y-3 pt-2 border-t">
                      <div className="grid grid-cols-1 gap-3">
                        <div>
                          <label className="block text-xs font-medium text-gray-700 mb-1">
                            Must Include (+)
                          </label>
                          <Input
                            value={advancedSearch.requiredTerms}
                            onChange={(e) => setAdvancedSearch(prev => ({...prev, requiredTerms: e.target.value}))}
                            placeholder="AI, justice (comma-separated)"
                            className="text-xs"
                            disabled={isCrawling}
                          />
                        </div>
                        
                        <div>
                          <label className="block text-xs font-medium text-gray-700 mb-1">
                            Must Exclude (-)
                          </label>
                          <Input
                            value={advancedSearch.excludedTerms}
                            onChange={(e) => setAdvancedSearch(prev => ({...prev, excludedTerms: e.target.value}))}
                            placeholder="sports, entertainment (comma-separated)"
                            className="text-xs"
                            disabled={isCrawling}
                          />
                        </div>
                        
                        <div>
                          <label className="block text-xs font-medium text-gray-700 mb-1">
                            Exact Phrase
                          </label>
                          <Input
                            value={advancedSearch.exactPhrase}
                            onChange={(e) => setAdvancedSearch(prev => ({...prev, exactPhrase: e.target.value}))}
                            placeholder="artificial intelligence law"
                            className="text-xs"
                            disabled={isCrawling}
                          />
                        </div>
                        
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block text-xs font-medium text-gray-700 mb-1">
                              File Type
                            </label>
                            <select
                              value={advancedSearch.fileType}
                              onChange={(e) => setAdvancedSearch(prev => ({...prev, fileType: e.target.value}))}
                              className="w-full px-2 py-1 text-xs border border-gray-300 rounded-md bg-white"
                              disabled={isCrawling}
                            >
                              <option value="">Any</option>
                              <option value="pdf">PDF</option>
                              <option value="doc">DOC</option>
                              <option value="docx">DOCX</option>
                              <option value="html">HTML</option>
                            </select>
                          </div>
                          
                          <div>
                            <label className="block text-xs font-medium text-gray-700 mb-1">
                              In Title
                            </label>
                            <Input
                              value={advancedSearch.inTitle}
                              onChange={(e) => setAdvancedSearch(prev => ({...prev, inTitle: e.target.value}))}
                              placeholder="justice"
                              className="text-xs"
                              disabled={isCrawling}
                            />
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block text-xs font-medium text-gray-700 mb-1">
                              Include Site
                            </label>
                            <Input
                              value={advancedSearch.includeSite}
                              onChange={(e) => setAdvancedSearch(prev => ({...prev, includeSite: e.target.value}))}
                              placeholder="gov.uk"
                              className="text-xs"
                              disabled={isCrawling}
                            />
                          </div>
                          
                          <div>
                            <label className="block text-xs font-medium text-gray-700 mb-1">
                              Exclude Site
                            </label>
                            <Input
                              value={advancedSearch.excludeSite}
                              onChange={(e) => setAdvancedSearch(prev => ({...prev, excludeSite: e.target.value}))}
                              placeholder="reddit.com"
                              className="text-xs"
                              disabled={isCrawling}
                            />
                          </div>
                        </div>
                        
                        <div>
                          <label className="block text-xs font-medium text-gray-700 mb-1">
                            In URL
                          </label>
                          <Input
                            value={advancedSearch.inUrl}
                            onChange={(e) => setAdvancedSearch(prev => ({...prev, inUrl: e.target.value}))}
                            placeholder="news"
                            className="text-xs"
                            disabled={isCrawling}
                          />
                        </div>
                        
                                                 <div>
                           <label className="block text-xs font-medium text-gray-700 mb-2">
                             Search Regions
                           </label>
                           <div className="grid grid-cols-2 gap-2 border rounded p-2">
                            {availableRegions.map((region) => (
                              <label key={region.code} className="flex items-center gap-2 text-xs cursor-pointer hover:bg-gray-50 p-1 rounded">
                                <input
                                  type="checkbox"
                                  checked={advancedSearch.regions.includes(region.code)}
                                  onChange={(e) => {
                                    const newRegions = e.target.checked
                                      ? [...advancedSearch.regions, region.code]
                                      : advancedSearch.regions.filter(r => r !== region.code)
                                    
                                    // Prevent deselecting all regions
                                    if (newRegions.length > 0) {
                                      setAdvancedSearch(prev => ({...prev, regions: newRegions}))
                                    }
                                  }}
                                  className="w-3 h-3"
                                  disabled={isCrawling}
                                />
                                <span className="flex items-center gap-1">
                                  <span>{region.flag}</span>
                                  <span>{region.name}</span>
                                </span>
                              </label>
                            ))}
                          </div>
                          <div className="flex items-center justify-between mt-2">
                            <p className="text-xs text-gray-500">
                              Selected: {advancedSearch.regions.length} region(s)
                            </p>
                            <div className="flex gap-1">
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => setAdvancedSearch(prev => ({...prev, regions: ["uk-en"]}))}
                                className="text-xs px-2 py-1 h-6"
                                disabled={isCrawling}
                              >
                                🇬🇧 UK Only
                              </Button>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => setAdvancedSearch(prev => ({...prev, regions: ["uk-en", "us-en"]}))}
                                className="text-xs px-2 py-1 h-6"
                                disabled={isCrawling}
                              >
                                🇬🇧🇺🇸 UK+US
                              </Button>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => setAdvancedSearch(prev => ({...prev, regions: ["uk-en", "us-en", "de-de", "fr-fr", "it-it", "es-es", "nl-nl"]}))}
                                className="text-xs px-2 py-1 h-6"
                                disabled={isCrawling}
                              >
                                🌍 EU+US
                              </Button>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded space-y-1">
                        <div><strong>Query:</strong> {buildSearchQuery() || "Enter search criteria above"}</div>
                        <div><strong>Regions:</strong> {advancedSearch.regions.map(code => 
                          availableRegions.find(r => r.code === code)?.flag + " " + availableRegions.find(r => r.code === code)?.name
                        ).join(", ")}</div>
                      </div>
                      
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => setAdvancedSearch({
                          requiredTerms: "",
                          excludedTerms: "",
                          exactPhrase: "",
                          fileType: "",
                          includeSite: "",
                          excludeSite: "",
                          inTitle: "",
                          inUrl: "",
                          regions: ["uk-en"]
                        })}
                        className="w-full text-xs"
                        disabled={isCrawling}
                      >
                        Clear Advanced Options
                      </Button>
                    </div>
                  )}
                  <Button onClick={handleCrawl} disabled={isCrawling || !buildSearchQuery().trim()} className="w-full">
                  {isCrawling ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Discovering Articles...
                    </>
                                      ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Discover Articles
                      </>
                    )}
                </Button>
                  {crawlStatus && <p className="text-sm text-gray-600 mt-2">{crawlStatus}</p>}
                </div>
              </CardContent>
            </Card>

            {/* Articles List - Only show when advanced search is not expanded */}
            {!showAdvancedSearch && (
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
                          {article.region && (
                            <span className="text-xs text-gray-500 flex items-center gap-1">
                              {getRegionInfo(article.region)?.flag} {getRegionInfo(article.region)?.name}
                            </span>
                          )}
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
            )}
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
