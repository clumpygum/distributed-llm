import { useState, useRef, useEffect } from 'react';
import { Send, Bot, ChevronDown, Moon, Sun, Info } from 'lucide-react';
import { ChatMessage } from './components/ChatMessage';
import { TypingIndicator } from './components/TypingIndicator';

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  device?: string;        // "nano" | "orin" | "error"
  reasoning?: string;     // routing_reasoning from backend
  method?: string;        // routing_method e.g. "hybrid", "heuristic_cached"
  confidence?: number;    // 0.0–1.0
  cacheHit?: boolean;     // true if served from routing/response cache
  tokens?: number;        // response token count
}

type RoutingAlgorithm = 'token-counting' | 'semantic' | 'hybrid' | 'heuristic' | 'perf';

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "System Online. Connected to Distributed Edge Cluster (Nano + Orin). Select a routing strategy and ask a question.",
      sender: 'bot',
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [routingAlgorithm, setRoutingAlgorithm] = useState<RoutingAlgorithm>('hybrid');
  const [showSampleQueries, setShowSampleQueries] = useState(true);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Stable session ID for this browser tab — lets the backend maintain
  // per-session conversation history correctly across multiple messages.
  const sessionId = useRef<string>(`session-${Date.now()}-${Math.random().toString(36).slice(2)}`);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const sampleQueries = [
    "What causes osteoporosis?",
    "What causes a headache?",
    "Give me the symptoms of IBS.",
    "I'm having trouble sleeping. Are there any ways to help?",
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const toggleDarkMode = () => setIsDarkMode((prev) => !prev);

  const handleSampleQueryClick = (query: string) => {
    setInputValue(query);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    setShowSampleQueries(false);

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const currentInput = inputValue;
    setInputValue('');
    setIsTyping(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: currentInput,
          strategy: routingAlgorithm,
          session_id: sessionId.current,   // so backend maintains per-tab history
        }),
      });

      const data = await response.json();

      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: data.reply ?? "No response received.",
        sender: 'bot',
        timestamp: new Date(),
        device:     data.device,
        reasoning:  data.reasoning,
        method:     data.method,           // new
        confidence: data.confidence,       // new
        cacheHit:   data.cache_hit,        // new
        tokens:     data.tokens,           // new
      };

      setMessages((prev) => [...prev, botResponse]);

    } catch (error) {
      console.error("API Error:", error);
      const errorMsg: Message = {
        id: Date.now().toString(),
        text: "Error: Could not connect to Jetson cluster. Is app.py running?",
        sender: 'bot',
        timestamp: new Date(),
        device: 'error',
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const algorithmLabels: Record<RoutingAlgorithm, string> = {
    'token-counting': 'Token Counting',
    'semantic':       'Semantic',
    'hybrid':         'Hybrid',
    'heuristic':      'Heuristic',
    'perf':           'Performance Aware',
  };

  return (
    <div className={`min-h-screen flex items-center justify-center p-4 ${isDarkMode ? 'bg-gray-900' : 'bg-gray-100'}`}>
      <div className={`w-full max-w-3xl flex flex-col rounded-2xl shadow-2xl overflow-hidden ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`} style={{ height: '90vh' }}>

        {/* Header */}
        <div className={`flex items-center justify-between px-6 py-4 border-b ${isDarkMode ? 'bg-gray-700 border-gray-600' : 'border-gray-200'}`}>
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${isDarkMode ? 'bg-blue-900 text-blue-300' : 'bg-blue-100 text-blue-600'}`}>
              <Bot className="w-5 h-5" />
            </div>
            <div>
              <h1 className={`font-semibold text-lg ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Edge Cluster Chat</h1>
              <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Nano + Orin · {algorithmLabels[routingAlgorithm]}</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Strategy dropdown */}
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setIsDropdownOpen((o) => !o)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium shadow border transition-all ${isDarkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-200 text-gray-700'}`}
              >
                {algorithmLabels[routingAlgorithm]}
                <ChevronDown className={`w-4 h-4 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
              </button>

              {isDropdownOpen && (
                <div className={`absolute right-0 top-full mt-1 w-52 rounded-lg shadow-xl border z-50 overflow-hidden ${isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'}`}>
                  {(Object.keys(algorithmLabels) as RoutingAlgorithm[]).map((key) => (
                    <div
                      key={key}
                      onClick={() => { setRoutingAlgorithm(key); setIsDropdownOpen(false); }}
                      className={`px-5 py-3 text-sm font-medium cursor-pointer transition-colors ${
                        routingAlgorithm === key
                          ? isDarkMode ? 'bg-blue-800 text-blue-200' : 'bg-blue-50 text-blue-700'
                          : isDarkMode ? 'text-gray-200 hover:bg-gray-600' : 'text-gray-700 hover:bg-gray-50'
                      }`}
                    >
                      {algorithmLabels[key]}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Dark mode toggle */}
            <button
              onClick={toggleDarkMode}
              className={`flex items-center justify-center px-3 py-2 rounded-lg text-sm shadow border transition-all ${isDarkMode ? 'bg-gray-600 border-gray-500 text-yellow-300' : 'bg-white border-gray-200 text-gray-700'}`}
            >
              {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Perf-aware info banner */}
        {routingAlgorithm === 'perf' && (
          <div className={`px-6 py-3 border-b flex items-start gap-3 ${isDarkMode ? 'bg-blue-900/20 border-blue-800 text-blue-200' : 'bg-blue-50 border-blue-200 text-blue-800'}`}>
            <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <p className="text-sm">
              <span className="font-semibold">Performance Aware Routing:</span> Routes based on live device latency; device may vary per message.
            </p>
          </div>
        )}

        {/* Messages */}
        <div className={`flex-1 min-h-0 overflow-y-auto p-6 ${isDarkMode ? 'bg-gray-800' : ''}`}>
          <div className="flex flex-col justify-end min-h-full space-y-4">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} isDarkMode={isDarkMode} />
          ))}

          {/* Sample queries */}
          {showSampleQueries && (
            <div className="space-y-3">
              <p className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Try asking:</p>
              <div className="grid grid-cols-2 gap-2">
                {sampleQueries.map((query, index) => (
                  <button
                    key={index}
                    onClick={() => handleSampleQueryClick(query)}
                    className={`text-left px-4 py-3 rounded-lg border text-sm transition-colors ${
                      isDarkMode
                        ? 'bg-gray-700 hover:bg-gray-600 border-gray-600 text-blue-300'
                        : 'bg-blue-50 hover:bg-blue-100 border-blue-200 text-blue-700'
                    }`}
                  >
                    {query}
                  </button>
                ))}
              </div>
            </div>
          )}

          {isTyping && <TypingIndicator isDarkMode={isDarkMode} />}
          <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input area */}
        <div className={`p-4 border-t ${isDarkMode ? 'bg-gray-700 border-gray-600' : 'border-gray-200 bg-gray-50'}`}>
          <div className="flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask something..."
              className={`flex-1 px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                isDarkMode ? 'bg-gray-600 border-gray-500 text-white placeholder-gray-400' : 'border-gray-300'
              }`}
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isTyping}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <Send className="w-4 h-4" />
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}