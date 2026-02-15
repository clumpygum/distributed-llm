import { useState, useRef, useEffect } from 'react';
import { Send, Bot, ChevronDown, Moon, Sun, Info } from 'lucide-react';
import { ChatMessage } from './components/ChatMessage';
import { TypingIndicator } from './components/TypingIndicator';

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  device?: string;       // "nano" or "orin"
  reasoning?: string;    // "tokens=500 > threshold"
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
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const sampleQueries = [
    "What causes osteoporosis?",
    "What causes a headache?",
    "Give me the symptoms of IBS.",
    "I’m having trouble sleeping. Are there any ways to help?",
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

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
      // Call Python Backend
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: currentInput,
          strategy: routingAlgorithm
        }),
      });

      const data = await response.json();

      // Add Bot Response with Metadata
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: data.reply,
        sender: 'bot',
        timestamp: new Date(),
        device: data.device,
        reasoning: data.reasoning
      };

      setMessages((prev) => [...prev, botResponse]);

    } catch (error) {
      console.error("API Error:", error);
      const errorMsg: Message = {
        id: Date.now().toString(),
        text: "Error: Could not connect to Jetson cluster. Is 'app.py' running on localhost:5000?",
        sender: 'bot',
        timestamp: new Date(),
        device: "error"
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSampleQueryClick = (query: string) => {
    setInputValue(query);
    setTimeout(() => {
      handleSendMessage();
    }, 100);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleDropdownClick = () => {
    setIsDropdownOpen(!isDropdownOpen);
  };

  const handleOutsideClick = (e: MouseEvent) => {
    if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
      setIsDropdownOpen(false);
    }
  };

  useEffect(() => {
    document.addEventListener('mousedown', handleOutsideClick);
    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
    };
  }, []);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div className={`min-h-screen ${isDarkMode ? 'bg-gray-900' : 'bg-gradient-to-br from-blue-50 to-cyan-50'} flex items-center justify-center p-4`}>
      <div className={`w-full max-w-4xl h-[600px] ${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-2xl shadow-2xl flex flex-col overflow-hidden`}>
        {/* Header */}
        <div className={`${isDarkMode ? 'bg-gradient-to-r from-gray-700 to-gray-600' : 'bg-gradient-to-r from-blue-600 to-cyan-600'} text-white p-6 flex items-center justify-between`}>
          <div className="flex items-center gap-3">
            <div className="bg-white/20 p-2 rounded-full">
              <Bot className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-semibold">MediBot</h1>
              <p className="text-sm opacity-90">Nano (Light) vs Orin (Heavy)</p>
            </div>
          </div>
          
          {/* Routing Algorithm Selector and Dark Mode Toggle */}
          <div className="flex items-center gap-2">
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={handleDropdownClick}
                className="flex items-center gap-2 bg-white text-gray-800 px-5 py-2.5 rounded-lg text-sm font-medium shadow-lg hover:shadow-xl transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-400 border border-gray-200"
              >
                <span>
                  {routingAlgorithm === 'token-counting' ? 'Token Counting' :
                   routingAlgorithm === 'heuristic' ? 'Heuristic' :
                   routingAlgorithm === 'semantic' ? 'Semantic' :
                   routingAlgorithm === 'perf' ? 'Performance Aware' : 'Hybrid'}
                </span>
                <ChevronDown className={`w-4 h-4 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
              </button>
              
              {isDropdownOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl border border-gray-200 overflow-hidden z-50">
                  <div
                    onClick={() => {
                      setRoutingAlgorithm('heuristic');
                      setIsDropdownOpen(false);
                    }}
                    className={`px-5 py-3 text-sm font-medium cursor-pointer transition-colors ${
                      routingAlgorithm === 'heuristic' ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    Heuristic
                  </div>
                  <div
                    onClick={() => {
                      setRoutingAlgorithm('token-counting');
                      setIsDropdownOpen(false);
                    }}
                    className={`px-5 py-3 text-sm font-medium cursor-pointer transition-colors ${
                      routingAlgorithm === 'token-counting' ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    Token Counting
                  </div>
                  <div
                    onClick={() => {
                      setRoutingAlgorithm('semantic');
                      setIsDropdownOpen(false);
                    }}
                    className={`px-5 py-3 text-sm font-medium cursor-pointer transition-colors ${
                      routingAlgorithm === 'semantic' ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    Semantic
                  </div>
                  <div
                    onClick={() => {
                      setRoutingAlgorithm('hybrid');
                      setIsDropdownOpen(false);
                    }}
                    className={`px-5 py-3 text-sm font-medium cursor-pointer transition-colors ${
                      routingAlgorithm === 'hybrid' ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    Hybrid
                  </div>
                  <div
                    onClick={() => {
                      setRoutingAlgorithm('perf');
                      setIsDropdownOpen(false);
                    }}
                    className={`px-5 py-3 text-sm font-medium cursor-pointer transition-colors ${
                      routingAlgorithm === 'perf' ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    Performance Aware
                  </div>
                </div>
              )}
            </div>
            
            {/* Dark Mode Toggle */}
            <button
              onClick={toggleDarkMode}
              className="flex items-center justify-center bg-white text-gray-800 px-5 py-2.5 rounded-lg text-sm font-medium shadow-lg hover:shadow-xl transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-400 border border-gray-200"
            >
              {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Routing Info Banner */}
        {routingAlgorithm === 'perf' && (
          <div className="px-6 py-3 border-b flex items-start gap-3 
            bg-blue-50 border-blue-200 text-blue-800 
            dark:bg-blue-900/20 dark:border-blue-800 dark:text-blue-200">
            
            <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm">
                <span className="font-semibold">Performance Aware Routing:</span> Routes based on live device performance; may vary per message.
              </p>
            </div>
          </div>
        )}

        {/* Messages Container */}
        <div className={`flex-1 overflow-y-auto p-6 space-y-4 ${isDarkMode ? 'bg-gray-800' : ''}`}>
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} isDarkMode={isDarkMode} />
          ))}
          
          {/* Sample Queries */}
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

        {/* Input Area */}
        <div className={`p-4 border-t ${isDarkMode ? 'bg-gray-700 border-gray-600' : 'border-gray-200 bg-gray-50'}`}>
          <div className="flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask something..."
              className={`flex-1 px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                isDarkMode 
                  ? 'bg-gray-600 border-gray-500 text-white placeholder-gray-400' 
                  : 'border-gray-300'
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