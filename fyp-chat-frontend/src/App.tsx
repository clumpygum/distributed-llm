import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, ChevronDown, Moon, Sun } from 'lucide-react';
import { ChatMessage } from './components/ChatMessage';
import { TypingIndicator } from './components/TypingIndicator';

// 1. Update Message Interface to include Device Info
export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  device?: string;       // "nano" or "orin"
  reasoning?: string;    // "tokens=500 > threshold"
}

type RoutingAlgorithm = 'token-counting' | 'semantic' | 'hybrid' | 'heuristic';

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
    "Hi",
    "What is the capital of France?",
    "Summarize the history of the Roman Empire in detail.",
    "Write a Python script to sort a list.",
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => { scrollToBottom(); }, [messages, isTyping]);

  // --- THE NEW API CONNECTION ---
  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    setShowSampleQueries(false);

    // 1. Add User Message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    try {
      // 2. Call Python Backend
      // Ensure your backend is running on port 5000
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: userMessage.text,
          strategy: routingAlgorithm // Send selected dropdown value
        }),
      });

      const data = await response.json();

      // 3. Add Bot Response with Metadata
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: data.reply,
        sender: 'bot',
        timestamp: new Date(),
        device: data.device,       // e.g. "orin"
        reasoning: data.reasoning  // e.g. "Sim=0.85"
      };

      setMessages((prev) => [...prev, botResponse]);

    } catch (error) {
      console.error("API Error:", error);
      const errorMsg: Message = {
        id: Date.now().toString(),
        text: "Error: Could not connect to Jetson cluster. Is 'app.py' running?",
        sender: 'bot',
        timestamp: new Date(),
        device: "error"
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  // ... (Keep your handlers for Dropdown, KeyPress, etc. exactly the same) ...
  const handleSampleQueryClick = (query: string) => {
    setInputValue(query);
    // Optional: auto-send
    // setTimeout(() => handleSendMessage(), 100); 
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const toggleDarkMode = () => setIsDarkMode(!isDarkMode);

  // ... (Keep your Return JSX structure exactly the same) ...
  
  return (
    <div className={`min-h-screen ${isDarkMode ? 'bg-gray-900' : 'bg-gradient-to-br from-blue-50 to-cyan-50'} flex items-center justify-center p-4`}>
       {/* ... Your existing JSX Layout ... */}
       {/* Just ensure <ChatMessage> receives the new message object with 'device' */}
       
       <div className={`w-full max-w-4xl h-[600px] ${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-2xl shadow-2xl flex flex-col overflow-hidden`}>
         {/* ... Header ... */}
         <div className={`${isDarkMode ? 'bg-gradient-to-r from-gray-700 to-gray-600' : 'bg-gradient-to-r from-blue-600 to-cyan-600'} text-white p-6 flex items-center justify-between`}>
             {/* ... Dropdown & Title ... */}
             <div className="flex items-center gap-3">
                <div className="bg-white/20 p-2 rounded-full"><Bot className="w-6 h-6" /></div>
                <div>
                  <h1 className="text-xl font-semibold">Edge Router Demo</h1>
                  <p className="text-sm opacity-90">Nano (Light) vs Orin (Heavy)</p>
                </div>
             </div>
             {/* ... (Keep Dropdown Code) ... */}
             <div className="relative" ref={dropdownRef}>
                  <button onClick={() => setIsDropdownOpen(!isDropdownOpen)} className="flex items-center gap-2 bg-white text-gray-800 px-5 py-2.5 rounded-lg text-sm font-medium shadow-lg cursor-pointer">
                    <span>{routingAlgorithm}</span> <ChevronDown className="w-4 h-4" />
                  </button>
                  {isDropdownOpen && (
                    <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl z-50">
                      {['token-counting', 'semantic', 'hybrid', 'heuristic'].map(algo => (
                        <div key={algo} onClick={() => { setRoutingAlgorithm(algo as any); setIsDropdownOpen(false); }} className="px-5 py-3 hover:bg-gray-50 cursor-pointer text-gray-800">
                          {algo}
                        </div>
                      ))}
                    </div>
                  )}
             </div>
         </div>

         {/* Messages Area */}
         <div className={`flex-1 overflow-y-auto p-6 space-y-4 ${isDarkMode ? 'bg-gray-800' : ''}`}>
           {messages.map((message) => (
             // Pass message to your ChatMessage component
             <ChatMessage key={message.id} message={message} isDarkMode={isDarkMode} />
           ))}
           {isTyping && <TypingIndicator isDarkMode={isDarkMode} />}
           <div ref={messagesEndRef} />
         </div>

         {/* Input Area */}
         <div className={`p-4 border-t ${isDarkMode ? 'bg-gray-700 border-gray-600' : 'border-gray-200 bg-gray-50'}`}>
            <div className="flex gap-2">
              <input 
                value={inputValue} 
                onChange={e => setInputValue(e.target.value)} 
                onKeyPress={handleKeyPress}
                className="flex-1 px-4 py-3 border rounded-lg"
                placeholder="Ask something..."
              />
              <button onClick={handleSendMessage} className="bg-blue-600 text-white px-6 py-3 rounded-lg"><Send className="w-4 h-4" /></button>
            </div>
         </div>
       </div>
    </div>
  );
}