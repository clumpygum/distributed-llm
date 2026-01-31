import { useState, useRef, useEffect } from 'react';
import { MessageCircle, Send, Bot, User, Settings, ChevronDown, Moon, Sun } from 'lucide-react';
import { ChatMessage } from './components/ChatMessage';
import { TypingIndicator } from './components/TypingIndicator';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

type RoutingAlgorithm = 'token-counting' | 'semantic' | 'hybrid' | 'heuristic';

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm your medical enquiry assistant. I can help answer general health questions, provide information about symptoms, and offer wellness advice. How can I assist you today?",
      sender: 'bot',
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [routingAlgorithm, setRoutingAlgorithm] = useState<RoutingAlgorithm>('heuristic');
  const [showSampleQueries, setShowSampleQueries] = useState(true);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const sampleQueries = [
    "What should I do about my headache?",
    "How much exercise do I need?",
    "I have a fever, what should I do?",
    "Tips for better sleep?",
    "How to manage stress and anxiety?",
    "Tell me about healthy eating",
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // Knowledge base for routing
  const knowledgeBase = [
    {
      category: 'headache',
      keywords: ['headache', 'head pain', 'migraine', 'head ache', 'head hurts'],
      semanticConcepts: ['pain', 'head', 'discomfort'],
      response: "Headaches can have various causes including stress, dehydration, lack of sleep, or eye strain. For mild headaches, try resting in a quiet, dark room, staying hydrated, and using over-the-counter pain relievers if appropriate. If headaches are severe, persistent, or accompanied by other symptoms like vision changes or fever, please consult a healthcare provider."
    },
    {
      category: 'fever',
      keywords: ['fever', 'temperature', 'high temp', 'hot', 'burning up'],
      semanticConcepts: ['infection', 'temperature', 'illness'],
      response: "A fever is typically a sign that your body is fighting an infection. For adults, a temperature above 100.4°F (38°C) is considered a fever. Rest, stay hydrated, and use fever-reducing medications like acetaminophen or ibuprofen if needed. Seek medical attention if the fever exceeds 103°F (39.4°C), lasts more than 3 days, or is accompanied by severe symptoms."
    },
    {
      category: 'cold-flu',
      keywords: ['cold', 'flu', 'cough', 'congestion', 'runny nose', 'sneezing'],
      semanticConcepts: ['respiratory', 'infection', 'symptoms'],
      response: "Common cold and flu symptoms include cough, congestion, sore throat, and fatigue. Rest, stay hydrated, and consider using over-the-counter medications for symptom relief. Flu symptoms tend to be more severe than cold symptoms. If you have difficulty breathing, persistent chest pain, or severe symptoms, please seek medical care."
    },
    {
      category: 'sore-throat',
      keywords: ['sore throat', 'throat pain', 'throat hurts', 'swallowing pain'],
      semanticConcepts: ['throat', 'pain', 'swallowing'],
      response: "Sore throats are often caused by viral infections, but can also be bacterial (like strep throat). Try warm liquids, throat lozenges, and gargling with salt water. If the sore throat is severe, lasts more than a week, or you have difficulty swallowing or breathing, please see a doctor."
    },
    {
      category: 'exercise',
      keywords: ['exercise', 'workout', 'fitness', 'gym', 'physical activity', 'training'],
      semanticConcepts: ['health', 'activity', 'fitness'],
      response: "Regular exercise is essential for good health. Adults should aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week, plus muscle-strengthening activities twice a week. Start slowly if you're new to exercise and gradually increase intensity. Always consult a doctor before starting a new exercise program if you have health concerns."
    },
    {
      category: 'diet',
      keywords: ['diet', 'nutrition', 'eating', 'food', 'healthy eating', 'meal'],
      semanticConcepts: ['nutrition', 'food', 'health'],
      response: "A balanced diet includes plenty of fruits, vegetables, whole grains, lean proteins, and healthy fats. Try to limit processed foods, added sugars, and excessive sodium. Stay hydrated by drinking plenty of water. Everyone's nutritional needs are different, so consider consulting a registered dietitian for personalized advice."
    },
    {
      category: 'sleep',
      keywords: ['sleep', 'insomnia', 'tired', 'fatigue', 'rest', 'sleeping'],
      semanticConcepts: ['rest', 'fatigue', 'sleep'],
      response: "Adults typically need 7-9 hours of sleep per night. For better sleep: maintain a consistent sleep schedule, create a relaxing bedtime routine, keep your bedroom cool and dark, avoid screens before bed, and limit caffeine and alcohol. If you consistently have trouble sleeping, consider speaking with a healthcare provider about possible sleep disorders."
    },
    {
      category: 'mental-health',
      keywords: ['stress', 'anxiety', 'mental health', 'depression', 'worried', 'nervous'],
      semanticConcepts: ['mental', 'emotional', 'psychological'],
      response: "Managing stress is important for overall health. Try techniques like deep breathing, meditation, regular exercise, adequate sleep, and connecting with others. It's normal to feel stressed sometimes, but if anxiety or stress is interfering with daily life, please reach out to a mental health professional. Remember, seeking help is a sign of strength."
    },
    {
      category: 'medication',
      keywords: ['medication', 'medicine', 'prescription', 'pills', 'drugs', 'meds'],
      semanticConcepts: ['treatment', 'medicine', 'pharmaceutical'],
      response: "Always take medications as prescribed by your healthcare provider. Don't stop taking prescription medications without consulting your doctor first. Store medications properly, check expiration dates, and be aware of potential side effects. If you have questions about your medications, contact your pharmacist or doctor."
    },
    {
      category: 'vaccination',
      keywords: ['vaccine', 'vaccination', 'immunization', 'shot', 'vaccinated'],
      semanticConcepts: ['prevention', 'immunity', 'protection'],
      response: "Vaccinations are an important part of preventive healthcare. They help protect you and your community from serious diseases. Common vaccines include flu shots, COVID-19 vaccines, and routine childhood immunizations. Talk to your healthcare provider about which vaccines are recommended for you based on your age, health status, and lifestyle."
    },
    {
      category: 'emergency',
      keywords: ['emergency', '911', 'urgent', 'critical', 'severe', 'chest pain'],
      semanticConcepts: ['urgent', 'emergency', 'critical'],
      response: "⚠️ For medical emergencies like chest pain, difficulty breathing, severe bleeding, stroke symptoms, or loss of consciousness, call emergency services (911 in the US) immediately. Don't wait or try to treat serious emergencies on your own. Time is critical in emergency situations."
    }
  ];

  const defaultResponse = "I can provide general health information, but I'm not a substitute for professional medical advice. For specific medical concerns, diagnosis, or treatment plans, please consult with a qualified healthcare provider. Is there a particular health topic you'd like to know more about? I can discuss symptoms, wellness tips, preventive care, and general health information.";

  // Token Counting Algorithm
  const tokenCountingRoute = (userMessage: string): string => {
    const lowerMessage = userMessage.toLowerCase();
    const tokens = lowerMessage.split(/\s+/);
    
    let bestMatch = { category: '', score: 0, response: defaultResponse };
    
    knowledgeBase.forEach((item) => {
      let score = 0;
      item.keywords.forEach((keyword) => {
        const keywordTokens = keyword.toLowerCase().split(/\s+/);
        keywordTokens.forEach((kToken) => {
          if (tokens.includes(kToken)) {
            score += 1;
          }
        });
      });
      
      if (score > bestMatch.score) {
        bestMatch = { category: item.category, score, response: item.response };
      }
    });
    
    return bestMatch.response;
  };

  // Semantic Algorithm
  const semanticRoute = (userMessage: string): string => {
    const lowerMessage = userMessage.toLowerCase();
    
    let bestMatch = { category: '', score: 0, response: defaultResponse };
    
    knowledgeBase.forEach((item) => {
      let score = 0;
      
      // Check semantic concepts
      item.semanticConcepts.forEach((concept) => {
        if (lowerMessage.includes(concept)) {
          score += 2; // Higher weight for semantic matches
        }
      });
      
      // Also check keywords with partial matching
      item.keywords.forEach((keyword) => {
        if (lowerMessage.includes(keyword)) {
          score += 3;
        }
      });
      
      if (score > bestMatch.score) {
        bestMatch = { category: item.category, score, response: item.response };
      }
    });
    
    return bestMatch.response;
  };

  // Hybrid Algorithm (combines token counting and semantic)
  const hybridRoute = (userMessage: string): string => {
    const lowerMessage = userMessage.toLowerCase();
    const tokens = lowerMessage.split(/\s+/);
    
    let bestMatch = { category: '', score: 0, response: defaultResponse };
    
    knowledgeBase.forEach((item) => {
      let tokenScore = 0;
      let semanticScore = 0;
      
      // Token counting component
      item.keywords.forEach((keyword) => {
        const keywordTokens = keyword.toLowerCase().split(/\s+/);
        keywordTokens.forEach((kToken) => {
          if (tokens.includes(kToken)) {
            tokenScore += 1;
          }
        });
      });
      
      // Semantic component
      item.semanticConcepts.forEach((concept) => {
        if (lowerMessage.includes(concept)) {
          semanticScore += 2;
        }
      });
      
      item.keywords.forEach((keyword) => {
        if (lowerMessage.includes(keyword)) {
          semanticScore += 3;
        }
      });
      
      // Combine scores (weighted average)
      const combinedScore = (tokenScore * 0.4) + (semanticScore * 0.6);
      
      if (combinedScore > bestMatch.score) {
        bestMatch = { category: item.category, score: combinedScore, response: item.response };
      }
    });
    
    return bestMatch.response;
  };

  // Heuristic Algorithm (original rule-based approach)
  const heuristicRoute = (userMessage: string): string => {
    const lowerMessage = userMessage.toLowerCase();

    // Symptom-related queries
    if (lowerMessage.includes('headache') || lowerMessage.includes('head pain')) {
      return "Headaches can have various causes including stress, dehydration, lack of sleep, or eye strain. For mild headaches, try resting in a quiet, dark room, staying hydrated, and using over-the-counter pain relievers if appropriate. If headaches are severe, persistent, or accompanied by other symptoms like vision changes or fever, please consult a healthcare provider.";
    }
    
    if (lowerMessage.includes('fever') || lowerMessage.includes('temperature')) {
      return "A fever is typically a sign that your body is fighting an infection. For adults, a temperature above 100.4°F (38°C) is considered a fever. Rest, stay hydrated, and use fever-reducing medications like acetaminophen or ibuprofen if needed. Seek medical attention if the fever exceeds 103°F (39.4°C), lasts more than 3 days, or is accompanied by severe symptoms.";
    }
    
    if (lowerMessage.includes('cold') || lowerMessage.includes('flu') || lowerMessage.includes('cough')) {
      return "Common cold and flu symptoms include cough, congestion, sore throat, and fatigue. Rest, stay hydrated, and consider using over-the-counter medications for symptom relief. Flu symptoms tend to be more severe than cold symptoms. If you have difficulty breathing, persistent chest pain, or severe symptoms, please seek medical care.";
    }

    if (lowerMessage.includes('sore throat') || lowerMessage.includes('throat pain')) {
      return "Sore throats are often caused by viral infections, but can also be bacterial (like strep throat). Try warm liquids, throat lozenges, and gargling with salt water. If the sore throat is severe, lasts more than a week, or you have difficulty swallowing or breathing, please see a doctor.";
    }

    // Wellness and prevention
    if (lowerMessage.includes('exercise') || lowerMessage.includes('workout') || lowerMessage.includes('fitness')) {
      return "Regular exercise is essential for good health. Adults should aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week, plus muscle-strengthening activities twice a week. Start slowly if you're new to exercise and gradually increase intensity. Always consult a doctor before starting a new exercise program if you have health concerns.";
    }

    if (lowerMessage.includes('diet') || lowerMessage.includes('nutrition') || lowerMessage.includes('eating')) {
      return "A balanced diet includes plenty of fruits, vegetables, whole grains, lean proteins, and healthy fats. Try to limit processed foods, added sugars, and excessive sodium. Stay hydrated by drinking plenty of water. Everyone's nutritional needs are different, so consider consulting a registered dietitian for personalized advice.";
    }

    if (lowerMessage.includes('sleep') || lowerMessage.includes('insomnia') || lowerMessage.includes('tired')) {
      return "Adults typically need 7-9 hours of sleep per night. For better sleep: maintain a consistent sleep schedule, create a relaxing bedtime routine, keep your bedroom cool and dark, avoid screens before bed, and limit caffeine and alcohol. If you consistently have trouble sleeping, consider speaking with a healthcare provider about possible sleep disorders.";
    }

    if (lowerMessage.includes('stress') || lowerMessage.includes('anxiety') || lowerMessage.includes('mental health')) {
      return "Managing stress is important for overall health. Try techniques like deep breathing, meditation, regular exercise, adequate sleep, and connecting with others. It's normal to feel stressed sometimes, but if anxiety or stress is interfering with daily life, please reach out to a mental health professional. Remember, seeking help is a sign of strength.";
    }

    // Medication questions
    if (lowerMessage.includes('medication') || lowerMessage.includes('medicine') || lowerMessage.includes('prescription')) {
      return "Always take medications as prescribed by your healthcare provider. Don't stop taking prescription medications without consulting your doctor first. Store medications properly, check expiration dates, and be aware of potential side effects. If you have questions about your medications, contact your pharmacist or doctor.";
    }

    // Vaccination
    if (lowerMessage.includes('vaccine') || lowerMessage.includes('vaccination') || lowerMessage.includes('immunization')) {
      return "Vaccinations are an important part of preventive healthcare. They help protect you and your community from serious diseases. Common vaccines include flu shots, COVID-19 vaccines, and routine childhood immunizations. Talk to your healthcare provider about which vaccines are recommended for you based on your age, health status, and lifestyle.";
    }

    // Emergency situations
    if (lowerMessage.includes('emergency') || lowerMessage.includes('911') || lowerMessage.includes('urgent')) {
      return "⚠️ For medical emergencies like chest pain, difficulty breathing, severe bleeding, stroke symptoms, or loss of consciousness, call emergency services (911 in the US) immediately. Don't wait or try to treat serious emergencies on your own. Time is critical in emergency situations.";
    }

    // General/default response
    return "I can provide general health information, but I'm not a substitute for professional medical advice. For specific medical concerns, diagnosis, or treatment plans, please consult with a qualified healthcare provider. Is there a particular health topic you'd like to know more about? I can discuss symptoms, wellness tips, preventive care, and general health information.";
  };

  const generateResponse = (userMessage: string): string => {
    switch (routingAlgorithm) {
      case 'token-counting':
        return tokenCountingRoute(userMessage);
      case 'semantic':
        return semanticRoute(userMessage);
      case 'hybrid':
        return hybridRoute(userMessage);
      case 'heuristic':
      default:
        return heuristicRoute(userMessage);
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    // Hide sample queries after first message
    setShowSampleQueries(false);

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    // Simulate typing delay
    setTimeout(() => {
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: generateResponse(inputValue),
        sender: 'bot',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botResponse]);
      setIsTyping(false);
    }, 1000 + Math.random() * 1000);
  };

  const handleSampleQueryClick = (query: string) => {
    setInputValue(query);
    // Optionally auto-send after a short delay
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
              <h1 className="text-xl font-semibold">Medical Enquiry Assistant</h1>
              <p className="text-sm opacity-90">Ask me about health and wellness</p>
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
                   routingAlgorithm === 'semantic' ? 'Semantic' : 'Hybrid'}
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

        {/* Messages Container */}
        <div className={`flex-1 overflow-y-auto p-6 space-y-4 ${isDarkMode ? 'bg-gray-800' : ''}`}>
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} isDarkMode={isDarkMode} />
          ))}
          
          {/* Sample Queries */}
          {showSampleQueries && (
            <div className="space-y-3">
              <p className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Try asking about:</p>
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

        {/* Disclaimer */}
        <div className={`px-6 py-3 border-t ${isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-amber-50 border-amber-100'}`}>
          <p className={`text-xs ${isDarkMode ? 'text-amber-300' : 'text-amber-800'}`}>
            ⚠️ This chatbot provides general health information only and is not a substitute for professional medical advice, diagnosis, or treatment.
          </p>
        </div>

        {/* Input Area */}
        <div className={`p-4 border-t ${isDarkMode ? 'bg-gray-700 border-gray-600' : 'border-gray-200 bg-gray-50'}`}>
          <div className="flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your health question here..."
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