import { Bot, User, Cpu, Zap } from 'lucide-react';

// Update interface to accept routing metadata
export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  device?: string;       // "nano" or "orin"
  reasoning?: string;    // e.g. "Sim=0.85 > 0.4"
}

interface ChatMessageProps {
  message: Message;
  isDarkMode?: boolean;
}

export function ChatMessage({ message, isDarkMode = false }: ChatMessageProps) {
  const isBot = message.sender === 'bot';

  return (
    <div className={`flex gap-3 ${isBot ? '' : 'flex-row-reverse'}`}>
      {/* Avatar Circle */}
      <div
        className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
          isBot 
            ? isDarkMode ? 'bg-blue-900 text-blue-300' : 'bg-blue-100 text-blue-600'
            : isDarkMode ? 'bg-gray-600 text-gray-300' : 'bg-gray-100 text-gray-600'
        }`}
      >
        {isBot ? <Bot className="w-5 h-5" /> : <User className="w-5 h-5" />}
      </div>

      {/* Message Content Area */}
      <div className={`flex-1 ${isBot ? '' : 'flex flex-col items-end'}`}>
        
        {/* The Text Bubble */}
        <div
          className={`inline-block max-w-[80%] p-4 rounded-2xl ${
            isBot
              ? isDarkMode 
                ? 'bg-gray-700 text-gray-200 rounded-tl-sm'
                : 'bg-gray-100 text-gray-800 rounded-tl-sm'
              : 'bg-blue-600 text-white rounded-tr-sm'
          }`}
        >
          <p className="whitespace-pre-wrap break-words">{message.text}</p>
        </div>

        {/* Footer: Timestamp + Routing Badge */}
        <div className="flex items-center gap-2 mt-1 px-2">
          
          {/* 1. Timestamp */}
          <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            {message.timestamp.toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>

          {/* 2. DEVICE BADGE (New Feature) - Only show for Bot */}
          {isBot && message.device && (
            <div className="flex items-center gap-2 border-l pl-2 ml-1 border-gray-300/50">
              {/* Badge */}
              <span className={`text-[10px] px-2 py-0.5 rounded-full flex items-center gap-1 font-semibold uppercase tracking-wide border ${
                message.device === 'orin' 
                  ? 'bg-purple-100 text-purple-700 border-purple-200 dark:bg-purple-900/30 dark:text-purple-300 dark:border-purple-700' 
                  : 'bg-green-100 text-green-700 border-green-200 dark:bg-green-900/30 dark:text-green-300 dark:border-green-700'
              }`}>
                {message.device === 'orin' ? <Zap size={10} /> : <Cpu size={10} />}
                {message.device}
              </span>
              
              {/* Reasoning (Optional - truncates if too long) */}
              {message.reasoning && (
                <span className={`text-[10px] truncate max-w-[150px] hidden sm:block ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  {message.reasoning}
                </span>
              )}
            </div>
          )}
        </div>
        
      </div>
    </div>
  );
}