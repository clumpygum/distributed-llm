import { Bot, User, Cpu } from 'lucide-react';

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  device?: string;
  reasoning?: string;
}

interface ChatMessageProps {
  message: Message;
  isDarkMode?: boolean;
}

export function ChatMessage({ message, isDarkMode = false }: ChatMessageProps) {
  const isBot = message.sender === 'bot';

  return (
    <div className={`flex gap-3 ${isBot ? '' : 'flex-row-reverse'}`}>
      <div
        className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
          isBot 
            ? isDarkMode ? 'bg-blue-900 text-blue-300' : 'bg-blue-100 text-blue-600'
            : isDarkMode ? 'bg-gray-600 text-gray-300' : 'bg-gray-100 text-gray-600'
        }`}
      >
        {isBot ? <Bot className="w-5 h-5" /> : <User className="w-5 h-5" />}
      </div>
      <div className={`flex-1 ${isBot ? '' : 'flex flex-col items-end'}`}>
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
          
          {/* Display Device and Reasoning for Bot Messages */}
          {isBot && (message.device || message.reasoning) && (
            <div className={`mt-3 pt-3 border-t text-xs ${
              isDarkMode ? 'border-gray-600' : 'border-gray-300'
            }`}>
              {message.device && (
                <div className="flex items-center gap-2 mb-1">
                  <Cpu className="w-3 h-3" />
                  <span className={isDarkMode ? 'text-blue-300' : 'text-blue-600'}>
                    Device: <strong>{message.device.toUpperCase()}</strong>
                  </span>
                </div>
              )}
              {message.reasoning && (
                <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-600'} italic`}>
                  {message.reasoning}
                </div>
              )}
            </div>
          )}
        </div>
        
        <span className={`text-xs mt-1 px-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          {message.timestamp.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>
    </div>
  );
}
