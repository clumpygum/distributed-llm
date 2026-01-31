import { Bot, User } from 'lucide-react';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
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