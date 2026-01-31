import { Bot } from 'lucide-react';

interface TypingIndicatorProps {
  isDarkMode?: boolean;
}

export function TypingIndicator({ isDarkMode = false }: TypingIndicatorProps) {
  return (
    <div className="flex gap-3">
      <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
        isDarkMode ? 'bg-blue-900 text-blue-300' : 'bg-blue-100 text-blue-600'
      }`}>
        <Bot className="w-5 h-5" />
      </div>
      <div className={`rounded-2xl rounded-tl-sm p-4 flex items-center gap-1 ${
        isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
      }`}>
        <div className="flex gap-1">
          <span className={`w-2 h-2 rounded-full animate-bounce ${
            isDarkMode ? 'bg-gray-400' : 'bg-gray-400'
          }`} style={{ animationDelay: '0ms' }}></span>
          <span className={`w-2 h-2 rounded-full animate-bounce ${
            isDarkMode ? 'bg-gray-400' : 'bg-gray-400'
          }`} style={{ animationDelay: '150ms' }}></span>
          <span className={`w-2 h-2 rounded-full animate-bounce ${
            isDarkMode ? 'bg-gray-400' : 'bg-gray-400'
          }`} style={{ animationDelay: '300ms' }}></span>
        </div>
      </div>
    </div>
  );
}