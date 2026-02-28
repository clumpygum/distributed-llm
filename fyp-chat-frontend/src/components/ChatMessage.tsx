import { Bot, User, Cpu, Zap } from 'lucide-react';
import { Message } from '../App';

interface ChatMessageProps {
  message: Message;
  isDarkMode?: boolean;
}

export function ChatMessage({ message, isDarkMode = false }: ChatMessageProps) {
  const isBot = message.sender === 'bot';
  const hasMetadata = isBot && (message.device || message.reasoning || message.method != null);

  // Colour-code device badge
  const deviceColour = () => {
    if (message.device === 'orin') return isDarkMode ? 'text-purple-300' : 'text-purple-600';
    if (message.device === 'nano') return isDarkMode ? 'text-blue-300' : 'text-blue-600';
    return isDarkMode ? 'text-red-400' : 'text-red-500';   // "error"
  };

  // Format confidence as a percentage string
  const confidencePct = message.confidence != null
    ? `${Math.round(message.confidence * 100)}%`
    : null;

  return (
    <div className={`flex gap-3 ${isBot ? '' : 'flex-row-reverse'}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
          isBot
            ? isDarkMode ? 'bg-blue-900 text-blue-300' : 'bg-blue-100 text-blue-600'
            : isDarkMode ? 'bg-gray-600 text-gray-300' : 'bg-gray-100 text-gray-600'
        }`}
      >
        {isBot ? <Bot className="w-5 h-5" /> : <User className="w-5 h-5" />}
      </div>

      {/* Bubble */}
      <div className={`min-w-0 ${isBot ? 'flex flex-col items-start' : 'flex flex-col items-end flex-1'}`}>
        <div
          className={`max-w-[80%] p-4 rounded-2xl ${
            isBot
              ? isDarkMode
                ? 'bg-gray-700 text-gray-200 rounded-tl-sm'
                : 'bg-gray-100 text-gray-800 rounded-tl-sm'
              : 'bg-blue-600 text-white rounded-tr-sm'
          }`}
        >
          <p className="whitespace-pre-wrap break-words">{message.text}</p>

          {/* Routing metadata panel — only shown on bot messages */}
          {hasMetadata && (
            <div className={`mt-3 pt-3 border-t text-xs space-y-1 ${isDarkMode ? 'border-gray-600' : 'border-gray-300'}`}>

              {/* Device + cache hit badge */}
              {message.device && (
                <div className="flex items-center gap-2">
                  <Cpu className="w-3 h-3 flex-shrink-0" />
                  <span className={`font-semibold ${deviceColour()}`}>
                    {message.device.toUpperCase()}
                  </span>
                  {message.cacheHit && (
                    <span className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium ${
                      isDarkMode ? 'bg-green-900 text-green-300' : 'bg-green-100 text-green-700'
                    }`}>
                      <Zap className="w-2.5 h-2.5" /> cache
                    </span>
                  )}
                </div>
              )}

              {/* Method + confidence on one line */}
              {(message.method || confidencePct) && (
                <div className={`flex items-center gap-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  {message.method && <span>Method: <span className="font-medium">{message.method}</span></span>}
                  {confidencePct && <span>· Confidence: <span className="font-medium">{confidencePct}</span></span>}
                  {message.tokens != null && <span>· {message.tokens} tok</span>}
                </div>
              )}

              {/* Reasoning string */}
              {message.reasoning && (
                <div className={`italic ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  {message.reasoning}
                </div>
              )}
            </div>
          )}
        </div>

        <span className={`text-xs mt-1 px-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  );
}