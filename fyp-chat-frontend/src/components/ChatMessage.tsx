import { Bot, User, Cpu, Zap } from 'lucide-react';
import { Message } from '../App';

interface ChatMessageProps {
  message: Message;
  isDarkMode?: boolean;
}

export function ChatMessage({ message, isDarkMode = false }: ChatMessageProps) {
  const isBot = message.sender === 'bot';
  const hasMetadata = isBot && (message.device || message.reasoning || message.method != null);

  const deviceColour = () => {
    if (message.device === 'orin') return isDarkMode ? 'text-purple-300' : 'text-purple-600';
    if (message.device === 'nano') return isDarkMode ? 'text-blue-300' : 'text-blue-600';
    return isDarkMode ? 'text-red-400' : 'text-red-500';
  };

  const confidencePct = message.confidence != null
    ? `${Math.round(message.confidence * 100)}%`
    : null;

  return (
    <div className={`flex gap-3 ${isBot ? '' : 'flex-row-reverse'}`}>

      {/* Avatar */}
      <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
        isBot
          ? isDarkMode ? 'bg-blue-900 text-blue-300' : 'bg-blue-100 text-blue-600'
          : isDarkMode ? 'bg-gray-600 text-gray-300' : 'bg-gray-100 text-gray-600'
      }`}>
        {isBot ? <Bot className="w-5 h-5" /> : <User className="w-5 h-5" />}
      </div>

      {/* Bubble + timestamp column — max-w on THIS div caps the total width */}
      <div className={`flex flex-col gap-1 max-w-[75%] ${isBot ? 'items-start' : 'items-end flex-1'}`}>

        {/* w-fit shrinks bubble to content; max-w-full respects parent cap */}
        <div className={`w-fit max-w-full p-4 rounded-2xl ${
          isBot
            ? isDarkMode
              ? 'bg-gray-700 text-gray-200 rounded-tl-sm'
              : 'bg-gray-100 text-gray-800 rounded-tl-sm'
            : 'bg-blue-600 text-white rounded-tr-sm'
        }`}>
          <p className="whitespace-pre-wrap break-words">{message.text}</p>

          {/* Routing metadata — bot messages only */}
          {hasMetadata && (
            <div className={`mt-3 pt-3 border-t text-xs space-y-1 ${isDarkMode ? 'border-gray-600' : 'border-gray-300'}`}>

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

              {(message.method || confidencePct) && (
                <div className={`flex items-center gap-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  {message.method && <span>Method: <span className="font-medium">{message.method}</span></span>}
                  {confidencePct && <span>· Confidence: <span className="font-medium">{confidencePct}</span></span>}
                  {message.tokens != null && <span>· {message.tokens} tok</span>}
                </div>
              )}

              {message.reasoning && (
                <div className={`italic ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  {message.reasoning}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Timestamp aligned under bubble */}
        <span className={`text-xs px-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>

    </div>
  );
}