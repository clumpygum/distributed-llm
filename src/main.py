from router import Router

class Chatbot:
    def __init__(self, strategy="token", config=None, threshold_fallback=100):
        # strategy: "token" | "semantic" | "heuristic" | "hybrid" | "perf"
        # config: dict passed into QueryRouter + Router cache flags
        self.router = Router(strategy=strategy, config=config or {}, threshold_fallback=threshold_fallback)
        self.conversation_history = []

    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    def chat(self):
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                self.router.orin.server_manager.stop_server()
                self.router.nano.server_manager.stop_server()
                break

            self.add_message("user", user_input)
            response, num_tokens, _ = self.router.route_query(self.conversation_history)

            assistant_text = response.get("response", "") if isinstance(response, dict) else str(response)
            self.add_message("assistant", assistant_text)

            print(f"Assistant: {assistant_text}", num_tokens)

if __name__ == "__main__":
    # Example: switch strategy here when chatting manually
    chatbot = Chatbot(
        strategy="semantic",
        config={
            # QueryRouter cache (routing cache)
            "cache_enabled": False,
            # Router response cache (inference cache)
            "enable_response_cache": False,
            "enable_failover": True,
        }
    )
    chatbot.chat()