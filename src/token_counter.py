import json
from litellm import token_counter

class TokenCounter:
    def count_tokens(self, msg):
        """Counts tokens for a single message."""
        msg_payload = [{"role": msg["role"], "content": msg["content"]}]
        return token_counter(model="ollama/phi3", messages=msg_payload)

    def get_context_size(self, context):
        """Gets token count for the entire context window."""
        return token_counter(model="ollama/phi3", messages=context)

