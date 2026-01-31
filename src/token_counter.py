import json
from litellm import token_counter

class TokenCounter:
    def count_tokens(self, msg):
        """Counts tokens for a single message."""
        msg_payload = [{"role": msg["role"], "content": json.dumps(msg)}]
        return token_counter(model="ollama/phi3", messages=msg_payload)

    def get_context_size(self, context):
        """Gets token count for the last 5 messages."""
        return sum(self.count_tokens(msg) for msg in context)


