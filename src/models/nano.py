import requests
from models.server_manager import ServerManager

nano_ip = "10.0.1.11"
local_port = 11134
nano_port = 5001
ssh_user = "nano"
ssh_port = 22

class Nano:
    def __init__(self):
        self.server_manager = ServerManager(nano_ip, nano_port, local_port, ssh_user, ssh_port)

    def process(self, query):
        """
        query is expected to be conversation_history:
        [{"role":"user","content":"..."}, ...]
        """
        if not self.server_manager.is_server_running():
            print("No running Nano server found, starting...")
            self.server_manager.start_server()

        url = f"http://127.0.0.1:{self.server_manager.local_port}/query"
        payload = {"query": query}

        try:
            # (connect_timeout=5s, read_timeout=180s)
            r = requests.post(url, json=payload, timeout=(5, 180))

            # If Flask returns non-JSON (e.g. traceback HTML), show it clearly
            ct = (r.headers.get("Content-Type") or "").lower()
            if "application/json" not in ct:
                return {"error": f"Non-JSON response ({r.status_code})", "body": r.text[:500]}

            return r.json()

        except requests.Timeout:
            return {"error": "Request timed out on Nano (model cold start / slow inference)."}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}