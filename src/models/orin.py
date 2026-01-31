import json
import time
import requests
from models.server_manager import ServerManager

orin_ip = "10.0.1.8"  
local_port = 11135 
orin_port = 5000 
ssh_user = "orin" 
ssh_port = 22

class Orin: 
    def __init__(self):
            self.server_manager = ServerManager(orin_ip, orin_port, local_port, ssh_user, ssh_port) 
            
    def process(self, query):
        """Routes a query to the Orin API."""
        if not self.server_manager.is_server_running():
            print("No running Orin server found, starting...")
            self.server_manager.start_server()

        url = f"http://localhost:{self.server_manager.local_port}/query"
        payload = {"query": query}

        try:
            response = requests.post(url, json=payload)
            return response.json() if response.text else {"error": "Empty response"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    


    
    