import csv
import os
import subprocess
import pexpect
from datetime import datetime
from main import Chatbot
from router import Router
from query_sets import query_sets
print("RUNNING TESTER:", os.path.abspath(__file__))

class ChatbotTester:
    def __init__(self, test_queries, context_thresholds, nano_ip, orin_ip, nano_port=22, orin_port=30006):
        self.chatbot = Chatbot(strategy="perf", config={"cache_enabled": False, "enable_response_cache": False, "enable_failover": True})
        self.test_queries = test_queries
        self.context_thresholds = context_thresholds
        self.nano_ip = nano_ip
        self.orin_ip = orin_ip
        self.nano_port = nano_port
        self.orin_port = orin_port

    def start_logging(self, device):
        """Start jtop logging on both Nano and Orin."""
        if device == "orin":
            device_ip = self.orin_ip
            device_port = self.orin_port
        elif device == "nano":
            device_ip = self.nano_ip
            device_port = self.nano_port

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # Start logging on Nano
            child = pexpect.spawn(f"ssh {device}@{device_ip} -p {device_port}", encoding='utf-8', timeout=30)
            child.expect(r'\$') 
            
            if device == "nano":
                child.sendline(f"sudo timedatectl set-time \"{current_time}\"")
                child.expect(r'\$') 

            child.sendline("cd murong")
            child.expect(r'\$') 

            child.sendline("nohup python3 -u logging_power.py > power.log 2>&1 &")
            child.expect(r'\$')  # Ensure command is accepted
            child.sendline("pgrep -f 'python3 -u logging_power.py'")
            child.expect(r'\d+')  # Look for process ID
            pid = child.match.group(0)
            print(f"{device} logging PID: {pid}")
            
            # Clean exit
            child.sendline("exit")
            child.expect(pexpect.EOF)
            
        except pexpect.EOF:
            print("EOF Error: Connection closed prematurely")
        except pexpect.TIMEOUT:
            print("Timeout Error: Command took too long")

    def stop_logging(self, device):
        """Stop jtop logging on both Nano and Orin."""
        if device == "orin":
            device_ip = self.orin_ip
            device_port = self.orin_port
        elif device == "nano":
            device_ip = self.nano_ip
            device_port = self.nano_port

        try:
            # First check if server process exists
            check_cmd = (
                f"ssh {device}@{device_ip} -p {device_port} "
                "'pgrep -f \"python3 -u logging_power.py\"'"
            )
            result = subprocess.run(
                check_cmd,
                shell=True,
                capture_output=True,
                text=True
            )

            # If process exists (pgrep found something)
            if result.returncode == 0:
                print(f"Found running logger with PID: {result.stdout.strip()}")
                # Now kill it gently
                kill_cmd = (
                    f"ssh {device}@{device_ip} -p {device_port} "
                    "'pkill -f \"python3 -u logging_power.py\"'"
                )
                kill_result = subprocess.run(
                    kill_cmd,
                    shell=True,
                    capture_output=True,
                    text=True
                )
                
                if kill_result.returncode != 0:
                    print(f"Warning: Kill command exited with {kill_result.returncode}")
                    print(f"Stderr: {kill_result.stderr.strip()}")
                else: 
                    print("Logging process killed successfully")
            else:
                print("No running logging process found")

        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def run_test(self):
            """Runs through all context thresholds for one query set."""
            query_log = []

            self.chatbot.router.nano.server_manager.start_server()
            self.chatbot.router.orin.server_manager.start_server()

            for threshold in self.context_thresholds:
                print(f"\nTesting with context threshold: {threshold} tokens")
                self.chatbot.router.set_threshold(threshold)
                self.chatbot.conversation_history = []
                
                for query in self.test_queries:
                    self.chatbot.add_message("user", query)

                    start_time = datetime.now()
                    response, response_tokens, device_used = self.chatbot.router.route_query(self.chatbot.conversation_history)  
                    end_time = datetime.now()

                    assistant_text = response.get("response", "") if isinstance(response, dict) else str(response)
                    print("Response:", assistant_text)
                    self.chatbot.add_message("assistant", assistant_text)

                    # Record in query log (index corresponds to index in test_queries)
                    query_log.append([threshold, device_used, start_time, end_time, response_tokens])
            
            # Stop the servers
            self.chatbot.router.nano.server_manager.stop_server()
            self.chatbot.router.orin.server_manager.stop_server()

            return query_log

    def retrieve_power_log(self, device):
        """SSH into the device (Nano or Orin) and retrieve the power logs to the host laptop."""
        remote_log_path = f"/home/{device}/murong/power.log"
        if device == "orin":
            device_ip = self.orin_ip
            device_port = self.orin_port
            local_log_path = "orin_power.log"
        elif device == "nano":
            device_ip = self.nano_ip
            device_port = self.nano_port
            local_log_path = "nano_power.log"

        try:
            # Check if the log exists on the remote device via SSH
            ssh_check = pexpect.spawn(f"ssh {device}@{device_ip} -p {device_port} test -f {remote_log_path}", timeout=10)
            ssh_check.wait()
            
            exit_status = ssh_check.exitstatus
            if exit_status != 0:
                print(f"Error: Log file not found on {device}!")
                print(f"Exit status: {exit_status}")
                return

            # Use SCP locally to copy the file
            scp_command = f"scp -P {device_port} {device}@{device_ip}:{remote_log_path} ./{local_log_path}"
            print(f"Retrieving log file from {device} using SCP...")
            
            child = pexpect.spawn(scp_command, encoding='utf-8', timeout=30)
            child.wait()
            
            if child.exitstatus == 0:
                print(f"Power log successfully retrieved to {local_log_path}.")
            else:
                print("Failed to retrieve log.")

        except pexpect.EOF:
            print("EOF Error: Connection closed prematurely")
        except pexpect.TIMEOUT:
            print("Timeout Error: Command took too long")
        except Exception as e:
            print(f"An error occurred: {e}")
        
    def parse_power_log(self, device):
        """Parses the power.log file and returns a dictionary {timestamp: power_value}."""
        if device == "orin":
            log_path = "orin_power.log"
        elif device == "nano":
            log_path = "nano_power.log"
        
        power_data = {}

        with open(log_path, 'r') as file:
            next(file)
            for line in file:
                parts = line.strip().rsplit(":", 1)
                if len(parts) == 2:
                    timestamp_str, power_str = parts
                    try:
                        timestamp_str = timestamp_str.strip()
                        power_str = power_str.strip()
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                        power = int(power_str.strip())
                        power_data[timestamp] = power
                    except ValueError:
                        print(f"Skipping malformed line: {line}")
        return power_data

    def calculate_energy(self, query_log, nano_power_data, orin_power_data):
        """
        Returns total energy used by each device for the set of queries in query_log.

        query_log: List of queries with [device_used, start_time, end_time]
        power_data: Dictionary of power readings with {timestamp: power_value}
        """

        query_results = []

        for threshold, device, start_time, end_time, response_tokens in query_log:
            power_data = nano_power_data if device == "nano" else orin_power_data if device == "orin" else None

            # Filter power readings within the query's time range
            relevant_powers = [p for t, p in power_data.items() if start_time <= t <= end_time]
            print('\n relevent powers:', relevant_powers)
            
            # Based on trapezoidal rule for numerical integration
            energy = sum(relevant_powers)  # mili Joules 
            latency = round((end_time - start_time).total_seconds() * 1000)  # ms

            query_results.append([threshold, device, start_time, end_time, energy, latency, response_tokens])
            print("query results:", query_results[-1])
        
        # Calculate total latency and energy for each device at each threshold
        results = {}

        for threshold, device, start_time, end_time, energy, latency, response_tokens in query_results:
            if threshold not in results:
                results[threshold] = {"nano": [0, 0, 0, 0], "orin": [0, 0, 0, 0]}  # [latency, energy, avg power, total_tokens_generated]

            if device in results[threshold]:
                results[threshold][device][0] += latency  # Total Latency
                results[threshold][device][1] += energy   # Total Energy
                results[threshold][device][3] += response_tokens   # Total Tokens Generated

        # Compute Average Power (mW) = Total Energy (mJ) / Total Latency (ms)
        for threshold in results:
            for device in ["nano", "orin"]:
                total_latency = results[threshold][device][0]
                total_energy = results[threshold][device][1]
                avg_power = (total_energy / total_latency) if total_latency > 0 else 0  # in W
                results[threshold][device][2] = round(avg_power, 3)  # Store rounded average power

        return results 

    
    
    def save_results(self, results, query_set_name, output_file="final_results.csv"):
        """
        Saves the final results to a CSV file.

        results: Dictionary structured as {threshold: {device: [latency, energy]}}
        query_set_name: Name of the query set used in the test
        output_file: File to save the results (default: "final_results.csv")
        """
        file_exists = os.path.exists(output_file)

        with open(output_file, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Write header if file is newly created
            if not file_exists:
                writer.writerow(["Query Set", "Context Threshold",
                                "Nano Latency (ms)", "Nano Energy (mJ)", "Nano Avg Power (W)", "Nano Tokens Generated",
                                "Orin Latency (ms)", "Orin Energy (mJ)", "Orin Avg Power (W)", "Orin Tokens Generated"])

            # Write results for each threshold
            for threshold, device_results in results.items():
                print(f"Results for threshold {threshold}: {device_results}")
                writer.writerow([
                    query_set_name, threshold,
                    device_results["nano"][0], device_results["nano"][1], device_results["nano"][2], device_results["nano"][3], # Latency, Energy, Avg Power
                    device_results["orin"][0], device_results["orin"][1], device_results["orin"][2], device_results["orin"][3]
                ])


        print(f"Final results saved to {output_file}")

            
# PYTHONPATH=src python3 src/tests/chatbot_tester.py
if __name__ == "__main__":
    tester = ChatbotTester(query_sets["personal_health"], [4000], "10.0.1.10", "10.0.1.8", nano_port=22, orin_port=22)

    # start logging on both devices
    tester.start_logging("nano")
    tester.start_logging("orin")

    query_log = tester.run_test()

    # stop logging on both devices
    tester.stop_logging("nano")
    tester.stop_logging("orin")

    # retrieve and parse power logs
    tester.retrieve_power_log("nano")
    nano_power_data = tester.parse_power_log("nano")

    tester.retrieve_power_log("orin")
    orin_power_data = tester.parse_power_log("orin")

    results = tester.calculate_energy(query_log, nano_power_data, orin_power_data)
    print("Results: ", results)
    tester.save_results(results, "personal_health")

    
