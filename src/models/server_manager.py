import shlex
import time
import pexpect
import subprocess
import socket
import requests  # Added for API health check

class ServerManager:
    def __init__(self, device_ip, device_port, local_port, ssh_user, ssh_port):
        self.ssh_user = ssh_user
        self.device_ip = device_ip
        self.local_port = local_port
        self.device_port = device_port
        self.ssh_port = ssh_port
        self.ssh_tunnel = None

    # ----------------
    # Fast readiness checks (NO inference)
    # ----------------
    def _wait_local_port_open(self, host="127.0.0.1", timeout_s=120) -> bool:
        """
        Wait until the SSH tunnel is accepting TCP connections locally.
        This does NOT call /query (so it won’t block on model load).
        """
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                with socket.create_connection((host, self.local_port), timeout=1):
                    return True
            except OSError:
                time.sleep(0.5)
        return False

    def _start_tunnel(self):
        """
        Start SSH tunnel as a managed process.
        IMPORTANT: do NOT use -f, since we want Popen to manage it.
        """
        tunnel_cmd = (
            f"ssh -N -L {self.local_port}:localhost:{self.device_port} "
            f"{self.ssh_user}@{self.device_ip} -p {self.ssh_port}"
        )
        proc = subprocess.Popen(
            shlex.split(tunnel_cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.ssh_tunnel = proc
        print(f"Tunnel PID: {proc.pid}")
        return proc

    def is_server_running(self):
        """
        Lightweight check: is the tunnel port open?
        (Not: does /query respond)
        """
        try:
            with socket.create_connection(("127.0.0.1", self.local_port), timeout=1):
                return True
        except OSError:
            return False

    # ----------------
    # Main API
    # ----------------
    def start_server(self):
        """
        Starts remote Flask api and local SSH tunnel.
        Readiness = local tunnel port open AND API healthy.
        """

        # If port already open, assume tunnel+API likely up
        if self.is_server_running():
            print(f"Tunnel already open on localhost:{self.local_port}")
            return None

        ssh_cmd = f"ssh {self.ssh_user}@{self.device_ip} -p {self.ssh_port}"

        # 1) Start remote server
        try:
            child = pexpect.spawn(ssh_cmd, encoding="utf-8", timeout=180)

            # Wait for prompt (supports $ or #)
            child.expect([r"\$", r"#"])

            child.sendline("cd murong")
            child.expect([r"\$", r"#"])

            # Activate venv (prompt may vary)
            child.sendline("source venv/bin/activate")
            child.expect([r"\(venv\)", r"\$", r"#"])

            # Start server (nohup)
            child.sendline("nohup python3 -u api.py > flask.log 2>&1 &")
            child.expect([r"\$", r"#"])

            # STRICT PID fetch: print PID on its own line
            child.sendline("pgrep -f 'python3 -u api.py' | head -n 1")
            # Expect a line that is ONLY digits
            child.expect(r"\r\n(\d+)\r\n")
            pid = child.match.group(1)
            print(f"Server PID: {pid}")

            child.sendline("exit")
            child.expect(pexpect.EOF)

        except pexpect.EOF:
            print("EOF Error: Connection closed prematurely")
        except pexpect.TIMEOUT:
            print("Timeout Error: Command took too long while starting remote server")

        # 2) Start tunnel
        try:
            tunnel_process = self._start_tunnel()

            # Wait for tunnel port to open (Nano may be slow)
            if self._wait_local_port_open(timeout_s=180):
                print("Tunnel is up (local port open).")

                # --- HEALTH CHECK ADDED HERE ---
                print("Waiting for Flask API health check...")
                # We try for up to 15 seconds for the API to actually respond
                for _ in range(15):
                    try:
                        # Try hitting the health endpoint
                        requests.get(f"http://127.0.0.1:{self.local_port}/health", timeout=1)
                        print("Remote API is healthy (HTTP 200).")
                        return tunnel_process
                    except Exception:
                        # Wait a bit for Flask to finish importing torch/transformers/etc.
                        time.sleep(1)
                
                print("Warning: API health check failed after 15s (connection refused or timeout), but proceeding...")
                return tunnel_process
                # -------------------------------

            print("Tunnel timed out (local port never opened).")
            return tunnel_process

        except Exception as e:
            print(f"Tunnel error: {str(e)}")
            return None

    def stop_server(self):
        """Stops the Flask server and tunnel reliably."""
        try:
            # Stop remote server
            check_cmd = (
                f"ssh {self.ssh_user}@{self.device_ip} -p {self.ssh_port} "
                "'pgrep -f \"python3 -u api.py\"'"
            )
            result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"Found running server with PID(s): {result.stdout.strip()}")
                kill_cmd = (
                    f"ssh {self.ssh_user}@{self.device_ip} -p {self.ssh_port} "
                    "'pkill -f \"python3 -u api.py\"'"
                )
                kill_result = subprocess.run(kill_cmd, shell=True, capture_output=True, text=True)
                if kill_result.returncode != 0:
                    print(f"Warning: Kill command exited with {kill_result.returncode}")
                    print(f"Stderr: {kill_result.stderr.strip()}")
                else:
                    print("Server process killed successfully")
            else:
                print("No running server process found")

            # Stop tunnel
            if self.ssh_tunnel is not None:
                try:
                    self.ssh_tunnel.terminate()
                    self.ssh_tunnel.wait(timeout=3)
                    print("SSH tunnel stopped successfully (managed process)")
                except Exception:
                    try:
                        self.ssh_tunnel.kill()
                    except Exception:
                        pass
                finally:
                    self.ssh_tunnel = None
            else:
                # fallback kill by pattern
                subprocess.run(
                    f"pkill -f '{self.local_port}:localhost:{self.device_port}'",
                    shell=True
                )

        except Exception as e:
            print(f"Error during cleanup: {str(e)}")