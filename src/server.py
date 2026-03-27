"""
Server process management for LLM inference servers.
Supports vLLM with extensibility for SGLang.
"""

import fcntl
import os
import select
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests
import torch


@dataclass
class ServerConfig:
    """Configuration for an inference server."""
    
    engine: str  # "vllm" or "sglang"
    model_id: str
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str = "auto"
    gpu_mem_util: float = 0.90
    max_model_len: int = 4096
    extra_args: List[str] = field(default_factory=list)

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


class ServerProcess:
    """
    Manages an inference server process (vLLM or SGLang).
    
    Handles starting, monitoring, and stopping the server with proper
    cleanup and error handling.
    """
    
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None
        self._startup_logs: List[str] = []
        
    def start(self, timeout_s: int = 600):
        """
        Start the server and wait until it's ready.
        
        Args:
            timeout_s: Maximum seconds to wait for server startup
            
        Raises:
            RuntimeError: If server crashes during startup
            TimeoutError: If server doesn't become ready in time
        """
        if self.cfg.engine == "vllm":
            self._start_vllm()
        elif self.cfg.engine == "sglang":
            self._start_sglang()
        else:
            raise ValueError(f"Unknown engine: {self.cfg.engine}")
        
        self._wait_until_ready(timeout_s)
    
    def _start_vllm(self):
        """Start vLLM server."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.cfg.model_id,
            "--host", "0.0.0.0",
            "--port", str(self.cfg.port),
            "--dtype", self.cfg.dtype,
            "--gpu-memory-utilization", str(self.cfg.gpu_mem_util),
            "--max-model-len", str(self.cfg.max_model_len),
        ] + self.cfg.extra_args

        print(f"\n🚀 Starting vLLM server on port {self.cfg.port}:")
        print("   " + " ".join(cmd))

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "HF_TOKEN": os.environ.get("HF_TOKEN", "")},
        )

        # Set non-blocking IO for log streaming
        fd = self.proc.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    
    def _start_sglang(self):
        """Start SGLang server."""
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.cfg.model_id,
            "--host", "0.0.0.0",
            "--port", str(self.cfg.port),
            "--dtype", self.cfg.dtype,
            "--mem-fraction-static", str(self.cfg.gpu_mem_util),
            "--context-length", str(self.cfg.max_model_len),
        ] + self.cfg.extra_args

        print(f"\n🚀 Starting SGLang server on port {self.cfg.port}:")
        print("   " + " ".join(cmd))

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "HF_TOKEN": os.environ.get("HF_TOKEN", "")},
        )

        fd = self.proc.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    def _wait_until_ready(self, timeout_s: int = 600):
        """
        Wait for server to become ready by polling the models endpoint.
        
        Streams server logs during startup for debugging.
        """
        url = f"http://localhost:{self.cfg.port}/v1/models"
        t0 = time.time()
        self._startup_logs = []

        while True:
            # Stream server output
            if self.proc and self.proc.stdout:
                try:
                    ready, _, _ = select.select([self.proc.stdout], [], [], 0.1)
                    while ready:
                        line = self.proc.stdout.readline()
                        if line:
                            line = line.rstrip()
                            self._startup_logs.append(line)
                            print(f"  [{self.cfg.engine}] {line}")
                            if len(self._startup_logs) > 300:
                                self._startup_logs = self._startup_logs[-300:]
                        ready, _, _ = select.select([self.proc.stdout], [], [], 0.1)
                except Exception:
                    pass

            # Check if process crashed
            if self.proc and self.proc.poll() is not None:
                time.sleep(0.5)
                if self.proc.stdout:
                    for line in self.proc.stdout:
                        self._startup_logs.append(line.rstrip())
                        print(f"  [{self.cfg.engine}] {line.rstrip()}")
                raise RuntimeError(
                    f"Server crashed (exit code {self.proc.returncode}). "
                    f"Last logs:\n" + "\n".join(self._startup_logs[-80:])
                )

            # Check if ready
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    print(f"✅ {self.cfg.engine} ready on port {self.cfg.port}")
                    return
            except Exception:
                pass

            # Check timeout
            if time.time() - t0 > timeout_s:
                raise TimeoutError(
                    f"Server not ready after {timeout_s}s. "
                    f"Last logs:\n" + "\n".join(self._startup_logs[-80:])
                )

            time.sleep(0.3)

    def stop(self):
        """Stop the server process and clean up resources."""
        if not self.proc:
            return
        
        print(f"🧹 Stopping {self.cfg.engine} on port {self.cfg.port}...")
        
        try:
            self.proc.terminate()
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        
        self.proc = None
        time.sleep(2)
        
        # Clear GPU cache
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    
    def get_startup_logs(self) -> List[str]:
        """Return logs captured during server startup."""
        return self._startup_logs.copy()
    
    def is_running(self) -> bool:
        """Check if the server process is still running."""
        if self.proc is None:
            return False
        return self.proc.poll() is None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure server is stopped."""
        self.stop()
        return False
