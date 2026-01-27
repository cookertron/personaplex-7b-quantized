# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Server Manager for PersonaPlex GUI

Handles starting, stopping, and monitoring the PersonaPlex server process.
"""

import os
import sys
import subprocess
import signal
import threading
import time
import socket
from pathlib import Path
from typing import Callable
from dataclasses import dataclass
from enum import Enum


class ServerStatus(Enum):
    """Server status states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class ServerInfo:
    """Information about the server state."""
    status: ServerStatus
    pid: int | None = None
    port: int = 8998
    url: str = ""
    error_message: str = ""
    gpu_name: str = ""
    vram_used: str = ""
    vram_total: str = ""


class ServerManager:
    """Manages the PersonaPlex server process."""

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._status = ServerStatus.STOPPED
        self._error_message = ""
        self._log_lines: list[str] = []
        self._max_log_lines = 500
        self._log_callbacks: list[Callable[[str], None]] = []
        self._status_callbacks: list[Callable[[ServerInfo], None]] = []
        self._log_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def status(self) -> ServerStatus:
        """Get current server status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._status == ServerStatus.RUNNING

    @property
    def logs(self) -> list[str]:
        """Get recent log lines."""
        return self._log_lines.copy()

    def add_log_callback(self, callback: Callable[[str], None]):
        """Add a callback for new log lines."""
        self._log_callbacks.append(callback)

    def add_status_callback(self, callback: Callable[[ServerInfo], None]):
        """Add a callback for status changes."""
        self._status_callbacks.append(callback)

    def _notify_log(self, line: str):
        """Notify all log callbacks."""
        for callback in self._log_callbacks:
            try:
                callback(line)
            except Exception:
                pass

    def _notify_status(self):
        """Notify all status callbacks."""
        info = self.get_info()
        for callback in self._status_callbacks:
            try:
                callback(info)
            except Exception:
                pass

    def _update_status(self, status: ServerStatus, error: str = ""):
        """Update status and notify callbacks."""
        self._status = status
        self._error_message = error
        self._notify_status()

    def _log_reader_thread(self):
        """Background thread to read server logs."""
        if self._process is None:
            return

        try:
            for line in iter(self._process.stdout.readline, ''):
                if self._stop_event.is_set():
                    break
                if line:
                    line = line.rstrip()
                    self._log_lines.append(line)
                    if len(self._log_lines) > self._max_log_lines:
                        self._log_lines.pop(0)
                    self._notify_log(line)

                    # Check for server ready message
                    if "Access the Web UI" in line or "Application startup complete" in line:
                        self._update_status(ServerStatus.RUNNING)
        except Exception as e:
            self._log_lines.append(f"Log reader error: {e}")

    def start(
        self,
        port: int = 8998,
        host: str = "localhost",
        quantize: str = "4bit",
        device: str = "cuda",
    ) -> bool:
        """Start the PersonaPlex server.

        Args:
            port: Port to run server on
            host: Host to bind to
            quantize: Quantization level (4bit or 8bit)
            device: Device to run on (cuda or cpu)

        Returns:
            True if server started successfully
        """
        if self._process is not None and self._process.poll() is None:
            self._log_lines.append("Server is already running")
            return False

        self._stop_event.clear()
        self._update_status(ServerStatus.STARTING)
        self._log_lines.clear()
        self._log_lines.append(f"Starting server on {host}:{port} with {quantize} quantization...")

        # Build command
        cmd = [
            sys.executable, "-m", "moshi.server",
            "--host", host,
            "--port", str(port),
            "--device", device,
        ]

        if quantize:
            cmd.extend(["--quantize", quantize])

        # Set environment
        env = os.environ.copy()

        try:
            # Start the process
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=Path(__file__).parent.parent,
            )

            self._log_lines.append(f"Server process started with PID {self._process.pid}")

            # Start log reader thread
            self._log_thread = threading.Thread(target=self._log_reader_thread, daemon=True)
            self._log_thread.start()

            # Wait a bit and check if process is still running
            time.sleep(2)
            if self._process.poll() is not None:
                # Process exited
                self._update_status(ServerStatus.ERROR, "Server process exited unexpectedly")
                return False

            return True

        except Exception as e:
            self._update_status(ServerStatus.ERROR, str(e))
            self._log_lines.append(f"Failed to start server: {e}")
            return False

    def stop(self) -> bool:
        """Stop the PersonaPlex server.

        Returns:
            True if server stopped successfully
        """
        if self._process is None:
            self._update_status(ServerStatus.STOPPED)
            return True

        self._stop_event.set()
        self._log_lines.append("Stopping server...")

        try:
            # Try graceful shutdown first
            self._process.terminate()

            # Wait up to 10 seconds for graceful shutdown
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                self._log_lines.append("Force killing server...")
                self._process.kill()
                self._process.wait(timeout=5)

            self._process = None
            self._update_status(ServerStatus.STOPPED)
            self._log_lines.append("Server stopped")
            return True

        except Exception as e:
            self._log_lines.append(f"Error stopping server: {e}")
            self._update_status(ServerStatus.ERROR, str(e))
            return False

    def restart(self, **kwargs) -> bool:
        """Restart the server with new settings.

        Args:
            **kwargs: Arguments to pass to start()

        Returns:
            True if restart successful
        """
        self.stop()
        time.sleep(1)
        return self.start(**kwargs)

    def get_info(self) -> ServerInfo:
        """Get current server information."""
        pid = self._process.pid if self._process else None
        port = 8998  # TODO: Get from actual config

        info = ServerInfo(
            status=self._status,
            pid=pid,
            port=port,
            url=f"http://localhost:{port}" if self._status == ServerStatus.RUNNING else "",
            error_message=self._error_message,
        )

        # Try to get GPU info
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(',')]
                if len(parts) >= 3:
                    info.gpu_name = parts[0]
                    info.vram_used = parts[1]
                    info.vram_total = parts[2]
        except Exception:
            pass

        return info

    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def check_health(self) -> bool:
        """Check if the server is responding."""
        if self._process is None or self._process.poll() is not None:
            return False

        try:
            import urllib.request
            with urllib.request.urlopen(f"http://localhost:8998/", timeout=5) as response:
                return response.status == 200
        except Exception:
            return False


# Global server manager instance
_server_instance: ServerManager | None = None


def get_server_manager() -> ServerManager:
    """Get the global server manager instance."""
    global _server_instance
    if _server_instance is None:
        _server_instance = ServerManager()
    return _server_instance
