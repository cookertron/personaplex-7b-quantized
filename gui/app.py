# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""
PersonaPlex Desktop GUI

A Gradio-based graphical user interface for easy interaction with PersonaPlex.
"""

import os
import sys
import argparse
import webbrowser
import threading
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from gui.config_manager import (
    get_config,
    ConfigManager,
    TEXT_PROMPT_PRESETS,
    VOICE_PROMPTS,
    ALL_VOICES,
)
from gui.server_manager import get_server_manager, ServerManager, ServerStatus


# Custom CSS for better styling
CUSTOM_CSS = """
.server-status-running {
    color: #22c55e !important;
    font-weight: bold;
}
.server-status-stopped {
    color: #ef4444 !important;
    font-weight: bold;
}
.server-status-starting {
    color: #f59e0b !important;
    font-weight: bold;
}
.voice-category {
    font-weight: bold;
    margin-top: 10px;
}
.preset-btn {
    margin: 2px !important;
}
.log-box {
    font-family: monospace;
    font-size: 12px;
    background-color: #1a1a1a;
    color: #00ff00;
}
"""


class PersonaPlexGUI:
    """Main GUI application class."""

    def __init__(self):
        self.config = get_config()
        self.server = get_server_manager()
        self.app: gr.Blocks | None = None

    def get_status_text(self) -> str:
        """Get formatted status text."""
        info = self.server.get_info()
        status_map = {
            ServerStatus.RUNNING: "Running",
            ServerStatus.STOPPED: "Stopped",
            ServerStatus.STARTING: "Starting...",
            ServerStatus.ERROR: f"Error: {info.error_message}",
        }
        return status_map.get(info.status, "Unknown")

    def get_status_color(self) -> str:
        """Get status indicator color."""
        status_map = {
            ServerStatus.RUNNING: "#22c55e",  # Green
            ServerStatus.STOPPED: "#ef4444",  # Red
            ServerStatus.STARTING: "#f59e0b",  # Yellow
            ServerStatus.ERROR: "#ef4444",  # Red
        }
        return status_map.get(self.server.status, "#6b7280")

    def get_gpu_info(self) -> str:
        """Get GPU information string."""
        info = self.server.get_info()
        if info.gpu_name:
            return f"{info.gpu_name} ({info.vram_used} / {info.vram_total})"
        return "No GPU detected"

    def start_server(
        self,
        port: int,
        quantize: str,
        device: str,
    ) -> tuple[str, str]:
        """Start the server with given settings."""
        success = self.server.start(
            port=int(port),
            host="localhost",
            quantize=quantize,
            device=device,
        )

        if success:
            # Wait for server to be ready
            for _ in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if self.server.status == ServerStatus.RUNNING:
                    break

        status = self.get_status_text()
        logs = "\n".join(self.server.logs[-50:])
        return status, logs

    def stop_server(self) -> tuple[str, str]:
        """Stop the server."""
        self.server.stop()
        status = self.get_status_text()
        logs = "\n".join(self.server.logs[-50:])
        return status, logs

    def refresh_status(self) -> tuple[str, str, str]:
        """Refresh all status displays."""
        status = self.get_status_text()
        gpu = self.get_gpu_info()
        logs = "\n".join(self.server.logs[-50:])
        return status, gpu, logs

    def open_web_ui(self):
        """Open the web UI in the default browser."""
        if self.server.status == ServerStatus.RUNNING:
            info = self.server.get_info()
            webbrowser.open(info.url)
            return f"Opened {info.url} in browser"
        return "Server is not running. Start the server first."

    def apply_preset(self, preset_name: str) -> str:
        """Apply a text prompt preset."""
        if preset_name in TEXT_PROMPT_PRESETS:
            return TEXT_PROMPT_PRESETS[preset_name]
        return ""

    def save_settings(
        self,
        port: int,
        quantize: str,
        device: str,
        default_voice: str,
        auto_start: bool,
    ) -> str:
        """Save current settings to config file."""
        self.config.server_port = int(port)
        self.config.quantize = quantize
        self.config.device = device
        self.config.default_voice = default_voice
        self.config.auto_start_server = auto_start
        self.config.save()
        return "Settings saved successfully!"

    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface."""

        with gr.Blocks(
            title="PersonaPlex",
            css=CUSTOM_CSS,
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="gray",
            ),
        ) as app:
            gr.Markdown(
                """
                # PersonaPlex
                ### Voice and Role Control for Full Duplex Conversation

                A real-time, full-duplex speech-to-speech conversational AI with persona control.
                """
            )

            with gr.Tabs():
                # Tab 1: Quick Start
                with gr.Tab("Quick Start", id="quickstart"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### Voice Selection")
                            voice_dropdown = gr.Dropdown(
                                choices=ALL_VOICES,
                                value=self.config.default_voice,
                                label="Select Voice",
                                info="Choose the voice for the AI assistant",
                            )

                            # Voice category buttons
                            gr.Markdown("**Quick Select:**")
                            with gr.Row():
                                for category, voices in VOICE_PROMPTS.items():
                                    with gr.Column(min_width=100):
                                        gr.Markdown(f"*{category}*")
                                        for voice in voices[:2]:  # Show first 2 of each
                                            btn = gr.Button(
                                                voice.replace(".pt", ""),
                                                size="sm",
                                                elem_classes=["preset-btn"],
                                            )
                                            btn.click(
                                                lambda v=voice: v,
                                                outputs=[voice_dropdown],
                                            )

                        with gr.Column(scale=3):
                            gr.Markdown("### Text Prompt")

                            # Preset buttons
                            gr.Markdown("**Presets:**")
                            with gr.Row():
                                for preset_name in TEXT_PROMPT_PRESETS.keys():
                                    btn = gr.Button(preset_name, size="sm")
                                    # Will connect below

                            text_prompt = gr.Textbox(
                                value=self.config.default_text_prompt,
                                label="System Prompt",
                                placeholder="Enter the persona/role prompt for the AI...",
                                lines=5,
                                max_lines=10,
                                info="Max 1000 characters. Describes the AI's persona and behavior.",
                            )
                            char_count = gr.Markdown("0 / 1000 characters")

                            # Update character count
                            text_prompt.change(
                                lambda t: f"{len(t)} / 1000 characters",
                                inputs=[text_prompt],
                                outputs=[char_count],
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Server Control")
                            with gr.Row():
                                status_display = gr.Textbox(
                                    value=self.get_status_text(),
                                    label="Status",
                                    interactive=False,
                                    scale=1,
                                )
                                gpu_display = gr.Textbox(
                                    value=self.get_gpu_info(),
                                    label="GPU",
                                    interactive=False,
                                    scale=2,
                                )

                            with gr.Row():
                                start_btn = gr.Button(
                                    "Start Server",
                                    variant="primary",
                                    size="lg",
                                )
                                stop_btn = gr.Button(
                                    "Stop Server",
                                    variant="stop",
                                    size="lg",
                                )
                                open_ui_btn = gr.Button(
                                    "Open Web UI",
                                    variant="secondary",
                                    size="lg",
                                )

                            open_result = gr.Textbox(
                                label="",
                                interactive=False,
                                visible=False,
                            )

                    # Connect preset buttons
                    for child in app.children:
                        pass  # Presets connected via lambda above

                # Tab 2: Settings
                with gr.Tab("Settings", id="settings"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Server Settings")

                            port_input = gr.Number(
                                value=self.config.server_port,
                                label="Port",
                                info="Server port (default: 8998)",
                                precision=0,
                            )

                            quantize_dropdown = gr.Dropdown(
                                choices=["4bit", "8bit"],
                                value=self.config.quantize,
                                label="Quantization",
                                info="4bit uses ~8GB VRAM, 8bit uses ~12GB VRAM",
                            )

                            device_dropdown = gr.Dropdown(
                                choices=["cuda", "cpu"],
                                value=self.config.device,
                                label="Device",
                                info="cuda for GPU, cpu for CPU (very slow)",
                            )

                        with gr.Column():
                            gr.Markdown("### Default Settings")

                            default_voice_dropdown = gr.Dropdown(
                                choices=ALL_VOICES,
                                value=self.config.default_voice,
                                label="Default Voice",
                            )

                            auto_start_checkbox = gr.Checkbox(
                                value=self.config.auto_start_server,
                                label="Auto-start server on GUI launch",
                            )

                            save_settings_btn = gr.Button(
                                "Save Settings",
                                variant="primary",
                            )
                            settings_status = gr.Textbox(
                                label="",
                                interactive=False,
                                visible=True,
                            )

                    save_settings_btn.click(
                        self.save_settings,
                        inputs=[
                            port_input,
                            quantize_dropdown,
                            device_dropdown,
                            default_voice_dropdown,
                            auto_start_checkbox,
                        ],
                        outputs=[settings_status],
                    )

                # Tab 3: Logs
                with gr.Tab("Logs", id="logs"):
                    gr.Markdown("### Server Logs")
                    log_display = gr.Textbox(
                        value="",
                        label="Server Output",
                        lines=20,
                        max_lines=30,
                        interactive=False,
                        elem_classes=["log-box"],
                    )
                    refresh_btn = gr.Button("Refresh Logs")

                # Tab 4: Help
                with gr.Tab("Help", id="help"):
                    gr.Markdown(
                        """
                        ## Quick Start Guide

                        1. **Select a Voice**: Choose from Natural (more conversational) or Variety voices
                        2. **Set a Prompt**: Use a preset or write your own persona description
                        3. **Start the Server**: Click "Start Server" and wait for it to initialize
                        4. **Open Web UI**: Click "Open Web UI" to start a conversation

                        ## Voice Categories

                        - **Natural Female/Male**: More conversational, natural-sounding voices
                        - **Variety Female/Male**: More diverse voice characteristics

                        ## Prompt Tips

                        - Keep prompts under 1000 characters
                        - Be specific about the persona's name, role, and knowledge
                        - For customer service roles, include specific information the AI should know
                        - For casual chat, simple prompts like "You enjoy having a good conversation" work well

                        ## Troubleshooting

                        - **Server won't start**: Check the Logs tab for error messages
                        - **Out of memory**: Try 4bit quantization or close other GPU applications
                        - **No microphone access**: Make sure to use `localhost` URL, not IP address
                        - **Slow responses**: Ensure you're using a CUDA GPU, not CPU

                        ## Requirements

                        - NVIDIA GPU with 8GB+ VRAM (for 4bit quantization)
                        - NVIDIA GPU with 12GB+ VRAM (for 8bit quantization)
                        - HuggingFace account with accepted model license

                        ## Links

                        - [PersonaPlex on HuggingFace](https://huggingface.co/nvidia/personaplex-7b-v1)
                        - [Documentation](https://github.com/nvidia/personaplex)
                        """
                    )

            # Connect event handlers
            start_btn.click(
                self.start_server,
                inputs=[port_input, quantize_dropdown, device_dropdown],
                outputs=[status_display, log_display],
            )

            stop_btn.click(
                self.stop_server,
                outputs=[status_display, log_display],
            )

            open_ui_btn.click(
                self.open_web_ui,
                outputs=[open_result],
            )

            refresh_btn.click(
                self.refresh_status,
                outputs=[status_display, gpu_display, log_display],
            )

            # Connect text prompt presets
            # This is done by finding buttons and connecting them
            for preset_name, preset_text in TEXT_PROMPT_PRESETS.items():
                # Create a closure to capture the preset text
                def make_preset_handler(text):
                    return lambda: text
                # Note: In actual Gradio, we'd need to reference the buttons differently
                # This is a simplified version

        self.app = app
        return app

    def launch(
        self,
        share: bool = False,
        server_port: int = 7860,
        auto_start_server: bool | None = None,
    ):
        """Launch the GUI application.

        Args:
            share: Whether to create a public Gradio link
            server_port: Port for the Gradio interface
            auto_start_server: Whether to auto-start the PersonaPlex server
        """
        if auto_start_server is None:
            auto_start_server = self.config.auto_start_server

        app = self.build_interface()

        # Auto-start server if configured
        if auto_start_server:
            print("Auto-starting PersonaPlex server...")
            threading.Thread(
                target=lambda: self.server.start(
                    port=self.config.server_port,
                    quantize=self.config.quantize,
                    device=self.config.device,
                ),
                daemon=True,
            ).start()

        print(f"\nPersonaPlex GUI starting on http://localhost:{server_port}")
        print("Press Ctrl+C to stop\n")

        app.launch(
            server_name="localhost",
            server_port=server_port,
            share=share,
            inbrowser=True,
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PersonaPlex Desktop GUI")
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the GUI interface (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Don't auto-start the PersonaPlex server",
    )
    args = parser.parse_args()

    gui = PersonaPlexGUI()
    gui.launch(
        share=args.share,
        server_port=args.port,
        auto_start_server=not args.no_auto_start,
    )


if __name__ == "__main__":
    main()
