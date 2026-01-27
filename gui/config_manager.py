# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Configuration Manager for PersonaPlex GUI

Handles loading, saving, and managing configuration from config.yaml
"""

import os
from pathlib import Path
from typing import Any
import yaml


DEFAULT_CONFIG = {
    "server": {
        "host": "localhost",
        "port": 8998,
        "quantize": "4bit",
        "device": "cuda",
    },
    "defaults": {
        "voice_prompt": "NATF2.pt",
        "text_prompt": "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
    },
    "model": {
        "hf_repo": "nvidia/personaplex-7b-v1",
    },
    "gui": {
        "theme": "default",
        "auto_start_server": True,
    },
}


# Voice prompt categories for the GUI
VOICE_PROMPTS = {
    "Natural Female": ["NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt"],
    "Natural Male": ["NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt"],
    "Variety Female": ["VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt"],
    "Variety Male": ["VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt"],
}

# All voices as a flat list
ALL_VOICES = [voice for voices in VOICE_PROMPTS.values() for voice in voices]

# Preset text prompts
TEXT_PROMPT_PRESETS = {
    "Assistant": "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
    "Medical Office": "You work for Dr. Jones's medical office and your name is Sarah. Information: Verify patient name. Office hours: Monday-Friday 8am-5pm. Next available appointment: Tomorrow at 2pm. Bring insurance card and photo ID.",
    "Bank": "You work for First Neuron Bank and your name is Michael. Information: Verify customer account number. Current hours: 9am-5pm weekdays. ATM available 24/7. New accounts require two forms of ID.",
    "Restaurant": "You work for Jerusalem Shakshuka which is a restaurant and your name is Owen Foster. Information: There are two shakshuka options: Classic (poached eggs, $9.50) and Spicy (scrambled eggs with jalapenos, $10.25). Sides include warm pita ($2.50) and Israeli salad ($3).",
    "Astronaut": "You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex. You are already dealing with a reactor core meltdown on a Mars mission. Several ship systems are failing, and continued instability will lead to catastrophic failure. You explain what is happening and you urgently ask for help thinking through how to stabilize the reactor.",
    "Casual Chat": "You enjoy having a good conversation.",
}


class ConfigManager:
    """Manages configuration for PersonaPlex."""

    def __init__(self, config_path: str | Path | None = None):
        """Initialize the config manager.

        Args:
            config_path: Path to config.yaml. If None, uses default location.
        """
        if config_path is None:
            # Default to config.yaml in the project root
            self.config_path = Path(__file__).parent.parent / "config.yaml"
        else:
            self.config_path = Path(config_path)

        self._config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from file or return defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded = yaml.safe_load(f) or {}
                # Merge with defaults to ensure all keys exist
                return self._merge_configs(DEFAULT_CONFIG, loaded)
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()

    def _merge_configs(self, default: dict, override: dict) -> dict:
        """Deep merge override into default."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def save(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., "server.port")
            default: Default value if key not found

        Returns:
            The configuration value or default
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """Set a configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., "server.port")
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    @property
    def server_host(self) -> str:
        return self.get("server.host", "localhost")

    @server_host.setter
    def server_host(self, value: str):
        self.set("server.host", value)

    @property
    def server_port(self) -> int:
        return self.get("server.port", 8998)

    @server_port.setter
    def server_port(self, value: int):
        self.set("server.port", value)

    @property
    def quantize(self) -> str:
        return self.get("server.quantize", "4bit")

    @quantize.setter
    def quantize(self, value: str):
        self.set("server.quantize", value)

    @property
    def device(self) -> str:
        return self.get("server.device", "cuda")

    @device.setter
    def device(self, value: str):
        self.set("server.device", value)

    @property
    def default_voice(self) -> str:
        return self.get("defaults.voice_prompt", "NATF2.pt")

    @default_voice.setter
    def default_voice(self, value: str):
        self.set("defaults.voice_prompt", value)

    @property
    def default_text_prompt(self) -> str:
        return self.get("defaults.text_prompt", TEXT_PROMPT_PRESETS["Assistant"])

    @default_text_prompt.setter
    def default_text_prompt(self, value: str):
        self.set("defaults.text_prompt", value)

    @property
    def hf_repo(self) -> str:
        return self.get("model.hf_repo", "nvidia/personaplex-7b-v1")

    @property
    def auto_start_server(self) -> bool:
        return self.get("gui.auto_start_server", True)

    @auto_start_server.setter
    def auto_start_server(self, value: bool):
        self.set("gui.auto_start_server", value)

    def get_all_voices(self) -> list[str]:
        """Get list of all available voice prompts."""
        return ALL_VOICES

    def get_voice_categories(self) -> dict[str, list[str]]:
        """Get voice prompts organized by category."""
        return VOICE_PROMPTS

    def get_text_presets(self) -> dict[str, str]:
        """Get preset text prompts."""
        return TEXT_PROMPT_PRESETS


# Global config instance
_config_instance: ConfigManager | None = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance
