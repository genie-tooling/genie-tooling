# src/genie_tooling/prompts/impl/file_system_prompt_registry.py
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

from genie_tooling.prompts.abc import PromptRegistryPlugin
from genie_tooling.prompts.types import PromptIdentifier

logger = logging.getLogger(__name__)

class FileSystemPromptRegistryPlugin(PromptRegistryPlugin):
    plugin_id: str = "file_system_prompt_registry_v1"
    description: str = "Loads prompt templates from a specified directory structure on the file system."

    _base_path: Optional[Path] = None
    _template_suffix: str = ".prompt" # e.g., my_prompt.v1.prompt or my_prompt.prompt

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        base_path_str = cfg.get("base_path", "./prompt_templates")
        self._template_suffix = cfg.get("template_suffix", self._template_suffix)
        
        try:
            self._base_path = Path(base_path_str).resolve()
            if not self._base_path.is_dir():
                logger.warning(f"{self.plugin_id}: Base path '{self._base_path}' does not exist or is not a directory. Creating it.")
                self._base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"{self.plugin_id}: Initialized. Base path: {self._base_path}, Suffix: {self._template_suffix}")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error setting up base path '{base_path_str}': {e}", exc_info=True)
            self._base_path = None

    async def get_template_content(self, name: str, version: Optional[str] = None) -> Optional[str]:
        if not self._base_path:
            logger.error(f"{self.plugin_id}: Base path not configured or invalid. Cannot load template.")
            return None

        # Construct filename: name.v<version>.suffix or name.suffix
        filename_parts = [name]
        if version:
            filename_parts.append(f"v{version}")
        
        # Try with version first, then without if version was provided but not found
        possible_filenames = []
        if version:
            possible_filenames.append(f"{'.'.join(filename_parts)}{self._template_suffix}")
        possible_filenames.append(f"{name}{self._template_suffix}") # Fallback or if no version

        for fname in possible_filenames:
            template_file = self._base_path / fname
            if await asyncio.to_thread(template_file.is_file): # Use to_thread for sync Path.is_file
                try:
                    async with aiofiles.open(template_file, mode="r", encoding="utf-8") as f:
                        content = await f.read()
                    logger.debug(f"{self.plugin_id}: Loaded template '{name}' (version: {version or 'latest'}) from '{template_file}'.")
                    return content
                except Exception as e:
                    logger.error(f"{self.plugin_id}: Error reading template file '{template_file}': {e}", exc_info=True)
                    return None
        
        logger.warning(f"{self.plugin_id}: Template '{name}' (version: {version or 'any'}) not found in '{self._base_path}' with suffix '{self._template_suffix}'. Tried: {possible_filenames}")
        return None

    async def list_available_templates(self) -> List[PromptIdentifier]:
        if not self._base_path:
            logger.error(f"{self.plugin_id}: Base path not configured. Cannot list templates.")
            return []

        templates: List[PromptIdentifier] = []
        try:
            # Path.rglob is synchronous, run in executor
            loop = asyncio.get_running_loop()
            files = await loop.run_in_executor(None, list, self._base_path.rglob(f"*{self._template_suffix}"))

            for file_path in files:
                if await asyncio.to_thread(file_path.is_file): # Check if it's a file asynchronously
                    name_parts = file_path.name.removesuffix(self._template_suffix).split('.')
                    base_name = name_parts[0]
                    version: Optional[str] = None
                    if len(name_parts) > 1 and name_parts[1].startswith('v') and name_parts[1][1:].isdigit():
                        version = name_parts[1][1:]
                    
                    # For description, we could try to read a comment from the file or a companion .meta file
                    # For now, keeping it simple.
                    templates.append(PromptIdentifier(name=base_name, version=version, description=f"Template loaded from {file_path.name}"))
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error listing templates in '{self._base_path}': {e}", exc_info=True)
        
        logger.debug(f"{self.plugin_id}: Found {len(templates)} templates.")
        return templates

    async def teardown(self) -> None:
        logger.debug(f"{self.plugin_id}: Teardown complete.")
