# ares-mini/vllm_manager.py

import asyncio
import sys
import os
import json
from typing import List, Dict, Optional, Any

# Es importante que el logger se configure antes de que se use
def get_logger():
    import logging
    logger = logging.getLogger("ares-mini.vllm_manager")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger()

class VLLMServerManager:
    """Gestiona el ciclo de vida de un servidor vLLM como un subproceso."""
    def __init__(self, config: AresMiniConfig):
        self.config: Dict[str, Any] = config.model_dump().get("vllm_server", {})
        self.process: Optional[asyncio.subprocess.Process] = None
        self.server_ready = asyncio.Event()

    def _build_command(self) -> List[str]:
        """Construye la lista de argumentos para vLLM a partir de la configuración."""
        if not self.config.get("enabled", False) or not self.config.get("model"):
            return []

        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        
        # Mapeo de claves de config a flags de vLLM
        # Esto nos da control total desde el TOML
        arg_map = {
            "model": "--model",
            "host": "--host",
            "port": "--port",
            "tensor_parallel_size": "--tensor-parallel-size",
            "gpu_memory_utilization": "--gpu-memory-utilization",
            "max_model_len": "--max-model-len",
            "dtype": "--dtype",
            "enforce_eager": "--enforce-eager",
            "chat_template": "--chat-template", # <-- AÑADIDO
        }

        for key, flag in arg_map.items():
            if key in self.config:
                command.extend([flag, str(self.config[key])])

        # Argumentos extra que no tienen un valor simple
        if self.config.get("disable_log_requests"):
            command.append("--disable-log-requests")
        
        # ¡Importante! Aquí desactivamos explícitamente modos que no queremos
        # para los modelos base. vLLM no tiene un flag para "modo razonamiento",
        # pero sí para `tool-call-parser`, que es lo que lo activa. Al NO añadirlo,
        # nos aseguramos de que no se use.
        # Si un modelo lo requiriera, se añadiría aquí condicionalmente.
        
        logger.info(f"Comando vLLM construido: {' '.join(command)}")
        return command

    async def _monitor_stream(self, stream: asyncio.StreamReader, prefix: str):
        """Lee y loguea la salida del subproceso para detectar cuándo está listo."""
        ready_message = "Uvicorn running on"
        while not stream.at_eof():
            try:
                line = await stream.readline()
                if not line: break
                decoded_line = line.decode(errors='ignore').strip()
                if decoded_line:
                    logger.info(f"[{prefix}] {decoded_line}")
                    if ready_message in decoded_line and not self.server_ready.is_set():
                        logger.info("¡Servidor vLLM detectado como listo!")
                        self.server_ready.set()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error leyendo stream {prefix}: {e}")
                break

    async def start_server(self, timeout: int = 300):
        """Inicia el servidor vLLM y espera a que esté listo."""
        command = self._build_command()
        if not command or (self.process and self.process.returncode is None):
            logger.info("El servidor vLLM ya está en ejecución o está deshabilitado.")
            return

        logger.info("Iniciando servidor vLLM autogestionado...")
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # Monitorear stdout y stderr en segundo plano
        asyncio.create_task(self._monitor_stream(self.process.stdout, "VLLM_OUT"))
        asyncio.create_task(self._monitor_stream(self.process.stderr, "VLLM_ERR"))

        try:
            await asyncio.wait_for(self.server_ready.wait(), timeout=timeout)
            logger.info("Servidor vLLM confirmado como listo.")
        except asyncio.TimeoutError:
            logger.error(f"El servidor vLLM no estuvo listo en {timeout} segundos. Abortando.")
            await self.stop_server()
            raise RuntimeError("Timeout esperando al servidor vLLM.")

    async def stop_server(self):
        """Detiene el servidor vLLM de forma segura."""
        if self.process and self.process.returncode is None:
            logger.info("Deteniendo el servidor vLLM...")
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=15)
                logger.info("Servidor vLLM detenido correctamente.")
            except asyncio.TimeoutError:
                logger.warning("El servidor no respondió. Forzando detención (kill)...")
                self.process.kill()
            self.process = None