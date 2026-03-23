"""
MCP adapter that stores/verifies local context and notifies a local Ollama instance.
- Stores context artifacts locally (./mcp_store/)
- Computes a compact embedding key (SHA256) and a simple anomaly/confidence score
- Optionally notifies local Ollama via its /api/chat endpoint with a short metadata message
- Minimal, dependency-light: requests, joblib, numpy, hashlib, json, os
"""

import os
from dotenv import load_dotenv
import json
import hashlib
import time
from typing import Dict, Any, Optional, Sequence
import numpy as np
import joblib
import requests
from esn_reservoir import ESN
from logger import get_logger

# Initialize logger for mcp_adapter
logger = get_logger("mcp_adapter")

# Load environment variables
load_dotenv()
env = os.environ

MCP_STORE_DIR = os.path.join(os.getcwd(), "mcp_store")
os.makedirs(MCP_STORE_DIR, exist_ok=True)

# Default local Ollama endpoint (adjust if your Ollama uses a different port)
OLLAMA_CHAT_URL = env.get("OLLAMA_CHAT_URL")
OLLAMA_MODEL = env.get("OLLAMA_MODEL")


def _embedding_hash(embedding: Sequence[float]) -> str:
    """Return a short hex digest for an embedding (used as stable id)."""
    b = np.asarray(embedding, dtype=np.float32).tobytes()
    return hashlib.sha256(b).hexdigest()


def save_context_local(context_id: str, payload: Dict[str, Any]) -> str:
    """Persist context payload to disk as JSON and return filepath."""
    filename = f"{context_id}.json"
    path = os.path.join(MCP_STORE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def notify_ollama(context_id: str, meta: Dict[str, Any], ollama_url: str = OLLAMA_CHAT_URL, model: str = OLLAMA_MODEL) -> Optional[Dict[str, Any]]:
    """
    Send a short metadata notification to local Ollama chat API.
    This is a lightweight notification only; it does not attempt to store the embedding in Ollama.
    Returns Ollama response JSON on success, otherwise None.
    """
    try:
        # Keep message short: id, score, tag, timestamp
        msg = {
            "role": "user",
            "content": f"[MCP] new_context id={context_id} score={meta.get('score'):.4f} tag={meta.get('tag','')}"
        }
        payload = {"model": model, "messages": [msg]}
        resp = requests.post(ollama_url, json=payload, timeout=6)
        resp.raise_for_status()
        
        # Handle response more robustly
        try:
            result = resp.json()
            logger.debug(f"Ollama response received: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Ollama JSON response: {e}")
            logger.debug(f"Raw response content: {resp.text[:500]}...")
            # Return basic success info even if JSON parsing fails
            return {"status": "sent", "raw_response": resp.text[:200]}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to Ollama failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in notify_ollama: {e}")
        return None


class MCPAdapter:
    """
    Simple MCP adapter class.
    Usage:
      adapter = MCPAdapter(ollama_notify=True)
      ctx = adapter.push_context(window_id, embedding, score=0.12, tag="anomaly")
    """

    def __init__(self, ollama_notify: bool = True, ollama_model: str = OLLAMA_MODEL, ollama_url: str = OLLAMA_CHAT_URL):
        self.ollama_notify = ollama_notify
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

    def push_context(self, raw_window: Sequence[float], embedding: Sequence[float], score: float, tag: str = "") -> Dict[str, Any]:
        """
        Save context locally and optionally notify Ollama.
        Returns a dict with context metadata including local path and optional ollama response.
        """
        emb_hash = _embedding_hash(embedding)
        context_id = f"ctx-{emb_hash[:12]}"
        timestamp = int(time.time())

        payload = {
            "id": context_id,
            "timestamp": timestamp,
            "tag": tag,
            "score": float(score),
            "embedding_hash": emb_hash,
            "embedding_len": len(embedding),
            "window_summary": {
                "length": len(raw_window),
                "mean": float(np.mean(raw_window)),
                "std": float(np.std(raw_window)),
            },
            # store raw window and embedding for reproducibility
            "raw_window": list(map(float, raw_window)),
            "embedding": list(map(float, embedding)),
        }

        path = save_context_local(context_id, payload)
        payload["path"] = path

        ollama_resp = None
        if self.ollama_notify:
            ollama_resp = notify_ollama(context_id, {"score": score, "tag": tag}, ollama_url=self.ollama_url, model=self.ollama_model)
            payload["ollama_notified"] = bool(ollama_resp)
            payload["ollama_response"] = ollama_resp

        return payload


def simple_anomaly_score(embedding: Sequence[float]) -> float:
    """
    A tiny anomaly/confidence score: normalized L2 distance from zero or from a small running baseline.
    Replace with a learned readout or distance-to-cluster centroid for production.
    """
    emb = np.asarray(embedding, dtype=np.float32)
    score = float(np.linalg.norm(emb) / (np.sqrt(len(emb)) + 1e-9))
    # normalize to 0..1 roughly using a soft cap
    return float(np.tanh(score))


def load_readout_and_embedder(model_path: str):
    """
    Convenience loader for a saved reservoir artifact that exposes an embed(window) function.
    Expected artifact saved with joblib containing keys for different reservoir types:
    - ESN: 'esn', 'readout', 'mu', 'sigma', 'window_size'
    - LSM: 'lsm', 'readout', 'mu', 'sigma', 'window_size'  
    - Physical: 'physical_reservoir', 'readout', 'mu', 'sigma', 'window_size'
    Returns a callable embed(window) -> embedding (1D float array) and readout object.
    """
    
    artifact = joblib.load(model_path)
    
    # Try different reservoir keys in order of preference
    reservoir = artifact.get("esn") or artifact.get("lsm") or artifact.get("physical_reservoir")
    readout = artifact.get("readout")
    mu = artifact.get("mu", 0.0)
    sigma = artifact.get("sigma", 1.0)
    window_size = artifact.get("window_size", None)

    if reservoir is None:
        raise ValueError("Reservoir artifact missing reservoir key (expected 'esn', 'lsm', or 'physical_reservoir')")

    if window_size is not None:
        logger.info(f"Loaded reservoir model with window_size: {window_size}")
    else:
        logger.warning("No window_size found in model artifact - validation disabled")

    def embed(window: Sequence[float]):
        # validate window size if specified
        if window_size is not None and len(window) != window_size:
            raise ValueError(f"Expected window size {window_size}, got {len(window)}")
        
        # normalize
        arr = np.asarray(window, dtype=np.float32)
        arr_n = (arr - mu) / (sigma + 1e-9)
        # Reservoir expects sequence shaped (T, 1)
        seq = arr_n.reshape(len(arr_n), 1)
        states = reservoir.run(seq)
        # use last state as embedding
        return states[-1].astype(float)

    return embed, readout
