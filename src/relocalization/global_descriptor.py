"""
Global Descriptor Extractor: Fast place recognition via MixVPR or foundation models.

Provides image-level embeddings for retrieval-based place recognition.
Supports MixVPR (best performance) and DINOv2 (zero-shot fallback).
"""

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


class GlobalDescriptorExtractor:
    """Fast place recognition via global image descriptors.
    
    Supports:
    - MixVPR: Task-specific learned descriptors (best for in-distribution scenes)
    - DINOv2: Foundation model descriptors (better zero-shot generalization)
    
    All descriptors are L2-normalized for efficient dot-product matching.
    """
    
    def __init__(self, model_type: str = "mixvpr",
                 device: str = "cuda",
                 cache_dir: Optional[str] = None):
        """Initialize global descriptor extractor.
        
        Args:
            model_type: "mixvpr" or "dinov2"
            device: "cuda" or "cpu"
            cache_dir: Optional directory to cache model weights
        """
        self.model_type = model_type.lower()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.model_type == "mixvpr":
            self._load_mixvpr_model()
        elif self.model_type == "dinov2":
            self._load_dinov2_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.eval()
        self.model = self.model.to(self.device)
    
    def _load_mixvpr_model(self) -> None:
        """Load MixVPR model from HuggingFace or local cache."""
        try:
            from transformers import AutoModel
            
            model_name = "gnutelford/MixVPR-G-65536"
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
            self.embedding_dim = 65536
            self.input_size = 384  # MixVPR input size
        except ImportError:
            raise ImportError(
                "transformers library required for MixVPR. "
                "Install: pip install transformers"
            )
    
    def _load_dinov2_model(self) -> None:
        """Load DINOv2 model from Meta (zero-shot, more general)."""
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.embedding_dim = 384
            self.input_size = 518  # DINOv2 typical input
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv2 model: {e}. "
                "Try: python -m torch.hub.load facebookresearch/dinov2:main dinov2_vits14"
            )
    
    @torch.no_grad()
    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract global descriptor from image.
        
        Args:
            image_rgb: (H, W, 3) uint8 in RGB
        
        Returns:
            descriptor: (D,) float32 L2-normalized
        """
        # Preprocess image
        image_tensor = self._preprocess_image(image_rgb)
        image_tensor = image_tensor.to(self.device)
        
        # Extract descriptor
        if self.model_type == "mixvpr":
            descriptor = self.model(image_tensor)
        elif self.model_type == "dinov2":
            # DINOv2 returns (B, N, D) where N = num_patches
            # We take CLS token (first token) and global average
            try:
                with torch.cuda.amp.autocast():
                    features = self.model.forward_features(image_tensor)
                    # Try multiple possible return formats
                    if isinstance(features, dict) and 'x' in features:
                        descriptor = features['x'][:, 0, :]  # CLS token
                    elif isinstance(features, torch.Tensor):
                        descriptor = features[:, 0, :]  # Direct tensor
                    else:
                        descriptor = features[:, 0, :]  # Fallback
            except Exception:
                # Fallback: direct forward pass
                descriptor = self.model(image_tensor)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # L2 normalize
        descriptor = F.normalize(descriptor, p=2, dim=1)
        
        return descriptor[0].cpu().numpy().astype(np.float32)
    
    def _preprocess_image(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Preprocess image to model input format.
        
        Args:
            image_rgb: (H, W, 3) uint8
        
        Returns:
            (1, 3, H, W) float32 tensor
        """
        # Resize to model input size
        if image_rgb.shape[:2] != (self.input_size, self.input_size):
            image_rgb = cv2.resize(
                image_rgb,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # (3, H, W)
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.unsqueeze(0)  # (1, 3, H, W)
    
    def find_top_k_similar(self, query_desc: np.ndarray,
                          corpus_descs: np.ndarray,
                          k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Fast kNN via cosine similarity.
        
        Assumes both query and corpus are L2-normalized.
        
        Args:
            query_desc: (D,) query descriptor
            corpus_descs: (N, D) corpus of descriptors
            k: Number of top matches to return
        
        Returns:
            top_indices: (k,) indices into corpus, sorted by similarity
            similarities: (k,) cosine similarity scores
        """
        # Cosine similarity = dot product on normalized vectors
        similarities = corpus_descs @ query_desc
        
        # Get top-k indices
        top_indices = np.argsort(-similarities)[:k]
        top_similarities = similarities[top_indices]
        
        return top_indices, top_similarities
    
    def benchmark_extract(self, image_rgb: np.ndarray) -> float:
        """Benchmark descriptor extraction time.
        
        Args:
            image_rgb: Test image
        
        Returns:
            Extraction time in milliseconds
        """
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        
        for _ in range(10):
            _ = self.extract(image_rgb)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.perf_counter() - t0) * 1000 / 10
        
        return elapsed
