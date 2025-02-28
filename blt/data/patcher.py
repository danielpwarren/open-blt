import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class PatchingMode(str, Enum):
    """Defines different modes for creating patches"""
    ENTROPY = "entropy"  # Create patches based on entropy
    STATIC = "static"    # Create fixed-size patches
    SPACE = "space"      # Create patches based on whitespace

class Patcher:
    """
    Creates patches from a sequence of tokens.
    
    A patch is a contiguous sequence of tokens that will be processed together.
    Different patching modes determine how tokens are grouped into patches.
    """
    def __init__(
        self,
        patch_size: float = 4,
        patching_mode: str = "static",
        patching_threshold: Optional[float] = None,
        max_patch_length: Optional[int] = None,
        monotonicity: bool = False,
    ):
        """
        Initialize a Patcher.
        
        Args:
            patch_size: Average patch size for static patching or the minimum
                        patch length for entropy-based patching.
            patching_mode: Mode for creating patches ("static", "entropy", or "space").
            patching_threshold: Threshold for entropy-based patching.
            max_patch_length: Maximum length of a patch.
            monotonicity: Whether to enforce monotonicity in entropy-based patching.
        """
        self.patch_size = patch_size
        self.patching_mode = patching_mode
        self.threshold = patching_threshold
        self.max_patch_length = max_patch_length
        self.monotonicity = monotonicity
        
    def _compute_entropy(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Compute the entropy of token predictions.
        
        Args:
            preds: Token prediction scores [batch_size, seq_len, vocab_size]
            
        Returns:
            Entropy values [batch_size, seq_len]
        """
        log_probs = F.log_softmax(preds, dim=-1)
        probs = torch.exp(log_probs)
        p_log_p = log_probs * probs
        return -p_log_p.sum(dim=-1)
    
    def _create_static_patches(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Create patches of fixed size.
        
        Args:
            tokens: Input tokens [batch_size, seq_len]
            
        Returns:
            Patch lengths [batch_size, num_patches]
        """
        bs, seq_len = tokens.shape
        
        # Calculate number of patches and patch lengths
        patch_size = int(self.patch_size)
        num_patches = math.ceil(seq_len / patch_size)
        
        # Create tensor of patch lengths
        patch_lengths = torch.full(
            (bs, num_patches),
            patch_size,
            dtype=torch.long,
            device=tokens.device
        )
        
        # Adjust the last patch length if it's smaller
        if seq_len % patch_size != 0:
            patch_lengths[:, -1] = seq_len % patch_size
            
        return patch_lengths
    
    def _create_space_patches(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Create patches based on whitespace (e.g., space, newline, tab).
        
        Args:
            tokens: Input tokens [batch_size, seq_len]
            
        Returns:
            Patch lengths [batch_size, num_patches]
        """
        bs, seq_len = tokens.shape
        
        # Space tokens (common ASCII whitespace: space, tab, newline)
        # Adjusted for the offset in the tokenizer (typically +5)
        offset = 5  # This should match the tokenizer's offset
        space_tokens = [32 + offset, 9 + offset, 10 + offset, 13 + offset]
        
        # Process each sequence in the batch
        batch_patch_lengths = []
        for b in range(bs):
            seq = tokens[b]
            
            # Find whitespace positions
            space_positions = [i for i, t in enumerate(seq) if t in space_tokens]
            space_positions = [-1] + space_positions + [seq_len - 1]
            
            # Calculate patch lengths
            patch_lengths = []
            for i in range(1, len(space_positions)):
                length = space_positions[i] - space_positions[i-1]
                
                # Apply max_patch_length if specified
                if self.max_patch_length is not None and length > self.max_patch_length:
                    # Split into multiple patches
                    full_patches = length // self.max_patch_length
                    for _ in range(full_patches):
                        patch_lengths.append(self.max_patch_length)
                    remainder = length % self.max_patch_length
                    if remainder > 0:
                        patch_lengths.append(remainder)
                else:
                    patch_lengths.append(length)
            
            batch_patch_lengths.append(patch_lengths)
        
        # Pad to the maximum number of patches in the batch
        max_num_patches = max(len(patches) for patches in batch_patch_lengths)
        padded_lengths = []
        for lengths in batch_patch_lengths:
            padded = lengths + [0] * (max_num_patches - len(lengths))
            padded_lengths.append(padded)
        
        return torch.tensor(padded_lengths, device=tokens.device)
    
    def _create_entropy_patches(
        self, 
        tokens: torch.Tensor, 
        preds: torch.Tensor
    ) -> torch.Tensor:
        """
        Create patches based on prediction entropy.
        
        Args:
            tokens: Input tokens [batch_size, seq_len]
            preds: Token predictions [batch_size, seq_len, vocab_size]
            
        Returns:
            Patch lengths [batch_size, num_patches]
        """
        bs, seq_len = tokens.shape
        
        # Compute entropy from predictions
        entropies = self._compute_entropy(preds)
        
        # Find positions with high entropy (potential patch boundaries)
        batch_patch_lengths = []
        for b in range(bs):
            entropy_seq = entropies[b]
            
            # Find positions where entropy exceeds threshold
            if self.monotonicity:
                # Start new patch when entropy increases above threshold
                entropy_diff = torch.cat([
                    torch.tensor([0.0], device=entropy_seq.device),
                    entropy_seq[1:] - entropy_seq[:-1]
                ])
                high_entropy_positions = [i for i, diff in enumerate(entropy_diff) if diff > self.threshold]
            else:
                # Start new patch when entropy exceeds threshold
                high_entropy_positions = [i for i, e in enumerate(entropy_seq) if e > self.threshold]
            
            # Always start with position 0
            if 0 not in high_entropy_positions:
                high_entropy_positions = [0] + high_entropy_positions
                
            # Sort positions
            high_entropy_positions.sort()
            
            # Add sequence end
            if seq_len - 1 not in high_entropy_positions:
                high_entropy_positions.append(seq_len - 1)
                
            # Calculate patch lengths
            patch_lengths = []
            for i in range(1, len(high_entropy_positions)):
                length = high_entropy_positions[i] - high_entropy_positions[i-1] + 1
                
                # Apply max_patch_length if specified
                if self.max_patch_length is not None and length > self.max_patch_length:
                    # Split into multiple patches
                    full_patches = length // self.max_patch_length
                    for _ in range(full_patches):
                        patch_lengths.append(self.max_patch_length)
                    remainder = length % self.max_patch_length
                    if remainder > 0:
                        patch_lengths.append(remainder)
                else:
                    patch_lengths.append(length)
            
            batch_patch_lengths.append(patch_lengths)
            
        # Pad to the maximum number of patches in the batch
        max_num_patches = max(len(patches) for patches in batch_patch_lengths)
        padded_lengths = []
        for lengths in batch_patch_lengths:
            padded = lengths + [0] * (max_num_patches - len(lengths))
            padded_lengths.append(padded)
        
        return torch.tensor(padded_lengths, device=tokens.device)
        
    def patch(
        self,
        tokens: torch.Tensor,
        preds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Patch a sequence of tokens.
        
        Args:
            tokens: Input tokens [batch_size, seq_len]
            preds: Token predictions for entropy-based patching [batch_size, seq_len, vocab_size]
            
        Returns:
            Tuple of (patch_lengths, scores)
                patch_lengths: Lengths of each patch [batch_size, num_patches]
                scores: Entropy scores (if computed) or None
        """
        scores = None
        
        if self.patching_mode == PatchingMode.STATIC:
            # Create patches of fixed size
            patch_lengths = self._create_static_patches(tokens)
        elif self.patching_mode == PatchingMode.ENTROPY:
            if preds is None:
                raise ValueError("Token predictions required for entropy-based patching")
            # Create patches based on entropy
            scores = self._compute_entropy(preds)
            patch_lengths = self._create_entropy_patches(tokens, preds)
        elif self.patching_mode == PatchingMode.SPACE:
            # Create patches based on whitespace
            patch_lengths = self._create_space_patches(tokens)
        else:
            raise ValueError(f"Unsupported patching mode: {self.patching_mode}")
        
        # Validate patch lengths - sum should equal sequence length
        sum_lengths = patch_lengths.sum(dim=1)
        if not torch.all(sum_lengths == tokens.shape[1]):
            # This can be a warning in practice, but for testing we'll make it an error
            raise ValueError(
                f"Sum of patch lengths ({sum_lengths}) does not match sequence length ({tokens.shape[1]})"
            )
            
        return patch_lengths, scores 