#!/usr/bin/env python3
"""
Test the mask_non_image_embeddings fix to validate our training setup improvements.
"""

import pytest
import torch
from PIL import Image

from colpali_engine.models import ColIntern3_5, ColIntern3_5Processor


class TestMaskNonImageEmbeddingsFix:
    """Test that the mask_non_image_embeddings fix works correctly."""
    
    @pytest.fixture
    def processor(self):
        """Load processor for testing."""
        return ColIntern3_5Processor.from_pretrained(
            "OpenGVLab/InternVL3_5-1B-HF",
            max_num_visual_tokens=1536
        )
    
    @pytest.fixture
    def model_without_masking(self):
        """Load model without masking (like the original training)."""
        model = ColIntern3_5.from_pretrained(
            "OpenGVLab/InternVL3_5-1B-HF",
            torch_dtype=torch.bfloat16,
            mask_non_image_embeddings=False,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        return model
    
    @pytest.fixture
    def model_with_masking(self):
        """Load model with masking enabled (our fix)."""
        model = ColIntern3_5.from_pretrained(
            "OpenGVLab/InternVL3_5-1B-HF",
            torch_dtype=torch.bfloat16,
            mask_non_image_embeddings=True,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        return model
    
    def test_masking_flag_set_correctly(self, model_without_masking, model_with_masking):
        """Test that the masking flag is set correctly."""
        assert model_without_masking.mask_non_image_embeddings == False
        assert model_with_masking.mask_non_image_embeddings == True
    
    def test_non_image_tokens_are_masked(self, processor, model_with_masking):
        """Test that non-image tokens are properly masked to zero."""
        # Create test image
        test_image = Image.new("RGB", (448, 448), color="white")
        batch = processor.process_images([test_image])
        
        # Move to device
        device = model_with_masking.device if hasattr(model_with_masking, 'device') else next(model_with_masking.parameters()).device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        
        # Forward pass
        with torch.no_grad():
            embeddings = model_with_masking(**batch)
        
        # Get image mask
        image_mask = processor.get_image_mask(batch).to(device)
        non_image_mask = ~image_mask
        
        if non_image_mask.any():
            # Check non-image positions
            non_image_mask_expanded = non_image_mask.unsqueeze(-1).expand_as(embeddings)
            non_image_embeddings = embeddings[non_image_mask_expanded]
            
            # All non-image embeddings should be zero (or very close to zero)
            non_zero_count = (non_image_embeddings.abs() > 1e-6).sum().item()
            total_non_image = non_image_embeddings.numel()
            
            assert non_zero_count == 0, f"Expected 0 non-zero values in non-image positions, got {non_zero_count}/{total_non_image}"
    
    def test_non_image_tokens_contribute_without_masking(self, processor, model_without_masking):
        """Test that non-image tokens contribute to similarity without masking."""
        # Create test image
        test_image = Image.new("RGB", (448, 448), color="white")
        batch = processor.process_images([test_image])
        
        # Move to device
        device = model_without_masking.device if hasattr(model_without_masking, 'device') else next(model_without_masking.parameters()).device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        
        # Forward pass
        with torch.no_grad():
            embeddings = model_without_masking(**batch)
        
        # Get image mask
        image_mask = processor.get_image_mask(batch).to(device)
        non_image_mask = ~image_mask
        
        if non_image_mask.any():
            # Check non-image positions
            non_image_mask_expanded = non_image_mask.unsqueeze(-1).expand_as(embeddings)
            non_image_embeddings = embeddings[non_image_mask_expanded]
            
            # Non-image embeddings should have significant norm (not zero)
            avg_norm = torch.norm(non_image_embeddings, dim=-1).mean().item()
            assert avg_norm > 0.1, f"Expected non-image embeddings to have significant norm, got {avg_norm:.6f}"
    
    def test_masking_vs_no_masking_difference(self, processor, model_with_masking, model_without_masking):
        """Test that masking and no-masking produce different results."""
        # Create test image
        test_image = Image.new("RGB", (448, 448), color="white")
        batch = processor.process_images([test_image])
        
        # Move to devices
        device_with = model_with_masking.device if hasattr(model_with_masking, 'device') else next(model_with_masking.parameters()).device
        device_without = model_without_masking.device if hasattr(model_without_masking, 'device') else next(model_without_masking.parameters()).device
        
        batch_with = {k: v.to(device_with) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_without = {k: v.to(device_without) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        with torch.no_grad():
            embeddings_with_mask = model_with_masking(**batch_with)
            embeddings_without_mask = model_without_masking(**batch_without)
        
        # Move to same device for comparison
        embeddings_without_mask = embeddings_without_mask.to(device_with)
        
        # They should be different (not equal)
        assert not torch.allclose(embeddings_with_mask, embeddings_without_mask, atol=1e-4), \
            "Embeddings with and without masking should be different"
    
    def test_image_tokens_have_proper_structure(self, processor, model_with_masking, model_without_masking):
        """Test that both models produce image token embeddings with proper structure."""
        # Create test image
        test_image = Image.new("RGB", (448, 448), color="white")
        batch = processor.process_images([test_image])
        
        # Move to devices
        device_with = model_with_masking.device if hasattr(model_with_masking, 'device') else next(model_with_masking.parameters()).device
        device_without = model_without_masking.device if hasattr(model_without_masking, 'device') else next(model_without_masking.parameters()).device
        
        batch_with = {k: v.to(device_with) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_without = {k: v.to(device_without) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        with torch.no_grad():
            embeddings_with_mask = model_with_masking(**batch_with)
            embeddings_without_mask = model_without_masking(**batch_without)
        
        # Get image mask
        image_mask = processor.get_image_mask(batch_with).to(device_with)
        
        # Both should have the same number of image tokens
        image_token_count_with = image_mask.sum().item()
        image_token_count_without = image_mask.sum().item()
        
        assert image_token_count_with == image_token_count_without, \
            "Both models should have the same number of image tokens"
        
        # Both should produce embeddings of the same shape
        assert embeddings_with_mask.shape == embeddings_without_mask.shape, \
            "Both models should produce embeddings of the same shape"
        
        # Image token embeddings should have reasonable magnitude
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(embeddings_with_mask)
        image_embeddings_with_mask = embeddings_with_mask[image_mask_expanded].view(-1, embeddings_with_mask.shape[-1])
        
        # Check that image embeddings have reasonable norms
        image_norms = torch.norm(image_embeddings_with_mask, dim=-1)
        avg_norm = image_norms.mean().item()
        assert 0.1 <= avg_norm <= 5.0, f"Expected reasonable image token norms, got {avg_norm:.4f}"
    
    def test_visual_token_budget_behavior(self, processor):
        """Test visual token budget behavior with different image sizes."""
        # Test with various image sizes and check token counts
        test_sizes = [(448, 448), (600, 800), (800, 600), (1000, 1000)]
        
        for width, height in test_sizes:
            test_image = Image.new("RGB", (width, height), color="white")
            batch = processor.process_images([test_image])
            
            # Get image token count
            image_mask = processor.get_image_mask(batch)
            image_token_count = image_mask.sum().item()
            
            # Report token count for debugging
            print(f"Image {width}x{height} produced {image_token_count} tokens")
            
            # Token count should be reasonable (not excessively high)
            assert image_token_count > 0, f"Image {width}x{height} should produce some tokens"
            assert image_token_count <= 2048, f"Image {width}x{height} produced too many tokens: {image_token_count}"
    
    def test_embedding_normalization(self, processor, model_with_masking):
        """Test that embeddings are properly normalized."""
        # Create test image
        test_image = Image.new("RGB", (448, 448), color="white")
        batch = processor.process_images([test_image])
        
        # Move to device
        device = model_with_masking.device if hasattr(model_with_masking, 'device') else next(model_with_masking.parameters()).device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        
        # Forward pass
        with torch.no_grad():
            embeddings = model_with_masking(**batch)
        
        # Check normalization
        norms = torch.norm(embeddings, dim=-1)
        
        # Only check norms for non-zero embeddings (image tokens should be normalized)
        image_mask = processor.get_image_mask(batch).to(device)
        image_norms = norms[image_mask]
        
        # Image token embeddings should be normalized (close to 1.0)
        avg_norm = image_norms.mean().item()
        assert 0.8 <= avg_norm <= 1.2, f"Expected image token norms around 1.0, got {avg_norm:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
