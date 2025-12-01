import torch
import pytest
from eit_lossless import AdvancedEITLossless


@pytest.mark.parametrize("strategy", ["prefix", "random", "suffix"])
@pytest.mark.parametrize("ratio", [0.5, 0.75, 0.9, 0.95])
def test_lossless_recovery(strategy, ratio):
    """Test 100% exact recovery for all strategies and ratios"""
    eit = AdvancedEITLossless(freeze_strategy=strategy, freeze_ratio=ratio)
    tokens = torch.randn(2, 1024, 2048)
    
    frozen, count = eit.freeze(tokens)
    assert count > 0, "Must freeze some tokens"
    
    processed = frozen + torch.randn_like(frozen) * 0.1
    restored = eit.restore(processed)
    
    mse = torch.mean((tokens - restored)**2).item()
    assert mse < 1e-5, f"MSE too high: {mse}"
    assert torch.allclose(tokens, restored, atol=1e-6), "Must be lossless"


def test_memory_efficiency():
    """Test memory savings from freezing"""
    eit = AdvancedEITLossless(freeze_ratio=0.9)
    large_tokens = torch.randn(1, 1_000, 512)

    frozen, count = eit.freeze(large_tokens)
    assert count == 900, f"Expected 900 frozen, got {count}"


def test_batch_processing():
    """Test batch processing"""
    eit = AdvancedEITLossless(freeze_ratio=0.8)
    batch_size = 4
    seq_len = 512
    d_model = 1024
    
    tokens = torch.randn(batch_size, seq_len, d_model)
    frozen, count = eit.freeze(tokens)
    
    assert frozen.shape == tokens.shape
    assert count == int(batch_size * seq_len * 0.8)


def test_clear():
    """Test clearing state"""
    eit = AdvancedEITLossless(freeze_ratio=0.9)
    tokens = torch.randn(1, 512, 512)
    
    eit.freeze(tokens)
    assert eit.frozen_count > 0
    
    eit.clear()
    assert eit.frozen_count == 0
    assert eit.backup is None
    assert eit.freeze_mask is None


def test_dtype_preservation():
    """Test that output dtype matches input"""
    eit = AdvancedEITLossless(freeze_ratio=0.9)
    
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        tokens = torch.randn(1, 512, 512, dtype=dtype)
        frozen, _ = eit.freeze(tokens)
        assert frozen.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])