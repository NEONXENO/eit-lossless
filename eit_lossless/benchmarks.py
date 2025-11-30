"""
EIT Lossless Benchmarks
1M and 10M token performance tests
"""

import torch
import time
from .core import AdvancedEITLossless


def benchmark_1m_tokens():
    """Benchmark 1M token context"""
    print("🚀 Benchmarking 1M tokens...")
    
    eit = AdvancedEITLossless(freeze_ratio=0.9)
    embeddings = torch.randn(1, 1_000_000, 4096, dtype=torch.float16, device='cuda')
    
    start = time.time()
    frozen, count = eit.freeze(embeddings)
    freeze_time = time.time() - start
    
    print(f"❄️  Frozen: {count:,} / 1,000,000 tokens")
    print(f"⏱️  Freeze time: {freeze_time*1000:.1f}ms")
    print(f"💾 Memory saved: {count * 4096 * 2 / 1e9:.1f}GB")
    
    return count, freeze_time


def benchmark_10m_tokens():
    """Benchmark 10M token context"""
    print("\n🔥 Benchmarking 10M tokens...")
    
    eit = AdvancedEITLossless(freeze_ratio=0.95)
    embeddings = torch.randn(1, 10_000_000, 4096, dtype=torch.float16, device='cuda')
    
    start = time.time()
    frozen, count = eit.freeze(embeddings)
    freeze_time = time.time() - start
    
    print(f"❄️  Frozen: {count:,} / 10,000,000 tokens")
    print(f"⏱️  Freeze time: {freeze_time*1000:.1f}ms")
    print(f"💾 Memory saved: {count * 4096 * 2 / 1e9:.1f}GB")
    
    return count, freeze_time


if __name__ == "__main__":
    benchmark_1m_tokens()
    benchmark_10m_tokens()