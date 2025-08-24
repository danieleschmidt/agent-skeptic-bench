#!/usr/bin/env python3
"""
Standalone performance demo for usage metrics optimization.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SimpleMemoryCache:
    """Simple in-memory cache for demo."""
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}  # key -> (value, timestamp)
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            self.misses += 1
            return None
        
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            self.misses += 1
            return None
        
        self.hits += 1
        return value
    
    def set(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, time.time())
    
    def stats(self) -> Dict[str, Any]:
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class SimpleBatchProcessor:
    """Simple batch processor for demo."""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.batch = []
        self.processed_count = 0
    
    def add_item(self, item: Dict[str, Any]) -> bool:
        """Add item to batch, return True if batch was flushed."""
        self.batch.append(item)
        
        if len(self.batch) >= self.batch_size:
            self.flush()
            return True
        
        return False
    
    def flush(self) -> int:
        """Flush current batch."""
        if not self.batch:
            return 0
        
        batch_size = len(self.batch)
        
        # Simulate processing
        storage_dir = Path("data/optimized_batches")
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        batch_file = storage_dir / f"batch_{int(time.time())}.json"
        
        with open(batch_file, "w") as f:
            json.dump(self.batch, f, indent=2, default=str)
        
        self.processed_count += batch_size
        self.batch.clear()
        
        print(f"    💾 Flushed batch of {batch_size} items")
        return batch_size


async def demo_performance_optimization():
    """Demo performance optimization features."""
    print("⚡ Performance Optimization Demo")
    print("-" * 40)
    
    # Test caching performance
    print("\n1. Cache Performance Test:")
    cache = SimpleMemoryCache(max_size=50, ttl=10)
    
    # Populate cache
    for i in range(20):
        cache.set(f"key_{i}", {"value": f"data_{i}", "timestamp": time.time()})
    
    # Test cache hits
    start_time = time.time()
    for i in range(100):
        key = f"key_{i % 20}"
        result = cache.get(key)
        if result is None and i < 20:  # Should hit for first 20
            cache.set(key, {"value": f"data_{i}", "timestamp": time.time()})
    
    cache_time = time.time() - start_time
    stats = cache.stats()
    
    print(f"   Cache operations: 100 in {cache_time:.3f}s")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"   Avg time per op: {cache_time/100*1000:.2f}ms")
    
    # Test batch processing
    print("\n2. Batch Processing Test:")
    batch_processor = SimpleBatchProcessor(batch_size=5)
    
    start_time = time.time()
    for i in range(23):  # Will create multiple batches
        item = {
            "id": f"item_{i}",
            "timestamp": datetime.utcnow().isoformat(),
            "data": f"test_data_{i}"
        }
        
        was_flushed = batch_processor.add_item(item)
        if was_flushed:
            print(f"    ⚡ Auto-flush triggered at item {i}")
    
    # Flush remaining
    remaining = batch_processor.flush()
    if remaining > 0:
        print(f"    🏁 Final flush: {remaining} items")
    
    batch_time = time.time() - start_time
    print(f"   Batch processing: 23 items in {batch_time:.3f}s")
    print(f"   Total processed: {batch_processor.processed_count}")
    print(f"   Avg time per item: {batch_time/23*1000:.2f}ms")
    
    # Test concurrent processing
    print("\n3. Concurrent Processing Test:")
    
    async def process_item(item_id: int, delay: float = 0.1) -> Dict[str, Any]:
        """Simulate processing an item."""
        await asyncio.sleep(delay)
        return {
            "item_id": item_id,
            "processed_at": datetime.utcnow().isoformat(),
            "processing_time": delay
        }
    
    # Sequential processing
    start_time = time.time()
    sequential_results = []
    for i in range(10):
        result = await process_item(i, 0.05)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    # Concurrent processing
    start_time = time.time()
    concurrent_tasks = [process_item(i, 0.05) for i in range(10)]
    concurrent_results = await asyncio.gather(*concurrent_tasks)
    concurrent_time = time.time() - start_time
    
    print(f"   Sequential: 10 items in {sequential_time:.3f}s")
    print(f"   Concurrent: 10 items in {concurrent_time:.3f}s")
    print(f"   Speedup: {sequential_time/concurrent_time:.1f}x")
    
    # Test scaling simulation
    print("\n4. Auto-Scaling Simulation:")
    
    class SimpleAutoScaler:
        def __init__(self):
            self.instances = 1
            self.max_instances = 5
            
        def should_scale_up(self, load: float) -> bool:
            return load > 0.8 and self.instances < self.max_instances
        
        def should_scale_down(self, load: float) -> bool:
            return load < 0.3 and self.instances > 1
        
        def scale_up(self):
            if self.instances < self.max_instances:
                self.instances += 1
                return True
            return False
        
        def scale_down(self):
            if self.instances > 1:
                self.instances -= 1
                return True
            return False
    
    scaler = SimpleAutoScaler()
    
    # Simulate varying load
    load_pattern = [0.2, 0.4, 0.7, 0.9, 0.95, 0.85, 0.6, 0.3, 0.1]
    
    for i, load in enumerate(load_pattern):
        print(f"   Step {i+1}: Load={load:.1%}, Instances={scaler.instances}")
        
        if scaler.should_scale_up(load):
            if scaler.scale_up():
                print(f"    🔼 Scaled up to {scaler.instances} instances")
        elif scaler.should_scale_down(load):
            if scaler.scale_down():
                print(f"    🔽 Scaled down to {scaler.instances} instances")
        
        await asyncio.sleep(0.1)
    
    print(f"   Final configuration: {scaler.instances} instances")
    
    return True


async def demo_export_performance():
    """Demo export performance optimization."""
    print("\n📤 Export Performance Demo")
    print("-" * 40)
    
    # Create test data
    test_data = []
    for i in range(1000):
        test_data.append({
            "session_id": f"session_{i:04d}",
            "user_id": f"user_{i % 10}",
            "timestamp": datetime.utcnow().isoformat(),
            "evaluations": i % 15 + 1,
            "duration": 10.0 + (i % 100),
            "score": 0.6 + (i % 40) / 100
        })
    
    print(f"Created {len(test_data)} test records")
    
    # Test different export formats
    export_dir = Path("exports/performance_test")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    formats = ["json", "csv"]
    
    for fmt in formats:
        print(f"\n  Testing {fmt.upper()} export:")
        
        start_time = time.time()
        
        if fmt == "json":
            output_file = export_dir / "performance_test.json"
            with open(output_file, "w") as f:
                json.dump({
                    "export_info": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "record_count": len(test_data),
                        "format": fmt
                    },
                    "data": test_data
                }, f, indent=2, default=str)
        
        elif fmt == "csv":
            import csv
            output_file = export_dir / "performance_test.csv"
            
            with open(output_file, "w", newline="") as f:
                if test_data:
                    fieldnames = test_data[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(test_data)
        
        export_time = time.time() - start_time
        file_size = output_file.stat().st_size / 1024  # KB
        
        print(f"    Time: {export_time:.3f}s")
        print(f"    Size: {file_size:.1f} KB")
        print(f"    Rate: {len(test_data)/export_time:.0f} records/second")
        print(f"    File: {output_file}")


async def main():
    """Run all performance demos."""
    print("🎯 GENERATION 3: Performance Optimization Demos")
    print("=" * 60)
    
    try:
        await demo_performance_optimization()
        await demo_export_performance()
        
        print(f"\n✅ All performance tests completed successfully!")
        print(f"📊 Key achievements:")
        print(f"  - ⚡ High-performance caching system")
        print(f"  - 📦 Efficient batch processing")
        print(f"  - 🚀 Concurrent operation handling")
        print(f"  - 📈 Auto-scaling simulation")
        print(f"  - 📤 Optimized export performance")
        
    except Exception as e:
        print(f"❌ Performance demo failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        sys.exit(1)