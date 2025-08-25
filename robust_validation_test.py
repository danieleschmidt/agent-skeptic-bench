#!/usr/bin/env python3
"""
Generation 2 Robustness Validation Test
Tests enhanced error handling, security, and monitoring features
"""

import asyncio
import sys
import traceback
from datetime import datetime

def test_generation_2_robustness():
    """Test Generation 2 robustness features."""
    print("🛡️  GENERATION 2: ROBUSTNESS VALIDATION")
    print("=" * 60)
    
    test_results = {
        'exception_handling': False,
        'input_validation': False,
        'model_validation': False,
        'async_cache': False,
        'rate_limiting': False,
        'health_monitoring': False,
        'security_components': False
    }
    
    # Test 1: Exception Handling
    try:
        from src.agent_skeptic_bench.exceptions import ScenarioNotFoundError
        
        try:
            raise ScenarioNotFoundError('test_scenario')
        except ScenarioNotFoundError as e:
            print(f"✅ Exception handling: {e.scenario_id}")
            test_results['exception_handling'] = True
    except Exception as e:
        print(f"❌ Exception handling failed: {e}")
    
    # Test 2: Input Validation
    try:
        from src.agent_skeptic_bench.security import InputValidator
        validator = InputValidator()
        result = validator.validate_text("Safe input text")
        print(f"✅ Input validation: {len(result)} chars validated")
        test_results['input_validation'] = True
    except Exception as e:
        print(f"❌ Input validation failed: {e}")
    
    # Test 3: Model Validation
    try:
        from src.agent_skeptic_bench.models import AgentConfig, AgentProvider
        config = AgentConfig(
            provider=AgentProvider.MOCK,
            model_name="test-model",
            api_key="test-key",
            temperature=0.5
        )
        print(f"✅ Model validation: {config.model_name}")
        test_results['model_validation'] = True
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
    
    # Test 4: Async Cache Management
    async def test_cache():
        try:
            from src.agent_skeptic_bench.cache import CacheManager
            cache = CacheManager()
            await cache.set("test_key", {"data": "test_value"})
            result = await cache.get("test_key")
            print(f"✅ Async cache: {result['data']}")
            return True
        except Exception as e:
            print(f"❌ Async cache failed: {e}")
            return False
    
    test_results['async_cache'] = asyncio.run(test_cache())
    
    # Test 5: Rate Limiting
    async def test_rate_limiting():
        try:
            from src.agent_skeptic_bench.security.rate_limiting import (
                RateLimiter, RateLimitConfig, RateLimitStrategy, RateLimitScope
            )
            
            limiter = RateLimiter()
            config = RateLimitConfig(
                name="test_limit",
                strategy=RateLimitStrategy.FIXED_WINDOW,
                scope=RateLimitScope.USER,
                limit=100,
                window_seconds=60
            )
            limiter.add_config(config)
            
            result = await limiter.check_rate_limit("test_limit", "user123")
            print(f"✅ Rate limiting: allowed={result.allowed}")
            return True
        except Exception as e:
            print(f"❌ Rate limiting failed: {e}")
            return False
    
    test_results['rate_limiting'] = asyncio.run(test_rate_limiting())
    
    # Test 6: Health Monitoring
    try:
        from src.agent_skeptic_bench.monitoring.health import HealthChecker
        checker = HealthChecker()
        health = checker.get_health_summary()
        print(f"✅ Health monitoring: {health['overall_status']}")
        test_results['health_monitoring'] = True
    except Exception as e:
        print(f"❌ Health monitoring failed: {e}")
    
    # Test 7: Security Components
    try:
        from src.agent_skeptic_bench.security import AuthenticationManager
        auth = AuthenticationManager()
        print("✅ Security components: AuthenticationManager loaded")
        test_results['security_components'] = True
    except Exception as e:
        print(f"❌ Security components failed: {e}")
    
    # Summary
    print("\n📊 ROBUSTNESS TEST RESULTS")
    print("-" * 40)
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests >= total_tests * 0.8:  # 80% pass rate
        print("🎉 GENERATION 2 ROBUSTNESS: VALIDATION SUCCESSFUL!")
        return True
    else:
        print("⚠️  GENERATION 2 ROBUSTNESS: NEEDS IMPROVEMENT")
        return False

if __name__ == "__main__":
    success = test_generation_2_robustness()
    sys.exit(0 if success else 1)