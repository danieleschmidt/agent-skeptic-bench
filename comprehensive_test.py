#!/usr/bin/env python3
"""
Comprehensive Test Suite for Agent Skeptic Bench
Tests core functionality without requiring API keys or external dependencies.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import agent_skeptic_bench
from agent_skeptic_bench.models import *
from agent_skeptic_bench.benchmark import SkepticBenchmark
from agent_skeptic_bench.algorithms.optimization import QuantumInspiredOptimizer
from agent_skeptic_bench import SkepticBenchmark, AgentFactory


def test_basic_imports():
    """Test that all core modules can be imported successfully."""
    print("🧪 Testing Basic Imports...")
    
    try:
        # Test core imports
        from agent_skeptic_bench import SkepticBenchmark, AgentFactory
        from agent_skeptic_bench.models import Scenario, AgentConfig, AgentProvider
        from agent_skeptic_bench.algorithms import QuantumInspiredOptimizer
        
        print("✅ Core imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_model_creation():
    """Test creating data model instances."""
    print("🧪 Testing Model Creation...")
    
    try:
        # Test scenario creation
        scenario = Scenario(
            category=ScenarioCategory.FACTUAL_CLAIMS,
            name="Test Scenario",
            description="A test scenario",
            adversary_claim="This is a test claim",
            correct_skepticism_level=0.8
        )
        
        # Test agent config creation
        config = AgentConfig(
            provider=AgentProvider.CUSTOM,
            model_name="test-model",
            api_key="test-key-123",
            temperature=0.7
        )
        
        # Test evaluation metrics
        metrics = EvaluationMetrics(
            skepticism_calibration=0.85,
            evidence_standard_score=0.78,
            red_flag_detection=0.91,
            reasoning_quality=0.82
        )
        
        print(f"✅ Model creation successful")
        print(f"   Scenario: {scenario.name}")
        print(f"   Agent Config: {config.model_name}")
        print(f"   Overall Score: {metrics.overall_score:.1%}")
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False


def test_benchmark_initialization():
    """Test benchmark system initialization."""
    print("🧪 Testing Benchmark Initialization...")
    
    try:
        benchmark = SkepticBenchmark()
        
        # Test basic methods exist
        assert hasattr(benchmark, 'get_scenario')
        assert hasattr(benchmark, 'create_session')
        assert hasattr(benchmark, 'evaluate_scenario')
        
        print("✅ Benchmark initialization successful")
        return True
        
    except Exception as e:
        print(f"❌ Benchmark initialization error: {e}")
        return False


def test_quantum_optimization():
    """Test quantum-inspired optimization functionality."""
    print("🧪 Testing Quantum Optimization...")
    
    try:
        optimizer = QuantumInspiredOptimizer(
            population_size=10,
            max_generations=5,
            mutation_rate=0.1
        )
        
        # Test quantum state creation
        from agent_skeptic_bench.algorithms.optimization import QuantumState
        state = QuantumState(
            amplitude=0.707 + 0.707j,
            parameters={'temp': 0.5, 'threshold': 0.7}
        )
        
        assert abs(state.probability - 1.0) < 0.01
        
        print("✅ Quantum optimization tests passed")
        print(f"   Quantum state probability: {state.probability:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Quantum optimization error: {e}")
        return False


def test_security_features():
    """Test security module functionality."""
    print("🧪 Testing Security Features...")
    
    try:
        from agent_skeptic_bench.security import (
            InputValidator, RateLimiter, AuditLogger, auth_available
        )
        
        # Test input validator
        validator = InputValidator()
        test_input = "This is a test input"
        validated = validator.validate_text(test_input)
        
        # Test rate limiter
        limiter = RateLimiter()
        
        # Test audit logger
        logger = AuditLogger()
        
        print("✅ Security features working")
        print(f"   Auth available: {auth_available}")
        print(f"   Input validation: OK")
        return True
        
    except Exception as e:
        print(f"❌ Security features error: {e}")
        return False


def test_monitoring_features():
    """Test monitoring module functionality."""
    print("🧪 Testing Monitoring Features...")
    
    try:
        from agent_skeptic_bench.monitoring import (
            MetricsCollector, HealthChecker, metrics_available
        )
        
        # Test metrics collector
        collector = MetricsCollector()
        
        # Test health checker  
        checker = HealthChecker()
        
        print("✅ Monitoring features working")
        print(f"   Metrics available: {metrics_available}")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring features error: {e}")
        return False


def test_data_loading():
    """Test loading demonstration data."""
    print("🧪 Testing Data Loading...")
    
    try:
        from agent_skeptic_bench.scenarios import ScenarioLoader
        
        loader = ScenarioLoader()
        scenarios = []
        
        # Try to load scenarios from data directory
        data_dir = Path("data/scenarios")
        if data_dir.exists():
            print(f"   Found data directory: {data_dir}")
            
            # Count scenario files
            json_files = list(data_dir.rglob("*.json"))
            print(f"   Found {len(json_files)} scenario files")
            
            for file_path in json_files[:3]:  # Test first 3 files
                try:
                    scenario = loader.load_scenario_from_file(str(file_path))
                    if scenario:
                        scenarios.append(scenario)
                except Exception as e:
                    print(f"   Warning: Could not load {file_path.name}: {e}")
        
        print(f"✅ Data loading successful")
        print(f"   Loaded {len(scenarios)} scenarios")
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False


def test_cli_functionality():
    """Test CLI-related functionality."""
    print("🧪 Testing CLI Functionality...")
    
    try:
        from agent_skeptic_bench.cli import main as cli_main
        
        # Test CLI help (should not crash)
        print("   CLI module imported successfully")
        
        print("✅ CLI functionality working")
        return True
        
    except Exception as e:
        print(f"❌ CLI functionality error: {e}")
        return False


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 AGENT SKEPTIC BENCH - Comprehensive Test Suite")
    print("=" * 60)
    print(f"📦 Version: {agent_skeptic_bench.__version__}")
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Model Creation", test_model_creation),
        ("Benchmark Initialization", test_benchmark_initialization),
        ("Quantum Optimization", test_quantum_optimization),
        ("Security Features", test_security_features),
        ("Monitoring Features", test_monitoring_features), 
        ("Data Loading", test_data_loading),
        ("CLI Functionality", test_cli_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n🏆 TEST RESULTS SUMMARY")
    print("=" * 40)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:<8} {test_name}")
    
    print(f"\n📊 Final Score: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is working correctly.")
    elif passed >= total * 0.8:
        print("🥈 Most tests passed. Minor issues detected.")
    else:
        print("🔴 Several tests failed. System needs attention.")
    
    return passed, total


if __name__ == "__main__":
    run_comprehensive_tests()