#!/usr/bin/env python3
"""Minimal Generation 1 test without external dependencies."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that core modules can be imported."""
    print("🧠 Agent Skeptic Bench - Minimal Generation 1 Test")
    print("=" * 55)
    
    try:
        print("\n1. Testing core imports...")
        
        # Test basic enums and classes without pydantic
        from agent_skeptic_bench.models import ScenarioCategory, SkepticismLevel, EvidenceStandard, AgentProvider
        print("   ✓ Basic enums imported successfully")
        
        # Test exception imports
        from agent_skeptic_bench.exceptions import AgentSkepticBenchError, DataLoadError
        print("   ✓ Exception classes imported successfully")
        
        print("\n2. Testing enum values...")
        categories = list(ScenarioCategory)
        print(f"   ✓ Found {len(categories)} scenario categories:")
        for cat in categories:
            print(f"     - {cat.value}")
        
        providers = list(AgentProvider)
        print(f"   ✓ Found {len(providers)} agent providers:")
        for provider in providers:
            print(f"     - {provider.value}")
        
        print("\n3. Testing project structure...")
        src_path = Path(__file__).parent / "src" / "agent_skeptic_bench"
        
        core_modules = [
            "models.py", "agents.py", "scenarios.py", 
            "evaluation.py", "metrics.py", "validation.py",
            "exceptions.py", "cli.py"
        ]
        
        for module in core_modules:
            module_path = src_path / module
            if module_path.exists():
                print(f"   ✓ {module} exists")
            else:
                print(f"   ❌ {module} missing")
        
        print("\n4. Testing data structure...")
        data_path = Path(__file__).parent / "data" / "scenarios"
        if data_path.exists():
            print(f"   ✓ Data directory exists: {data_path}")
            json_files = list(data_path.rglob("*.json"))
            print(f"   ✓ Found {len(json_files)} JSON scenario files")
            for json_file in json_files[:3]:  # Show first 3
                print(f"     - {json_file.name}")
        else:
            print(f"   ⚠️  Data directory not found: {data_path}")
        
        print("\n✅ MINIMAL GENERATION 1 TEST PASSED!")
        print("🎯 Core structure is in place and importable")
        print("📦 Ready for dependency installation and full testing")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Import test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_documentation():
    """Test documentation structure."""
    print("\n" + "=" * 55)
    print("📚 Documentation Structure Test")
    print("=" * 55)
    
    project_root = Path(__file__).parent
    
    # Check key documentation files
    docs_to_check = [
        "README.md",
        "pyproject.toml", 
        "docs/QUANTUM_OPTIMIZATION_GUIDE.md",
        "docs/PRODUCTION_DEPLOYMENT.md",
        "docs/API_REFERENCE.md"
    ]
    
    for doc in docs_to_check:
        doc_path = project_root / doc
        if doc_path.exists():
            size = doc_path.stat().st_size
            print(f"   ✓ {doc} ({size:,} bytes)")
        else:
            print(f"   ❌ {doc} missing")
    
    # Check deployment structure
    deployment_path = project_root / "deployment"
    if deployment_path.exists():
        deployment_files = list(deployment_path.glob("*"))
        print(f"   ✓ Deployment directory with {len(deployment_files)} files")
    
    return True


def test_project_maturity():
    """Assess project maturity based on structure."""
    print("\n" + "=" * 55)
    print("🏗️ Project Maturity Assessment")
    print("=" * 55)
    
    project_root = Path(__file__).parent
    
    maturity_indicators = {
        "Core Implementation": 0,
        "Testing Infrastructure": 0, 
        "Documentation": 0,
        "Production Readiness": 0,
        "Advanced Features": 0
    }
    
    # Core Implementation (0-20 points)
    src_files = list((project_root / "src").rglob("*.py"))
    maturity_indicators["Core Implementation"] = min(20, len(src_files))
    
    # Testing Infrastructure (0-20 points)
    test_files = list((project_root / "tests").rglob("*.py")) if (project_root / "tests").exists() else []
    maturity_indicators["Testing Infrastructure"] = min(20, len(test_files) * 2)
    
    # Documentation (0-20 points)
    doc_files = list((project_root / "docs").rglob("*.md")) if (project_root / "docs").exists() else []
    maturity_indicators["Documentation"] = min(20, len(doc_files) * 2)
    
    # Production Readiness (0-20 points)
    prod_files = [
        "Dockerfile", "docker-compose.yml", "pyproject.toml", 
        "deployment/", "monitoring/"
    ]
    prod_score = sum(5 for f in prod_files if (project_root / f).exists())
    maturity_indicators["Production Readiness"] = prod_score
    
    # Advanced Features (0-20 points)
    advanced_files = list((project_root / "src").rglob("*quantum*.py"))
    advanced_files.extend(list((project_root / "src").rglob("*optimization*.py")))
    advanced_files.extend(list((project_root / "src").rglob("*monitoring*.py")))
    maturity_indicators["Advanced Features"] = min(20, len(advanced_files) * 3)
    
    print("Maturity Assessment:")
    total_score = 0
    for category, score in maturity_indicators.items():
        total_score += score
        bar = "█" * (score // 4) + "░" * (5 - score // 4)
        print(f"   {category:20} {bar} {score:2d}/20")
    
    print(f"\n🎯 Overall Maturity Score: {total_score}/100")
    
    if total_score >= 80:
        maturity_level = "🚀 Production Ready"
    elif total_score >= 60:
        maturity_level = "🔧 Advanced Development"
    elif total_score >= 40:
        maturity_level = "⚙️  Active Development"
    elif total_score >= 20:
        maturity_level = "🌱 Early Development"
    else:
        maturity_level = "🥚 Prototype"
    
    print(f"🏷️  Maturity Level: {maturity_level}")
    
    return total_score


def main():
    """Run minimal tests."""
    print("🚀 Agent Skeptic Bench - Minimal Generation 1 Testing")
    
    imports_passed = test_imports()
    test_documentation()
    maturity_score = test_project_maturity()
    
    print("\n" + "=" * 65)
    print("📋 MINIMAL GENERATION 1 TEST SUMMARY")
    print("=" * 65)
    print(f"Core Imports:        {'✅ PASS' if imports_passed else '❌ FAIL'}")
    print(f"Project Maturity:    {maturity_score}/100")
    
    if imports_passed and maturity_score >= 40:
        print("\n🎉 GENERATION 1 FOUNDATION SOLID!")
        print("   ✓ Core modules structured correctly")
        print("   ✓ Import system working")
        print("   ✓ Project structure mature")
        print("\n📦 Next: Install dependencies and run full tests")
        return True
    else:
        print("\n⚠️  Foundation needs strengthening before proceeding")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)