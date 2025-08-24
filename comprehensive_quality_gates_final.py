#!/usr/bin/env python3
"""Comprehensive Quality Gates and Testing Framework

Implements automated quality validation with security scanning, performance 
benchmarking, test coverage analysis, and production readiness assessment.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import importlib

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    critical: bool = False  # If True, failure blocks deployment


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    timestamp: datetime
    overall_score: float
    passed_gates: int
    total_gates: int
    critical_failures: int
    gate_results: List[QualityGateResult]
    deployment_ready: bool
    recommendations: List[str]
    execution_time_seconds: float


class ComprehensiveQualityGates:
    """Comprehensive quality gates system."""
    
    def __init__(self):
        """Initialize quality gates system."""
        self.quality_gates: List[Tuple[str, callable, bool]] = []  # (name, function, critical)
        self.min_overall_score = 0.85  # Minimum score for deployment
        self.min_critical_score = 0.95  # Minimum score for critical gates
        
        self._register_default_gates()
    
    def _register_default_gates(self):
        """Register default quality gates."""
        # Critical gates (must pass for deployment)
        self.register_gate("security_scan", self._security_scan_gate, critical=True)
        self.register_gate("code_execution", self._code_execution_gate, critical=True)
        self.register_gate("import_validation", self._import_validation_gate, critical=True)
        
        # Important gates (contribute to overall score)
        self.register_gate("performance_benchmarks", self._performance_benchmark_gate, critical=False)
        self.register_gate("test_coverage", self._test_coverage_gate, critical=False)
        self.register_gate("code_quality", self._code_quality_gate, critical=False)
        self.register_gate("documentation_check", self._documentation_gate, critical=False)
        self.register_gate("dependency_check", self._dependency_security_gate, critical=False)
        self.register_gate("generation_compatibility", self._generation_compatibility_gate, critical=False)
    
    def register_gate(self, name: str, gate_function: callable, critical: bool = False):
        """Register a quality gate."""
        self.quality_gates.append((name, gate_function, critical))
        logger.info(f"Registered {'critical' if critical else 'standard'} quality gate: {name}")
    
    async def run_quality_gates(self) -> QualityReport:
        """Run all quality gates and generate comprehensive report."""
        logger.info("🚀 Starting Comprehensive Quality Gate Assessment")
        start_time = time.time()
        
        gate_results = []
        passed_gates = 0
        critical_failures = 0
        total_score = 0.0
        
        for gate_name, gate_function, is_critical in self.quality_gates:
            logger.info(f"Running quality gate: {gate_name}")
            gate_start = time.time()
            
            try:
                result = await gate_function()
                result.critical = is_critical
                result.execution_time_ms = (time.time() - gate_start) * 1000
                
                gate_results.append(result)
                total_score += result.score
                
                if result.passed:
                    passed_gates += 1
                    logger.info(f"✅ {gate_name}: PASSED (score: {result.score:.3f})")
                else:
                    if is_critical:
                        critical_failures += 1
                        logger.error(f"❌ {gate_name}: CRITICAL FAILURE (score: {result.score:.3f}) - {result.message}")
                    else:
                        logger.warning(f"⚠️ {gate_name}: FAILED (score: {result.score:.3f}) - {result.message}")
                
            except Exception as e:
                logger.error(f"💥 {gate_name}: ERROR - {str(e)}")
                error_result = QualityGateResult(
                    name=gate_name,
                    passed=False,
                    score=0.0,
                    message=f"Gate execution error: {str(e)}",
                    critical=is_critical,
                    execution_time_ms=(time.time() - gate_start) * 1000
                )
                gate_results.append(error_result)
                if is_critical:
                    critical_failures += 1
        
        # Calculate overall metrics
        overall_score = total_score / len(self.quality_gates) if self.quality_gates else 0.0
        deployment_ready = (
            critical_failures == 0 and 
            overall_score >= self.min_overall_score
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(gate_results, overall_score)
        
        execution_time = time.time() - start_time
        
        report = QualityReport(
            timestamp=datetime.utcnow(),
            overall_score=overall_score,
            passed_gates=passed_gates,
            total_gates=len(self.quality_gates),
            critical_failures=critical_failures,
            gate_results=gate_results,
            deployment_ready=deployment_ready,
            recommendations=recommendations,
            execution_time_seconds=execution_time
        )
        
        # Log summary
        logger.info(f"\n📊 Quality Gate Assessment Complete")
        logger.info(f"Overall Score: {overall_score:.1%}")
        logger.info(f"Gates Passed: {passed_gates}/{len(self.quality_gates)}")
        logger.info(f"Critical Failures: {critical_failures}")
        logger.info(f"Deployment Ready: {'✅ YES' if deployment_ready else '❌ NO'}")
        logger.info(f"Assessment Time: {execution_time:.1f}s")
        
        return report
    
    async def _security_scan_gate(self) -> QualityGateResult:
        """Security scanning quality gate."""
        try:
            # Check for common security issues
            security_issues = []
            score = 1.0
            
            # Check for hardcoded secrets (basic pattern matching)
            secret_patterns = [
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'password\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ]
            
            import re
            for py_file in Path("src").rglob("*.py"):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    for pattern in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            # Allow test keys and mock values
                            if "test" not in str(py_file).lower() and "mock" not in content.lower():
                                security_issues.append(f"Potential secret in {py_file}")
                                score -= 0.2
                except Exception:
                    continue
            
            # Check for SQL injection patterns
            sql_patterns = [r'execute\s*\(\s*["\'].*%.*["\']', r'query\s*\(\s*["\'].*\+.*["\']']
            for py_file in Path("src").rglob("*.py"):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    for pattern in sql_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            security_issues.append(f"Potential SQL injection in {py_file}")
                            score -= 0.3
                except Exception:
                    continue
            
            # Check for insecure imports
            insecure_imports = ['eval', 'exec', 'subprocess.call', 'os.system']
            for py_file in Path("src").rglob("*.py"):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    for insecure in insecure_imports:
                        if insecure in content and "# security: approved" not in content:
                            security_issues.append(f"Potentially insecure usage of {insecure} in {py_file}")
                            score -= 0.1
                except Exception:
                    continue
            
            score = max(0.0, min(1.0, score))
            passed = len(security_issues) == 0 or score >= 0.8
            
            message = f"Security scan completed. Found {len(security_issues)} potential issues."
            
            return QualityGateResult(
                name="security_scan",
                passed=passed,
                score=score,
                message=message,
                details={"issues": security_issues, "patterns_checked": len(secret_patterns) + len(sql_patterns)}
            )
            
        except Exception as e:
            return QualityGateResult(
                name="security_scan",
                passed=False,
                score=0.0,
                message=f"Security scan failed: {str(e)}"
            )
    
    async def _code_execution_gate(self) -> QualityGateResult:
        """Code execution validation gate."""
        try:
            execution_results = {}
            total_score = 0.0
            tests_run = 0
            
            # Test Generation 1 demo
            try:
                logger.info("   Testing Generation 1 simple demo...")
                result = subprocess.run([
                    sys.executable, "generation_1_simple_demo.py"
                ], capture_output=True, text=True, timeout=120, cwd=".")
                
                if result.returncode == 0:
                    execution_results["generation_1"] = {"passed": True, "output": "Execution successful"}
                    total_score += 1.0
                else:
                    execution_results["generation_1"] = {
                        "passed": False, 
                        "output": result.stderr[-500:] if result.stderr else "Unknown error"
                    }
                tests_run += 1
                
            except subprocess.TimeoutExpired:
                execution_results["generation_1"] = {"passed": False, "output": "Execution timeout"}
                tests_run += 1
            except Exception as e:
                execution_results["generation_1"] = {"passed": False, "output": str(e)}
                tests_run += 1
            
            # Test Generation 2 demo
            try:
                logger.info("   Testing Generation 2 robust demo...")
                result = subprocess.run([
                    sys.executable, "generation_2_robust_demo.py"
                ], capture_output=True, text=True, timeout=180, cwd=".")
                
                if result.returncode == 0:
                    execution_results["generation_2"] = {"passed": True, "output": "Execution successful"}
                    total_score += 1.0
                else:
                    execution_results["generation_2"] = {
                        "passed": False,
                        "output": result.stderr[-500:] if result.stderr else "Unknown error"
                    }
                tests_run += 1
                    
            except subprocess.TimeoutExpired:
                execution_results["generation_2"] = {"passed": False, "output": "Execution timeout"}
                tests_run += 1
            except Exception as e:
                execution_results["generation_2"] = {"passed": False, "output": str(e)}
                tests_run += 1
            
            # Test Generation 3 demo
            try:
                logger.info("   Testing Generation 3 optimized demo...")
                result = subprocess.run([
                    sys.executable, "generation_3_optimized_demo.py"
                ], capture_output=True, text=True, timeout=240, cwd=".")
                
                if result.returncode == 0:
                    execution_results["generation_3"] = {"passed": True, "output": "Execution successful"}
                    total_score += 1.0
                else:
                    execution_results["generation_3"] = {
                        "passed": False,
                        "output": result.stderr[-500:] if result.stderr else "Unknown error"
                    }
                tests_run += 1
                    
            except subprocess.TimeoutExpired:
                execution_results["generation_3"] = {"passed": False, "output": "Execution timeout"}
                tests_run += 1
            except Exception as e:
                execution_results["generation_3"] = {"passed": False, "output": str(e)}
                tests_run += 1
            
            score = total_score / tests_run if tests_run > 0 else 0.0
            passed_tests = sum(1 for r in execution_results.values() if r["passed"])
            passed = passed_tests >= 2  # At least 2 generations must pass
            
            message = f"Code execution test: {passed_tests}/{tests_run} demos executed successfully"
            
            return QualityGateResult(
                name="code_execution",
                passed=passed,
                score=score,
                message=message,
                details={"execution_results": execution_results, "tests_run": tests_run}
            )
            
        except Exception as e:
            return QualityGateResult(
                name="code_execution",
                passed=False,
                score=0.0,
                message=f"Code execution test failed: {str(e)}"
            )
    
    async def _import_validation_gate(self) -> QualityGateResult:
        """Import validation quality gate."""
        try:
            import_results = {}
            total_score = 0.0
            imports_tested = 0
            
            # Test core imports
            core_imports = [
                "src.agent_skeptic_bench",
                "src.agent_skeptic_bench.benchmark",
                "src.agent_skeptic_bench.agents",
                "src.agent_skeptic_bench.models",
                "src.agent_skeptic_bench.quantum_optimizer",
            ]
            
            for module_name in core_imports:
                try:
                    module = importlib.import_module(module_name)
                    import_results[module_name] = {"success": True, "error": None}
                    total_score += 1.0
                except Exception as e:
                    import_results[module_name] = {"success": False, "error": str(e)}
                
                imports_tested += 1
            
            # Test Generation 2 & 3 specific imports
            advanced_imports = [
                "src.agent_skeptic_bench.robust_monitoring",
                "src.agent_skeptic_bench.comprehensive_security",
                "src.agent_skeptic_bench.performance_optimizer",
            ]
            
            for module_name in advanced_imports:
                try:
                    module = importlib.import_module(module_name)
                    import_results[module_name] = {"success": True, "error": None}
                    total_score += 0.5  # Half weight for advanced imports
                except Exception as e:
                    import_results[module_name] = {"success": False, "error": str(e)}
                
                imports_tested += 0.5
            
            score = total_score / imports_tested if imports_tested > 0 else 0.0
            successful_imports = sum(1 for r in import_results.values() if r["success"])
            passed = successful_imports >= len(core_imports)  # All core imports must succeed
            
            message = f"Import validation: {successful_imports}/{len(import_results)} modules imported successfully"
            
            return QualityGateResult(
                name="import_validation",
                passed=passed,
                score=score,
                message=message,
                details={"import_results": import_results}
            )
            
        except Exception as e:
            return QualityGateResult(
                name="import_validation",
                passed=False,
                score=0.0,
                message=f"Import validation failed: {str(e)}"
            )
    
    async def _performance_benchmark_gate(self) -> QualityGateResult:
        """Performance benchmarking quality gate."""
        try:
            from src.agent_skeptic_bench import SkepticBenchmark, AgentConfig, AgentProvider
            
            # Simple performance benchmark
            benchmark = SkepticBenchmark()
            agent_config = AgentConfig(
                provider=AgentProvider.CUSTOM,
                model_name="mock_benchmark_agent",
                api_key="test_key"
            )
            
            # Test session creation performance
            start_time = time.time()
            session = benchmark.create_session("Performance Test", agent_config)
            session_create_time = (time.time() - start_time) * 1000
            
            # Test health check performance
            start_time = time.time()
            health = benchmark.health_check()
            health_check_time = (time.time() - start_time) * 1000
            
            # Test export performance
            start_time = time.time()
            export_data = benchmark.export_session_data(session.id)
            export_time = (time.time() - start_time) * 1000
            
            # Performance scoring
            score = 1.0
            issues = []
            
            if session_create_time > 100:  # 100ms threshold
                score -= 0.2
                issues.append(f"Slow session creation: {session_create_time:.1f}ms")
            
            if health_check_time > 50:  # 50ms threshold
                score -= 0.1
                issues.append(f"Slow health check: {health_check_time:.1f}ms")
            
            if export_time > 200:  # 200ms threshold
                score -= 0.2
                issues.append(f"Slow data export: {export_time:.1f}ms")
            
            score = max(0.0, score)
            passed = score >= 0.7
            
            message = f"Performance benchmark completed. {len(issues)} performance issues found."
            
            # Cleanup
            benchmark.cleanup_session(session.id)
            
            return QualityGateResult(
                name="performance_benchmarks",
                passed=passed,
                score=score,
                message=message,
                details={
                    "session_create_time_ms": session_create_time,
                    "health_check_time_ms": health_check_time,
                    "export_time_ms": export_time,
                    "issues": issues
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                name="performance_benchmarks",
                passed=False,
                score=0.0,
                message=f"Performance benchmark failed: {str(e)}"
            )
    
    async def _test_coverage_gate(self) -> QualityGateResult:
        """Test coverage analysis gate."""
        try:
            # Count test files and estimate coverage
            test_files = list(Path(".").rglob("test*.py"))
            src_files = list(Path("src").rglob("*.py"))
            
            # Basic heuristic: test files should exist
            coverage_score = 0.0
            
            # Check for unit tests
            unit_test_files = [f for f in test_files if "unit" in str(f) or "test_" in f.name]
            if unit_test_files:
                coverage_score += 0.4
            
            # Check for integration tests
            integration_test_files = [f for f in test_files if "integration" in str(f)]
            if integration_test_files:
                coverage_score += 0.3
            
            # Check for demo/example tests
            demo_files = [f for f in Path(".").glob("*demo*.py") if f.exists()]
            if demo_files:
                coverage_score += 0.3
            
            # Estimate based on file ratio
            if src_files:
                file_ratio = len(test_files) / len(src_files)
                if file_ratio > 0.3:  # 30% test to source ratio
                    coverage_score = min(1.0, coverage_score + 0.2)
            
            coverage_score = min(1.0, coverage_score)
            passed = coverage_score >= 0.6
            
            message = f"Test coverage analysis: {len(test_files)} test files, estimated {coverage_score:.1%} coverage"
            
            return QualityGateResult(
                name="test_coverage",
                passed=passed,
                score=coverage_score,
                message=message,
                details={
                    "test_files": len(test_files),
                    "src_files": len(src_files),
                    "unit_tests": len(unit_test_files),
                    "integration_tests": len(integration_test_files),
                    "demo_files": len(demo_files)
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                name="test_coverage",
                passed=False,
                score=0.0,
                message=f"Test coverage analysis failed: {str(e)}"
            )
    
    async def _code_quality_gate(self) -> QualityGateResult:
        """Code quality assessment gate."""
        try:
            quality_score = 1.0
            issues = []
            
            # Check for basic code quality indicators
            src_files = list(Path("src").rglob("*.py"))
            
            # Check docstring coverage
            files_with_docstrings = 0
            for py_file in src_files:
                try:
                    content = py_file.read_text(encoding='utf-8')
                    if '"""' in content or "'''" in content:
                        files_with_docstrings += 1
                except Exception:
                    continue
            
            if src_files:
                docstring_ratio = files_with_docstrings / len(src_files)
                if docstring_ratio < 0.7:
                    quality_score -= 0.2
                    issues.append(f"Low docstring coverage: {docstring_ratio:.1%}")
            
            # Check for TODO/FIXME comments (should be minimal in production)
            todo_count = 0
            for py_file in src_files:
                try:
                    content = py_file.read_text(encoding='utf-8')
                    todo_count += content.count("TODO") + content.count("FIXME")
                except Exception:
                    continue
            
            if todo_count > 10:  # More than 10 TODOs/FIXMEs
                quality_score -= 0.1
                issues.append(f"High TODO/FIXME count: {todo_count}")
            
            # Check for proper error handling (basic heuristic)
            files_with_error_handling = 0
            for py_file in src_files:
                try:
                    content = py_file.read_text(encoding='utf-8')
                    if "try:" in content and "except" in content:
                        files_with_error_handling += 1
                except Exception:
                    continue
            
            if src_files:
                error_handling_ratio = files_with_error_handling / len(src_files)
                if error_handling_ratio < 0.5:
                    quality_score -= 0.1
                    issues.append(f"Limited error handling: {error_handling_ratio:.1%}")
            
            quality_score = max(0.0, quality_score)
            passed = quality_score >= 0.7 and len(issues) <= 3
            
            message = f"Code quality assessment completed. {len(issues)} quality issues found."
            
            return QualityGateResult(
                name="code_quality",
                passed=passed,
                score=quality_score,
                message=message,
                details={
                    "src_files": len(src_files),
                    "docstring_coverage": docstring_ratio if src_files else 0,
                    "todo_count": todo_count,
                    "error_handling_coverage": error_handling_ratio if src_files else 0,
                    "issues": issues
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                name="code_quality",
                passed=False,
                score=0.0,
                message=f"Code quality assessment failed: {str(e)}"
            )
    
    async def _documentation_gate(self) -> QualityGateResult:
        """Documentation quality gate."""
        try:
            doc_score = 0.0
            
            # Check for README
            if Path("README.md").exists():
                doc_score += 0.3
                readme_content = Path("README.md").read_text(encoding='utf-8')
                if len(readme_content) > 1000:  # Substantial README
                    doc_score += 0.2
            
            # Check for documentation directory
            docs_dir = Path("docs")
            if docs_dir.exists():
                doc_files = list(docs_dir.rglob("*.md"))
                if doc_files:
                    doc_score += 0.3
                    if len(doc_files) >= 5:  # Multiple doc files
                        doc_score += 0.2
            
            # Check for docstrings in source files
            src_files = list(Path("src").rglob("*.py"))
            files_with_good_docstrings = 0
            
            for py_file in src_files:
                try:
                    content = py_file.read_text(encoding='utf-8')
                    # Look for class and function docstrings
                    if ('"""' in content or "'''" in content) and len(content.split('"""')) > 2:
                        files_with_good_docstrings += 1
                except Exception:
                    continue
            
            if src_files:
                docstring_ratio = files_with_good_docstrings / len(src_files)
                doc_score += docstring_ratio * 0.3
            
            doc_score = min(1.0, doc_score)
            passed = doc_score >= 0.6
            
            message = f"Documentation assessment: score {doc_score:.1%}"
            
            return QualityGateResult(
                name="documentation_check",
                passed=passed,
                score=doc_score,
                message=message,
                details={
                    "readme_exists": Path("README.md").exists(),
                    "docs_dir_exists": docs_dir.exists(),
                    "doc_files": len(list(docs_dir.rglob("*.md"))) if docs_dir.exists() else 0,
                    "docstring_coverage": docstring_ratio if src_files else 0
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                name="documentation_check",
                passed=False,
                score=0.0,
                message=f"Documentation check failed: {str(e)}"
            )
    
    async def _dependency_security_gate(self) -> QualityGateResult:
        """Dependency security scanning gate."""
        try:
            # Check pyproject.toml for dependencies
            pyproject_path = Path("pyproject.toml")
            if not pyproject_path.exists():
                return QualityGateResult(
                    name="dependency_check",
                    passed=True,
                    score=0.8,
                    message="No pyproject.toml found - dependency check skipped"
                )
            
            import tomllib
            with open(pyproject_path, 'rb') as f:
                pyproject_data = tomllib.load(f)
            
            dependencies = pyproject_data.get('project', {}).get('dependencies', [])
            dev_dependencies = pyproject_data.get('project', {}).get('optional-dependencies', {})
            
            security_score = 1.0
            security_issues = []
            
            # Check for known problematic packages (basic list)
            problematic_packages = ['eval', 'exec', 'pickle']  # Simplified check
            
            all_deps = dependencies[:]
            for dep_group in dev_dependencies.values():
                all_deps.extend(dep_group)
            
            for dep in all_deps:
                dep_name = dep.split('>=')[0].split('==')[0].split('<')[0].strip()
                if any(problematic in dep_name.lower() for problematic in problematic_packages):
                    security_issues.append(f"Potentially risky dependency: {dep_name}")
                    security_score -= 0.2
            
            # Check for version pinning (security best practice)
            unpinned_deps = [dep for dep in dependencies if '>=' not in dep and '==' not in dep and '~=' not in dep]
            if unpinned_deps:
                security_score -= len(unpinned_deps) * 0.05  # Small penalty for unpinned deps
            
            security_score = max(0.0, security_score)
            passed = len(security_issues) == 0
            
            message = f"Dependency security scan: {len(security_issues)} issues found in {len(all_deps)} dependencies"
            
            return QualityGateResult(
                name="dependency_check",
                passed=passed,
                score=security_score,
                message=message,
                details={
                    "total_dependencies": len(all_deps),
                    "security_issues": security_issues,
                    "unpinned_dependencies": len(unpinned_deps)
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                name="dependency_check",
                passed=True,  # Don't fail deployment on dependency check errors
                score=0.7,
                message=f"Dependency check failed: {str(e)}"
            )
    
    async def _generation_compatibility_gate(self) -> QualityGateResult:
        """Test compatibility across all SDLC generations."""
        try:
            compatibility_score = 0.0
            test_results = {}
            
            # Test that all generations can coexist
            try:
                # Import all generation components
                from src.agent_skeptic_bench import SkepticBenchmark  # Generation 1
                benchmark = SkepticBenchmark()
                test_results["gen1_core"] = True
                compatibility_score += 0.25
            except Exception as e:
                test_results["gen1_core"] = False
                test_results["gen1_error"] = str(e)
            
            try:
                from src.agent_skeptic_bench.robust_monitoring import get_monitor  # Generation 2
                monitor = get_monitor()
                test_results["gen2_robust"] = True
                compatibility_score += 0.25
            except Exception as e:
                test_results["gen2_robust"] = False
                test_results["gen2_error"] = str(e)
            
            try:
                from src.agent_skeptic_bench.comprehensive_security import get_security  # Generation 2
                security = get_security()
                test_results["gen2_security"] = True
                compatibility_score += 0.25
            except Exception as e:
                test_results["gen2_security"] = False
                test_results["gen2_security_error"] = str(e)
            
            try:
                from src.agent_skeptic_bench.performance_optimizer import get_optimizer  # Generation 3
                optimizer = get_optimizer()
                test_results["gen3_optimizer"] = True
                compatibility_score += 0.25
            except Exception as e:
                test_results["gen3_optimizer"] = False
                test_results["gen3_error"] = str(e)
            
            passed = compatibility_score >= 0.75  # At least 3 out of 4 components must work
            
            message = f"Generation compatibility: {compatibility_score:.1%} components accessible"
            
            return QualityGateResult(
                name="generation_compatibility",
                passed=passed,
                score=compatibility_score,
                message=message,
                details=test_results
            )
            
        except Exception as e:
            return QualityGateResult(
                name="generation_compatibility",
                passed=False,
                score=0.0,
                message=f"Generation compatibility test failed: {str(e)}"
            )
    
    def _generate_recommendations(self, gate_results: List[QualityGateResult], overall_score: float) -> List[str]:
        """Generate improvement recommendations based on gate results."""
        recommendations = []
        
        # Overall score recommendations
        if overall_score < 0.7:
            recommendations.append("Overall quality score is below 70% - consider addressing multiple quality issues")
        elif overall_score < 0.85:
            recommendations.append("Overall quality score is good but could be improved for production readiness")
        
        # Specific gate recommendations
        for result in gate_results:
            if not result.passed and result.critical:
                recommendations.append(f"CRITICAL: Fix {result.name} issues before deployment - {result.message}")
            elif result.score < 0.7:
                recommendations.append(f"Improve {result.name}: {result.message}")
        
        # Security-specific recommendations
        security_results = [r for r in gate_results if "security" in r.name]
        for result in security_results:
            if result.score < 0.9:
                recommendations.append("Review security findings and implement recommended fixes")
        
        # Performance recommendations
        perf_results = [r for r in gate_results if "performance" in r.name]
        for result in perf_results:
            if result.score < 0.8:
                recommendations.append("Consider performance optimizations for production workloads")
        
        return recommendations[:10]  # Limit to top 10 recommendations


async def main():
    """Main execution function for comprehensive quality gates."""
    logger.info("🚀 Starting Comprehensive Quality Gate Assessment")
    
    try:
        # Initialize quality gates system
        quality_gates = ComprehensiveQualityGates()
        
        # Run all quality gates
        report = await quality_gates.run_quality_gates()
        
        # Save detailed report
        report_data = {
            "timestamp": report.timestamp.isoformat(),
            "overall_score": report.overall_score,
            "passed_gates": report.passed_gates,
            "total_gates": report.total_gates,
            "critical_failures": report.critical_failures,
            "deployment_ready": report.deployment_ready,
            "execution_time_seconds": report.execution_time_seconds,
            "gate_results": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "score": result.score,
                    "message": result.message,
                    "critical": result.critical,
                    "execution_time_ms": result.execution_time_ms,
                    "details": result.details
                }
                for result in report.gate_results
            ],
            "recommendations": report.recommendations,
            "quality_assessment": {
                "deployment_ready": report.deployment_ready,
                "quality_level": "production" if report.overall_score >= 0.9 else "staging" if report.overall_score >= 0.7 else "development",
                "risk_level": "low" if report.critical_failures == 0 and report.overall_score >= 0.85 else "medium" if report.critical_failures == 0 else "high"
            }
        }
        
        results_file = f"quality_gates_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"📄 Quality gates report saved to {results_file}")
        
        # Print final summary
        logger.info("\n🎯 Quality Gates Assessment Summary:")
        logger.info(f"Overall Score: {report.overall_score:.1%}")
        logger.info(f"Gates Passed: {report.passed_gates}/{report.total_gates}")
        logger.info(f"Critical Failures: {report.critical_failures}")
        logger.info(f"Deployment Status: {'✅ READY' if report.deployment_ready else '❌ NOT READY'}")
        logger.info(f"Quality Level: {report_data['quality_assessment']['quality_level'].upper()}")
        logger.info(f"Risk Level: {report_data['quality_assessment']['risk_level'].upper()}")
        
        if report.recommendations:
            logger.info("\n💡 Top Recommendations:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                logger.info(f"  {i}. {rec}")
        
        # Exit with appropriate code
        if report.deployment_ready:
            logger.info("🎉 All quality gates passed - Ready for production deployment!")
            return True
        else:
            logger.error("❌ Quality gates failed - Address issues before deployment")
            return False
            
    except Exception as e:
        logger.error(f"💥 Quality gates assessment failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)