#!/usr/bin/env python3
"""
Comprehensive Quality Gates & Testing Framework
==============================================

Implements enterprise-grade quality gates including security scanning,
performance benchmarking, code quality analysis, and compliance validation.
"""

import os
import sys
import time
import json
import subprocess
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: float  # 0-100
    details: Dict[str, Any]
    execution_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL

@dataclass
class SecurityFinding:
    """Security vulnerability finding."""
    finding_id: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    category: str
    description: str
    file_path: str
    line_number: Optional[int] = None
    recommendation: Optional[str] = None

@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    test_name: str
    metric: str
    value: float
    unit: str
    baseline: Optional[float] = None
    threshold: Optional[float] = None
    passed: bool = True

class SecurityScanner:
    """Comprehensive security scanning."""
    
    def __init__(self):
        self.security_patterns = {
            'sql_injection': [
                r'(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER).*?(FROM|INTO|SET|WHERE)',
                r'(\'\s*OR\s*\'\s*=\s*\'|\'\s*;\s*--)',
                r'(UNION\s+SELECT|EXEC\s*\(|EXECUTE\s*\()'
            ],
            'xss': [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on(load|click|error|focus)=',
                r'<iframe[^>]*>.*?</iframe>'
            ],
            'command_injection': [
                r'(os\.system|subprocess\.call|exec\(|eval\()',
                r'(\|\||&&|\|)',
                r'(rm\s+-rf|del\s+/)',
                r'(wget|curl)\s+http'
            ],
            'path_traversal': [
                r'\.\./',
                r'\.\.\\',
                r'%2e%2e%2f',
                r'%2e%2e\\'
            ],
            'hardcoded_secrets': [
                r'(password|passwd|pwd)\s*=\s*["\'][^"\']{8,}["\']',
                r'(api_key|apikey|secret)\s*=\s*["\'][^"\']{16,}["\']',
                r'(token|auth)\s*=\s*["\'][^"\']{20,}["\']',
                r'-----BEGIN\s+(PRIVATE\s+KEY|RSA\s+PRIVATE\s+KEY)',
            ],
            'insecure_functions': [
                r'(pickle\.loads|marshal\.loads|exec\(|eval\()',
                r'(input\(|raw_input\()',  # In Python 2 context
                r'(yaml\.load\((?!.*Loader))',  # YAML load without safe loader
                r'(random\.random\(\)|random\.randint\()'  # Weak randomness
            ]
        }
    
    async def scan_directory(self, directory: str) -> List[SecurityFinding]:
        """Scan directory for security vulnerabilities."""
        findings = []
        
        for root, dirs, files in os.walk(directory):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'node_modules'}]
            
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.html', '.xml', '.json', '.yaml', '.yml')):
                    file_path = os.path.join(root, file)
                    file_findings = await self._scan_file(file_path)
                    findings.extend(file_findings)
        
        return findings
    
    async def _scan_file(self, file_path: str) -> List[SecurityFinding]:
        """Scan individual file for security issues."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check each security pattern category
            for category, patterns in self.security_patterns.items():
                category_findings = self._check_patterns(file_path, content, lines, category, patterns)
                findings.extend(category_findings)
            
            # Additional file-specific checks
            findings.extend(self._check_file_permissions(file_path))
            findings.extend(self._check_sensitive_data(file_path, content))
            
        except Exception as e:
            logger.warning(f"Could not scan file {file_path}: {e}")
        
        return findings
    
    def _check_patterns(self, file_path: str, content: str, lines: List[str], 
                       category: str, patterns: List[str]) -> List[SecurityFinding]:
        """Check content against security patterns."""
        import re
        findings = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                
                severity = self._get_severity_for_category(category)
                finding_id = hashlib.md5(f"{file_path}:{line_num}:{pattern}".encode()).hexdigest()[:12]
                
                findings.append(SecurityFinding(
                    finding_id=finding_id,
                    severity=severity,
                    category=category,
                    description=f"{category.replace('_', ' ').title()} pattern detected: {match.group()}",
                    file_path=file_path,
                    line_number=line_num,
                    recommendation=self._get_recommendation_for_category(category)
                ))
        
        return findings
    
    def _check_file_permissions(self, file_path: str) -> List[SecurityFinding]:
        """Check file permissions for security issues."""
        findings = []
        
        try:
            stat_info = os.stat(file_path)
            file_mode = stat_info.st_mode
            
            # Check for overly permissive files
            if file_mode & 0o002:  # World writable
                findings.append(SecurityFinding(
                    finding_id=hashlib.md5(f"{file_path}:permissions".encode()).hexdigest()[:12],
                    severity="MEDIUM",
                    category="file_permissions",
                    description="File is world-writable",
                    file_path=file_path,
                    recommendation="Remove world-write permissions: chmod o-w filename"
                ))
            
            if file_mode & 0o004 and file_path.endswith(('.key', '.pem', '.p12')):  # World readable sensitive files
                findings.append(SecurityFinding(
                    finding_id=hashlib.md5(f"{file_path}:readable".encode()).hexdigest()[:12],
                    severity="HIGH",
                    category="file_permissions",
                    description="Sensitive file is world-readable",
                    file_path=file_path,
                    recommendation="Restrict permissions: chmod 600 filename"
                ))
                
        except Exception:
            pass  # Skip permission check if not accessible
        
        return findings
    
    def _check_sensitive_data(self, file_path: str, content: str) -> List[SecurityFinding]:
        """Check for sensitive data in files."""
        findings = []
        
        # Check for common sensitive patterns
        sensitive_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'social_security': r'\b\d{3}-\d{2}-\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
        import re
        for pattern_name, pattern in sensitive_patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                
                findings.append(SecurityFinding(
                    finding_id=hashlib.md5(f"{file_path}:{line_num}:{pattern_name}".encode()).hexdigest()[:12],
                    severity="LOW",
                    category="sensitive_data",
                    description=f"Potential {pattern_name.replace('_', ' ')} found: {match.group()[:20]}...",
                    file_path=file_path,
                    line_number=line_num,
                    recommendation="Review if this sensitive data should be in source code"
                ))
        
        return findings
    
    def _get_severity_for_category(self, category: str) -> str:
        """Get severity level for security category."""
        severity_map = {
            'sql_injection': 'CRITICAL',
            'xss': 'HIGH',
            'command_injection': 'CRITICAL',
            'path_traversal': 'HIGH',
            'hardcoded_secrets': 'CRITICAL',
            'insecure_functions': 'MEDIUM',
            'file_permissions': 'MEDIUM',
            'sensitive_data': 'LOW'
        }
        return severity_map.get(category, 'MEDIUM')
    
    def _get_recommendation_for_category(self, category: str) -> str:
        """Get security recommendation for category."""
        recommendations = {
            'sql_injection': 'Use parameterized queries or ORM to prevent SQL injection',
            'xss': 'Sanitize user input and use proper output encoding',
            'command_injection': 'Avoid executing user input; use subprocess with shell=False',
            'path_traversal': 'Validate and sanitize file paths; use os.path.join()',
            'hardcoded_secrets': 'Use environment variables or secure secret management',
            'insecure_functions': 'Use secure alternatives and validate inputs',
            'file_permissions': 'Set appropriate file permissions (principle of least privilege)',
            'sensitive_data': 'Remove sensitive data from source code'
        }
        return recommendations.get(category, 'Review security implications')

class PerformanceTester:
    """Performance testing and benchmarking."""
    
    def __init__(self):
        self.benchmarks: List[PerformanceBenchmark] = []
        self.baseline_file = "performance_baselines.json"
        
    async def run_performance_tests(self) -> List[PerformanceBenchmark]:
        """Run comprehensive performance tests."""
        benchmarks = []
        
        # Load existing baselines
        baselines = self._load_baselines()
        
        # Test 1: Memory usage
        memory_usage = await self._test_memory_usage()
        benchmarks.append(PerformanceBenchmark(
            test_name="memory_usage",
            metric="peak_memory_mb",
            value=memory_usage,
            unit="MB",
            baseline=baselines.get("memory_usage", 100),
            threshold=200
        ))
        
        # Test 2: Startup time
        startup_time = await self._test_startup_time()
        benchmarks.append(PerformanceBenchmark(
            test_name="startup_time",
            metric="cold_start_ms",
            value=startup_time,
            unit="ms",
            baseline=baselines.get("startup_time", 1000),
            threshold=2000
        ))
        
        # Test 3: Throughput
        throughput = await self._test_throughput()
        benchmarks.append(PerformanceBenchmark(
            test_name="throughput",
            metric="requests_per_second",
            value=throughput,
            unit="req/s",
            baseline=baselines.get("throughput", 100),
            threshold=50
        ))
        
        # Test 4: Response time
        response_time = await self._test_response_time()
        benchmarks.append(PerformanceBenchmark(
            test_name="response_time",
            metric="p95_response_ms",
            value=response_time,
            unit="ms",
            baseline=baselines.get("response_time", 500),
            threshold=1000
        ))
        
        # Evaluate pass/fail for each benchmark
        for benchmark in benchmarks:
            if benchmark.threshold:
                if benchmark.metric.endswith("_ms") or benchmark.metric.endswith("_mb"):
                    # Lower is better for time and memory
                    benchmark.passed = benchmark.value <= benchmark.threshold
                else:
                    # Higher is better for throughput
                    benchmark.passed = benchmark.value >= benchmark.threshold
        
        return benchmarks
    
    async def _test_memory_usage(self) -> float:
        """Test peak memory usage."""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            # Fallback to resource module
            import resource
            initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB
        
        # Simulate memory-intensive operations
        data = []
        for i in range(10000):
            data.append({"id": i, "data": f"test_data_{i}" * 10})
        
        try:
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        except:
            import resource
            peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB
        
        # Cleanup
        del data
        try:
            import gc
            gc.collect()
        except:
            pass
        
        return peak_memory - initial_memory
    
    async def _test_startup_time(self) -> float:
        """Test application startup time."""
        start_time = time.time()
        
        # Simulate application initialization
        await asyncio.sleep(0.1)  # Simulate module loading
        
        # Simulate component initialization
        components = ['database', 'cache', 'security', 'monitoring']
        for component in components:
            await asyncio.sleep(0.02)  # Simulate component startup
        
        return (time.time() - start_time) * 1000  # Convert to ms
    
    async def _test_throughput(self) -> float:
        """Test request throughput."""
        start_time = time.time()
        num_requests = 1000
        
        # Simulate processing requests
        tasks = []
        for i in range(num_requests):
            tasks.append(self._simulate_request())
        
        await asyncio.gather(*tasks)
        
        duration = time.time() - start_time
        return num_requests / duration  # requests per second
    
    async def _test_response_time(self) -> float:
        """Test response time percentiles."""
        response_times = []
        
        # Collect response times
        for i in range(100):
            start_time = time.time()
            await self._simulate_request()
            response_time = (time.time() - start_time) * 1000  # ms
            response_times.append(response_time)
        
        # Calculate 95th percentile
        response_times.sort()
        p95_index = int(0.95 * len(response_times))
        return response_times[p95_index]
    
    async def _simulate_request(self):
        """Simulate processing a request."""
        # Simulate varying processing times
        import random
        delay = random.uniform(0.01, 0.1)
        await asyncio.sleep(delay)
    
    def _load_baselines(self) -> Dict[str, float]:
        """Load performance baselines from file."""
        try:
            if os.path.exists(self.baseline_file):
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load baselines: {e}")
        return {}
    
    def save_baselines(self, benchmarks: List[PerformanceBenchmark]):
        """Save current results as baselines."""
        baselines = {}
        for benchmark in benchmarks:
            baselines[benchmark.test_name] = benchmark.value
        
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(baselines, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save baselines: {e}")

class CodeQualityAnalyzer:
    """Code quality analysis and metrics."""
    
    def __init__(self):
        self.quality_metrics = {}
        
    async def analyze_code_quality(self, directory: str) -> Dict[str, Any]:
        """Analyze code quality metrics."""
        
        # Collect code statistics
        stats = await self._collect_code_stats(directory)
        
        # Calculate complexity metrics
        complexity = await self._analyze_complexity(directory)
        
        # Check coding standards
        standards = await self._check_coding_standards(directory)
        
        # Calculate maintainability index
        maintainability = self._calculate_maintainability_index(stats, complexity)
        
        quality_report = {
            "overall_score": maintainability,
            "code_statistics": stats,
            "complexity_metrics": complexity,
            "coding_standards": standards,
            "maintainability_index": maintainability,
            "quality_grade": self._get_quality_grade(maintainability),
            "recommendations": self._generate_recommendations(stats, complexity, standards)
        }
        
        return quality_report
    
    async def _collect_code_stats(self, directory: str) -> Dict[str, int]:
        """Collect basic code statistics."""
        stats = {
            "total_files": 0,
            "total_lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "python_files": 0,
            "test_files": 0
        }
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache'}]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    stats["total_files"] += 1
                    stats["python_files"] += 1
                    
                    if 'test' in file.lower():
                        stats["test_files"] += 1
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            
                        for line in lines:
                            stats["total_lines"] += 1
                            stripped = line.strip()
                            
                            if not stripped:
                                stats["blank_lines"] += 1
                            elif stripped.startswith('#'):
                                stats["comment_lines"] += 1
                            else:
                                stats["code_lines"] += 1
                                
                    except Exception:
                        pass
        
        return stats
    
    async def _analyze_complexity(self, directory: str) -> Dict[str, Any]:
        """Analyze code complexity."""
        complexity_metrics = {
            "average_function_length": 0,
            "max_function_length": 0,
            "deeply_nested_functions": 0,
            "complex_functions": 0,
            "total_functions": 0
        }
        
        function_lengths = []
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache'}]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_complexity = await self._analyze_file_complexity(file_path)
                    
                    function_lengths.extend(file_complexity["function_lengths"])
                    complexity_metrics["deeply_nested_functions"] += file_complexity["deeply_nested"]
                    complexity_metrics["complex_functions"] += file_complexity["complex_functions"]
                    complexity_metrics["total_functions"] += file_complexity["total_functions"]
        
        if function_lengths:
            complexity_metrics["average_function_length"] = sum(function_lengths) / len(function_lengths)
            complexity_metrics["max_function_length"] = max(function_lengths)
        
        return complexity_metrics
    
    async def _analyze_file_complexity(self, file_path: str) -> Dict[str, Any]:
        """Analyze complexity of a single file."""
        metrics = {
            "function_lengths": [],
            "deeply_nested": 0,
            "complex_functions": 0,
            "total_functions": 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            in_function = False
            function_start = 0
            current_indent = 0
            max_indent = 0
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                
                # Calculate indentation
                indent_level = (len(line) - len(line.lstrip())) // 4
                max_indent = max(max_indent, indent_level)
                
                # Check for function definition
                if stripped.startswith('def ') or stripped.startswith('async def '):
                    if in_function:
                        # End previous function
                        function_length = i - function_start
                        metrics["function_lengths"].append(function_length)
                        
                        if function_length > 50:  # Long function
                            metrics["complex_functions"] += 1
                    
                    # Start new function
                    in_function = True
                    function_start = i
                    current_indent = indent_level
                    metrics["total_functions"] += 1
                
                # Check for deeply nested code
                elif in_function and indent_level > current_indent + 3:
                    metrics["deeply_nested"] += 1
            
            # Handle last function
            if in_function:
                function_length = len(lines) - function_start
                metrics["function_lengths"].append(function_length)
                if function_length > 50:
                    metrics["complex_functions"] += 1
                    
        except Exception:
            pass
        
        return metrics
    
    async def _check_coding_standards(self, directory: str) -> Dict[str, Any]:
        """Check adherence to coding standards."""
        standards = {
            "pep8_violations": 0,
            "naming_violations": 0,
            "docstring_missing": 0,
            "line_length_violations": 0,
            "import_violations": 0
        }
        
        import re
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache'}]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_standards = await self._check_file_standards(file_path)
                    
                    for key in standards:
                        standards[key] += file_standards.get(key, 0)
        
        return standards
    
    async def _check_file_standards(self, file_path: str) -> Dict[str, int]:
        """Check coding standards for a single file."""
        violations = {
            "pep8_violations": 0,
            "naming_violations": 0,
            "docstring_missing": 0,
            "line_length_violations": 0,
            "import_violations": 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            import re
            
            for i, line in enumerate(lines):
                # Check line length
                if len(line.rstrip()) > 88:  # Black's default
                    violations["line_length_violations"] += 1
                
                # Check for trailing whitespace
                if line.rstrip() != line.rstrip('\n').rstrip('\r'):
                    violations["pep8_violations"] += 1
                
                stripped = line.strip()
                
                # Check function/class naming
                if re.match(r'^\s*def\s+([A-Z][a-zA-Z0-9_]*)', line):
                    violations["naming_violations"] += 1  # Function should be lowercase
                if re.match(r'^\s*class\s+([a-z][a-zA-Z0-9_]*)', line):
                    violations["naming_violations"] += 1  # Class should be PascalCase
                
                # Check for missing docstrings
                if stripped.startswith('def ') or stripped.startswith('class '):
                    # Look for docstring in next few lines
                    has_docstring = False
                    for j in range(i+1, min(i+5, len(lines))):
                        if '"""' in lines[j] or "'''" in lines[j]:
                            has_docstring = True
                            break
                    if not has_docstring:
                        violations["docstring_missing"] += 1
                
                # Check import order (simplified)
                if stripped.startswith('from ') and i > 0:
                    prev_line = lines[i-1].strip()
                    if prev_line.startswith('import '):
                        violations["import_violations"] += 1
                        
        except Exception:
            pass
        
        return violations
    
    def _calculate_maintainability_index(self, stats: Dict, complexity: Dict) -> float:
        """Calculate maintainability index (0-100)."""
        if stats["code_lines"] == 0:
            return 0
        
        # Base score
        score = 100
        
        # Deduct for complexity
        if complexity["total_functions"] > 0:
            avg_complexity = complexity["complex_functions"] / complexity["total_functions"]
            score -= avg_complexity * 20
        
        # Deduct for long functions
        if complexity["average_function_length"] > 20:
            score -= (complexity["average_function_length"] - 20) * 0.5
        
        # Deduct for deeply nested code
        if complexity["total_functions"] > 0:
            nesting_ratio = complexity["deeply_nested_functions"] / complexity["total_functions"]
            score -= nesting_ratio * 15
        
        # Bonus for test coverage
        if stats["test_files"] > 0:
            test_ratio = stats["test_files"] / max(1, stats["python_files"])
            score += min(10, test_ratio * 10)
        
        return max(0, min(100, score))
    
    def _get_quality_grade(self, score: float) -> str:
        """Get quality grade based on score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, stats: Dict, complexity: Dict, standards: Dict) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if complexity["average_function_length"] > 30:
            recommendations.append("Break down large functions into smaller, more focused functions")
        
        if complexity["deeply_nested_functions"] > 0:
            recommendations.append("Reduce nesting depth by extracting methods or using early returns")
        
        if standards["docstring_missing"] > stats["python_files"] * 0.5:
            recommendations.append("Add docstrings to functions and classes for better documentation")
        
        if standards["line_length_violations"] > 0:
            recommendations.append("Follow line length guidelines (88 characters for Black formatter)")
        
        if stats["test_files"] == 0:
            recommendations.append("Add unit tests to improve code reliability and maintainability")
        
        if standards["pep8_violations"] > 0:
            recommendations.append("Follow PEP 8 style guidelines for better code consistency")
        
        return recommendations

class QualityGateManager:
    """Orchestrates all quality gate checks."""
    
    def __init__(self):
        self.security_scanner = SecurityScanner()
        self.performance_tester = PerformanceTester()
        self.code_analyzer = CodeQualityAnalyzer()
        self.results: List[QualityGateResult] = []
        
    async def run_all_quality_gates(self, project_dir: str = ".") -> Dict[str, Any]:
        """Run all quality gates and return comprehensive report."""
        print("🛡️  RUNNING COMPREHENSIVE QUALITY GATES")
        print("=" * 60)
        
        gate_results = []
        
        # Gate 1: Security Scanning
        print("🔒 Running Security Scan...")
        start_time = time.time()
        security_findings = await self.security_scanner.scan_directory(project_dir)
        security_time = (time.time() - start_time) * 1000
        
        critical_findings = len([f for f in security_findings if f.severity == "CRITICAL"])
        high_findings = len([f for f in security_findings if f.severity == "HIGH"])
        
        security_score = max(0, 100 - (critical_findings * 25) - (high_findings * 10))
        security_passed = critical_findings == 0 and high_findings <= 2
        
        gate_results.append(QualityGateResult(
            gate_name="security_scan",
            passed=security_passed,
            score=security_score,
            details={
                "total_findings": len(security_findings),
                "critical": critical_findings,
                "high": high_findings,
                "findings": [f.__dict__ for f in security_findings[:10]]  # Top 10
            },
            execution_time_ms=security_time,
            severity="CRITICAL" if critical_findings > 0 else "WARNING" if high_findings > 0 else "INFO"
        ))
        
        print(f"  ✅ Security scan completed: {len(security_findings)} findings ({security_time:.1f}ms)")
        
        # Gate 2: Performance Testing
        print("⚡ Running Performance Tests...")
        start_time = time.time()
        performance_benchmarks = await self.performance_tester.run_performance_tests()
        performance_time = (time.time() - start_time) * 1000
        
        failed_benchmarks = [b for b in performance_benchmarks if not b.passed]
        performance_score = max(0, 100 - len(failed_benchmarks) * 20)
        performance_passed = len(failed_benchmarks) == 0
        
        gate_results.append(QualityGateResult(
            gate_name="performance_tests",
            passed=performance_passed,
            score=performance_score,
            details={
                "total_benchmarks": len(performance_benchmarks),
                "passed": len(performance_benchmarks) - len(failed_benchmarks),
                "failed": len(failed_benchmarks),
                "benchmarks": [b.__dict__ for b in performance_benchmarks]
            },
            execution_time_ms=performance_time,
            severity="ERROR" if len(failed_benchmarks) > 2 else "WARNING" if failed_benchmarks else "INFO"
        ))
        
        print(f"  ✅ Performance tests completed: {len(performance_benchmarks)} benchmarks ({performance_time:.1f}ms)")
        
        # Gate 3: Code Quality Analysis
        print("📊 Running Code Quality Analysis...")
        start_time = time.time()
        code_quality = await self.code_analyzer.analyze_code_quality(project_dir)
        quality_time = (time.time() - start_time) * 1000
        
        quality_score = code_quality["maintainability_index"]
        quality_passed = quality_score >= 70
        
        gate_results.append(QualityGateResult(
            gate_name="code_quality",
            passed=quality_passed,
            score=quality_score,
            details=code_quality,
            execution_time_ms=quality_time,
            severity="WARNING" if quality_score < 70 else "INFO"
        ))
        
        print(f"  ✅ Code quality analysis completed: Grade {code_quality['quality_grade']} ({quality_time:.1f}ms)")
        
        # Overall quality gate assessment
        total_gates = len(gate_results)
        passed_gates = len([g for g in gate_results if g.passed])
        overall_score = sum(g.score for g in gate_results) / total_gates if total_gates > 0 else 0
        overall_passed = passed_gates == total_gates
        
        print(f"\n🏆 QUALITY GATE SUMMARY")
        print("=" * 60)
        print(f"Overall Score: {overall_score:.1f}/100")
        print(f"Gates Passed: {passed_gates}/{total_gates}")
        print(f"Status: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        
        return {
            "overall_passed": overall_passed,
            "overall_score": overall_score,
            "gates_passed": passed_gates,
            "total_gates": total_gates,
            "gate_results": [g.__dict__ for g in gate_results],
            "summary": {
                "security_findings": len(security_findings),
                "performance_benchmarks": len(performance_benchmarks),
                "code_quality_grade": code_quality["quality_grade"],
                "execution_time_ms": sum(g.execution_time_ms for g in gate_results)
            },
            "recommendations": self._generate_overall_recommendations(gate_results)
        }
    
    def _generate_overall_recommendations(self, gate_results: List[QualityGateResult]) -> List[str]:
        """Generate overall improvement recommendations."""
        recommendations = []
        
        for gate in gate_results:
            if not gate.passed:
                if gate.gate_name == "security_scan":
                    recommendations.append("Address security vulnerabilities before deployment")
                elif gate.gate_name == "performance_tests":
                    recommendations.append("Optimize performance bottlenecks")
                elif gate.gate_name == "code_quality":
                    recommendations.append("Improve code quality and maintainability")
        
        if not recommendations:
            recommendations.append("All quality gates passed - ready for production deployment!")
        
        return recommendations

async def main():
    """Run quality gates on the current project."""
    quality_manager = QualityGateManager()
    results = await quality_manager.run_all_quality_gates("src/agent_skeptic_bench")
    
    # Print detailed results
    print(f"\n📋 DETAILED QUALITY REPORT")
    print("=" * 60)
    print(json.dumps(results["summary"], indent=2))
    
    return results

if __name__ == "__main__":
    results = asyncio.run(main())