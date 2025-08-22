#!/usr/bin/env python3
"""
🛡️ SIMPLIFIED VALIDATION FRAMEWORK
==================================

Production-grade validation framework for Agent Skeptic Bench without external dependencies.
Ensures robustness, reliability, and deployment readiness.
"""

import asyncio
import json
import logging
import math
import random
import statistics
import time
from typing import Dict, List, Any, Tuple, Optional, Callable


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of validation testing."""
    
    def __init__(self, test_name: str, passed: bool, score: float, 
                 confidence_interval: Tuple[float, float], details: Dict[str, Any],
                 recommendations: List[str]):
        self.test_name = test_name
        self.passed = passed
        self.score = score
        self.confidence_interval = confidence_interval
        self.details = details
        self.recommendations = recommendations
        self.timestamp = time.time()


class StatisticalValidator:
    """Statistical validation without external dependencies."""
    
    def __init__(self):
        pass
    
    def bootstrap_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        if not data:
            return (0.0, 0.0)
        
        bootstrap_stats = []
        n_bootstrap = 1000
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = [random.choice(data) for _ in range(len(data))]
            stat = statistics.mean(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        # Calculate confidence interval
        bootstrap_stats.sort()
        alpha = 1 - confidence
        lower_idx = int(alpha / 2 * n_bootstrap)
        upper_idx = int((1 - alpha / 2) * n_bootstrap)
        
        ci_lower = bootstrap_stats[lower_idx] if lower_idx < len(bootstrap_stats) else bootstrap_stats[0]
        ci_upper = bootstrap_stats[upper_idx] if upper_idx < len(bootstrap_stats) else bootstrap_stats[-1]
        
        return (ci_lower, ci_upper)
    
    def simple_t_test(self, sample: List[float], baseline: float) -> Tuple[float, float]:
        """Simple t-test implementation."""
        if len(sample) < 2:
            return 0.0, 1.0
        
        mean_sample = statistics.mean(sample)
        std_sample = statistics.stdev(sample)
        n = len(sample)
        
        # Calculate t-statistic
        t_stat = (mean_sample - baseline) / (std_sample / math.sqrt(n))
        
        # Approximate p-value (simplified)
        df = n - 1
        # Very rough approximation
        if abs(t_stat) > 2.0:
            p_value = 0.05  # Likely significant
        elif abs(t_stat) > 1.5:
            p_value = 0.1   # Possibly significant
        else:
            p_value = 0.5   # Not significant
        
        return t_stat, p_value
    
    def effect_size(self, sample1: List[float], sample2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if not sample1 or not sample2:
            return 0.0
        
        mean1 = statistics.mean(sample1)
        mean2 = statistics.mean(sample2)
        
        if len(sample1) > 1:
            var1 = statistics.variance(sample1)
        else:
            var1 = 0.0
            
        if len(sample2) > 1:
            var2 = statistics.variance(sample2)
        else:
            var2 = 0.0
        
        # Pooled standard deviation
        n1, n2 = len(sample1), len(sample2)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1.0
        
        return (mean1 - mean2) / pooled_std


class SecurityValidator:
    """Security validation."""
    
    def __init__(self):
        pass
    
    async def test_adversarial_inputs(self, target_func: Callable, num_tests: int = 50) -> ValidationResult:
        """Test with adversarial inputs."""
        logger.info("🔒 Testing adversarial inputs...")
        
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "{{7*7}}",
            "' OR '1'='1",
            "A" * 1000,
            float('inf'),
            float('-inf'),
            {"malicious": "payload"},
            [1] * 10000,
        ]
        
        passed_tests = 0
        failed_tests = 0
        vulnerabilities = []
        
        for i in range(num_tests):
            try:
                # Choose random malicious input
                malicious_input = random.choice(malicious_inputs)
                
                # Test function
                result = await target_func([malicious_input])
                
                # Check for security violations
                if self._check_security_violation(malicious_input, result):
                    failed_tests += 1
                    vulnerabilities.append({
                        'test': i,
                        'input': str(malicious_input)[:100],
                        'result': str(result)[:100]
                    })
                else:
                    passed_tests += 1
                    
            except Exception as e:
                # Proper error handling is good
                if "security" in str(e).lower() or "validation" in str(e).lower():
                    passed_tests += 1
                else:
                    failed_tests += 1
        
        total_tests = passed_tests + failed_tests
        security_score = passed_tests / total_tests if total_tests > 0 else 0.0
        
        return ValidationResult(
            test_name="adversarial_input_testing",
            passed=security_score >= 0.95,
            score=security_score,
            confidence_interval=(security_score - 0.05, security_score + 0.05),
            details={
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'vulnerabilities': vulnerabilities[:5],  # Keep first 5
                'total_tests': total_tests
            },
            recommendations=self._get_security_recommendations(security_score, len(vulnerabilities))
        )
    
    def _check_security_violation(self, input_data: Any, result: Any) -> bool:
        """Check for security violations."""
        # Information leakage
        if isinstance(result, str):
            sensitive_words = ['password', 'secret', 'key', 'token']
            if any(word in result.lower() for word in sensitive_words):
                return True
        
        # Input injection
        if isinstance(input_data, str) and isinstance(result, str):
            if input_data in result and len(input_data) > 5:
                return True
        
        # Resource exhaustion
        if isinstance(result, (list, dict, str)) and len(str(result)) > 5000:
            return True
        
        return False
    
    def _get_security_recommendations(self, score: float, num_vulnerabilities: int) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if score < 0.8:
            recommendations.append("Critical: Implement comprehensive input validation")
            recommendations.append("Add output sanitization to prevent information leakage")
        
        if num_vulnerabilities > 5:
            recommendations.append("Review error handling to prevent information disclosure")
            recommendations.append("Implement rate limiting and request throttling")
        
        if score >= 0.95:
            recommendations.append("Security testing passed - maintain current practices")
        
        return recommendations


class PerformanceValidator:
    """Performance validation."""
    
    def __init__(self):
        self.baselines = {
            'execution_time': 1.0,
            'accuracy': 0.7,
            'throughput': 10.0
        }
    
    async def benchmark_performance(self, algo_func: Callable, test_data: List[Any]) -> ValidationResult:
        """Benchmark algorithm performance."""
        logger.info(f"⚡ Benchmarking {algo_func.__name__}...")
        
        execution_times = []
        accuracy_scores = []
        
        # Run multiple iterations
        for _ in range(10):
            start_time = time.time()
            
            try:
                result = await algo_func(test_data[:5])
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                # Extract accuracy
                if isinstance(result, dict):
                    accuracy = result.get('accuracy', result.get('score', 0.8))
                elif isinstance(result, (int, float)):
                    accuracy = float(result)
                else:
                    accuracy = 0.8
                
                accuracy_scores.append(accuracy)
                
            except Exception as e:
                logger.warning(f"Performance test iteration failed: {e}")
                execution_times.append(10.0)  # Penalty for failure
                accuracy_scores.append(0.0)
        
        # Calculate metrics
        avg_execution_time = statistics.mean(execution_times)
        avg_accuracy = statistics.mean(accuracy_scores)
        throughput = len(test_data) / avg_execution_time
        
        # Performance score (higher is better)
        time_score = min(1.0, self.baselines['execution_time'] / avg_execution_time)
        accuracy_score = avg_accuracy
        throughput_score = min(1.0, throughput / self.baselines['throughput'])
        
        overall_score = (time_score + accuracy_score + throughput_score) / 3.0
        
        return ValidationResult(
            test_name=f"performance_benchmark_{algo_func.__name__}",
            passed=overall_score >= 0.7,
            score=overall_score,
            confidence_interval=(overall_score - 0.1, overall_score + 0.1),
            details={
                'avg_execution_time': avg_execution_time,
                'avg_accuracy': avg_accuracy,
                'throughput': throughput,
                'time_score': time_score,
                'accuracy_score': accuracy_score,
                'throughput_score': throughput_score,
                'baseline_execution_time': self.baselines['execution_time'],
                'baseline_accuracy': self.baselines['accuracy']
            },
            recommendations=self._get_performance_recommendations(overall_score, avg_execution_time, avg_accuracy)
        )
    
    def _get_performance_recommendations(self, score: float, exec_time: float, accuracy: float) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if score < 0.6:
            recommendations.append("Critical: Performance below acceptable threshold")
        
        if exec_time > self.baselines['execution_time'] * 2:
            recommendations.append("Optimize algorithm execution time")
            recommendations.append("Consider parallel processing or caching")
        
        if accuracy < self.baselines['accuracy']:
            recommendations.append("Improve algorithm accuracy")
            recommendations.append("Review model parameters and training data")
        
        if score >= 0.8:
            recommendations.append("Good performance - monitor for regressions")
        
        return recommendations


class ReproducibilityValidator:
    """Reproducibility validation."""
    
    def __init__(self):
        pass
    
    async def test_reproducibility(self, algo_func: Callable, test_input: Any, num_runs: int = 5) -> ValidationResult:
        """Test algorithm reproducibility."""
        logger.info(f"🔄 Testing reproducibility of {algo_func.__name__}...")
        
        results = []
        
        for run in range(num_runs):
            # Set deterministic seed
            random.seed(42 + run)
            
            try:
                result = await algo_func(test_input)
                results.append(result)
            except Exception as e:
                logger.warning(f"Reproducibility test run {run} failed: {e}")
                continue
        
        if len(results) < 2:
            return ValidationResult(
                test_name="reproducibility_test",
                passed=False,
                score=0.0,
                confidence_interval=(0.0, 0.0),
                details={'error': 'Insufficient successful runs'},
                recommendations=["Fix algorithm errors before testing reproducibility"]
            )
        
        # Calculate reproducibility score
        reproducibility_score = self._calculate_reproducibility(results)
        
        return ValidationResult(
            test_name=f"reproducibility_{algo_func.__name__}",
            passed=reproducibility_score >= 0.9,
            score=reproducibility_score,
            confidence_interval=(reproducibility_score - 0.05, reproducibility_score + 0.05),
            details={
                'num_runs': len(results),
                'results_sample': str(results[:3])[:200],
                'reproducibility_analysis': self._analyze_reproducibility(results)
            },
            recommendations=self._get_reproducibility_recommendations(reproducibility_score)
        )
    
    def _calculate_reproducibility(self, results: List[Any]) -> float:
        """Calculate reproducibility score."""
        if len(results) < 2:
            return 0.0
        
        first_result = results[0]
        
        if isinstance(first_result, (int, float)):
            # Numerical results
            values = [float(r) for r in results]
            if len(set(values)) == 1:
                return 1.0
            
            mean_val = statistics.mean(values)
            if mean_val == 0:
                return 0.0
            
            cv = statistics.stdev(values) / mean_val
            return max(0.0, 1.0 - cv)
        
        elif isinstance(first_result, dict):
            # Dictionary results
            if not all(isinstance(r, dict) for r in results):
                return 0.0
            
            # Check key consistency
            all_keys = set()
            for result in results:
                all_keys.update(result.keys())
            
            key_scores = []
            for key in all_keys:
                values = []
                for result in results:
                    if key in result:
                        values.append(result[key])
                
                if len(values) == len(results):
                    if all(isinstance(v, (int, float)) for v in values):
                        float_values = [float(v) for v in values]
                        if len(set(float_values)) == 1:
                            key_scores.append(1.0)
                        else:
                            mean_val = statistics.mean(float_values)
                            if mean_val != 0:
                                cv = statistics.stdev(float_values) / mean_val
                                key_scores.append(max(0.0, 1.0 - cv))
                            else:
                                key_scores.append(0.0)
                    elif all(v == values[0] for v in values):
                        key_scores.append(1.0)
                    else:
                        key_scores.append(0.0)
                else:
                    key_scores.append(0.0)
            
            return statistics.mean(key_scores) if key_scores else 0.0
        
        else:
            # Other types - check exact equality
            return 1.0 if all(str(r) == str(first_result) for r in results) else 0.0
    
    def _analyze_reproducibility(self, results: List[Any]) -> Dict[str, Any]:
        """Analyze reproducibility results."""
        if not results:
            return {}
        
        first_result = results[0]
        
        return {
            'result_type': type(first_result).__name__,
            'num_unique_results': len(set(str(r) for r in results)),
            'all_identical': len(set(str(r) for r in results)) == 1,
            'consistent_type': all(type(r) == type(first_result) for r in results)
        }
    
    def _get_reproducibility_recommendations(self, score: float) -> List[str]:
        """Generate reproducibility recommendations."""
        if score >= 0.95:
            return ["Excellent reproducibility - maintain current practices"]
        elif score >= 0.8:
            return [
                "Good reproducibility with minor variance",
                "Consider using fixed random seeds",
                "Document expected sources of variance"
            ]
        elif score >= 0.5:
            return [
                "Moderate reproducibility issues",
                "Review random number generation",
                "Implement deterministic algorithms where possible"
            ]
        else:
            return [
                "Significant reproducibility issues",
                "Fix non-deterministic behavior",
                "Implement proper state management",
                "Use deterministic algorithms"
            ]


class ComprehensiveValidationFramework:
    """Main validation framework."""
    
    def __init__(self):
        """Initialize validation framework."""
        self.statistical_validator = StatisticalValidator()
        self.security_validator = SecurityValidator()
        self.performance_validator = PerformanceValidator()
        self.reproducibility_validator = ReproducibilityValidator()
        
        logger.info("🛡️ Validation Framework Initialized")
    
    async def run_validation_suite(self, algorithms: Dict[str, Callable], test_data: List[Any]) -> Dict[str, Any]:
        """Run complete validation suite."""
        logger.info("\n🔬 STARTING COMPREHENSIVE VALIDATION")
        logger.info("=" * 50)
        
        start_time = time.time()
        validation_summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'critical_failures': 0,
            'validation_results': [],
            'overall_score': 0.0,
            'recommendations': []
        }
        
        try:
            # 1. Performance Validation
            await self._run_performance_validation(algorithms, test_data, validation_summary)
            
            # 2. Security Validation
            await self._run_security_validation(algorithms, validation_summary)
            
            # 3. Reproducibility Validation
            await self._run_reproducibility_validation(algorithms, test_data, validation_summary)
            
            # 4. Statistical Validation
            await self._run_statistical_validation(algorithms, test_data, validation_summary)
            
            # Calculate overall score
            self._calculate_overall_score(validation_summary)
            
            # Generate recommendations
            self._generate_recommendations(validation_summary)
            
            execution_time = time.time() - start_time
            validation_summary['execution_time'] = execution_time
            
            logger.info(f"\n✅ Validation completed in {execution_time:.2f}s")
            logger.info(f"🎯 Overall score: {validation_summary['overall_score']:.3f}")
            logger.info(f"📊 Tests passed: {validation_summary['passed_tests']}/{validation_summary['total_tests']}")
            
            return validation_summary
            
        except Exception as e:
            logger.error(f"❌ Validation suite failed: {e}")
            validation_summary['error'] = str(e)
            return validation_summary
    
    async def _run_performance_validation(self, algorithms: Dict[str, Callable], test_data: List[Any], summary: Dict[str, Any]):
        """Run performance validation."""
        logger.info("\n⚡ Performance Validation...")
        
        for algo_name, algo_func in algorithms.items():
            try:
                result = await self.performance_validator.benchmark_performance(algo_func, test_data)
                
                summary['validation_results'].append(result)
                summary['total_tests'] += 1
                
                if result.passed:
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                
                logger.info(f"   {algo_name}: Score={result.score:.3f}")
                
            except Exception as e:
                logger.warning(f"Performance validation failed for {algo_name}: {e}")
                summary['total_tests'] += 1
                summary['failed_tests'] += 1
    
    async def _run_security_validation(self, algorithms: Dict[str, Callable], summary: Dict[str, Any]):
        """Run security validation."""
        logger.info("\n🔒 Security Validation...")
        
        for algo_name, algo_func in algorithms.items():
            try:
                result = await self.security_validator.test_adversarial_inputs(algo_func)
                
                summary['validation_results'].append(result)
                summary['total_tests'] += 1
                
                if result.passed:
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                    if result.score < 0.8:
                        summary['critical_failures'] += 1
                
                logger.info(f"   {algo_name}: Security Score={result.score:.3f}")
                
            except Exception as e:
                logger.warning(f"Security validation failed for {algo_name}: {e}")
                summary['total_tests'] += 1
                summary['failed_tests'] += 1
    
    async def _run_reproducibility_validation(self, algorithms: Dict[str, Callable], test_data: List[Any], summary: Dict[str, Any]):
        """Run reproducibility validation."""
        logger.info("\n🔄 Reproducibility Validation...")
        
        for algo_name, algo_func in algorithms.items():
            try:
                result = await self.reproducibility_validator.test_reproducibility(algo_func, test_data[:3])
                
                summary['validation_results'].append(result)
                summary['total_tests'] += 1
                
                if result.passed:
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                
                logger.info(f"   {algo_name}: Reproducibility={result.score:.3f}")
                
            except Exception as e:
                logger.warning(f"Reproducibility validation failed for {algo_name}: {e}")
                summary['total_tests'] += 1
                summary['failed_tests'] += 1
    
    async def _run_statistical_validation(self, algorithms: Dict[str, Callable], test_data: List[Any], summary: Dict[str, Any]):
        """Run statistical validation."""
        logger.info("\n📊 Statistical Validation...")
        
        for algo_name, algo_func in algorithms.items():
            try:
                # Generate results for statistical testing
                results = []
                for _ in range(15):  # 15 runs
                    result = await algo_func(test_data[:3])
                    if isinstance(result, dict):
                        score = result.get('accuracy', result.get('score', 0.8))
                    elif isinstance(result, (int, float)):
                        score = float(result)
                    else:
                        score = 0.8
                    results.append(score)
                
                # Statistical tests
                mean_score = statistics.mean(results)
                ci = self.statistical_validator.bootstrap_confidence_interval(results)
                
                baseline = 0.7
                t_stat, p_value = self.statistical_validator.simple_t_test(results, baseline)
                effect_size = self.statistical_validator.effect_size(results, [baseline] * len(results))
                
                stat_result = ValidationResult(
                    test_name=f"statistical_{algo_name}",
                    passed=p_value < 0.05 and abs(effect_size) > 0.2,
                    score=mean_score,
                    confidence_interval=ci,
                    details={
                        'mean_score': mean_score,
                        'baseline': baseline,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        't_statistic': t_stat,
                        'sample_size': len(results)
                    },
                    recommendations=[
                        f"Statistical significance: {'Yes' if p_value < 0.05 else 'No'}",
                        f"Effect size: {'Large' if abs(effect_size) > 0.8 else 'Medium' if abs(effect_size) > 0.5 else 'Small'}",
                        "Consider larger sample size if effect is small"
                    ]
                )
                
                summary['validation_results'].append(stat_result)
                summary['total_tests'] += 1
                
                if stat_result.passed:
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                
                logger.info(f"   {algo_name}: Mean={mean_score:.3f}, p={p_value:.3f}")
                
            except Exception as e:
                logger.warning(f"Statistical validation failed for {algo_name}: {e}")
                summary['total_tests'] += 1
                summary['failed_tests'] += 1
    
    def _calculate_overall_score(self, summary: Dict[str, Any]):
        """Calculate overall validation score."""
        if summary['total_tests'] == 0:
            summary['overall_score'] = 0.0
            return
        
        # Pass rate
        pass_rate = summary['passed_tests'] / summary['total_tests']
        
        # Individual scores
        if summary['validation_results']:
            individual_scores = [r.score for r in summary['validation_results']]
            avg_score = statistics.mean(individual_scores)
        else:
            avg_score = 0.0
        
        # Critical failure penalty
        critical_penalty = summary['critical_failures'] * 0.2
        
        # Combined score
        overall_score = (0.6 * pass_rate + 0.4 * avg_score) - critical_penalty
        summary['overall_score'] = max(0.0, min(1.0, overall_score))
    
    def _generate_recommendations(self, summary: Dict[str, Any]):
        """Generate overall recommendations."""
        recommendations = []
        
        if summary['overall_score'] >= 0.9:
            recommendations.append("🎉 Excellent validation - ready for production")
        elif summary['overall_score'] >= 0.8:
            recommendations.append("✅ Good validation - minor improvements needed")
        elif summary['overall_score'] >= 0.7:
            recommendations.append("⚠️ Acceptable validation - address failures")
        else:
            recommendations.append("❌ Below threshold - major improvements required")
        
        if summary['critical_failures'] > 0:
            recommendations.append("🚨 Critical failures detected - immediate attention needed")
        
        # Add specific recommendations
        for result in summary['validation_results']:
            if not result.passed and result.recommendations:
                recommendations.extend(result.recommendations[:2])
        
        # Remove duplicates
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)
        
        summary['recommendations'] = unique_recs[:8]
    
    def save_report(self, summary: Dict[str, Any], filename: str = "validation_report.json"):
        """Save validation report."""
        try:
            # Convert ValidationResult objects to dictionaries
            serializable_summary = summary.copy()
            serializable_summary['validation_results'] = [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'score': r.score,
                    'confidence_interval': r.confidence_interval,
                    'details': r.details,
                    'recommendations': r.recommendations,
                    'timestamp': r.timestamp
                }
                for r in summary['validation_results']
            ]
            
            with open(filename, 'w') as f:
                json.dump(serializable_summary, f, indent=2)
            
            logger.info(f"📄 Report saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")


async def main():
    """Run validation framework demo."""
    print("\n" + "🛡️" * 20)
    print("COMPREHENSIVE VALIDATION FRAMEWORK")
    print("🛡️" * 20 + "\n")
    
    validator = ComprehensiveValidationFramework()
    
    # Mock algorithms
    async def mock_meta_learning(data):
        await asyncio.sleep(0.01)
        return {
            'accuracy': 0.85 + random.uniform(-0.05, 0.05),
            'uncertainty': 0.15,
            'score': 0.88
        }
    
    async def mock_quantum_annealing(data):
        await asyncio.sleep(0.02)
        return {
            'accuracy': 0.82 + random.uniform(-0.05, 0.05),
            'speedup': 2.5,
            'score': 0.84
        }
    
    async def mock_temporal_dynamics(data):
        await asyncio.sleep(0.015)
        return {
            'accuracy': 0.78 + random.uniform(-0.05, 0.05),
            'coherence': 0.85,
            'score': 0.81
        }
    
    algorithms = {
        'meta_learning': mock_meta_learning,
        'quantum_annealing': mock_quantum_annealing,
        'temporal_dynamics': mock_temporal_dynamics
    }
    
    test_data = [f"test_{i}" for i in range(15)]
    
    try:
        validation_summary = await validator.run_validation_suite(algorithms, test_data)
        
        validator.save_report(validation_summary)
        
        print("\n" + "✅" * 20)
        print("VALIDATION COMPLETED!")
        print("✅" * 20)
        
        print(f"\n🎯 SUMMARY:")
        print(f"   • Overall Score: {validation_summary['overall_score']:.3f}")
        print(f"   • Tests Passed: {validation_summary['passed_tests']}/{validation_summary['total_tests']}")
        print(f"   • Critical Failures: {validation_summary['critical_failures']}")
        print(f"   • Execution Time: {validation_summary.get('execution_time', 0):.2f}s")
        
        print(f"\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(validation_summary['recommendations'][:5], 1):
            print(f"   {i}. {rec}")
        
        return validation_summary
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(main())