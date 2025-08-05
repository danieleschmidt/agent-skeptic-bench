"""Enhanced validation utilities for Agent Skeptic Bench with quantum-inspired verification."""

import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import json

from .models import Scenario, ScenarioCategory, SkepticResponse, AgentConfig, EvaluationMetrics
from .exceptions import ValidationError, InvalidScenarioError


logger = logging.getLogger(__name__)


class ScenarioValidator:
    """Validates scenario data and structure."""
    
    def __init__(self):
        self.required_fields = {
            'id', 'category', 'name', 'description', 'adversary_claim',
            'correct_skepticism_level', 'good_evidence_requests', 'red_flags'
        }
        self.optional_fields = {'metadata'}
        
        # ID pattern: category_number_descriptor
        self.id_pattern = re.compile(r'^[a-zA-Z_]+_\d{3}_[a-zA-Z_]+$')
    
    def validate_scenario_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate raw scenario data dictionary."""
        errors = []
        
        # Check required fields
        missing_fields = self.required_fields - set(data.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Check unknown fields
        all_fields = self.required_fields | self.optional_fields
        unknown_fields = set(data.keys()) - all_fields
        if unknown_fields:
            errors.append(f"Unknown fields: {', '.join(unknown_fields)}")
        
        # Validate specific fields
        if 'id' in data:
            errors.extend(self._validate_id(data['id']))
        
        if 'category' in data:
            errors.extend(self._validate_category(data['category']))
        
        if 'correct_skepticism_level' in data:
            errors.extend(self._validate_skepticism_level(data['correct_skepticism_level']))
        
        if 'good_evidence_requests' in data:
            errors.extend(self._validate_evidence_requests(data['good_evidence_requests']))
        
        if 'red_flags' in data:
            errors.extend(self._validate_red_flags(data['red_flags']))
        
        if 'metadata' in data:
            errors.extend(self._validate_metadata(data['metadata']))
        
        return errors
    
    def validate_scenario(self, scenario: Scenario) -> List[str]:
        """Validate a Scenario object."""
        errors = []
        
        # Validate ID format
        if not self.id_pattern.match(scenario.id):
            errors.append(f"Invalid ID format: {scenario.id}")
        
        # Validate skepticism level
        if not 0.0 <= scenario.correct_skepticism_level <= 1.0:
            errors.append(f"Skepticism level must be between 0.0 and 1.0: {scenario.correct_skepticism_level}")
        
        # Validate evidence requests
        if not scenario.good_evidence_requests or len(scenario.good_evidence_requests) == 0:
            errors.append("Must have at least one good evidence request")
        
        # Validate red flags
        if not scenario.red_flags or len(scenario.red_flags) == 0:
            errors.append("Must have at least one red flag")
        
        # Validate claim length
        if len(scenario.adversary_claim) < 20:
            errors.append("Adversary claim too short (minimum 20 characters)")
        elif len(scenario.adversary_claim) > 1000:
            errors.append("Adversary claim too long (maximum 1000 characters)")
        
        # Validate name and description
        if len(scenario.name) < 5:
            errors.append("Scenario name too short (minimum 5 characters)")
        
        if len(scenario.description) < 20:
            errors.append("Scenario description too short (minimum 20 characters)")
        
        return errors
    
    def _validate_id(self, scenario_id: str) -> List[str]:
        """Validate scenario ID."""
        errors = []
        
        if not isinstance(scenario_id, str):
            errors.append(f"ID must be string, got {type(scenario_id)}")
            return errors
        
        if not self.id_pattern.match(scenario_id):
            errors.append(f"ID must match pattern 'category_###_descriptor': {scenario_id}")
        
        if len(scenario_id) > 50:
            errors.append(f"ID too long (max 50 chars): {scenario_id}")
        
        return errors
    
    def _validate_category(self, category: str) -> List[str]:
        """Validate scenario category."""
        errors = []
        
        if not isinstance(category, str):
            errors.append(f"Category must be string, got {type(category)}")
            return errors
        
        try:
            ScenarioCategory(category)
        except ValueError:
            valid_categories = [c.value for c in ScenarioCategory]
            errors.append(f"Invalid category '{category}', must be one of: {valid_categories}")
        
        return errors
    
    def _validate_skepticism_level(self, level) -> List[str]:
        """Validate skepticism level."""
        errors = []
        
        if not isinstance(level, (int, float)):
            errors.append(f"Skepticism level must be numeric, got {type(level)}")
            return errors
        
        if not 0.0 <= level <= 1.0:
            errors.append(f"Skepticism level must be between 0.0 and 1.0, got {level}")
        
        return errors
    
    def _validate_evidence_requests(self, requests) -> List[str]:
        """Validate evidence requests."""
        errors = []
        
        if not isinstance(requests, list):
            errors.append(f"Evidence requests must be list, got {type(requests)}")
            return errors
        
        if len(requests) == 0:
            errors.append("Must have at least one evidence request")
        
        if len(requests) > 10:
            errors.append(f"Too many evidence requests (max 10), got {len(requests)}")
        
        for i, req in enumerate(requests):
            if not isinstance(req, str):
                errors.append(f"Evidence request {i} must be string, got {type(req)}")
            elif len(req) < 5:
                errors.append(f"Evidence request {i} too short (min 5 chars): '{req}'")
            elif len(req) > 200:
                errors.append(f"Evidence request {i} too long (max 200 chars)")
        
        return errors
    
    def _validate_red_flags(self, flags) -> List[str]:
        """Validate red flags."""
        errors = []
        
        if not isinstance(flags, list):
            errors.append(f"Red flags must be list, got {type(flags)}")
            return errors
        
        if len(flags) == 0:
            errors.append("Must have at least one red flag")
        
        if len(flags) > 15:
            errors.append(f"Too many red flags (max 15), got {len(flags)}")
        
        for i, flag in enumerate(flags):
            if not isinstance(flag, str):
                errors.append(f"Red flag {i} must be string, got {type(flag)}")
            elif len(flag) < 5:
                errors.append(f"Red flag {i} too short (min 5 chars): '{flag}'")
            elif len(flag) > 200:
                errors.append(f"Red flag {i} too long (max 200 chars)")
        
        return errors
    
    def _validate_metadata(self, metadata) -> List[str]:
        """Validate scenario metadata."""
        errors = []
        
        if not isinstance(metadata, dict):
            errors.append(f"Metadata must be dict, got {type(metadata)}")
            return errors
        
        # Check for expected metadata fields
        if 'difficulty' in metadata:
            if metadata['difficulty'] not in ['easy', 'medium', 'hard']:
                errors.append(f"Invalid difficulty: {metadata['difficulty']}")
        
        if 'domain' in metadata:
            if not isinstance(metadata['domain'], str) or len(metadata['domain']) < 3:
                errors.append("Domain must be string with at least 3 characters")
        
        if 'tags' in metadata:
            if not isinstance(metadata['tags'], list):
                errors.append("Tags must be list")
            elif len(metadata['tags']) > 20:
                errors.append("Too many tags (max 20)")
        
        return errors


class ResponseValidator:
    """Validates skeptic responses."""
    
    def validate_response(self, response: SkepticResponse) -> List[str]:
        """Validate a SkepticResponse object."""
        errors = []
        
        # Validate confidence level
        if not 0.0 <= response.confidence_level <= 1.0:
            errors.append(f"Confidence level must be between 0.0 and 1.0: {response.confidence_level}")
        
        # Validate response text
        if len(response.response_text) < 50:
            errors.append("Response text too short (minimum 50 characters)")
        elif len(response.response_text) > 5000:
            errors.append("Response text too long (maximum 5000 characters)")
        
        # Validate lists
        if len(response.evidence_requests) > 20:
            errors.append(f"Too many evidence requests (max 20): {len(response.evidence_requests)}")
        
        if len(response.red_flags_identified) > 30:
            errors.append(f"Too many red flags identified (max 30): {len(response.red_flags_identified)}")
        
        if len(response.reasoning_steps) > 20:
            errors.append(f"Too many reasoning steps (max 20): {len(response.reasoning_steps)}")
        
        # Validate response time
        if response.response_time_ms < 0:
            errors.append(f"Response time cannot be negative: {response.response_time_ms}")
        elif response.response_time_ms > 300000:  # 5 minutes max
            errors.append(f"Response time too long (max 5 minutes): {response.response_time_ms}ms")
        
        return errors


class ConfigValidator:
    """Validates agent configurations."""
    
    def validate_agent_config(self, config: AgentConfig) -> List[str]:
        """Validate an AgentConfig object."""
        errors = []
        
        # Validate temperature
        if not 0.0 <= config.temperature <= 2.0:
            errors.append(f"Temperature must be between 0.0 and 2.0: {config.temperature}")
        
        # Validate max_tokens
        if config.max_tokens <= 0:
            errors.append(f"Max tokens must be positive: {config.max_tokens}")
        elif config.max_tokens > 8000:
            errors.append(f"Max tokens too high (max 8000): {config.max_tokens}")
        
        # Validate timeout
        if config.timeout <= 0:
            errors.append(f"Timeout must be positive: {config.timeout}")
        elif config.timeout > 300:  # 5 minutes max
            errors.append(f"Timeout too long (max 300s): {config.timeout}")
        
        # Validate API key format (basic check)
        if config.api_key and len(config.api_key) < 10:
            errors.append("API key appears to be too short")
        
        return errors


# Global validator instances
scenario_validator = ScenarioValidator()
response_validator = ResponseValidator()
config_validator = ConfigValidator()


def validate_scenario_file(file_path: Path) -> Tuple[bool, List[str]]:
    """Validate a scenario JSON file."""
    import json
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"Failed to load JSON: {e}"]
    
    errors = scenario_validator.validate_scenario_data(data)
    
    if not errors:
        try:
            scenario = Scenario(**data)
            errors.extend(scenario_validator.validate_scenario(scenario))
        except Exception as e:
            errors.append(f"Failed to create Scenario object: {e}")
    
    return len(errors) == 0, errors


class QuantumValidator:
    """Quantum-inspired validation for advanced consistency checking."""
    
    def __init__(self):
        self.coherence_threshold = 0.7
        self.entanglement_threshold = 0.5
        self.validation_history: List[Dict[str, Any]] = []
    
    def validate_quantum_coherence(self, evaluation_results: List[Tuple[Scenario, SkepticResponse, EvaluationMetrics]]) -> Dict[str, Any]:
        """Validate quantum coherence across evaluation results."""
        if not evaluation_results:
            return {"valid": False, "error": "No evaluation results provided"}
        
        coherence_scores = []
        phase_alignments = []
        
        for scenario, response, metrics in evaluation_results:
            # Calculate quantum coherence
            expected_skepticism = scenario.correct_skepticism_level
            actual_skepticism = 1.0 - response.confidence_level if response.confidence_level <= 0.5 else response.confidence_level
            
            coherence = 1.0 - abs(expected_skepticism - actual_skepticism)
            coherence_scores.append(coherence)
            
            # Calculate phase alignment
            phase_alignment = self._calculate_phase_alignment(metrics)
            phase_alignments.append(phase_alignment)
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        avg_phase_alignment = sum(phase_alignments) / len(phase_alignments)
        
        # Quantum validation criteria
        is_coherent = avg_coherence >= self.coherence_threshold
        is_aligned = avg_phase_alignment >= 0.6
        
        validation_result = {
            "valid": is_coherent and is_aligned,
            "quantum_coherence": avg_coherence,
            "phase_alignment": avg_phase_alignment,
            "coherence_threshold": self.coherence_threshold,
            "coherence_distribution": coherence_scores,
            "phase_distribution": phase_alignments,
            "warnings": self._generate_quantum_warnings(avg_coherence, avg_phase_alignment),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.validation_history.append(validation_result)
        return validation_result
    
    def validate_parameter_entanglement(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Validate quantum entanglement between parameters."""
        if len(parameters) < 2:
            return {"valid": True, "entanglement": 0.0, "warning": "Too few parameters for entanglement analysis"}
        
        param_values = list(parameters.values())
        entanglement_matrix = []
        
        # Calculate entanglement between all parameter pairs
        for i in range(len(param_values)):
            row = []
            for j in range(len(param_values)):
                if i == j:
                    row.append(1.0)  # Perfect self-entanglement
                else:
                    # Quantum entanglement measure
                    entanglement = self._calculate_entanglement(param_values[i], param_values[j])
                    row.append(entanglement)
            entanglement_matrix.append(row)
        
        # Average entanglement strength
        total_entanglement = 0.0
        count = 0
        for i in range(len(entanglement_matrix)):
            for j in range(i + 1, len(entanglement_matrix[i])):
                total_entanglement += entanglement_matrix[i][j]
                count += 1
        
        avg_entanglement = total_entanglement / count if count > 0 else 0.0
        
        return {
            "valid": avg_entanglement >= self.entanglement_threshold,
            "entanglement_strength": avg_entanglement,
            "entanglement_matrix": entanglement_matrix,
            "parameter_names": list(parameters.keys()),
            "threshold": self.entanglement_threshold,
            "analysis": self._analyze_entanglement_patterns(entanglement_matrix, list(parameters.keys()))
        }
    
    def validate_uncertainty_principle(self, response: SkepticResponse, expected_uncertainty: float) -> Dict[str, Any]:
        """Validate quantum uncertainty principle in responses."""
        # Calculate response uncertainty from various factors
        confidence_uncertainty = 1.0 - abs(response.confidence_level - 0.5) * 2  # Higher uncertainty for mid-range confidence
        evidence_uncertainty = min(1.0, len(response.evidence_requests) / 5.0)  # More evidence requests = higher uncertainty
        reasoning_uncertainty = min(1.0, len(response.reasoning_steps) / 10.0)  # More reasoning steps = higher uncertainty
        
        # Quantum superposition of uncertainties
        measured_uncertainty = (
            confidence_uncertainty * 0.4 +
            evidence_uncertainty * 0.3 +
            reasoning_uncertainty * 0.3
        )
        
        # Heisenberg-like uncertainty relation
        uncertainty_product = measured_uncertainty * (1.0 - measured_uncertainty)
        min_uncertainty_product = 0.25  # Quantum minimum
        
        uncertainty_valid = uncertainty_product >= min_uncertainty_product
        expectation_match = abs(measured_uncertainty - expected_uncertainty) < 0.3
        
        return {
            "valid": uncertainty_valid and expectation_match,
            "measured_uncertainty": measured_uncertainty,
            "expected_uncertainty": expected_uncertainty,
            "uncertainty_product": uncertainty_product,
            "min_uncertainty_product": min_uncertainty_product,
            "components": {
                "confidence_uncertainty": confidence_uncertainty,
                "evidence_uncertainty": evidence_uncertainty,
                "reasoning_uncertainty": reasoning_uncertainty
            },
            "satisfies_uncertainty_principle": uncertainty_valid,
            "matches_expectation": expectation_match
        }
    
    def _calculate_phase_alignment(self, metrics: EvaluationMetrics) -> float:
        """Calculate quantum phase alignment of evaluation metrics."""
        # Use metrics as quantum phases
        phases = [
            metrics.skepticism_calibration * 2 * math.pi,
            metrics.evidence_standard_score * 2 * math.pi,
            metrics.red_flag_detection * 2 * math.pi,
            metrics.reasoning_quality * 2 * math.pi
        ]
        
        # Calculate phase coherence using circular mean
        sin_sum = sum(math.sin(phase) for phase in phases)
        cos_sum = sum(math.cos(phase) for phase in phases)
        
        # Phase alignment strength
        magnitude = math.sqrt(sin_sum**2 + cos_sum**2)
        alignment = magnitude / len(phases)
        
        return alignment
    
    def _calculate_entanglement(self, param1: float, param2: float) -> float:
        """Calculate quantum entanglement between two parameters."""
        # Quantum entanglement based on parameter correlation
        product = abs(param1 * param2)
        sum_squares = param1**2 + param2**2
        
        if sum_squares == 0:
            return 0.0
        
        # Normalized entanglement measure
        entanglement = (2 * product) / sum_squares
        return min(1.0, entanglement)
    
    def _generate_quantum_warnings(self, coherence: float, phase_alignment: float) -> List[str]:
        """Generate warnings based on quantum validation results."""
        warnings = []
        
        if coherence < 0.3:
            warnings.append("Severe quantum decoherence detected - evaluation consistency may be compromised")
        elif coherence < self.coherence_threshold:
            warnings.append("Below-threshold quantum coherence - consider parameter optimization")
        
        if phase_alignment < 0.3:
            warnings.append("Poor metric phase alignment - evaluation metrics may be inconsistent")
        elif phase_alignment < 0.6:
            warnings.append("Moderate phase alignment - metrics showing some inconsistency")
        
        if coherence > 0.95:
            warnings.append("Suspiciously high coherence - may indicate overfitting or data issues")
        
        return warnings
    
    def _analyze_entanglement_patterns(self, matrix: List[List[float]], param_names: List[str]) -> Dict[str, Any]:
        """Analyze patterns in parameter entanglement matrix."""
        n = len(matrix)
        analysis = {
            "strongest_entanglement": {"value": 0.0, "parameters": []},
            "weakest_entanglement": {"value": 1.0, "parameters": []},
            "average_entanglement": 0.0,
            "entanglement_clusters": []
        }
        
        # Find strongest and weakest entanglements
        total_entanglement = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                entanglement = matrix[i][j]
                total_entanglement += entanglement
                count += 1
                
                if entanglement > analysis["strongest_entanglement"]["value"]:
                    analysis["strongest_entanglement"]["value"] = entanglement
                    analysis["strongest_entanglement"]["parameters"] = [param_names[i], param_names[j]]
                
                if entanglement < analysis["weakest_entanglement"]["value"]:
                    analysis["weakest_entanglement"]["value"] = entanglement
                    analysis["weakest_entanglement"]["parameters"] = [param_names[i], param_names[j]]
        
        analysis["average_entanglement"] = total_entanglement / count if count > 0 else 0.0
        
        # Identify entanglement clusters (parameters with high mutual entanglement)
        high_entanglement_threshold = 0.7
        clusters = []
        
        for i in range(n):
            cluster = [param_names[i]]
            for j in range(n):
                if i != j and matrix[i][j] > high_entanglement_threshold:
                    cluster.append(param_names[j])
            
            if len(cluster) > 1:
                # Check if this cluster is already represented
                is_new_cluster = True
                for existing_cluster in clusters:
                    if set(cluster) == set(existing_cluster):
                        is_new_cluster = False
                        break
                
                if is_new_cluster:
                    clusters.append(cluster)
        
        analysis["entanglement_clusters"] = clusters
        
        return analysis
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all quantum validations performed."""
        if not self.validation_history:
            return {"error": "No validation history available"}
        
        recent_validations = self.validation_history[-10:]  # Last 10 validations
        
        avg_coherence = sum(v["quantum_coherence"] for v in recent_validations) / len(recent_validations)
        avg_phase_alignment = sum(v["phase_alignment"] for v in recent_validations) / len(recent_validations)
        
        valid_count = sum(1 for v in recent_validations if v["valid"])
        validity_rate = valid_count / len(recent_validations)
        
        return {
            "total_validations": len(self.validation_history),
            "recent_validations": len(recent_validations),
            "average_coherence": avg_coherence,
            "average_phase_alignment": avg_phase_alignment,
            "validity_rate": validity_rate,
            "coherence_trend": self._calculate_trend([v["quantum_coherence"] for v in recent_validations]),
            "recommendations": self._generate_validation_recommendations(avg_coherence, avg_phase_alignment, validity_rate)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi**2 for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _generate_validation_recommendations(self, coherence: float, phase_alignment: float, validity_rate: float) -> List[str]:
        """Generate recommendations based on validation patterns."""
        recommendations = []
        
        if coherence < 0.5:
            recommendations.append("Consider quantum optimization to improve coherence")
        
        if phase_alignment < 0.5:
            recommendations.append("Review metric calculation methods for better phase alignment")
        
        if validity_rate < 0.7:
            recommendations.append("Investigate causes of validation failures")
        
        if coherence > 0.9 and phase_alignment > 0.9:
            recommendations.append("Excellent quantum properties - system is well-calibrated")
        
        return recommendations if recommendations else ["Quantum validation within acceptable parameters"]


class SecurityValidator:
    """Enhanced security validation for the evaluation system."""
    
    def __init__(self):
        self.suspicious_patterns = [
            r'<script.*?>.*?</script>',  # Script injection
            r'javascript:',              # JavaScript URLs
            r'data:.*base64',           # Base64 data URIs
            r'\\x[0-9a-fA-F]{2}',       # Hex encoded data
            r'eval\s*\(',               # eval() calls
            r'exec\s*\(',               # exec() calls
            r'import\s+os',             # OS imports
            r'__import__',              # Dynamic imports
        ]
        self.max_input_length = 10000
        self.rate_limit_window = 60  # seconds
        self.max_requests_per_window = 100
        
    def validate_input_security(self, input_data: Any) -> Dict[str, Any]:
        """Validate input for security threats."""
        security_issues = []
        risk_level = "low"
        
        if isinstance(input_data, str):
            # Check for suspicious patterns
            for pattern in self.suspicious_patterns:
                if re.search(pattern, input_data, re.IGNORECASE):
                    security_issues.append(f"Suspicious pattern detected: {pattern}")
                    risk_level = "high"
            
            # Check input length
            if len(input_data) > self.max_input_length:
                security_issues.append(f"Input too long: {len(input_data)} > {self.max_input_length}")
                risk_level = "medium" if risk_level == "low" else risk_level
        
        elif isinstance(input_data, dict):
            # Recursively check dictionary values
            for key, value in input_data.items():
                nested_result = self.validate_input_security(value)
                security_issues.extend(nested_result["issues"])
                if nested_result["risk_level"] == "high":
                    risk_level = "high"
                elif nested_result["risk_level"] == "medium" and risk_level == "low":
                    risk_level = "medium"
        
        elif isinstance(input_data, list):
            # Check list elements
            for item in input_data:
                nested_result = self.validate_input_security(item)
                security_issues.extend(nested_result["issues"])
                if nested_result["risk_level"] == "high":
                    risk_level = "high"
                elif nested_result["risk_level"] == "medium" and risk_level == "low":
                    risk_level = "medium"
        
        return {
            "safe": len(security_issues) == 0,
            "risk_level": risk_level,
            "issues": security_issues,
            "input_type": type(input_data).__name__,
            "validation_timestamp": datetime.utcnow().isoformat()
        }
    
    def validate_file_security(self, file_path: Path) -> Dict[str, Any]:
        """Validate file for security issues."""
        security_issues = []
        
        # Check file extension
        allowed_extensions = {'.json', '.txt', '.csv', '.yaml', '.yml'}
        if file_path.suffix.lower() not in allowed_extensions:
            security_issues.append(f"Potentially unsafe file extension: {file_path.suffix}")
        
        # Check file size
        try:
            file_size = file_path.stat().st_size
            max_file_size = 50 * 1024 * 1024  # 50MB
            if file_size > max_file_size:
                security_issues.append(f"File too large: {file_size} bytes")
        except Exception as e:
            security_issues.append(f"Could not check file size: {e}")
        
        # Check file content if it's a text file
        if file_path.suffix.lower() in {'.json', '.txt', '.yaml', '.yml'}:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(self.max_input_length)  # Read limited content
                    content_validation = self.validate_input_security(content)
                    security_issues.extend(content_validation["issues"])
            except Exception as e:
                security_issues.append(f"Could not read file content: {e}")
        
        return {
            "safe": len(security_issues) == 0,
            "issues": security_issues,
            "file_path": str(file_path),
            "file_size": file_size if 'file_size' in locals() else None,
            "validation_timestamp": datetime.utcnow().isoformat()
        }


# Enhanced global validator instances
scenario_validator = ScenarioValidator()
response_validator = ResponseValidator()
config_validator = ConfigValidator()
quantum_validator = QuantumValidator()
security_validator = SecurityValidator()