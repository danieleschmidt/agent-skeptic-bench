"""Validation utilities for Agent Skeptic Bench."""

import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .models import Scenario, ScenarioCategory, SkepticResponse, AgentConfig
from .exceptions import ValidationError, InvalidScenarioError


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