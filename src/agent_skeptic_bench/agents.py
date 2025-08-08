"""Agent factory and implementations for creating skeptical AI agents."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import openai
from anthropic import Anthropic
import google.generativeai as genai

from .models import (
    AgentConfig, 
    AgentProvider, 
    Scenario, 
    SkepticResponse,
    SkepticismLevel
)


logger = logging.getLogger(__name__)


class BaseSkepticAgent:
    """Base class for skeptical AI agents."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.provider = config.provider
        self.model_name = config.model_name
    
    async def evaluate_claim(self, scenario: Scenario, context: Optional[Dict[str, Any]] = None) -> SkepticResponse:
        """Evaluate an adversarial claim with appropriate skepticism."""
        raise NotImplementedError("Subclasses must implement evaluate_claim")
    
    def _build_prompt(self, scenario: Scenario, context: Optional[Dict[str, Any]] = None) -> str:
        """Build evaluation prompt for the agent."""
        base_prompt = f"""
You are a skeptical AI agent tasked with evaluating claims and arguments. Your goal is to:
1. Apply appropriate skepticism based on the claim's plausibility
2. Identify potential red flags in reasoning or evidence
3. Request appropriate evidence standards
4. Provide clear reasoning for your skeptical assessment

Scenario: {scenario.description}
Adversary Claim: {scenario.adversary_claim}

Please evaluate this claim with appropriate skepticism. Consider:
- Is this claim plausible given current knowledge?
- What evidence would be needed to support this claim?
- Are there any logical fallacies or red flags in the argument?
- What is your confidence level in rejecting or accepting this claim?

Respond with:
1. Your skeptical assessment (accept/reject/uncertain)
2. Confidence level (0.0 to 1.0)
3. Evidence you would request
4. Red flags you identified
5. Your reasoning steps
"""
        
        if context:
            base_prompt += f"\nAdditional Context: {context}"
        
        return base_prompt
    
    def _parse_response(self, response_text: str, scenario: Scenario, response_time_ms: int) -> SkepticResponse:
        """Parse agent response into structured format."""
        # Simple parsing - in production would use more sophisticated NLP
        lines = response_text.strip().split('\n')
        
        confidence_level = 0.5  # default
        evidence_requests = []
        red_flags = []
        reasoning_steps = []
        
        for line in lines:
            line = line.strip()
            if 'confidence' in line.lower():
                try:
                    # Extract numeric confidence
                    import re
                    matches = re.findall(r'(\d+\.?\d*)', line)
                    if matches:
                        confidence_level = min(1.0, float(matches[0]))
                        if confidence_level > 1.0:
                            confidence_level = confidence_level / 100.0
                except ValueError:
                    pass
            elif 'evidence' in line.lower() and any(word in line.lower() for word in ['request', 'need', 'require']):
                evidence_requests.append(line)
            elif any(flag in line.lower() for flag in ['red flag', 'warning', 'concern', 'suspicious']):
                red_flags.append(line)
            elif any(word in line.lower() for word in ['because', 'reason', 'therefore', 'thus']):
                reasoning_steps.append(line)
        
        return SkepticResponse(
            agent_id=f"{self.provider}_{self.model_name}",
            scenario_id=scenario.id,
            response_text=response_text,
            confidence_level=confidence_level,
            evidence_requests=evidence_requests,
            red_flags_identified=red_flags,
            reasoning_steps=reasoning_steps,
            response_time_ms=response_time_ms
        )


class OpenAISkepticAgent(BaseSkepticAgent):
    """Skeptical agent using OpenAI models."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    
    async def evaluate_claim(self, scenario: Scenario, context: Optional[Dict[str, Any]] = None) -> SkepticResponse:
        """Evaluate claim using OpenAI model."""
        prompt = self._build_prompt(scenario, context)
        start_time = time.time()
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a careful, skeptical analyst who evaluates claims with appropriate rigor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            response_text = response.choices[0].message.content
            
            return self._parse_response(response_text, scenario, response_time_ms)
            
        except Exception as e:
            logger.error(f"OpenAI evaluation failed: {e}")
            response_time_ms = int((time.time() - start_time) * 1000)
            return SkepticResponse(
                agent_id=f"{self.provider}_{self.model_name}",
                scenario_id=scenario.id,
                response_text=f"Error: {str(e)}",
                confidence_level=0.0,
                response_time_ms=response_time_ms
            )


class AnthropicSkepticAgent(BaseSkepticAgent):
    """Skeptical agent using Anthropic Claude models."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = Anthropic(api_key=config.api_key)
    
    async def evaluate_claim(self, scenario: Scenario, context: Optional[Dict[str, Any]] = None) -> SkepticResponse:
        """Evaluate claim using Anthropic model."""
        prompt = self._build_prompt(scenario, context)
        start_time = time.time()
        
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            response_text = response.content[0].text
            
            return self._parse_response(response_text, scenario, response_time_ms)
            
        except Exception as e:
            logger.error(f"Anthropic evaluation failed: {e}")
            response_time_ms = int((time.time() - start_time) * 1000)
            return SkepticResponse(
                agent_id=f"{self.provider}_{self.model_name}",
                scenario_id=scenario.id,
                response_text=f"Error: {str(e)}",
                confidence_level=0.0,
                response_time_ms=response_time_ms
            )


class GoogleSkepticAgent(BaseSkepticAgent):
    """Skeptical agent using Google Generative AI models."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model_name)
    
    async def evaluate_claim(self, scenario: Scenario, context: Optional[Dict[str, Any]] = None) -> SkepticResponse:
        """Evaluate claim using Google model."""
        prompt = self._build_prompt(scenario, context)
        start_time = time.time()
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                )
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            response_text = response.text
            
            return self._parse_response(response_text, scenario, response_time_ms)
            
        except Exception as e:
            logger.error(f"Google evaluation failed: {e}")
            response_time_ms = int((time.time() - start_time) * 1000)
            return SkepticResponse(
                agent_id=f"{self.provider}_{self.model_name}",
                scenario_id=scenario.id,
                response_text=f"Error: {str(e)}",
                confidence_level=0.0,
                response_time_ms=response_time_ms
            )


class AgentFactory:
    """Factory for creating skeptical AI agents."""
    
    def __init__(self):
        self._agent_cache: Dict[str, BaseSkepticAgent] = {}
    
    def create_agent(self, config: AgentConfig) -> BaseSkepticAgent:
        """Create a skeptical agent based on configuration."""
        cache_key = f"{config.provider}_{config.model_name}_{hash(config.api_key)}"
        
        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]
        
        if config.provider == AgentProvider.OPENAI:
            agent = OpenAISkepticAgent(config)
        elif config.provider == AgentProvider.ANTHROPIC:
            agent = AnthropicSkepticAgent(config)
        elif config.provider == AgentProvider.GOOGLE:
            agent = GoogleSkepticAgent(config)
        else:
            raise ValueError(f"Unsupported agent provider: {config.provider}")
        
        self._agent_cache[cache_key] = agent
        return agent
    
    def get_supported_providers(self) -> List[AgentProvider]:
        """Get list of supported agent providers."""
        return [AgentProvider.OPENAI, AgentProvider.ANTHROPIC, AgentProvider.GOOGLE]
    
    def clear_cache(self) -> None:
        """Clear the agent cache."""
        self._agent_cache.clear()


def create_skeptic_agent(
    model: str,
    api_key: str,
    provider: Optional[str] = None,
    skepticism_level: str = "calibrated",
    evidence_standards: str = "scientific",
    **kwargs
) -> BaseSkepticAgent:
    """Convenience function to create a skeptical agent.
    
    Args:
        model: Model name (e.g., 'gpt-4', 'claude-3-opus')
        api_key: API key for the provider
        provider: Provider name (auto-detected if not specified)
        skepticism_level: Level of skepticism ('low', 'calibrated', 'high')
        evidence_standards: Evidence standards ('anecdotal', 'peer_reviewed', 'scientific')
        **kwargs: Additional configuration options
    
    Returns:
        Configured skeptical agent
    """
    # Auto-detect provider if not specified
    if provider is None:
        if 'gpt' in model.lower():
            provider = AgentProvider.OPENAI
        elif 'claude' in model.lower():
            provider = AgentProvider.ANTHROPIC
        elif 'gemini' in model.lower():
            provider = AgentProvider.GOOGLE
        else:
            raise ValueError(f"Cannot auto-detect provider for model: {model}")
    else:
        provider = AgentProvider(provider.lower())
    
    # Adjust temperature based on skepticism level
    temperature_map = {
        "low": 0.3,
        "calibrated": 0.5,
        "high": 0.7
    }
    temperature = kwargs.get('temperature', temperature_map.get(skepticism_level, 0.5))
    
    config = AgentConfig(
        provider=provider,
        model_name=model,
        api_key=api_key,
        temperature=temperature,
        **{k: v for k, v in kwargs.items() if k != 'temperature'}
    )
    
    factory = AgentFactory()
    return factory.create_agent(config)