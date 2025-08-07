"""Agent factory and implementations for creating skeptical AI agents."""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import openai
from anthropic import Anthropic
import google.generativeai as genai
from PIL import Image
import base64
import io

from .models import (
    AgentConfig, 
    AgentProvider, 
    Scenario, 
    SkepticResponse,
    SkepticismLevel,
    MultiModalInput,
    AdaptationMetrics
)


logger = logging.getLogger(__name__)


class BaseSkepticAgent:
    """Base class for skeptical AI agents with multi-modal and adaptive capabilities."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.provider = config.provider
        self.model_name = config.model_name
        self.adaptation_history: List[AdaptationMetrics] = []
        self.performance_window = timedelta(hours=1)
        self.adaptation_threshold = 0.1
        self.multimodal_cache = {}
        
        # Initialize adaptive parameters
        self.adaptive_temperature = config.temperature
        self.adaptive_confidence_threshold = 0.5
        self.context_memory = []
        self.max_context_length = 10
    
    async def evaluate_claim(self, scenario: Scenario, context: Optional[Dict[str, Any]] = None) -> SkepticResponse:
        """Evaluate an adversarial claim with appropriate skepticism."""
        raise NotImplementedError("Subclasses must implement evaluate_claim")
    
    async def evaluate_multimodal_claim(self, 
                                       scenario: Scenario, 
                                       inputs: List[MultiModalInput],
                                       context: Optional[Dict[str, Any]] = None) -> SkepticResponse:
        """Evaluate claims with multiple input modalities (text, image, audio, video)."""
        # Process different input types
        processed_inputs = await self._process_multimodal_inputs(inputs)
        
        # Enhanced context with multimodal data
        enhanced_context = context or {}
        enhanced_context['multimodal_inputs'] = processed_inputs
        
        # Use standard evaluation with enhanced context
        response = await self.evaluate_claim(scenario, enhanced_context)
        
        # Apply multimodal-specific skepticism adjustments
        adjusted_response = await self._adjust_for_multimodal_skepticism(response, inputs)
        
        return adjusted_response
    
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
        
        # Add multimodal context if available
        if context and 'multimodal_inputs' in context:
            multimodal_info = "\nMultimodal Evidence Analysis:\n"
            for input_data in context['multimodal_inputs']:
                multimodal_info += f"- {input_data['type']}: {input_data['analysis']}\n"
            base_prompt += multimodal_info
        
        # Add adaptive context from performance history
        if self.context_memory:
            recent_patterns = self._analyze_recent_patterns()
            if recent_patterns:
                base_prompt += f"\nRecent Performance Insights: {recent_patterns}\n"
        
        return base_prompt
    
    async def _parse_response(self, response_text: str, scenario: Scenario, response_time_ms: int) -> SkepticResponse:
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
        
        # Create base response
        response = SkepticResponse(
            agent_id=f"{self.provider}_{self.model_name}",
            scenario_id=scenario.id,
            response_text=response_text,
            confidence_level=confidence_level,
            evidence_requests=evidence_requests,
            red_flags_identified=red_flags,
            reasoning_steps=reasoning_steps,
            response_time_ms=response_time_ms
        )
        
        # Apply adaptive learning
        await self._update_adaptation_metrics(response, scenario)
        
        return response
    
    async def _process_multimodal_inputs(self, inputs: List[MultiModalInput]) -> List[Dict[str, Any]]:
        """Process different types of multimodal inputs."""
        processed = []
        
        for input_item in inputs:
            if input_item.type == "image":
                analysis = await self._analyze_image(input_item.data)
            elif input_item.type == "audio":
                analysis = await self._analyze_audio(input_item.data)
            elif input_item.type == "video":
                analysis = await self._analyze_video(input_item.data)
            elif input_item.type == "document":
                analysis = await self._analyze_document(input_item.data)
            else:
                analysis = "Unsupported input type"
            
            processed.append({
                'type': input_item.type,
                'analysis': analysis,
                'metadata': input_item.metadata or {}
            })
        
        return processed
    
    async def _analyze_image(self, image_data: Union[str, bytes]) -> str:
        """Analyze image for potential deception or manipulation."""
        try:
            # Convert to PIL Image for analysis
            if isinstance(image_data, str):
                # Assume base64 encoded
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
                
            image = Image.open(io.BytesIO(image_bytes))
            
            # Basic image analysis for skepticism
            analysis = []
            
            # Check image metadata for manipulation signs
            if hasattr(image, '_getexif') and image._getexif():
                analysis.append("Image contains EXIF data - verify authenticity")
            
            # Basic format and quality checks
            if image.format in ['JPEG', 'PNG']:
                analysis.append(f"Standard {image.format} format - examine for compression artifacts")
            
            # Size and resolution analysis
            width, height = image.size
            if width * height > 5000000:  # High resolution
                analysis.append("High resolution image - may contain hidden details")
            
            return "; ".join(analysis) if analysis else "Image appears standard - verify source and context"
            
        except Exception as e:
            return f"Image analysis failed: {str(e)}"
    
    async def _analyze_audio(self, audio_data: Union[str, bytes]) -> str:
        """Analyze audio for potential manipulation or synthetic content."""
        # Placeholder for audio analysis - would integrate with audio processing libraries
        return "Audio analysis: Check for synthetic speech patterns, background consistency, and edit points"
    
    async def _analyze_video(self, video_data: Union[str, bytes]) -> str:
        """Analyze video for deepfakes or manipulation."""
        # Placeholder for video analysis - would integrate with deepfake detection
        return "Video analysis: Examine for deepfake artifacts, temporal inconsistencies, and editing signatures"
    
    async def _analyze_document(self, document_data: Union[str, bytes]) -> str:
        """Analyze document for authenticity and manipulation."""
        # Placeholder for document analysis
        return "Document analysis: Verify formatting consistency, metadata integrity, and content authenticity"
    
    async def _adjust_for_multimodal_skepticism(self, 
                                               response: SkepticResponse, 
                                               inputs: List[MultiModalInput]) -> SkepticResponse:
        """Adjust skepticism based on multimodal input analysis."""
        # Increase skepticism for visual content (easier to manipulate)
        visual_inputs = [inp for inp in inputs if inp.type in ['image', 'video']]
        if visual_inputs:
            # Slightly increase skepticism for visual content
            adjusted_confidence = min(1.0, response.confidence_level * 1.1)
            response.confidence_level = adjusted_confidence
            response.red_flags_identified.append("Visual content present - increased scrutiny applied")
        
        return response
    
    async def _update_adaptation_metrics(self, response: SkepticResponse, scenario: Scenario):
        """Update adaptation metrics based on evaluation performance."""
        current_time = datetime.utcnow()
        
        # Calculate performance metrics
        expected_skepticism = getattr(scenario, 'correct_skepticism_level', 0.5)
        skepticism_error = abs(response.confidence_level - expected_skepticism)
        
        # Store adaptation metrics
        metrics = AdaptationMetrics(
            timestamp=current_time,
            scenario_id=scenario.id,
            expected_skepticism=expected_skepticism,
            actual_skepticism=response.confidence_level,
            error_magnitude=skepticism_error,
            response_time_ms=response.response_time_ms
        )
        
        self.adaptation_history.append(metrics)
        
        # Keep only recent history
        cutoff_time = current_time - self.performance_window
        self.adaptation_history = [
            m for m in self.adaptation_history if m.timestamp > cutoff_time
        ]
        
        # Update adaptive parameters if needed
        if len(self.adaptation_history) >= 5:
            await self._adjust_adaptive_parameters()
    
    async def _adjust_adaptive_parameters(self):
        """Adjust agent parameters based on recent performance."""
        if not self.adaptation_history:
            return
        
        # Calculate average error over recent evaluations
        recent_errors = [m.error_magnitude for m in self.adaptation_history[-5:]]
        avg_error = sum(recent_errors) / len(recent_errors)
        
        # Adjust temperature based on performance
        if avg_error > self.adaptation_threshold:
            # High error - increase exploration (higher temperature)
            self.adaptive_temperature = min(1.0, self.adaptive_temperature * 1.05)
        else:
            # Good performance - slight exploitation (lower temperature)
            self.adaptive_temperature = max(0.1, self.adaptive_temperature * 0.98)
        
        # Adjust confidence threshold
        avg_response_time = sum(m.response_time_ms for m in self.adaptation_history[-5:]) / 5
        if avg_response_time > 3000:  # Slow responses
            # Lower confidence threshold for faster decisions
            self.adaptive_confidence_threshold = max(0.3, self.adaptive_confidence_threshold * 0.95)
        
        logger.info(f"Adaptive parameters updated: temp={self.adaptive_temperature:.3f}, "
                   f"conf_threshold={self.adaptive_confidence_threshold:.3f}")
    
    def _analyze_recent_patterns(self) -> Optional[str]:
        """Analyze recent evaluation patterns for contextual insights."""
        if len(self.adaptation_history) < 3:
            return None
        
        recent_metrics = self.adaptation_history[-3:]
        
        # Analyze error trends
        errors = [m.error_magnitude for m in recent_metrics]
        if all(e1 > e2 for e1, e2 in zip(errors[:-1], errors[1:])):
            return "Improving accuracy trend detected - maintaining current approach"
        elif all(e1 < e2 for e1, e2 in zip(errors[:-1], errors[1:])):
            return "Declining accuracy detected - consider increasing skepticism"
        
        # Analyze response time patterns
        times = [m.response_time_ms for m in recent_metrics]
        avg_time = sum(times) / len(times)
        if avg_time > 5000:
            return "Extended response times - consider simpler evaluation approach"
        
        return None
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of agent's adaptation behavior."""
        if not self.adaptation_history:
            return {"status": "No adaptation data available"}
        
        recent_metrics = self.adaptation_history[-10:] if len(self.adaptation_history) >= 10 else self.adaptation_history
        
        return {
            "total_evaluations": len(self.adaptation_history),
            "recent_evaluations": len(recent_metrics),
            "average_error": sum(m.error_magnitude for m in recent_metrics) / len(recent_metrics),
            "average_response_time": sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics),
            "current_temperature": self.adaptive_temperature,
            "current_confidence_threshold": self.adaptive_confidence_threshold,
            "adaptation_active": len(self.adaptation_history) >= 5
        }


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
                temperature=self.adaptive_temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            response_text = response.choices[0].message.content
            
            return await self._parse_response(response_text, scenario, response_time_ms)
            
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
                temperature=self.adaptive_temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            response_text = response.content[0].text
            
            return await self._parse_response(response_text, scenario, response_time_ms)
            
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
                    temperature=self.adaptive_temperature,
                    max_output_tokens=self.config.max_tokens,
                )
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            response_text = response.text
            
            return await self._parse_response(response_text, scenario, response_time_ms)
            
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