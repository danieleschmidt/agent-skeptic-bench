# API Reference

## Overview

The Agent Skeptic Bench provides a comprehensive REST API for evaluating AI agent skepticism with quantum-inspired optimization capabilities. This reference covers all endpoints, request/response formats, and integration examples.

## Base URL

```
Production: https://api.agent-skeptic-bench.com
Development: http://localhost:8000
```

## Authentication

### JWT Token Authentication

Most endpoints require JWT authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <jwt_token>
```

### API Key Authentication

Alternative authentication using API keys:

```http
X-API-Key: <your_api_key>
```

### Obtain JWT Token

```http
POST /auth/token
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## Core API Endpoints

### Health and Status

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0.0",
  "quantum_optimization": "enabled",
  "database": "connected",
  "cache": "connected"
}
```

#### Readiness Check
```http
GET /ready
```

**Response:**
```json
{
  "ready": true,
  "services": {
    "database": "ready",
    "cache": "ready",
    "quantum_optimizer": "ready"
  }
}
```

### Session Management

#### Create Evaluation Session
```http
POST /sessions
Authorization: Bearer <token>
Content-Type: application/json

{
  "session_name": "my_evaluation_session",
  "agent_config": {
    "provider": "openai",
    "model_name": "gpt-4",
    "api_key": "sk-...",
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "evaluation_config": {
    "scenario_categories": ["factual_claims", "authority_appeals"],
    "quantum_optimization": true,
    "auto_calibration": true
  }
}
```

**Response:**
```json
{
  "session_id": "sess_123456789",
  "status": "created",
  "agent_config": {
    "provider": "openai",
    "model_name": "gpt-4",
    "temperature": 0.7,
    "quantum_enhanced": true
  },
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### List Sessions
```http
GET /sessions
Authorization: Bearer <token>
```

**Query Parameters:**
- `status` (optional): Filter by status (active, completed, failed)
- `limit` (optional): Number of results (default: 20, max: 100)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "sess_123456789",
      "session_name": "my_evaluation_session",
      "status": "active",
      "created_at": "2024-01-15T10:30:00Z",
      "scenarios_completed": 15,
      "total_scenarios": 50
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

#### Get Session Details
```http
GET /sessions/{session_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "session_id": "sess_123456789",
  "session_name": "my_evaluation_session",
  "status": "active",
  "agent_config": {
    "provider": "openai",
    "model_name": "gpt-4",
    "temperature": 0.7,
    "quantum_optimized": true
  },
  "evaluation_progress": {
    "scenarios_completed": 15,
    "total_scenarios": 50,
    "success_rate": 0.87,
    "average_skepticism_calibration": 0.82
  },
  "quantum_insights": {
    "coherence_level": 0.91,
    "entanglement_strength": 0.73,
    "optimization_fitness": 0.89
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T11:45:00Z"
}
```

### Scenario Management

#### List Available Scenarios
```http
GET /scenarios
Authorization: Bearer <token>
```

**Query Parameters:**
- `category` (optional): Filter by category
- `difficulty` (optional): Filter by difficulty (easy, medium, hard)
- `tags[]` (optional): Filter by tags (can specify multiple)

**Response:**
```json
{
  "scenarios": [
    {
      "id": "scenario_001",
      "category": "factual_claims",
      "title": "Climate Change Denial",
      "description": "A claim about climate change being a hoax",
      "difficulty": "medium",
      "correct_skepticism_level": 0.9,
      "tags": ["climate", "science", "misinformation"],
      "metadata": {
        "evidence_quality": 0.2,
        "plausibility": 0.1,
        "complexity": "medium"
      }
    }
  ],
  "total": 150,
  "categories": ["factual_claims", "authority_appeals", "statistical_claims"],
  "difficulties": ["easy", "medium", "hard"]
}
```

#### Get Scenario Details
```http
GET /scenarios/{scenario_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": "scenario_001",
  "category": "factual_claims",
  "title": "Climate Change Denial",
  "description": "A detailed scenario about climate change denial claims...",
  "adversary_claim": "Climate change is a hoax perpetrated by scientists for grant money.",
  "context": "Social media post with 10K shares",
  "correct_skepticism_level": 0.9,
  "evaluation_criteria": {
    "red_flags": ["scientific consensus ignored", "cherry-picking data"],
    "evidence_requests": ["peer-reviewed studies", "scientific consensus"],
    "reasoning_quality": "Should identify logical fallacies"
  },
  "metadata": {
    "evidence_quality": 0.2,
    "plausibility": 0.1,
    "complexity": "medium",
    "created_by": "expert_reviewer",
    "validation_status": "verified"
  }
}
```

### Evaluation Execution

#### Run Single Evaluation
```http
POST /sessions/{session_id}/evaluate
Authorization: Bearer <token>
Content-Type: application/json

{
  "scenario_id": "scenario_001",
  "evaluation_options": {
    "include_quantum_analysis": true,
    "generate_insights": true,
    "timeout_seconds": 30
  }
}
```

**Response:**
```json
{
  "evaluation_id": "eval_987654321",
  "session_id": "sess_123456789",
  "scenario_id": "scenario_001",
  "status": "completed",
  "agent_response": {
    "response_text": "I need to examine this claim carefully...",
    "confidence_level": 0.85,
    "skepticism_level": 0.92,
    "evidence_requests": [
      "peer-reviewed climate studies",
      "temperature data verification",
      "consensus among climate scientists"
    ],
    "reasoning_steps": [
      "Identified appeal to conspiracy",
      "Recognized scientific consensus contradiction",
      "Requested empirical evidence"
    ],
    "response_time_ms": 1250
  },
  "evaluation_metrics": {
    "skepticism_calibration": 0.89,
    "evidence_standard_score": 0.92,
    "red_flag_detection": 0.87,
    "reasoning_quality": 0.94,
    "overall_score": 0.91
  },
  "quantum_analysis": {
    "coherence_measurement": 0.93,
    "entanglement_with_previous": 0.78,
    "uncertainty_compliance": true,
    "optimization_contribution": 0.15
  },
  "execution_time_ms": 1347,
  "timestamp": "2024-01-15T10:35:00Z"
}
```

#### Run Batch Evaluation
```http
POST /sessions/{session_id}/evaluate/batch
Authorization: Bearer <token>
Content-Type: application/json

{
  "scenario_ids": ["scenario_001", "scenario_002", "scenario_003"],
  "evaluation_options": {
    "parallel_execution": true,
    "max_concurrent": 3,
    "include_quantum_analysis": true,
    "timeout_seconds": 60
  }
}
```

**Response:**
```json
{
  "batch_id": "batch_555666777",
  "session_id": "sess_123456789",
  "status": "running",
  "total_scenarios": 3,
  "completed_scenarios": 0,
  "estimated_completion": "2024-01-15T10:38:00Z",
  "progress_url": "/sessions/sess_123456789/batch/batch_555666777/progress"
}
```

#### Get Batch Progress
```http
GET /sessions/{session_id}/batch/{batch_id}/progress
Authorization: Bearer <token>
```

**Response:**
```json
{
  "batch_id": "batch_555666777",
  "status": "running",
  "progress": {
    "total_scenarios": 3,
    "completed_scenarios": 2,
    "failed_scenarios": 0,
    "progress_percentage": 66.7
  },
  "completed_evaluations": [
    {
      "evaluation_id": "eval_111",
      "scenario_id": "scenario_001",
      "status": "completed",
      "overall_score": 0.91
    },
    {
      "evaluation_id": "eval_222",
      "scenario_id": "scenario_002", 
      "status": "completed",
      "overall_score": 0.87
    }
  ],
  "running_evaluations": [
    {
      "evaluation_id": "eval_333",
      "scenario_id": "scenario_003",
      "status": "running",
      "started_at": "2024-01-15T10:36:30Z"
    }
  ]
}
```

### Quantum Optimization

#### Get Quantum Insights
```http
GET /sessions/{session_id}/quantum-insights
Authorization: Bearer <token>
```

**Response:**
```json
{
  "session_id": "sess_123456789",
  "quantum_insights": {
    "overall_coherence": 0.91,
    "parameter_entanglement": {
      "temperature_threshold": 0.73,
      "evidence_weight_threshold": 0.68,
      "temperature_evidence": 0.82
    },
    "optimization_performance": {
      "fitness_progression": [0.65, 0.72, 0.78, 0.85, 0.89],
      "convergence_generation": 42,
      "stability_metric": 0.94
    },
    "uncertainty_compliance": {
      "responses_compliant": 47,
      "total_responses": 50,
      "compliance_rate": 0.94
    },
    "recommendations": [
      "Consider increasing evidence weight for improved calibration",
      "Current parameter entanglement shows good stability",
      "Quantum coherence is within optimal range"
    ]
  },
  "generated_at": "2024-01-15T11:00:00Z"
}
```

#### Optimize Agent Parameters
```http
POST /sessions/{session_id}/optimize
Authorization: Bearer <token>
Content-Type: application/json

{
  "target_metrics": {
    "skepticism_calibration": 0.90,
    "evidence_standard_score": 0.85,
    "red_flag_detection": 0.88,
    "reasoning_quality": 0.92
  },
  "optimization_config": {
    "max_generations": 100,
    "population_size": 20,
    "mutation_rate": 0.1,
    "enable_quantum_tunneling": true
  }
}
```

**Response:**
```json
{
  "optimization_id": "opt_888999000",
  "session_id": "sess_123456789",
  "status": "running",
  "target_metrics": {
    "skepticism_calibration": 0.90,
    "evidence_standard_score": 0.85,
    "red_flag_detection": 0.88,
    "reasoning_quality": 0.92
  },
  "estimated_completion": "2024-01-15T11:15:00Z",
  "progress_url": "/sessions/sess_123456789/optimization/opt_888999000/progress"
}
```

#### Get Optimization Progress
```http
GET /sessions/{session_id}/optimization/{optimization_id}/progress
Authorization: Bearer <token>
```

**Response:**
```json
{
  "optimization_id": "opt_888999000",
  "status": "running",
  "progress": {
    "current_generation": 35,
    "max_generations": 100,
    "progress_percentage": 35.0,
    "current_best_fitness": 0.87,
    "fitness_history": [0.65, 0.68, 0.71, "...", 0.87],
    "convergence_status": "improving"
  },
  "current_best_parameters": {
    "temperature": 0.73,
    "skepticism_threshold": 0.68,
    "evidence_weight": 1.24
  },
  "quantum_metrics": {
    "population_coherence": 0.89,
    "parameter_entanglement": 0.76,
    "convergence_stability": 0.91
  }
}
```

#### Predict Scenario Difficulty
```http
POST /predict-difficulty
Authorization: Bearer <token>
Content-Type: application/json

{
  "scenario_id": "scenario_001",
  "agent_parameters": {
    "temperature": 0.7,
    "skepticism_threshold": 0.6,
    "evidence_weight": 1.2
  }
}
```

**Response:**
```json
{
  "scenario_id": "scenario_001",
  "predicted_difficulty": 0.73,
  "confidence": 0.89,
  "difficulty_factors": {
    "complexity": 0.65,
    "evidence_ambiguity": 0.78,
    "reasoning_requirements": 0.71
  },
  "recommendations": {
    "parameter_adjustments": {
      "evidence_weight": "increase to 1.4 for better performance"
    },
    "expected_performance": 0.84
  },
  "quantum_prediction_confidence": 0.91
}
```

### Results and Analytics

#### Get Session Results
```http
GET /sessions/{session_id}/results
Authorization: Bearer <token>
```

**Query Parameters:**
- `format` (optional): Response format (json, csv, detailed)
- `include_quantum` (optional): Include quantum analysis (default: true)

**Response:**
```json
{
  "session_id": "sess_123456789",
  "session_summary": {
    "session_name": "my_evaluation_session",
    "total_evaluations": 50,
    "completed_evaluations": 50,
    "success_rate": 0.88,
    "average_scores": {
      "skepticism_calibration": 0.85,
      "evidence_standard_score": 0.82,
      "red_flag_detection": 0.87,
      "reasoning_quality": 0.89,
      "overall_score": 0.86
    },
    "execution_time_total_ms": 67340,
    "average_response_time_ms": 1347
  },
  "detailed_results": [
    {
      "evaluation_id": "eval_001",
      "scenario_id": "scenario_001",
      "overall_score": 0.91,
      "metrics": {
        "skepticism_calibration": 0.89,
        "evidence_standard_score": 0.92,
        "red_flag_detection": 0.87,
        "reasoning_quality": 0.94
      },
      "quantum_analysis": {
        "coherence_measurement": 0.93,
        "entanglement_contribution": 0.78
      },
      "timestamp": "2024-01-15T10:35:00Z"
    }
  ],
  "quantum_insights": {
    "overall_coherence": 0.91,
    "optimization_effectiveness": 0.87,
    "parameter_stability": 0.94
  },
  "generated_at": "2024-01-15T12:00:00Z"
}
```

#### Export Results
```http
GET /sessions/{session_id}/export
Authorization: Bearer <token>
```

**Query Parameters:**
- `format` (required): Export format (json, csv, xlsx, pdf)
- `include_details` (optional): Include detailed analysis (default: true)
- `include_quantum` (optional): Include quantum metrics (default: true)

**Response:**
```http
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="session_results_sess_123456789.csv"

<binary_data>
```

### Analytics and Reporting

#### Get Performance Analytics
```http
GET /analytics/performance
Authorization: Bearer <token>
```

**Query Parameters:**
- `time_range` (optional): Time range (24h, 7d, 30d, custom)
- `start_date` (optional): Start date for custom range
- `end_date` (optional): End date for custom range

**Response:**
```json
{
  "time_range": "7d",
  "performance_metrics": {
    "total_evaluations": 1250,
    "average_score": 0.84,
    "success_rate": 0.89,
    "average_response_time_ms": 1428,
    "quantum_optimization_improvement": 0.12
  },
  "trend_analysis": {
    "score_trend": "improving",
    "response_time_trend": "stable",
    "quantum_coherence_trend": "improving"
  },
  "top_performing_scenarios": [
    {
      "scenario_id": "scenario_015",
      "category": "statistical_claims",
      "average_score": 0.94
    }
  ],
  "improvement_opportunities": [
    {
      "area": "evidence_standard_score",
      "current_average": 0.79,
      "improvement_potential": 0.15
    }
  ]
}
```

#### Get Quantum Analytics
```http
GET /analytics/quantum
Authorization: Bearer <token>
```

**Response:**
```json
{
  "quantum_performance": {
    "average_coherence": 0.87,
    "optimization_success_rate": 0.92,
    "parameter_stability": 0.89,
    "convergence_efficiency": 0.84
  },
  "optimization_statistics": {
    "total_optimizations": 45,
    "average_generations_to_convergence": 38,
    "best_fitness_achieved": 0.96,
    "parameter_entanglement_strength": 0.73
  },
  "insights": [
    "Quantum optimization shows 12% improvement over classical methods",
    "Parameter entanglement indicates good optimization stability",
    "Coherence levels suggest reliable quantum-enhanced predictions"
  ]
}
```

## WebSocket API

### Real-time Evaluation Updates

Connect to WebSocket for real-time updates:

```javascript
const ws = new WebSocket('wss://api.agent-skeptic-bench.com/ws');

// Authentication
ws.send(JSON.stringify({
  type: 'auth',
  token: 'your_jwt_token'
}));

// Subscribe to session updates
ws.send(JSON.stringify({
  type: 'subscribe',
  session_id: 'sess_123456789'
}));

// Handle messages
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

**Message Types:**
- `evaluation_started`: Evaluation begins
- `evaluation_completed`: Evaluation finished
- `optimization_progress`: Parameter optimization update
- `quantum_insight`: New quantum analysis available
- `session_complete`: All evaluations finished

## Error Handling

### Standard Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid agent configuration",
    "details": {
      "field": "temperature",
      "issue": "Value must be between 0.1 and 1.0"
    },
    "request_id": "req_123456789",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200  | OK | Request successful |
| 201  | Created | Resource created successfully |
| 400  | Bad Request | Invalid request format or parameters |
| 401  | Unauthorized | Authentication required or invalid |
| 403  | Forbidden | Insufficient permissions |
| 404  | Not Found | Resource not found |
| 422  | Unprocessable Entity | Validation errors |
| 429  | Too Many Requests | Rate limit exceeded |
| 500  | Internal Server Error | Server error |
| 503  | Service Unavailable | Service temporarily unavailable |

### Common Error Codes

| Error Code | Description |
|------------|-------------|
| `AUTHENTICATION_REQUIRED` | JWT token missing or invalid |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions |
| `SESSION_NOT_FOUND` | Session ID does not exist |
| `SCENARIO_NOT_FOUND` | Scenario ID does not exist |
| `VALIDATION_ERROR` | Request validation failed |
| `QUANTUM_OPTIMIZATION_FAILED` | Quantum optimization encountered error |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `SERVICE_UNAVAILABLE` | Dependent service unavailable |

## Rate Limiting

The API implements rate limiting to ensure fair usage:

**Limits:**
- **Standard users**: 1000 requests per hour
- **Premium users**: 5000 requests per hour
- **Enterprise users**: Custom limits

**Headers:**
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
```

## SDK and Client Libraries

### Python SDK

```python
from agent_skeptic_bench import SkepticBenchClient

client = SkepticBenchClient(
    base_url="https://api.agent-skeptic-bench.com",
    api_key="your_api_key"
)

# Create session
session = client.create_session(
    name="my_evaluation",
    agent_config={
        "provider": "openai",
        "model_name": "gpt-4",
        "temperature": 0.7
    }
)

# Run evaluation
result = client.evaluate(
    session_id=session.id,
    scenario_id="scenario_001"
)

print(f"Score: {result.overall_score}")
```

### JavaScript SDK

```javascript
import { SkepticBenchClient } from '@agent-skeptic-bench/js-sdk';

const client = new SkepticBenchClient({
  baseUrl: 'https://api.agent-skeptic-bench.com',
  apiKey: 'your_api_key'
});

// Create session
const session = await client.createSession({
  name: 'my_evaluation',
  agentConfig: {
    provider: 'openai',
    modelName: 'gpt-4',
    temperature: 0.7
  }
});

// Run evaluation
const result = await client.evaluate({
  sessionId: session.id,
  scenarioId: 'scenario_001'
});

console.log(`Score: ${result.overallScore}`);
```

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **Endpoint**: `/openapi.json`
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

---

For integration examples and advanced usage patterns, see the `examples/` directory in the repository.
For authentication setup and API key management, refer to the Authentication section in the main documentation.