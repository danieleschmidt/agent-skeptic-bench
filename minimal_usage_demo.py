#!/usr/bin/env python3
"""
Minimal demo of usage metrics tracking without external dependencies.
"""

import json
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


@dataclass
class SimpleUsageMetrics:
    """Simple usage metrics without external dependencies."""
    
    timestamp: str
    session_id: str
    user_id: Optional[str] = None
    agent_provider: Optional[str] = None
    model: Optional[str] = None
    evaluation_count: int = 0
    total_duration: float = 0.0
    tokens_used: int = 0
    scenarios_completed: List[str] = None
    categories_used: List[str] = None
    performance_scores: Dict[str, List[float]] = None
    feature_usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.scenarios_completed is None:
            self.scenarios_completed = []
        if self.categories_used is None:
            self.categories_used = []
        if self.performance_scores is None:
            self.performance_scores = {}
        if self.feature_usage is None:
            self.feature_usage = {}


class SimpleUsageTracker:
    """Simple usage tracker implementation."""
    
    def __init__(self, storage_path: str = "data/usage_metrics"):
        """Initialize simple usage tracker."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.active_sessions: Dict[str, SimpleUsageMetrics] = {}
        
        print(f"📊 UsageTracker initialized: {self.storage_path}")
    
    def start_session(self, session_id: str, user_id: Optional[str] = None, 
                     agent_provider: Optional[str] = None, model: Optional[str] = None) -> None:
        """Start tracking a session."""
        # Basic validation
        if not session_id or len(session_id) < 3:
            raise ValueError("Session ID must be at least 3 characters")
        
        self.active_sessions[session_id] = SimpleUsageMetrics(
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            user_id=user_id,
            agent_provider=agent_provider,
            model=model
        )
        
        print(f"▶️  Started session {session_id}")
    
    def record_evaluation(self, session_id: str, scenario_id: str, category: str,
                         duration: float, score: float, tokens_used: int = 0) -> None:
        """Record an evaluation."""
        if session_id not in self.active_sessions:
            print(f"⚠️  Session {session_id} not found, creating...")
            self.start_session(session_id)
        
        metrics = self.active_sessions[session_id]
        metrics.evaluation_count += 1
        metrics.total_duration += duration
        metrics.tokens_used += tokens_used
        metrics.scenarios_completed.append(scenario_id)
        
        if category not in metrics.categories_used:
            metrics.categories_used.append(category)
        
        if "overall_score" not in metrics.performance_scores:
            metrics.performance_scores["overall_score"] = []
        metrics.performance_scores["overall_score"].append(score)
        
        print(f"📝 Recorded evaluation: {scenario_id} ({category}) - Score: {score:.3f}")
    
    def record_feature_usage(self, session_id: str, feature: str) -> None:
        """Record feature usage."""
        if session_id not in self.active_sessions:
            return
        
        metrics = self.active_sessions[session_id]
        if feature not in metrics.feature_usage:
            metrics.feature_usage[feature] = 0
        metrics.feature_usage[feature] += 1
        
        print(f"🔧 Feature used: {feature}")
    
    def end_session(self, session_id: str) -> Optional[SimpleUsageMetrics]:
        """End a session and save metrics."""
        if session_id not in self.active_sessions:
            return None
        
        metrics = self.active_sessions[session_id]
        
        # Calculate aggregated scores
        performance_scores_copy = metrics.performance_scores.copy()
        for score_type, scores in performance_scores_copy.items():
            if scores and isinstance(scores, list):
                metrics.performance_scores[f"{score_type}_avg"] = statistics.mean(scores)
                metrics.performance_scores[f"{score_type}_max"] = max(scores)
                metrics.performance_scores[f"{score_type}_min"] = min(scores)
        
        # Save to file
        self._save_metrics(metrics)
        
        # Remove from active tracking
        del self.active_sessions[session_id]
        
        print(f"🏁 Ended session {session_id}")
        return metrics
    
    def _save_metrics(self, metrics: SimpleUsageMetrics) -> None:
        """Save metrics to file."""
        try:
            timestamp = datetime.fromisoformat(metrics.timestamp)
            date_str = timestamp.strftime("%Y-%m-%d")
            file_path = self.storage_path / f"usage_metrics_{date_str}.jsonl"
            
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(metrics), default=str) + "\n")
            
            print(f"💾 Saved metrics to {file_path}")
            
        except Exception as e:
            print(f"❌ Failed to save metrics: {e}")
    
    def get_usage_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get usage summary."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        all_metrics = []
        
        try:
            for file_path in self.storage_path.glob("usage_metrics_*.jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            timestamp = datetime.fromisoformat(data["timestamp"])
                            
                            if timestamp >= cutoff_date:
                                all_metrics.append(data)
                        except Exception as e:
                            print(f"⚠️  Skipped invalid line: {e}")
        
        except Exception as e:
            print(f"❌ Failed to load metrics: {e}")
            return {"error": str(e)}
        
        if not all_metrics:
            return {"error": "No data found"}
        
        # Calculate summary
        total_sessions = len(all_metrics)
        total_evaluations = sum(m.get("evaluation_count", 0) for m in all_metrics)
        total_duration = sum(m.get("total_duration", 0) for m in all_metrics)
        total_tokens = sum(m.get("tokens_used", 0) for m in all_metrics)
        
        return {
            "total_sessions": total_sessions,
            "total_evaluations": total_evaluations,
            "total_duration": total_duration,
            "total_tokens": total_tokens,
            "avg_evaluations_per_session": total_evaluations / total_sessions if total_sessions > 0 else 0,
            "avg_session_duration": total_duration / total_sessions if total_sessions > 0 else 0
        }


def demo_usage_tracking():
    """Run the usage tracking demo."""
    print("🚀 Starting Usage Metrics Demo")
    
    # Initialize tracker
    tracker = SimpleUsageTracker()
    
    # Simulate user sessions
    print("\n👥 Simulating user sessions...")
    
    sessions = []
    for i in range(3):
        session_id = f"demo_session_{i+1:03d}"
        user_id = f"user_{(i % 2) + 1}"
        provider = ["openai", "anthropic", "google"][i % 3]
        model = f"model-{i+1}"
        
        tracker.start_session(session_id, user_id, provider, model)
        sessions.append(session_id)
        
        # Record feature usage
        tracker.record_feature_usage(session_id, "dashboard_view")
        tracker.record_feature_usage(session_id, "scenario_search")
        
        if i == 0:
            tracker.record_feature_usage(session_id, "data_export")
    
    # Simulate evaluations
    print("\n🔍 Simulating evaluations...")
    
    categories = ["factual_claims", "flawed_plans", "persuasion_attacks", "evidence_evaluation"]
    
    for i, session_id in enumerate(sessions):
        num_evals = (i + 1) * 2  # 2, 4, 6 evaluations
        
        for eval_num in range(num_evals):
            scenario_id = f"scenario_{eval_num + 1:03d}"
            category = categories[eval_num % len(categories)]
            duration = 1.5 + (eval_num * 0.2)
            score = min(0.65 + (eval_num * 0.05), 1.0)
            tokens = 120 + (eval_num * 15)
            
            tracker.record_evaluation(session_id, scenario_id, category, duration, score, tokens)
            time.sleep(0.1)  # Small delay to simulate real usage
    
    # End sessions
    print("\n🏁 Ending sessions...")
    
    final_metrics = []
    for session_id in sessions:
        metrics = tracker.end_session(session_id)
        if metrics:
            final_metrics.append(metrics)
    
    # Generate summary
    print("\n📈 Usage Summary:")
    summary = tracker.get_usage_summary()
    
    if "error" not in summary:
        print(f"  📊 Total Sessions: {summary['total_sessions']}")
        print(f"  📊 Total Evaluations: {summary['total_evaluations']}")
        print(f"  📊 Total Duration: {summary['total_duration']:.1f}s")
        print(f"  📊 Total Tokens: {summary['total_tokens']}")
        print(f"  📊 Avg Evaluations/Session: {summary['avg_evaluations_per_session']:.1f}")
        print(f"  📊 Avg Session Duration: {summary['avg_session_duration']:.1f}s")
    else:
        print(f"  ❌ Error: {summary['error']}")
    
    # Test export functionality
    print("\n💾 Testing Export Functionality...")
    
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    
    # Export summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = export_dir / f"usage_summary_{timestamp}.json"
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"  ✅ Exported summary: {summary_file}")
    
    # Export detailed data
    detailed_file = export_dir / f"detailed_usage_{timestamp}.json"
    
    detailed_data = {
        "export_info": {
            "timestamp": datetime.utcnow().isoformat(),
            "record_count": len(final_metrics),
            "format": "json"
        },
        "sessions": [asdict(m) for m in final_metrics]
    }
    
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, indent=2, default=str)
    
    print(f"  ✅ Exported detailed data: {detailed_file}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📁 Check these directories:")
    print(f"  - {tracker.storage_path} (raw metrics)")
    print(f"  - {export_dir} (exported reports)")


if __name__ == "__main__":
    demo_usage_tracking()