"""Data export functionality for Agent Skeptic Bench."""

import csv
import json
import logging
import statistics
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from ..models import BenchmarkSession, EvaluationResult, Scenario

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for data export."""

    format: str
    include_metadata: bool = True
    include_raw_responses: bool = False
    date_format: str = "%Y-%m-%d %H:%M:%S"
    flatten_nested: bool = True
    filter_fields: list[str] | None = None


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool
    file_path: Path | None = None
    records_exported: int = 0
    error_message: str | None = None
    export_time: datetime = None


class DataExporter(ABC):
    """Abstract base class for data exporters."""

    def __init__(self):
        """Initialize data exporter."""
        pass

    @abstractmethod
    async def export_evaluations(self, results: list[EvaluationResult],
                               output_path: Path, config: ExportConfig) -> ExportResult:
        """Export evaluation results."""
        pass

    @abstractmethod
    async def export_sessions(self, sessions: list[BenchmarkSession],
                            output_path: Path, config: ExportConfig) -> ExportResult:
        """Export benchmark sessions."""
        pass

    @abstractmethod
    async def export_scenarios(self, scenarios: list[Scenario],
                             output_path: Path, config: ExportConfig) -> ExportResult:
        """Export scenarios."""
        pass

    def _prepare_evaluation_data(self, results: list[EvaluationResult],
                               config: ExportConfig) -> list[dict[str, Any]]:
        """Prepare evaluation data for export."""
        exported_data = []

        for result in results:
            # Convert to dictionary
            data = {
                "id": result.id,
                "scenario_id": result.scenario_id,
                "agent_provider": result.agent_provider,
                "model": result.model,
                "evaluated_at": result.evaluated_at.strftime(config.date_format),
                "passed_evaluation": result.passed_evaluation,
                "evaluation_duration": getattr(result, 'evaluation_duration', None)
            }

            # Add metrics
            if config.flatten_nested:
                data.update({
                    "overall_score": result.metrics.overall_score,
                    "skepticism_calibration": result.metrics.skepticism_calibration,
                    "evidence_standard_score": result.metrics.evidence_standard_score,
                    "red_flag_detection": result.metrics.red_flag_detection,
                    "reasoning_quality": result.metrics.reasoning_quality
                })
            else:
                data["metrics"] = asdict(result.metrics)

            # Add response data
            if config.include_raw_responses:
                if config.flatten_nested:
                    data.update({
                        "response_decision": result.response.decision,
                        "response_confidence": result.response.confidence_level,
                        "response_reasoning": result.response.reasoning,
                        "evidence_requests_count": len(result.response.evidence_requests),
                        "red_flags_identified_count": len(result.response.red_flags_identified)
                    })

                    # Add lists as JSON strings for flat formats
                    data.update({
                        "evidence_requests": json.dumps(result.response.evidence_requests),
                        "red_flags_identified": json.dumps(result.response.red_flags_identified)
                    })
                else:
                    data["response"] = asdict(result.response)

            # Add scenario data if available
            if hasattr(result, 'scenario') and result.scenario:
                if config.flatten_nested:
                    data.update({
                        "scenario_title": result.scenario.title,
                        "scenario_category": str(result.scenario.category),
                        "scenario_difficulty": result.scenario.metadata.get('difficulty', 'unknown'),
                        "correct_skepticism_level": result.scenario.correct_skepticism_level
                    })
                else:
                    data["scenario"] = asdict(result.scenario)

            # Add metadata
            if config.include_metadata:
                data["export_timestamp"] = datetime.utcnow().strftime(config.date_format)

            # Filter fields if specified
            if config.filter_fields:
                data = {k: v for k, v in data.items() if k in config.filter_fields}

            exported_data.append(data)

        return exported_data

    def _prepare_session_data(self, sessions: list[BenchmarkSession],
                            config: ExportConfig) -> list[dict[str, Any]]:
        """Prepare session data for export."""
        exported_data = []

        for session in sessions:
            data = {
                "id": session.id,
                "agent_provider": session.agent_provider,
                "model": session.model,
                "created_at": session.created_at.strftime(config.date_format),
                "completed_at": session.completed_at.strftime(config.date_format) if session.completed_at else None,
                "status": session.status,
                "total_scenarios": len(session.scenario_ids),
                "completed_evaluations": getattr(session, 'completed_evaluations', 0),
                "overall_score": getattr(session, 'overall_score', None),
                "pass_rate": getattr(session, 'pass_rate', None)
            }

            if config.include_metadata:
                data.update({
                    "config": json.dumps(session.config) if hasattr(session, 'config') else None,
                    "metadata": json.dumps(session.metadata) if hasattr(session, 'metadata') else None,
                    "export_timestamp": datetime.utcnow().strftime(config.date_format)
                })

            if config.filter_fields:
                data = {k: v for k, v in data.items() if k in config.filter_fields}

            exported_data.append(data)

        return exported_data


class CSVExporter(DataExporter):
    """Exports data to CSV format."""

    async def export_evaluations(self, results: list[EvaluationResult],
                               output_path: Path, config: ExportConfig) -> ExportResult:
        """Export evaluation results to CSV."""
        try:
            data = self._prepare_evaluation_data(results, config)

            if not data:
                return ExportResult(
                    success=False,
                    error_message="No data to export",
                    export_time=datetime.utcnow()
                )

            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

            logger.info(f"Exported {len(data)} evaluation records to {output_path}")

            return ExportResult(
                success=True,
                file_path=output_path,
                records_exported=len(data),
                export_time=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Failed to export evaluations to CSV: {e}")
            return ExportResult(
                success=False,
                error_message=str(e),
                export_time=datetime.utcnow()
            )

    async def export_sessions(self, sessions: list[BenchmarkSession],
                            output_path: Path, config: ExportConfig) -> ExportResult:
        """Export benchmark sessions to CSV."""
        try:
            data = self._prepare_session_data(sessions, config)

            if not data:
                return ExportResult(
                    success=False,
                    error_message="No data to export",
                    export_time=datetime.utcnow()
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

            logger.info(f"Exported {len(data)} session records to {output_path}")

            return ExportResult(
                success=True,
                file_path=output_path,
                records_exported=len(data),
                export_time=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Failed to export sessions to CSV: {e}")
            return ExportResult(
                success=False,
                error_message=str(e),
                export_time=datetime.utcnow()
            )

    async def export_scenarios(self, scenarios: list[Scenario],
                             output_path: Path, config: ExportConfig) -> ExportResult:
        """Export scenarios to CSV."""
        try:
            data = []
            for scenario in scenarios:
                scenario_data = {
                    "id": scenario.id,
                    "title": scenario.title,
                    "description": scenario.description,
                    "category": str(scenario.category),
                    "correct_skepticism_level": scenario.correct_skepticism_level,
                    "red_flags": json.dumps(scenario.red_flags),
                    "metadata": json.dumps(scenario.metadata)
                }

                if config.filter_fields:
                    scenario_data = {k: v for k, v in scenario_data.items() if k in config.filter_fields}

                data.append(scenario_data)

            if not data:
                return ExportResult(
                    success=False,
                    error_message="No data to export",
                    export_time=datetime.utcnow()
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

            logger.info(f"Exported {len(data)} scenario records to {output_path}")

            return ExportResult(
                success=True,
                file_path=output_path,
                records_exported=len(data),
                export_time=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Failed to export scenarios to CSV: {e}")
            return ExportResult(
                success=False,
                error_message=str(e),
                export_time=datetime.utcnow()
            )


class JSONExporter(DataExporter):
    """Exports data to JSON format."""

    async def export_evaluations(self, results: list[EvaluationResult],
                               output_path: Path, config: ExportConfig) -> ExportResult:
        """Export evaluation results to JSON."""
        try:
            # For JSON, we can preserve nested structure
            config_copy = ExportConfig(
                format="json",
                include_metadata=config.include_metadata,
                include_raw_responses=config.include_raw_responses,
                date_format=config.date_format,
                flatten_nested=False,  # Keep nested for JSON
                filter_fields=config.filter_fields
            )

            data = self._prepare_evaluation_data(results, config_copy)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            export_data = {
                "export_info": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "record_count": len(data),
                    "format": "json",
                    "version": "1.0"
                },
                "evaluations": data
            }

            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(data)} evaluation records to {output_path}")

            return ExportResult(
                success=True,
                file_path=output_path,
                records_exported=len(data),
                export_time=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Failed to export evaluations to JSON: {e}")
            return ExportResult(
                success=False,
                error_message=str(e),
                export_time=datetime.utcnow()
            )

    async def export_sessions(self, sessions: list[BenchmarkSession],
                            output_path: Path, config: ExportConfig) -> ExportResult:
        """Export benchmark sessions to JSON."""
        try:
            config_copy = ExportConfig(
                format="json",
                include_metadata=config.include_metadata,
                date_format=config.date_format,
                flatten_nested=False,
                filter_fields=config.filter_fields
            )

            data = self._prepare_session_data(sessions, config_copy)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            export_data = {
                "export_info": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "record_count": len(data),
                    "format": "json",
                    "version": "1.0"
                },
                "sessions": data
            }

            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(data)} session records to {output_path}")

            return ExportResult(
                success=True,
                file_path=output_path,
                records_exported=len(data),
                export_time=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Failed to export sessions to JSON: {e}")
            return ExportResult(
                success=False,
                error_message=str(e),
                export_time=datetime.utcnow()
            )

    async def export_scenarios(self, scenarios: list[Scenario],
                             output_path: Path, config: ExportConfig) -> ExportResult:
        """Export scenarios to JSON."""
        try:
            data = [asdict(scenario) for scenario in scenarios]

            # Apply field filtering
            if config.filter_fields:
                data = [
                    {k: v for k, v in scenario_data.items() if k in config.filter_fields}
                    for scenario_data in data
                ]

            output_path.parent.mkdir(parents=True, exist_ok=True)

            export_data = {
                "export_info": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "record_count": len(data),
                    "format": "json",
                    "version": "1.0"
                },
                "scenarios": data
            }

            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(data)} scenario records to {output_path}")

            return ExportResult(
                success=True,
                file_path=output_path,
                records_exported=len(data),
                export_time=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Failed to export scenarios to JSON: {e}")
            return ExportResult(
                success=False,
                error_message=str(e),
                export_time=datetime.utcnow()
            )


class ExcelExporter(DataExporter):
    """Exports data to Excel format."""

    async def export_evaluations(self, results: list[EvaluationResult],
                               output_path: Path, config: ExportConfig) -> ExportResult:
        """Export evaluation results to Excel."""
        try:
            # Try to import pandas and openpyxl
            try:
                import pandas as pd
            except ImportError:
                logger.warning("pandas not available, falling back to CSV format")
                csv_exporter = CSVExporter()
                csv_path = output_path.with_suffix('.csv')
                return await csv_exporter.export_evaluations(results, csv_path, config)

            data = self._prepare_evaluation_data(results, config)

            if not data:
                return ExportResult(
                    success=False,
                    error_message="No data to export",
                    export_time=datetime.utcnow()
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create DataFrame and export to Excel
            df = pd.DataFrame(data)

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Evaluations', index=False)

                # Add a summary sheet
                summary_data = {
                    'Metric': ['Total Evaluations', 'Passed', 'Failed', 'Pass Rate', 'Average Score'],
                    'Value': [
                        len(results),
                        sum(1 for r in results if r.passed_evaluation),
                        sum(1 for r in results if not r.passed_evaluation),
                        f"{sum(1 for r in results if r.passed_evaluation) / len(results):.1%}" if results else "0%",
                        f"{sum(r.metrics.overall_score for r in results) / len(results):.3f}" if results else "0"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            logger.info(f"Exported {len(data)} evaluation records to {output_path}")

            return ExportResult(
                success=True,
                file_path=output_path,
                records_exported=len(data),
                export_time=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Failed to export evaluations to Excel: {e}")
            return ExportResult(
                success=False,
                error_message=str(e),
                export_time=datetime.utcnow()
            )

    async def export_sessions(self, sessions: list[BenchmarkSession],
                            output_path: Path, config: ExportConfig) -> ExportResult:
        """Export benchmark sessions to Excel."""
        try:
            try:
                import pandas as pd
            except ImportError:
                logger.warning("pandas not available, falling back to CSV format")
                csv_exporter = CSVExporter()
                csv_path = output_path.with_suffix('.csv')
                return await csv_exporter.export_sessions(sessions, csv_path, config)

            data = self._prepare_session_data(sessions, config)

            if not data:
                return ExportResult(
                    success=False,
                    error_message="No data to export",
                    export_time=datetime.utcnow()
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(data)
            df.to_excel(output_path, sheet_name='Sessions', index=False)

            logger.info(f"Exported {len(data)} session records to {output_path}")

            return ExportResult(
                success=True,
                file_path=output_path,
                records_exported=len(data),
                export_time=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Failed to export sessions to Excel: {e}")
            return ExportResult(
                success=False,
                error_message=str(e),
                export_time=datetime.utcnow()
            )

    async def export_scenarios(self, scenarios: list[Scenario],
                             output_path: Path, config: ExportConfig) -> ExportResult:
        """Export scenarios to Excel."""
        try:
            try:
                import pandas as pd
            except ImportError:
                logger.warning("pandas not available, falling back to CSV format")
                csv_exporter = CSVExporter()
                csv_path = output_path.with_suffix('.csv')
                return await csv_exporter.export_scenarios(scenarios, csv_path, config)

            data = []
            for scenario in scenarios:
                scenario_data = {
                    "id": scenario.id,
                    "title": scenario.title,
                    "description": scenario.description,
                    "category": str(scenario.category),
                    "correct_skepticism_level": scenario.correct_skepticism_level,
                    "red_flags_count": len(scenario.red_flags),
                    "red_flags": "; ".join(scenario.red_flags),
                    "difficulty": scenario.metadata.get('difficulty', 'unknown')
                }

                if config.filter_fields:
                    scenario_data = {k: v for k, v in scenario_data.items() if k in config.filter_fields}

                data.append(scenario_data)

            if not data:
                return ExportResult(
                    success=False,
                    error_message="No data to export",
                    export_time=datetime.utcnow()
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(data)
            df.to_excel(output_path, sheet_name='Scenarios', index=False)

            logger.info(f"Exported {len(data)} scenario records to {output_path}")

            return ExportResult(
                success=True,
                file_path=output_path,
                records_exported=len(data),
                export_time=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Failed to export scenarios to Excel: {e}")
            return ExportResult(
                success=False,
                error_message=str(e),
                export_time=datetime.utcnow()
            )


# Factory function for creating exporters
def create_exporter(format_type: str = "csv") -> DataExporter:
    """Create a data exporter of the specified type."""
    format_type = format_type.lower()

    if format_type == "csv":
        return CSVExporter()
    elif format_type == "json":
        return JSONExporter()
    elif format_type in ["excel", "xlsx"]:
        return ExcelExporter()
    else:
        raise ValueError(f"Unsupported export format: {format_type}")


# Convenience function for batch exports
async def export_evaluation_data(results: list[EvaluationResult],
                               output_dir: Path,
                               formats: list[str] = ["csv", "json"],
                               config: ExportConfig | None = None) -> dict[str, ExportResult]:
    """Export evaluation data in multiple formats."""
    if config is None:
        config = ExportConfig(format="multi")

    export_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for format_type in formats:
        try:
            exporter = create_exporter(format_type)
            output_path = output_dir / f"evaluations_{timestamp}.{format_type}"

            if format_type == "excel":
                output_path = output_path.with_suffix('.xlsx')

            result = await exporter.export_evaluations(results, output_path, config)
            export_results[format_type] = result

        except Exception as e:
            logger.error(f"Failed to export in {format_type} format: {e}")
            export_results[format_type] = ExportResult(
                success=False,
                error_message=str(e),
                export_time=datetime.utcnow()
            )

    return export_results


@dataclass
class UsageMetricsData:
    """Usage metrics data structure for export."""
    
    timestamp: datetime
    session_id: str
    user_id: str | None = None
    agent_provider: str | None = None
    model: str | None = None
    evaluation_count: int = 0
    total_duration: float = 0.0
    api_calls: int = 0
    tokens_used: int = 0
    scenarios_completed: List[str] = None
    categories_used: List[str] = None
    performance_scores: Dict[str, float] = None
    feature_usage: Dict[str, int] = None


class UsageMetricsExporter:
    """Specialized exporter for usage metrics and analytics."""
    
    def __init__(self):
        """Initialize usage metrics exporter."""
        pass
    
    async def export_usage_summary(self, metrics_data: List[UsageMetricsData],
                                  output_path: Path, 
                                  format_type: str = "json") -> ExportResult:
        """Export comprehensive usage summary."""
        try:
            if not metrics_data:
                return ExportResult(
                    success=False,
                    error_message="No usage metrics data to export",
                    export_time=datetime.utcnow()
                )
            
            # Calculate comprehensive analytics
            summary = self._calculate_usage_analytics(metrics_data)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format_type.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
            
            elif format_type.lower() == "csv":
                # Flatten summary for CSV export
                flattened_data = []
                
                # Overall statistics
                flattened_data.append({
                    "metric_type": "summary",
                    "metric_name": "total_sessions",
                    "value": summary["overall_stats"]["total_sessions"],
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                flattened_data.append({
                    "metric_type": "summary", 
                    "metric_name": "total_evaluations",
                    "value": summary["overall_stats"]["total_evaluations"],
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Provider statistics
                for provider, stats in summary["provider_stats"].items():
                    flattened_data.append({
                        "metric_type": "provider",
                        "metric_name": f"sessions_{provider}",
                        "value": stats["sessions"],
                        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                # Category statistics  
                for category, count in summary["category_usage"].items():
                    flattened_data.append({
                        "metric_type": "category",
                        "metric_name": f"usage_{category}",
                        "value": count,
                        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ["metric_type", "metric_name", "value", "timestamp"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(flattened_data)
            
            logger.info(f"Exported usage summary to {output_path}")
            
            return ExportResult(
                success=True,
                file_path=output_path,
                records_exported=len(metrics_data),
                export_time=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to export usage summary: {e}")
            return ExportResult(
                success=False,
                error_message=str(e),
                export_time=datetime.utcnow()
            )
    
    async def export_detailed_usage(self, metrics_data: List[UsageMetricsData],
                                   output_path: Path,
                                   format_type: str = "json") -> ExportResult:
        """Export detailed usage metrics data."""
        try:
            if not metrics_data:
                return ExportResult(
                    success=False,
                    error_message="No usage metrics data to export",
                    export_time=datetime.utcnow()
                )
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format_type.lower() == "json":
                export_data = {
                    "export_info": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "record_count": len(metrics_data),
                        "format": "json",
                        "version": "1.0"
                    },
                    "usage_metrics": [asdict(metrics) for metrics in metrics_data]
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
            
            elif format_type.lower() == "csv":
                # Flatten metrics for CSV
                flattened_data = []
                
                for metrics in metrics_data:
                    row = {
                        "timestamp": metrics.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "session_id": metrics.session_id,
                        "user_id": metrics.user_id or "",
                        "agent_provider": metrics.agent_provider or "",
                        "model": metrics.model or "",
                        "evaluation_count": metrics.evaluation_count,
                        "total_duration": metrics.total_duration,
                        "api_calls": metrics.api_calls,
                        "tokens_used": metrics.tokens_used,
                        "scenarios_completed": json.dumps(metrics.scenarios_completed or []),
                        "categories_used": json.dumps(metrics.categories_used or []),
                        "avg_score": metrics.performance_scores.get("overall_score_avg", 0),
                        "max_score": metrics.performance_scores.get("overall_score_max", 0),
                        "feature_usage": json.dumps(metrics.feature_usage or {})
                    }
                    flattened_data.append(row)
                
                with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                    if flattened_data:
                        fieldnames = flattened_data[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(flattened_data)
            
            logger.info(f"Exported {len(metrics_data)} detailed usage records to {output_path}")
            
            return ExportResult(
                success=True,
                file_path=output_path,
                records_exported=len(metrics_data),
                export_time=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to export detailed usage metrics: {e}")
            return ExportResult(
                success=False,
                error_message=str(e),
                export_time=datetime.utcnow()
            )
    
    def _calculate_usage_analytics(self, metrics_data: List[UsageMetricsData]) -> Dict[str, Any]:
        """Calculate comprehensive usage analytics."""
        if not metrics_data:
            return {"error": "No data available"}
        
        # Overall statistics
        total_sessions = len(metrics_data)
        total_evaluations = sum(m.evaluation_count for m in metrics_data)
        total_duration = sum(m.total_duration for m in metrics_data)
        total_tokens = sum(m.tokens_used for m in metrics_data)
        unique_users = len(set(m.user_id for m in metrics_data if m.user_id))
        
        # Time range analysis
        timestamps = [m.timestamp for m in metrics_data]
        time_range = {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat(),
            "span_days": (max(timestamps) - min(timestamps)).days
        }
        
        # Provider analysis
        provider_stats = {}
        for metrics in metrics_data:
            if metrics.agent_provider:
                if metrics.agent_provider not in provider_stats:
                    provider_stats[metrics.agent_provider] = {
                        "sessions": 0,
                        "evaluations": 0,
                        "duration": 0,
                        "tokens": 0,
                        "models": set()
                    }
                
                stats = provider_stats[metrics.agent_provider]
                stats["sessions"] += 1
                stats["evaluations"] += metrics.evaluation_count
                stats["duration"] += metrics.total_duration
                stats["tokens"] += metrics.tokens_used
                if metrics.model:
                    stats["models"].add(metrics.model)
        
        # Convert sets to lists for JSON serialization
        for stats in provider_stats.values():
            stats["models"] = list(stats["models"])
        
        # Category usage analysis
        category_usage = {}
        for metrics in metrics_data:
            for category in metrics.categories_used or []:
                category_usage[category] = category_usage.get(category, 0) + 1
        
        # Feature usage analysis
        feature_usage = {}
        for metrics in metrics_data:
            for feature, count in (metrics.feature_usage or {}).items():
                feature_usage[feature] = feature_usage.get(feature, 0) + count
        
        # Performance analysis
        all_scores = []
        for metrics in metrics_data:
            if "overall_score_avg" in (metrics.performance_scores or {}):
                all_scores.append(metrics.performance_scores["overall_score_avg"])
        
        performance_stats = {}
        if all_scores:
            performance_stats = {
                "mean_score": statistics.mean(all_scores),
                "median_score": statistics.median(all_scores),
                "std_score": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
                "min_score": min(all_scores),
                "max_score": max(all_scores)
            }
        
        # Daily usage patterns
        daily_usage = {}
        for metrics in metrics_data:
            day = metrics.timestamp.strftime("%Y-%m-%d")
            if day not in daily_usage:
                daily_usage[day] = {"sessions": 0, "evaluations": 0, "duration": 0}
            
            daily_usage[day]["sessions"] += 1
            daily_usage[day]["evaluations"] += metrics.evaluation_count
            daily_usage[day]["duration"] += metrics.total_duration
        
        return {
            "overall_stats": {
                "total_sessions": total_sessions,
                "total_evaluations": total_evaluations,
                "total_duration": total_duration,
                "total_tokens": total_tokens,
                "unique_users": unique_users,
                "avg_evaluations_per_session": total_evaluations / total_sessions if total_sessions > 0 else 0,
                "avg_session_duration": total_duration / total_sessions if total_sessions > 0 else 0,
                "avg_tokens_per_session": total_tokens / total_sessions if total_sessions > 0 else 0
            },
            "time_range": time_range,
            "provider_stats": provider_stats,
            "category_usage": dict(sorted(category_usage.items(), key=lambda x: x[1], reverse=True)),
            "feature_usage": dict(sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)),
            "performance_stats": performance_stats,
            "daily_usage": dict(sorted(daily_usage.items())),
            "export_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "data_points": total_sessions,
                "analysis_version": "1.0"
            }
        }


class AdvancedAnalyticsExporter:
    """Advanced analytics and reporting exporter."""
    
    def __init__(self):
        """Initialize advanced analytics exporter."""
        pass
    
    async def export_performance_trends(self, results: List[EvaluationResult],
                                      output_path: Path,
                                      time_window_days: int = 30) -> ExportResult:
        """Export performance trend analysis."""
        try:
            # Filter to time window
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            filtered_results = [r for r in results if r.evaluated_at >= cutoff_date]
            
            if not filtered_results:
                return ExportResult(
                    success=False,
                    error_message="No results in specified time window",
                    export_time=datetime.utcnow()
                )
            
            # Calculate trend data
            trend_data = self._calculate_trend_analysis(filtered_results)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(trend_data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Exported performance trends to {output_path}")
            
            return ExportResult(
                success=True,
                file_path=output_path,
                records_exported=len(filtered_results),
                export_time=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to export performance trends: {e}")
            return ExportResult(
                success=False,
                error_message=str(e),
                export_time=datetime.utcnow()
            )
    
    def _calculate_trend_analysis(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate detailed trend analysis."""
        if not results:
            return {"error": "No data for trend analysis"}
        
        # Sort by timestamp
        sorted_results = sorted(results, key=lambda r: r.evaluated_at)
        
        # Calculate time-based metrics
        daily_metrics = {}
        for result in sorted_results:
            day = result.evaluated_at.strftime("%Y-%m-%d")
            if day not in daily_metrics:
                daily_metrics[day] = {
                    "evaluations": 0,
                    "scores": [],
                    "providers": set(),
                    "categories": set()
                }
            
            daily_metrics[day]["evaluations"] += 1
            daily_metrics[day]["scores"].append(result.metrics.overall_score)
            daily_metrics[day]["providers"].add(result.agent_provider)
            
            if hasattr(result, 'scenario') and result.scenario:
                daily_metrics[day]["categories"].add(str(result.scenario.category))
        
        # Calculate daily averages
        for day_data in daily_metrics.values():
            if day_data["scores"]:
                day_data["avg_score"] = statistics.mean(day_data["scores"])
                day_data["score_std"] = statistics.stdev(day_data["scores"]) if len(day_data["scores"]) > 1 else 0
            day_data["providers"] = list(day_data["providers"])
            day_data["categories"] = list(day_data["categories"])
        
        # Calculate trends
        overall_scores = [r.metrics.overall_score for r in sorted_results]
        skepticism_scores = [r.metrics.skepticism_calibration for r in sorted_results]
        
        trend_analysis = {
            "time_period": {
                "start": sorted_results[0].evaluated_at.isoformat(),
                "end": sorted_results[-1].evaluated_at.isoformat(),
                "total_days": len(daily_metrics)
            },
            "overall_trends": {
                "score_trend": self._calculate_linear_trend(overall_scores),
                "skepticism_trend": self._calculate_linear_trend(skepticism_scores),
                "evaluation_volume_trend": self._calculate_volume_trend(daily_metrics)
            },
            "daily_metrics": daily_metrics,
            "performance_summary": {
                "total_evaluations": len(results),
                "avg_score": statistics.mean(overall_scores),
                "score_improvement": self._calculate_improvement(overall_scores),
                "consistency_score": 1.0 - (statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0)
            }
        }
        
        return trend_analysis
    
    def _calculate_linear_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate linear trend for a series of values."""
        if len(values) < 2:
            return {"slope": 0, "correlation": 0, "direction": "flat"}
        
        x = list(range(len(values)))
        n = len(values)
        
        # Calculate linear regression slope
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Calculate correlation coefficient
        mean_x = sum_x / n
        mean_y = sum_y / n
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, values))
        denominator_x = sum((xi - mean_x) ** 2 for xi in x)
        denominator_y = sum((yi - mean_y) ** 2 for yi in values)
        
        correlation = numerator / (denominator_x * denominator_y) ** 0.5 if denominator_x * denominator_y > 0 else 0
        
        direction = "increasing" if slope > 0.001 else "decreasing" if slope < -0.001 else "flat"
        
        return {
            "slope": slope,
            "correlation": correlation,
            "direction": direction,
            "start_value": values[0],
            "end_value": values[-1],
            "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
        }
    
    def _calculate_volume_trend(self, daily_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate evaluation volume trends."""
        if not daily_metrics:
            return {"trend": "no_data"}
        
        daily_counts = [day_data["evaluations"] for day_data in daily_metrics.values()]
        
        if len(daily_counts) < 2:
            return {"trend": "insufficient_data", "avg_daily": daily_counts[0] if daily_counts else 0}
        
        trend = self._calculate_linear_trend(daily_counts)
        
        return {
            "trend": trend["direction"],
            "slope": trend["slope"],
            "avg_daily": statistics.mean(daily_counts),
            "peak_day": max(daily_counts),
            "low_day": min(daily_counts),
            "consistency": 1.0 - (statistics.stdev(daily_counts) / statistics.mean(daily_counts) if statistics.mean(daily_counts) > 0 else 0)
        }
    
    def _calculate_improvement(self, scores: List[float]) -> float:
        """Calculate overall improvement in performance."""
        if len(scores) < 2:
            return 0.0
        
        # Compare first and last quartile
        first_quarter = scores[:len(scores)//4] if len(scores) >= 4 else scores[:1]
        last_quarter = scores[-len(scores)//4:] if len(scores) >= 4 else scores[-1:]
        
        if first_quarter and last_quarter:
            first_avg = statistics.mean(first_quarter)
            last_avg = statistics.mean(last_quarter)
            
            return (last_avg - first_avg) / first_avg if first_avg != 0 else 0
        
        return 0.0
