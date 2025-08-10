"""Data export functionality for Agent Skeptic Bench."""

import csv
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
