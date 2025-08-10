"""Simple data loader for scenario files."""

import json
import logging
from pathlib import Path

from .exceptions import DataLoadError, InvalidScenarioError
from .models import Scenario, ScenarioCategory
from .validation import validate_scenario_file


class SimpleDataLoader:
    """Simple file-based scenario data loader."""

    def __init__(self, data_dir: str | None = None, validate_on_load: bool = True):
        """Initialize with data directory."""
        if data_dir is None:
            # Default to data directory in repo root
            repo_root = Path(__file__).parent.parent.parent
            data_dir = repo_root / "data" / "scenarios"

        self.data_dir = Path(data_dir)
        self.validate_on_load = validate_on_load
        self._scenarios = {}
        self._loaded = False
        self.logger = logging.getLogger(__name__)

    def load_scenarios(self) -> dict[str, Scenario]:
        """Load all scenarios from JSON files."""
        if self._loaded:
            return self._scenarios

        self._scenarios = {}

        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return self._scenarios

        # Load scenarios from each category directory
        for category_dir in self.data_dir.iterdir():
            if not category_dir.is_dir():
                continue

            category_name = category_dir.name
            try:
                category = ScenarioCategory(category_name)
            except ValueError:
                print(f"Warning: Unknown category directory {category_name}")
                continue

            # Load JSON files in this category
            for json_file in category_dir.glob("*.json"):
                try:
                    # Validate file if requested
                    if self.validate_on_load:
                        is_valid, validation_errors = validate_scenario_file(json_file)
                        if not is_valid:
                            self.logger.error(f"Validation failed for {json_file}: {validation_errors}")
                            raise InvalidScenarioError(str(json_file), validation_errors)

                    with open(json_file, encoding='utf-8') as f:
                        data = json.load(f)

                    scenario = Scenario(
                        id=data["id"],
                        category=ScenarioCategory(data["category"]),
                        name=data["name"],
                        description=data["description"],
                        adversary_claim=data["adversary_claim"],
                        correct_skepticism_level=data["correct_skepticism_level"],
                        good_evidence_requests=data["good_evidence_requests"],
                        red_flags=data["red_flags"],
                        metadata=data.get("metadata", {})
                    )

                    self._scenarios[scenario.id] = scenario
                    self.logger.debug(f"Successfully loaded scenario: {scenario.id}")

                except (DataLoadError, InvalidScenarioError):
                    # Re-raise our custom exceptions
                    raise
                except Exception as e:
                    error_msg = f"Unexpected error loading scenario from {json_file}: {e}"
                    self.logger.error(error_msg)
                    raise DataLoadError(str(json_file), str(e))

        self._loaded = True
        print(f"Loaded {len(self._scenarios)} scenarios")
        return self._scenarios

    def get_scenario(self, scenario_id: str) -> Scenario | None:
        """Get specific scenario by ID."""
        scenarios = self.load_scenarios()
        return scenarios.get(scenario_id)

    def get_scenarios_by_category(self, category: ScenarioCategory) -> list[Scenario]:
        """Get all scenarios for a specific category."""
        scenarios = self.load_scenarios()
        return [s for s in scenarios.values() if s.category == category]

    def get_all_scenarios(self) -> list[Scenario]:
        """Get all loaded scenarios."""
        scenarios = self.load_scenarios()
        return list(scenarios.values())

# Global instance for easy access
_global_loader = None

def get_data_loader() -> SimpleDataLoader:
    """Get global data loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = SimpleDataLoader()
    return _global_loader
