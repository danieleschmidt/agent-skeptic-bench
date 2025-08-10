"""Machine learning algorithms for Agent Skeptic Bench."""

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..models import EvaluationResult, Scenario, ScenarioCategory

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for ML model performance."""

    mse: float
    mae: float
    r2: float
    cross_val_score_mean: float
    cross_val_score_std: float
    feature_importance: dict[str, float]
    training_time: float
    prediction_time: float


@dataclass
class MLPrediction:
    """ML model prediction result."""

    predicted_value: float
    confidence_interval: tuple[float, float]
    prediction_confidence: float
    feature_contributions: dict[str, float]
    model_version: str


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_type: str = "random_forest"
    test_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5
    feature_selection_k: int = 10
    hyperparameter_tuning: bool = True
    save_model: bool = True
    model_save_path: Path | None = None


class FeatureExtractor:
    """Extracts features from evaluation data for ML models."""

    def __init__(self):
        """Initialize feature extractor."""
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []

    def extract_features(self, results: list[EvaluationResult],
                        scenarios: list[Scenario] | None = None) -> pd.DataFrame:
        """Extract features from evaluation results."""
        features = []

        for result in results:
            feature_dict = self._extract_result_features(result)

            # Add scenario features if available
            if scenarios:
                scenario = next((s for s in scenarios if s.id == result.scenario_id), None)
                if scenario:
                    feature_dict.update(self._extract_scenario_features(scenario))

            features.append(feature_dict)

        df = pd.DataFrame(features)
        self.feature_names = list(df.columns)

        return df

    def _extract_result_features(self, result: EvaluationResult) -> dict[str, Any]:
        """Extract features from evaluation result."""
        return {
            # Basic features
            'agent_provider': result.agent_provider,
            'model': result.model,
            'passed_evaluation': int(result.passed_evaluation),

            # Metric features
            'overall_score': result.metrics.overall_score,
            'skepticism_calibration': result.metrics.skepticism_calibration,
            'evidence_standard_score': result.metrics.evidence_standard_score,
            'red_flag_detection': result.metrics.red_flag_detection,
            'reasoning_quality': result.metrics.reasoning_quality,

            # Response features
            'confidence_level': result.response.confidence_level,
            'evidence_requests_count': len(result.response.evidence_requests),
            'red_flags_identified_count': len(result.response.red_flags_identified),
            'reasoning_length': len(result.response.reasoning),

            # Temporal features
            'hour_of_day': result.evaluated_at.hour,
            'day_of_week': result.evaluated_at.weekday(),
            'days_since_epoch': (result.evaluated_at - datetime(1970, 1, 1)).days,

            # Decision features
            'decision_skeptical': int(result.response.decision == 'skeptical'),
            'decision_accepting': int(result.response.decision == 'accepting'),
            'decision_neutral': int(result.response.decision == 'neutral'),
        }

    def _extract_scenario_features(self, scenario: Scenario) -> dict[str, Any]:
        """Extract features from scenario."""
        return {
            # Scenario basic features
            'scenario_category': scenario.category.value,
            'correct_skepticism_level': scenario.correct_skepticism_level,
            'red_flags_count': len(scenario.red_flags),
            'title_length': len(scenario.title),
            'description_length': len(scenario.description),

            # Metadata features
            'difficulty': scenario.metadata.get('difficulty', 'unknown'),
            'domain': scenario.metadata.get('domain', 'general'),
            'scenario_tags_count': len(scenario.metadata.get('tags', [])),

            # Category one-hot encoding
            'category_misinformation': int(scenario.category == ScenarioCategory.MISINFORMATION),
            'category_fraud': int(scenario.category == ScenarioCategory.FRAUD),
            'category_manipulation': int(scenario.category == ScenarioCategory.MANIPULATION),
            'category_pseudoscience': int(scenario.category == ScenarioCategory.PSEUDOSCIENCE),
        }

    def prepare_features(self, df: pd.DataFrame, fit_encoders: bool = True) -> np.ndarray:
        """Prepare features for ML training/prediction."""
        df_encoded = df.copy()

        # Encode categorical variables
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            if col not in self.label_encoders:
                if fit_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    # Handle unseen categories
                    df_encoded[col] = df_encoded[col].map(
                        lambda x: self.label_encoders[col].transform([str(x)])[0]
                        if str(x) in self.label_encoders[col].classes_ else -1
                    )
            else:
                # Transform using existing encoder
                df_encoded[col] = df_encoded[col].map(
                    lambda x: self.label_encoders[col].transform([str(x)])[0]
                    if str(x) in self.label_encoders[col].classes_ else -1
                )

        # Scale numerical features
        if fit_encoders:
            features_scaled = self.scaler.fit_transform(df_encoded)
        else:
            features_scaled = self.scaler.transform(df_encoded)

        return features_scaled

    def get_feature_names(self) -> list[str]:
        """Get feature names."""
        return self.feature_names


class MLPredictor:
    """ML predictor for evaluation outcomes and scores."""

    def __init__(self, model_type: str = "random_forest"):
        """Initialize ML predictor."""
        self.model_type = model_type
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        self.model_metrics: ModelMetrics | None = None
        self.feature_selector = None

    def _create_model(self) -> Any:
        """Create ML model based on type."""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
        elif self.model_type == "linear_regression":
            return LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    async def train(self, results: list[EvaluationResult],
                   scenarios: list[Scenario],
                   target_column: str = "overall_score",
                   config: TrainingConfig = None) -> ModelMetrics:
        """Train ML model on evaluation data."""
        if config is None:
            config = TrainingConfig()

        start_time = datetime.now()

        # Extract features
        features_df = self.feature_extractor.extract_features(results, scenarios)
        target = features_df[target_column]
        features_df = features_df.drop(columns=[target_column])

        # Prepare features
        X = self.feature_extractor.prepare_features(features_df, fit_encoders=True)
        y = target.values

        # Feature selection
        if config.feature_selection_k and config.feature_selection_k < X.shape[1]:
            self.feature_selector = SelectKBest(score_func=f_regression, k=config.feature_selection_k)
            X = self.feature_selector.fit_transform(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )

        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=config.cross_validation_folds,
            scoring='neg_mean_squared_error'
        )

        # Feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.feature_extractor.get_feature_names()
            if self.feature_selector:
                selected_features = self.feature_selector.get_support()
                feature_names = [name for i, name in enumerate(feature_names) if selected_features[i]]

            for name, importance in zip(feature_names, self.model.feature_importances_, strict=False):
                feature_importance[name] = float(importance)

        training_time = (datetime.now() - start_time).total_seconds()

        self.model_metrics = ModelMetrics(
            mse=mse,
            mae=mae,
            r2=r2,
            cross_val_score_mean=-cv_scores.mean(),
            cross_val_score_std=cv_scores.std(),
            feature_importance=feature_importance,
            training_time=training_time,
            prediction_time=0.0
        )

        self.is_trained = True

        # Save model if requested
        if config.save_model and config.model_save_path:
            await self._save_model(config.model_save_path)

        logger.info(f"Model trained successfully. R²: {r2:.3f}, MAE: {mae:.3f}")

        return self.model_metrics

    async def predict(self, results: list[EvaluationResult],
                     scenarios: list[Scenario]) -> list[MLPrediction]:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        start_time = datetime.now()

        # Extract and prepare features
        features_df = self.feature_extractor.extract_features(results, scenarios)
        # Remove target column if present
        if 'overall_score' in features_df.columns:
            features_df = features_df.drop(columns=['overall_score'])

        X = self.feature_extractor.prepare_features(features_df, fit_encoders=False)

        # Apply feature selection
        if self.feature_selector:
            X = self.feature_selector.transform(X)

        # Make predictions
        predictions = self.model.predict(X)

        prediction_time = (datetime.now() - start_time).total_seconds()

        # Create prediction objects
        ml_predictions = []
        for i, pred in enumerate(predictions):
            # Calculate confidence interval (simplified)
            confidence_interval = (
                max(0.0, pred - 0.1),  # Lower bound
                min(1.0, pred + 0.1)   # Upper bound
            )

            # Feature contributions (simplified)
            feature_contributions = {}
            if hasattr(self.model, 'feature_importances_'):
                feature_names = self.feature_extractor.get_feature_names()
                if self.feature_selector:
                    selected_features = self.feature_selector.get_support()
                    feature_names = [name for j, name in enumerate(feature_names) if selected_features[j]]

                for j, (name, importance) in enumerate(zip(feature_names, self.model.feature_importances_, strict=False)):
                    if j < len(X[i]):
                        feature_contributions[name] = float(importance * X[i][j])

            ml_predictions.append(MLPrediction(
                predicted_value=float(pred),
                confidence_interval=confidence_interval,
                prediction_confidence=0.8,  # Simplified confidence score
                feature_contributions=feature_contributions,
                model_version=f"{self.model_type}_v1.0"
            ))

        # Update prediction time in metrics
        if self.model_metrics:
            self.model_metrics.prediction_time = prediction_time

        return ml_predictions

    async def predict_scenario_difficulty(self, scenario: Scenario) -> float:
        """Predict difficulty of a scenario."""
        # Create a mock evaluation result to extract features
        from ..models import EvaluationMetrics, SkepticResponse

        mock_response = SkepticResponse(
            decision="neutral",
            confidence_level=0.5,
            reasoning="Mock reasoning",
            evidence_requests=[],
            red_flags_identified=[]
        )

        mock_metrics = EvaluationMetrics(
            overall_score=0.5,
            skepticism_calibration=0.5,
            evidence_standard_score=0.5,
            red_flag_detection=0.5,
            reasoning_quality=0.5
        )

        mock_result = EvaluationResult(
            id="mock",
            scenario_id=scenario.id,
            agent_provider="mock",
            model="mock",
            response=mock_response,
            metrics=mock_metrics,
            passed_evaluation=True,
            evaluated_at=datetime.utcnow()
        )

        predictions = await self.predict([mock_result], [scenario])
        return predictions[0].predicted_value if predictions else 0.5

    async def _save_model(self, path: Path) -> None:
        """Save trained model to disk."""
        try:
            model_data = {
                'model': self.model,
                'feature_extractor': self.feature_extractor,
                'feature_selector': self.feature_selector,
                'model_type': self.model_type,
                'metrics': self.model_metrics,
                'trained_at': datetime.utcnow()
            }

            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    async def load_model(self, path: Path) -> None:
        """Load trained model from disk."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.feature_extractor = model_data['feature_extractor']
            self.feature_selector = model_data.get('feature_selector')
            self.model_type = model_data['model_type']
            self.model_metrics = model_data['metrics']
            self.is_trained = True

            logger.info(f"Model loaded from {path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the trained model."""
        if not self.is_trained:
            return {"status": "not_trained"}

        return {
            "status": "trained",
            "model_type": self.model_type,
            "metrics": self.model_metrics.__dict__ if self.model_metrics else None,
            "feature_count": len(self.feature_extractor.get_feature_names()),
            "feature_names": self.feature_extractor.get_feature_names()
        }


class ModelTrainer:
    """Advanced model trainer with hyperparameter tuning."""

    def __init__(self):
        """Initialize model trainer."""
        self.best_models: dict[str, MLPredictor] = {}
        self.training_history: list[dict[str, Any]] = []

    async def train_multiple_models(self, results: list[EvaluationResult],
                                  scenarios: list[Scenario],
                                  model_types: list[str] = None) -> dict[str, ModelMetrics]:
        """Train multiple model types and compare performance."""
        if model_types is None:
            model_types = ["random_forest", "gradient_boosting", "linear_regression"]

        model_results = {}

        for model_type in model_types:
            try:
                predictor = MLPredictor(model_type=model_type)
                metrics = await predictor.train(results, scenarios)

                model_results[model_type] = metrics
                self.best_models[model_type] = predictor

                self.training_history.append({
                    'timestamp': datetime.utcnow(),
                    'model_type': model_type,
                    'metrics': metrics.__dict__,
                    'data_size': len(results)
                })

            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                continue

        return model_results

    def get_best_model(self, metric: str = "r2") -> MLPredictor | None:
        """Get the best performing model based on specified metric."""
        if not self.best_models:
            return None

        best_model = None
        best_score = -float('inf')

        for model_type, predictor in self.best_models.items():
            if predictor.model_metrics:
                score = getattr(predictor.model_metrics, metric, 0)
                if score > best_score:
                    best_score = score
                    best_model = predictor

        return best_model

    def get_training_history(self) -> list[dict[str, Any]]:
        """Get training history."""
        return self.training_history
