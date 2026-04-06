"""
Configuration management for the MLOps platform.
"""

import os
import yaml
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from enum import Enum
from functools import lru_cache

from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.fields import FieldInfo
import dotenv

# Load environment variables
dotenv.load_dotenv()


class Environment(str, Enum):
    """Environment enumeration."""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_path: Optional[str] = Field(default=None)
    max_file_size: int = Field(default=10485760)  # 10MB
    backup_count: int = Field(default=5)
    json_format: bool = Field(default=False)


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="mlops")
    user: str = Field(default="postgres")
    password: str = Field(default="postgres")
    pool_size: int = Field(default=20)
    max_overflow: int = Field(default=30)
    pool_timeout: int = Field(default=30)
    pool_recycle: int = Field(default=3600)
    
    @property
    def url(self) -> str:
        """Get database URL."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
    
    @property
    def async_url(self) -> str:
        """Get async database URL."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )


class TimescaleDBConfig(BaseSettings):
    """TimescaleDB configuration."""
    
    enabled: bool = Field(default=True)
    hypertables: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        extra = "allow"


class StorageConfig(BaseSettings):
    """Storage configuration."""
    
    s3_endpoint: str = Field(default="http://localhost:9000")
    s3_access_key: str = Field(default="minioadmin")
    s3_secret_key: str = Field(default="minioadmin")
    s3_bucket: str = Field(default="mlops-data")
    s3_region: str = Field(default="us-east-1")
    s3_secure: bool = Field(default=False)
    
    local_data_dir: str = Field(default="./data")
    local_artifacts_dir: str = Field(default="./artifacts")
    
    @property
    def s3_config(self) -> Dict[str, Any]:
        """Get S3 configuration."""
        return {
            "endpoint_url": self.s3_endpoint,
            "aws_access_key_id": self.s3_access_key,
            "aws_secret_access_key": self.s3_secret_key,
            "region_name": self.s3_region
        }
    
    class Config:
        extra = "allow"


class MLflowConfig(BaseSettings):
    """MLflow configuration."""
    
    tracking_uri: str = Field(default="http://localhost:5000")
    registry_uri: str = Field(
        default="sqlite:///mlruns.db"
    )
    experiment_name: str = Field(default="default")
    artifact_location: str = Field(default="s3://mlflow-artifacts/")
    nested_runs: bool = Field(default=True)
    
    autolog_enabled: bool = Field(default=True)
    autolog_log_models: bool = Field(default=True)
    autolog_log_datasets: bool = Field(default=True)
    autolog_log_input_examples: bool = Field(default=True)
    
    class Config:
        extra = "allow"


class RayConfig(BaseSettings):
    """Ray configuration."""
    
    address: str = Field(default="auto")
    num_cpus: int = Field(default=4)
    num_gpus: int = Field(default=0)
    object_store_memory: int = Field(default=4000000000)  # 4GB
    dashboard_host: str = Field(default="0.0.0.0")
    dashboard_port: int = Field(default=8265)
    
    runtime_env: Dict[str, Any] = Field(
        default_factory=lambda: {
            "working_dir": "./src",
            "py_modules": [],
            "env_vars": {"PYTHONPATH": "./src"}
        }
    )
    
    train_scaling_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "num_workers": 2,
            "use_gpu": False,
            "resources_per_worker": {"CPU": 2, "GPU": 0}
        }
    )
    
    tune_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "num_samples": 20,
            "max_concurrent_trials": 4,
            "grace_period": 10,
            "reduction_factor": 3
        }
    )
    
    class Config:
        extra = "allow"


class FeastConfig(BaseSettings):
    """Feast configuration."""
    
    registry: str = Field(
        default="postgresql://postgres:postgres@localhost/feast"
    )
    project: str = Field(default="mlops_feature_store")
    provider: str = Field(default="local")
    
    online_store: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "redis",
            "connection": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            }
        }
    )
    
    offline_store: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "postgres",
            "connection": {
                "host": "localhost",
                "port": 5432,
                "database": "feast",
                "user": "postgres",
                "password": "postgres"
            }
        }
    )
    
    entity_key_serialization_version: int = Field(default=2)
    
    class Config:
        extra = "allow"


class DVCConfig(BaseSettings):
    """DVC configuration."""
    
    remote: str = Field(default="minio")
    
    cache: Dict[str, Any] = Field(
        default_factory=lambda: {
            "dir": "./.dvc/cache",
            "type": "hardlink,symlink"
        }
    )
    
    remotes: Dict[str, Any] = Field(
        default_factory=lambda: {
            "minio": {
                "type": "s3",
                "endpointurl": "http://localhost:9000",
                "access_key_id": "minioadmin",
                "secret_access_key": "minioadmin",
                "bucket": "dvc-store"
            }
        }
    )
    
    data: Dict[str, Any] = Field(
        default_factory=lambda: {
            "tracked_dirs": ["data/raw", "data/processed", "data/features"],
            "ignored_patterns": ["*.tmp", "*.log", "*.bak"]
        }
    )
    
    class Config:
        extra = "allow"


class GreatExpectationsConfig(BaseSettings):
    """Great Expectations configuration."""
    
    context_root_dir: str = Field(default="./great_expectations")
    
    expectation_suites: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "name": "raw_data_suite",
                "columns": ["id", "timestamp", "value"]
            },
            {
                "name": "processed_data_suite",
                "columns": ["feature_1", "feature_2", "label"]
            }
        ]
    )
    
    class Config:
        extra = "allow"


class DriftConfig(BaseSettings):
    """Drift detection configuration."""
    
    detection_methods: List[str] = Field(
        default_factory=lambda: ["ks_test", "psi", "classifier"]
    )
    
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "ks_test": 0.05,
            "psi": 0.1,
            "mmd": 0.05
        }
    )
    
    schedule: Dict[str, Any] = Field(
        default_factory=lambda: {
            "daily": True,
            "hour": 2,
            "minute": 0
        }
    )
    
    reference_data: Dict[str, Any] = Field(
        default_factory=lambda: {
            "window_days": 30,
            "update_frequency": "weekly"
        }
    )
    
    class Config:
        extra = "allow"


class PerformanceConfig(BaseSettings):
    """Performance monitoring configuration."""
    
    metrics: List[str] = Field(
        default_factory=lambda: [
            "accuracy", "precision", "recall", "f1",
            "auc", "mae", "mse", "r2"
        ]
    )
    
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.7,
            "f1": 0.7
        }
    )
    
    alerting: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "channels": ["slack", "email", "pagerduty"],
            "cooldown_minutes": 30
        }
    )
    
    class Config:
        extra = "allow"


class TracingConfig(BaseSettings):
    """Tracing configuration."""
    
    enabled: bool = Field(default=True)
    exporter: str = Field(default="jaeger")
    
    jaeger: Dict[str, Any] = Field(
        default_factory=lambda: {
            "host": "jaeger",
            "port": 6831
        }
    )
    
    sampling_rate: float = Field(default=0.1)
    
    class Config:
        extra = "allow"


class JWTConfig(BaseSettings):
    """JWT configuration."""
    
    secret_key: str = Field(
        default="your-super-secret-jwt-key-change-in-production"
    )
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    refresh_token_expire_days: int = Field(default=7)
    
    class Config:
        extra = "allow"


class CORSConfig(BaseSettings):
    """CORS configuration."""
    
    enabled: bool = Field(default=True)
    origins: List[str] = Field(default_factory=lambda: ["*"])
    methods: List[str] = Field(default_factory=lambda: ["*"])
    headers: List[str] = Field(default_factory=lambda: ["*"])
    credentials: bool = Field(default=True)
    
    class Config:
        extra = "allow"


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""
    
    enabled: bool = Field(default=True)
    requests_per_minute: int = Field(default=100)
    burst_limit: int = Field(default=20)
    
    class Config:
        extra = "allow"


class Neo4jConfig(BaseSettings):
    """Neo4j configuration."""
    
    uri: str = Field(default="bolt://localhost:7687")
    user: str = Field(default="neo4j")
    password: str = Field(default="password")
    database: str = Field(default="neo4j")
    
    schema: Dict[str, Any] = Field(
        default_factory=lambda: {
            "nodes": [
                {
                    "label": "Model",
                    "properties": ["name", "version", "framework", "status"]
                },
                {
                    "label": "Dataset",
                    "properties": ["name", "version", "source", "size"]
                },
                {
                    "label": "Feature",
                    "properties": ["name", "type", "importance"]
                },
                {
                    "label": "Experiment",
                    "properties": ["id", "name", "status", "metrics"]
                }
            ],
            "relationships": [
                {
                    "type": "TRAINED_WITH",
                    "from": "Model",
                    "to": "Dataset"
                },
                {
                    "type": "USES_FEATURE",
                    "from": "Model",
                    "to": "Feature"
                },
                {
                    "type": "PRODUCED_BY",
                    "from": "Model",
                    "to": "Experiment"
                }
            ]
        }
    )
    
    class Config:
        extra = "allow"


class AlertingConfig(BaseSettings):
    """Alerting configuration."""
    
    slack: Dict[str, Any] = Field(
        default_factory=lambda: {
            "webhook_url": "${SLACK_WEBHOOK_URL}",
            "channel": "#mlops-alerts",
            "username": "MLOps Bot",
            "icon_emoji": ":robot_face:"
        }
    )
    
    email: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "${EMAIL_USER}",
            "smtp_password": "${EMAIL_PASSWORD}",
            "from_email": "mlops@company.com",
            "to_emails": ["team@company.com"]
        }
    )
    
    pagerduty: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": False,
            "integration_key": "${PAGERDUTY_KEY}",
            "service_id": "${PAGERDUTY_SERVICE_ID}"
        }
    )
    
    rules: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "name": "data_drift_detected",
                "condition": "drift_score > 0.1",
                "severity": "high",
                "channels": ["slack", "email"]
            },
            {
                "name": "model_performance_degradation",
                "condition": "accuracy < 0.7",
                "severity": "critical",
                "channels": ["slack", "email", "pagerduty"]
            },
            {
                "name": "pipeline_failed",
                "condition": "status == 'failed'",
                "severity": "medium",
                "channels": ["slack"]
            },
            {
                "name": "resource_usage_high",
                "condition": "cpu_usage > 90 or memory_usage > 90",
                "severity": "warning",
                "channels": ["slack"]
            }
        ]
    )
    
    class Config:
        extra = "allow"


class WorkflowConfig(BaseSettings):
    """Workflow configuration."""
    
    data_processing: Dict[str, Any] = Field(
        default_factory=lambda: {
            "schedule": "0 2 * * *",  # Daily at 2 AM
            "timeout_minutes": 120,
            "retry_policy": {
                "max_retries": 3,
                "delay_seconds": 60,
                "backoff_factor": 2
            }
        }
    )
    
    training: Dict[str, Any] = Field(
        default_factory=lambda: {
            "schedule": "0 3 * * 0",  # Weekly on Sunday at 3 AM
            "timeout_minutes": 240,
            "retry_policy": {
                "max_retries": 2,
                "delay_seconds": 300,
                "backoff_factor": 2
            }
        }
    )
    
    monitoring: Dict[str, Any] = Field(
        default_factory=lambda: {
            "schedule": "*/30 * * * *",  # Every 30 minutes
            "timeout_minutes": 30,
            "retry_policy": {
                "max_retries": 1,
                "delay_seconds": 30
            }
        }
    )
    
    retraining: Dict[str, Any] = Field(
        default_factory=lambda: {
            "triggers": [
                {
                    "type": "drift",
                    "threshold": 0.1,
                    "cooldown_days": 1
                },
                {
                    "type": "performance",
                    "metric": "accuracy",
                    "threshold": 0.7,
                    "window_days": 7,
                    "cooldown_days": 2
                },
                {
                    "type": "schedule",
                    "cron": "0 4 * * 1"  # Weekly on Monday at 4 AM
                }
            ]
        }
    )
    
    class Config:
        extra = "allow"


class Settings(BaseSettings):
    """Main settings class."""
    
    # Environment
    environment: Environment = Field(default=Environment.DEV)
    
    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    api_debug: bool = Field(default=False)
    
    # Logging
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Database
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    timescaledb: TimescaleDBConfig = Field(default_factory=TimescaleDBConfig)
    
    # Storage
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    # MLOps
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    ray: RayConfig = Field(default_factory=RayConfig)
    feast: FeastConfig = Field(default_factory=FeastConfig)
    dvc: DVCConfig = Field(default_factory=DVCConfig)
    
    # Monitoring
    great_expectations: GreatExpectationsConfig = Field(
        default_factory=GreatExpectationsConfig
    )
    drift: DriftConfig = Field(default_factory=DriftConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    
    # Security
    jwt: JWTConfig = Field(default_factory=JWTConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    
    # Graph Database
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    
    # Alerting
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    
    # Workflows
    workflows: WorkflowConfig = Field(default_factory=WorkflowConfig)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False
        extra = "allow"
    
    @classmethod
    @lru_cache()
    def load(cls, env: Optional[str] = None) -> "Settings":
        """
        Load settings from YAML configuration file.
        
        Args:
            env: Environment name (dev, staging, prod)
            
        Returns:
            Settings instance
        """
        if env is None:
            env = os.getenv("ENVIRONMENT", "dev")
        
        config_path = Path("config/environments") / f"{env}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )
        
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        # Create settings instance
        settings = cls(**config_data)
        
        return settings
    
    def get_feast_config(self) -> Dict[str, Any]:
        """Get Feast configuration."""
        return {
            "registry": self.feast.registry,
            "project": self.feast.project,
            "provider": self.feast.provider,
            "online_store": self.feast.online_store,
            "offline_store": self.feast.offline_store,
            "entity_key_serialization_version": (
                self.feast.entity_key_serialization_version
            )
        }
    
    def get_ray_config(self) -> Dict[str, Any]:
        """Get Ray configuration."""
        return {
            "address": self.ray.address,
            "num_cpus": self.ray.num_cpus,
            "num_gpus": self.ray.num_gpus,
            "object_store_memory": self.ray.object_store_memory,
            "dashboard_host": self.ray.dashboard_host,
            "dashboard_port": self.ray.dashboard_port,
            "runtime_env": self.ray.runtime_env
        }
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return {
            "tracking_uri": self.mlflow.tracking_uri,
            "registry_uri": self.mlflow.registry_uri,
            "experiment_name": self.mlflow.experiment_name,
            "artifact_location": self.mlflow.artifact_location,
            "nested_runs": self.mlflow.nested_runs,
            "autolog": {
                "enabled": self.mlflow.autolog_enabled,
                "log_models": self.mlflow.autolog_log_models,
                "log_datasets": self.mlflow.autolog_log_datasets,
                "log_input_examples": self.mlflow.autolog_log_input_examples
            }
        }


# Global settings instance
settings = Settings.load()
