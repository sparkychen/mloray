"""
配置管理模块
企业级 MLOps 平台的集中配置管理
"""

import os
import yaml
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from enum import Enum
from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.fields import FieldInfo
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """运行环境枚举"""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class LoggingConfig(BaseSettings):
    """日志配置"""
    level: str = Field(default="INFO")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_path: Optional[str] = Field(default=None)
    max_bytes: int = Field(default=10 * 1024 * 1024)  # 10MB
    backup_count: int = Field(default=5)
    
    class Config:
        env_prefix = "LOGGING_"


class DatabaseConfig(BaseSettings):
    """数据库配置"""
    # PostgreSQL/TimescaleDB
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="mlops")
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="password")
    postgres_pool_size: int = Field(default=20)
    postgres_max_overflow: int = Field(default=10)
    postgres_pool_timeout: int = Field(default=30)
    postgres_pool_recycle: int = Field(default=3600)
    
    # TimescaleDB 特定配置
    timescaledb_enabled: bool = Field(default=True)
    timescaledb_compression: bool = Field(default=True)
    timescaledb_retention_days: int = Field(default=90)
    
    # 连接池配置
    pool_pre_ping: bool = Field(default=True)
    echo: bool = Field(default=False)
    
    class Config:
        env_prefix = "DATABASE_"
    
    @property
    def postgres_url(self) -> str:
        """获取同步 PostgreSQL 连接 URL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def async_postgres_url(self) -> str:
        """获取异步 PostgreSQL 连接 URL"""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def alembic_url(self) -> str:
        """获取 Alembic 迁移 URL"""
        return self.postgres_url


class StorageConfig(BaseSettings):
    """存储配置"""
    # S3 配置
    s3_endpoint: str = Field(default="https://s3.amazonaws.com")
    s3_access_key: str = Field(default="")
    s3_secret_key: str = Field(default="")
    s3_bucket: str = Field(default="mlops-data")
    s3_region: str = Field(default="us-east-1")
    s3_use_ssl: bool = Field(default=True)
    s3_verify_ssl: bool = Field(default=True)
    
    # 阿里云 OSS
    oss_enabled: bool = Field(default=False)
    oss_endpoint: str = Field(default="")
    oss_access_key_id: str = Field(default="")
    oss_access_key_secret: str = Field(default="")
    oss_bucket: str = Field(default="")
    
    # MinIO 配置
    use_minio: bool = Field(default=False)
    minio_endpoint: str = Field(default="http://localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    
    # 本地存储
    local_storage_path: str = Field(default="./data/storage")
    
    class Config:
        env_prefix = "STORAGE_"
    
    @validator("local_storage_path")
    def create_local_storage_path(cls, v: str) -> str:
        """创建本地存储目录"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    def get_s3_config(self) -> Dict[str, Any]:
        """获取 S3 配置"""
        if self.use_minio:
            return {
                "endpoint_url": self.minio_endpoint,
                "aws_access_key_id": self.minio_access_key,
                "aws_secret_access_key": self.minio_secret_key,
                "region_name": self.s3_region,
                "config": {
                    "s3": {"addressing_style": "path"},
                    "signature_version": "s3v4"
                }
            }
        elif self.oss_enabled:
            return {
                "endpoint_url": self.oss_endpoint,
                "aws_access_key_id": self.oss_access_key_id,
                "aws_secret_access_key": self.oss_access_key_secret,
                "region_name": self.s3_region
            }
        else:
            return {
                "endpoint_url": self.s3_endpoint,
                "aws_access_key_id": self.s3_access_key,
                "aws_secret_access_key": self.s3_secret_key,
                "region_name": self.s3_region
            }
    
    def get_storage_type(self) -> str:
        """获取存储类型"""
        if self.use_minio:
            return "minio"
        elif self.oss_enabled:
            return "oss"
        else:
            return "s3"


class MLflowConfig(BaseSettings):
    """MLflow 配置"""
    tracking_uri: str = Field(default="http://localhost:5000")
    registry_uri: str = Field(default="sqlite:///mlruns.db")
    artifact_store: str = Field(default="s3://mlops-artifacts")
    
    # 实验管理
    experiment_name: str = Field(default="default")
    experiment_creation_enabled: bool = Field(default=True)
    
    # 模型注册表
    model_registry_enabled: bool = Field(default=True)
    model_stage_transitions: Dict[str, List[str]] = Field(
        default={
            "None": ["Staging"],
            "Staging": ["Production", "Archived"],
            "Production": ["Archived"],
            "Archived": []
        }
    )
    
    # 跟踪配置
    autolog_enabled: bool = Field(default=True)
    autolog_max_workers: int = Field(default=10)
    
    class Config:
        env_prefix = "MLFLOW_"
    
    @validator("artifact_store")
    def validate_artifact_store(cls, v: str) -> str:
        """验证 artifact store 配置"""
        if v.startswith("s3://") and not v.startswith("s3://mlops"):
            logger.warning(f"Artifact store {v} may not be optimized for MLflow")
        return v


class RayConfig(BaseSettings):
    """Ray 配置"""
    # 集群配置
    address: str = Field(default="auto")
    node_ip_address: str = Field(default="127.0.0.1")
    num_cpus: Optional[int] = Field(default=None)
    num_gpus: Optional[int] = Field(default=None)
    
    # 资源限制
    memory: Optional[int] = Field(default=None)
    object_store_memory: Optional[int] = Field(default=None)
    
    # 调度配置
    _system_config: Dict[str, Any] = Field(
        default={
            "max_io_workers": 10,
            "object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}}
            ) if json else None
        }
    )
    
    # Dashboard
    dashboard_host: str = Field(default="0.0.0.0")
    dashboard_port: int = Field(default=8265)
    dashboard_grpc_port: int = Field(default=8266)
    
    # 监控
    metrics_export_port: int = Field(default=8080)
    enable_metrics_collection: bool = Field(default=True)
    
    class Config:
        env_prefix = "RAY_"
    
    def get_init_config(self) -> Dict[str, Any]:
        """获取 Ray 初始化配置"""
        config = {
            "address": self.address,
            "node_ip_address": self.node_ip_address,
            "dashboard_host": self.dashboard_host,
            "dashboard_port": self.dashboard_port,
            "dashboard_grpc_port": self.dashboard_grpc_port,
            "metrics_export_port": self.metrics_export_port,
            "_system_config": self._system_config,
        }
        
        if self.num_cpus:
            config["num_cpus"] = self.num_cpus
        if self.num_gpus:
            config["num_gpus"] = self.num_gpus
        if self.memory:
            config["memory"] = self.memory
        if self.object_store_memory:
            config["object_store_memory"] = self.object_store_memory
        
        return config


class DVCConfig(BaseSettings):
    """DVC 配置"""
    remote: str = Field(default="s3")
    remote_url: str = Field(default="s3://mlops-dvc")
    cache_dir: str = Field(default=".dvc/cache")
    cache_type: List[str] = Field(default=["hardlink", "symlink"])
    
    # 版本控制
    autocommit: bool = Field(default=False)
    autopush: bool = Field(default=False)
    
    # 大文件处理
    large_file_threshold: int = Field(default=100 * 1024 * 1024)  # 100MB
    compression_level: int = Field(default=6)
    
    class Config:
        env_prefix = "DVC_"
    
    @validator("cache_dir")
    def create_cache_dir(cls, v: str) -> str:
        """创建缓存目录"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)


class MonitoringConfig(BaseSettings):
    """监控配置"""
    # Prometheus
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090)
    
    # Grafana
    grafana_enabled: bool = Field(default=False)
    grafana_port: int = Field(default=3000)
    
    # 监控工具配置
    evidently_dash_port: int = Field(default=8050)
    phoenix_collector_endpoint: str = Field(default="http://localhost:6006")
    
    # WhyLabs
    whylabs_enabled: bool = Field(default=False)
    whylabs_api_key: Optional[str] = Field(default=None)
    whylabs_org_id: Optional[str] = Field(default=None)
    whylabs_dataset_id: Optional[str] = Field(default=None)
    
    # 告警
    alert_enabled: bool = Field(default=True)
    alert_webhook_url: Optional[str] = Field(default=None)
    alert_slack_webhook: Optional[str] = Field(default=None)
    alert_pagerduty_key: Optional[str] = Field(default=None)
    
    # 阈值配置
    drift_threshold: float = Field(default=0.1, ge=0, le=1)
    performance_threshold: float = Field(default=0.05, ge=0, le=1)
    data_quality_threshold: float = Field(default=0.95, ge=0, le=1)
    
    # 重试配置
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    
    class Config:
        env_prefix = "MONITORING_"


class FeatureStoreConfig(BaseSettings):
    """特征存储配置"""
    # Feast
    feast_registry_path: str = Field(default="feature_store/registry.db")
    feast_project: str = Field(default="mlops_feature_store")
    feast_provider: str = Field(default="local")
    
    # 在线存储
    feast_online_store_type: str = Field(default="redis")
    feast_online_store_config: Dict[str, Any] = Field(
        default={
            "type": "redis",
            "connection_string": "redis://localhost:6379"
        }
    )
    
    # 离线存储
    feast_offline_store_type: str = Field(default="file")
    feast_offline_store_config: Dict[str, Any] = Field(
        default={
            "type": "file",
            "path": "feature_store/data.parquet"
        }
    )
    
    # Doris
    doris_enabled: bool = Field(default=True)
    doris_host: str = Field(default="localhost")
    doris_port: int = Field(default=9030)
    doris_user: str = Field(default="root")
    doris_password: str = Field(default="")
    doris_database: str = Field(default="feast")
    doris_query_timeout: int = Field(default=300)
    
    # 特征注册
    auto_register_features: bool = Field(default=True)
    feature_validation_enabled: bool = Field(default=True)
    
    class Config:
        env_prefix = "FEATURE_STORE_"
    
    @property
    def doris_connection(self) -> Dict[str, Any]:
        """获取 Doris 连接配置"""
        return {
            "host": self.doris_host,
            "port": self.doris_port,
            "user": self.doris_user,
            "password": self.doris_password,
            "database": self.doris_database,
            "connect_timeout": 10,
            "read_timeout": self.doris_query_timeout
        }
    
    def get_feast_config(self) -> Dict[str, Any]:
        """获取 Feast 配置"""
        return {
            "registry": self.feast_registry_path,
            "project": self.feast_project,
            "provider": self.feast_provider,
            "online_store": self.feast_online_store_config,
            "offline_store": self.feast_offline_store_config,
        }


class Neo4jConfig(BaseSettings):
    """Neo4j 配置"""
    uri: str = Field(default="bolt://localhost:7687")
    user: str = Field(default="neo4j")
    password: str = Field(default="password")
    database: str = Field(default="neo4j")
    
    # 连接池
    max_connection_pool_size: int = Field(default=50)
    max_connection_lifetime: int = Field(default=3600)
    connection_acquisition_timeout: int = Field(default=60)
    
    # 加密
    encrypted: bool = Field(default=False)
    trust: str = Field(default="TRUST_ALL_CERTIFICATES")
    
    class Config:
        env_prefix = "NEO4J_"
    
    @property
    def connection_config(self) -> Dict[str, Any]:
        """获取连接配置"""
        return {
            "uri": self.uri,
            "auth": (self.user, self.password),
            "database": self.database,
            "max_connection_pool_size": self.max_connection_pool_size,
            "max_connection_lifetime": self.max_connection_lifetime,
            "connection_acquisition_timeout": self.connection_acquisition_timeout,
            "encrypted": self.encrypted,
            "trust": self.trust
        }


class APIConfig(BaseSettings):
    """API 配置"""
    # 服务器配置
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    reload: bool = Field(default=False)
    
    # CORS
    cors_origins: List[str] = Field(default=["*"])
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: List[str] = Field(default=["*"])
    cors_allow_headers: List[str] = Field(default=["*"])
    
    # 速率限制
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_default: str = Field(default="100/minute")
    rate_limit_storage_uri: str = Field(default="memory://")
    
    # 安全
    secret_key: str = Field(default="your-secret-key-change-in-production")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    refresh_token_expire_days: int = Field(default=7)
    
    # API 文档
    docs_url: str = Field(default="/docs")
    redoc_url: str = Field(default="/redoc")
    openapi_url: str = Field(default="/openapi.json")
    
    class Config:
        env_prefix = "API_"
    
    @validator("secret_key")
    def validate_secret_key(cls, v: str, values: Dict[str, Any]) -> str:
        """验证密钥"""
        if v == "your-secret-key-change-in-production":
            if values.get("environment") == Environment.PROD:
                raise ValueError("SECRET_KEY must be changed in production")
        return v


class CeleryConfig(BaseSettings):
    """Celery 配置"""
    # Broker
    broker_url: str = Field(default="redis://localhost:6379/0")
    result_backend: str = Field(default="redis://localhost:6379/1")
    
    # 任务配置
    task_serializer: str = Field(default="json")
    result_serializer: str = Field(default="json")
    accept_content: List[str] = Field(default=["json"])
    timezone: str = Field(default="UTC")
    
    # Worker 配置
    worker_concurrency: int = Field(default=4)
    worker_prefetch_multiplier: int = Field(default=4)
    worker_max_tasks_per_child: int = Field(default=1000)
    worker_max_memory_per_child: int = Field(default=120000)  # 120MB
    
    # 任务路由
    task_routes: Dict[str, Dict[str, str]] = Field(
        default={
            "src.core.training.*": {"queue": "training"},
            "src.core.data.*": {"queue": "data_processing"},
            "src.core.monitoring.*": {"queue": "monitoring"},
        }
    )
    
    # 监控
    flower_port: int = Field(default=5555)
    
    class Config:
        env_prefix = "CELERY_"


class Settings(BaseSettings):
    """主设置类"""
    
    # 环境配置
    environment: Environment = Field(default=Environment.DEV)
    debug: bool = Field(default=False)
    
    # 各模块配置
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    ray: RayConfig = Field(default_factory=RayConfig)
    dvc: DVCConfig = Field(default_factory=DVCConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    
    # 项目配置
    project_name: str = Field(default="Enterprise MLOps Platform")
    project_version: str = Field(default="1.0.0")
    project_description: str = Field(
        default="Enterprise-grade MLOps platform with full lifecycle management"
    )
    
    # 路径配置
    base_dir: Path = Field(default=Path(__file__).parent.parent.parent.parent)
    config_dir: Path = Field(default=Path("config"))
    data_dir: Path = Field(default=Path("data"))
    log_dir: Path = Field(default=Path("logs"))
    model_dir: Path = Field(default=Path("models"))
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False
        extra = "ignore"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.base_dir / self.data_dir,
            self.base_dir / self.log_dir,
            self.base_dir / self.model_dir,
            self.base_dir / "config",
            self.base_dir / "pipelines",
            self.base_dir / "schemas",
            self.base_dir / "tests",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @root_validator(pre=True)
    def load_config_files(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """从配置文件加载配置"""
        environment = values.get("environment", "dev")
        config_dir = Path(values.get("config_dir", "config")).absolute()
        
        # 加载环境配置
        env_file = config_dir / "environments" / f"{environment}.yaml"
        if env_file.exists():
            with open(env_file, "r") as f:
                env_config = yaml.safe_load(f) or {}
            
            # 深度合并配置
            for key, value in env_config.items():
                if key in values and isinstance(values[key], dict) and isinstance(value, dict):
                    # 深度合并字典
                    values[key] = {**values[key], **value}
                else:
                    values[key] = value
        
        return values
    
    def get_database_config(self) -> Dict[str, Any]:
        """获取数据库配置"""
        return {
            "url": self.database.postgres_url,
            "async_url": self.database.async_postgres_url,
            "pool_size": self.database.postgres_pool_size,
            "max_overflow": self.database.postgres_max_overflow,
            "pool_timeout": self.database.postgres_pool_timeout,
            "pool_recycle": self.database.postgres_pool_recycle,
            "echo": self.database.echo,
            "pool_pre_ping": self.database.pool_pre_ping,
        }
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """获取 MLflow 配置"""
        return {
            "tracking_uri": self.mlflow.tracking_uri,
            "registry_uri": self.mlflow.registry_uri,
            "artifact_store": self.mlflow.artifact_store,
            "experiment_name": self.mlflow.experiment_name,
            "autolog_enabled": self.mlflow.autolog_enabled,
        }
    
    def get_ray_config(self) -> Dict[str, Any]:
        """获取 Ray 配置"""
        return self.ray.get_init_config()
    
    def get_storage_config(self) -> Dict[str, Any]:
        """获取存储配置"""
        return {
            "type": self.storage.get_storage_type(),
            "s3_config": self.storage.get_s3_config(),
            "local_path": self.storage.local_storage_path,
            "bucket": self.storage.s3_bucket,
        }
    
    def get_feast_config(self) -> Dict[str, Any]:
        """获取 Feast 配置"""
        return self.feature_store.get_feast_config()
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取监控配置"""
        return {
            "prometheus_enabled": self.monitoring.prometheus_enabled,
            "grafana_enabled": self.monitoring.grafana_enabled,
            "evidently_port": self.monitoring.evidently_dash_port,
            "phoenix_endpoint": self.monitoring.phoenix_collector_endpoint,
            "whylabs_enabled": self.monitoring.whylabs_enabled,
            "drift_threshold": self.monitoring.drift_threshold,
            "performance_threshold": self.monitoring.performance_threshold,
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """获取 API 配置"""
        return {
            "host": self.api.host,
            "port": self.api.port,
            "workers": self.api.workers,
            "reload": self.api.reload,
            "docs_url": self.api.docs_url,
            "redoc_url": self.api.redoc_url,
            "openapi_url": self.api.openapi_url,
        }
    
    def get_celery_config(self) -> Dict[str, Any]:
        """获取 Celery 配置"""
        return {
            "broker_url": self.celery.broker_url,
            "result_backend": self.celery.result_backend,
            "task_serializer": self.celery.task_serializer,
            "result_serializer": self.celery.result_serializer,
            "worker_concurrency": self.celery.worker_concurrency,
            "flower_port": self.celery.flower_port,
        }


@lru_cache()
def get_settings() -> Settings:
    """
    获取设置实例（缓存）
    
    Returns:
        Settings: 设置实例
    """
    return Settings()


# 全局设置实例
settings = get_settings()
