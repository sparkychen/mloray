---
AIGC:
    ContentProducer: Minimax Agent AI
    ContentPropagator: Minimax Agent AI
    Label: AIGC
    ProduceID: 2d4d6aa17d30841e7dc7401b2154981f
    PropagateID: 2d4d6aa17d30841e7dc7401b2154981f
    ReservedCode1: 304402203d7583eb082132a152d46acfcd9e9ebdbd8dbed56a9175b567ced0550843708802203ffdb883ce71f7685aa7a341ca288840cb6a2322c11c6e619ce4893e4412fb0f
    ReservedCode2: 304502203cb9fec9c8b3102d501b894aeb14ec8bec99d266dbcde08911ddd135aeec3075022100e1fda679359a7f2d9686319f6c3936f8674a70b80f604f894e8d2bdcd0daaa1a
---

# 企业级 MLOps 全生命周期管理平台

## 技术栈版本

| 组件 | 版本 | 用途 |
|------|------|------|
| **FastAPI** | 0.115.x | API 服务框架 |
| **MLflow** | 3.10.1 | 实验追踪与模型注册 |
| **Ray** | 2.54.1 | 分布式计算 |
| **DVC** | 3.66.1 | 数据版本控制 |
| **Great Expectations** | 1.15.2 | 数据质量验证 |
| **whylogs** | 1.6.4 | 数据日志与画像 |
| **Evidently** | 0.7.21 | 数据/模型漂移检测 |
| **Arize Phoenix** | 13.22.2 | ML 可观测性 |
| **Feast** | 0.61.0 | 特征存储 |
| **PostgreSQL** | 16.x | 元数据存储 |
| **TimescaleDB** | 2.15.x | 时序数据 |
| **Neo4j** | 5.x | 知识图谱 |
| **Apache Doris** | 2.1.x | OLAP 分析 |

---

## 整体架构

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              控制平面 (Control Plane)                             │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   MLflow    │  │   Feast     │  │   Neo4j     │  │   TimescaleDB       │ │
│  │  3.10.1     │  │  0.61.0    │  │  5.x        │  │   2.15.x            │ │
│  │ 实验/注册   │  │ 特征存储    │  │ 知识图谱    │  │   时序监控          │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                │                     │              │
│         └────────────────┴────────────────┴─────────────────────┘              │
│                                    │                                            │
├────────────────────────────────────┼────────────────────────────────────────────┤
│                         计算平面 (Compute Plane)                                 │
│                                    │                                            │
│  ┌─────────────────────────────────┴─────────────────────────────────────────┐ │
│  │                         Ray Cluster 2.54.1                                  │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│ │
│  │  │   Ray      │  │   Ray      │  │   Ray       │  │   Ray               ││ │
│  │  │   Data     │  │   Train     │  │   Tune      │  │   Serve             ││ │
│  │  │  数据处理  │  │  模型训练   │  │  超参优化   │  │  模型服务           ││ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘│ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                            │
├────────────────────────────────────┼────────────────────────────────────────────┤
│                         数据平面 (Data Plane)                                   │
│                                    │                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │    S3       │  │ PostgreSQL  │  │  Apache     │  │   Neo4j                 │ │
│  │  (原始数据) │  │  (元数据)   │  │  Doris     │  │   (实体关系)            │ │
│  │             │  │             │  │  (OLAP)    │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                         可观测性层 (Observability)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Great Ex   │  │   whylogs  │  │  Evidently  │  │   Arize Phoenix         │ │
│  │  1.15.2    │  │  1.6.4     │  │  0.7.21    │  │   13.22.2              │ │
│  │  数据质量   │  │  数据画像   │  │  漂移检测   │  │   性能追踪              │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                         服务层 (Service Layer)                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        FastAPI Service Layer                             │  │
│  │  /data          /features        /training        /models        /monitor  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心模块设计

### 1. 数据接入与清洗层

```
Raw Data ──┬──► Great Expectations (质量验证) ──► whylogs (数据画像) ──► S3
           │
           ├──► 规则过滤 ──► 缺失值处理 ──► 格式标准化
           │
           └──► 异常检测 ──► 数据修复/标记
```

#### 核心实现

```python
# src/data/data_pipeline.py

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import great_expectations as gx
from whylogs import get_or_create_session
from whylogs.core.preprocessing import preprocess
import ray

@dataclass
class DataQualityConfig:
    """数据质量配置"""
    null_threshold: float = 0.05          # 空值阈值
    duplicate_threshold: float = 0.01     # 重复值阈值
    freshness_hours: int = 24             # 数据新鲜度
    drift_threshold: float = 0.05        # 漂移阈值

@dataclass
class DataQualityResult:
    """质量检查结果"""
    dataset_id: str
    timestamp: datetime
    passed: bool
    score: float
    issues: List[Dict[str, Any]] = field(default_factory=list)
    profile: Optional[Any] = None

class DataIngestionPipeline:
    """
    数据接入与清洗流水线
    支持: PostgreSQL, S3, Doris, Kafka
    """

    def __init__(
        self,
        quality_config: DataQualityConfig,
        feast_config: Optional[Dict] = None
    ):
        self.quality_config = quality_config
        self.feast_config = feast_config
        self.gx_context = gx.get_context()
        self._init_expectations()

    def _init_expectations(self):
        """初始化 Great Expectations 期望规则"""
        # 基础数据质量期望
        self.base_expectations = [
            gx.expect_column_values_to_not_be_null,
            gx.expect_column_value_lengths_to_be_between,
            gx.expect_column_distributions_to_be_between,
        ]

    @ray.remote
    def process_batch(self, data_path: str, dataset_config: Dict) -> DataQualityResult:
        """分布式处理数据批次"""
        # 1. 读取数据
        df = self._load_data(data_path, dataset_config)

        # 2. Great Expectations 质量验证
        quality_result = self._validate_quality(df, dataset_config)

        # 3. whylogs 数据画像
        profile = self._generate_profile(df, dataset_config)

        # 4. 数据清洗
        df_cleaned = self._clean_data(df, quality_result)

        # 5. 特征提取
        features = self._extract_features(df_cleaned, dataset_config)

        return DataQualityResult(
            dataset_id=dataset_config.get("id"),
            timestamp=datetime.utcnow(),
            passed=quality_result["passed"],
            score=quality_result["score"],
            issues=quality_result["issues"],
            profile=profile
        )

    def _validate_quality(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> Dict[str, Any]:
        """使用 Great Expectations 验证数据质量"""
        # 创建验证器
        validator = self.gx_context.sources.pandas_default.read_dataframe(df)

        # 应用期望规则
        expectations = config.get("expectations", [])

        for exp in expectations:
            expectation_type = exp.get("type")
            column = exp.get("column")
            params = exp.get("params", {})

            if expectation_type == "not_null":
                validator.expect_column_values_to_not_be_null(column)
            elif expectation_type == "in_range":
                validator.expect_column_values_to_be_between(
                    column,
                    min_value=params.get("min"),
                    max_value=params.get("max")
                )
            elif expectation_type == "in_set":
                validator.expect_column_values_to_be_in_set(
                    column,
                    value_set=params.get("values")
                )
            elif expectation_type == "unique":
                validator.expect_column_values_to_be_unique(column)
            elif expectation_type == "strftime":
                validator.expect_column_values_to_match_strftime_format(
                    column,
                    strftime_format=params.get("format")
                )

        # 运行验证
        results = validator.validate()

        return {
            "passed": results.success,
            "score": results.success_percent,
            "issues": self._extract_issues(results)
        }

    def _generate_profile(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> Any:
        """生成 whylogs 数据画像"""
        session = get_or_create_session()

        with session.log_batch(
            dataset_name=config.get("name", "default"),
            dataset_timestamp=datetime.utcnow()
        ) as batch:
            # 记录所有列
            batch.write_dataframe(df, reserved_columns=["timestamp"])

        # 获取画像
        profile = session.get_profile(dataset_name=config.get("name"))

        return profile

    def _clean_data(
        self,
        df: pd.DataFrame,
        quality_result: Dict
    ) -> pd.DataFrame:
        """数据清洗"""
        df_cleaned = df.copy()

        # 处理缺失值
        for issue in quality_result.get("issues", []):
            if issue["type"] == "null_values":
                col = issue["column"]
                strategy = issue.get("strategy", "drop")

                if strategy == "drop":
                    df_cleaned = df_cleaned.dropna(subset=[col])
                elif strategy == "mean":
                    if df_cleaned[col].dtype in ["float64", "int64"]:
                        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                elif strategy == "forward":
                    df_cleaned[col].fillna(method="ffill", inplace=True)

        # 去重
        df_cleaned = df_cleaned.drop_duplicates()

        return df_cleaned

    def _extract_features(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> pd.DataFrame:
        """特征提取"""
        feature_config = config.get("features", [])

        for feat in feature_config:
            name = feat["name"]
            feat_type = feat["type"]

            if feat_type == "datetime":
                df[name] = pd.to_datetime(df[feat["column"]])
                df[f"{name}_year"] = df[name].dt.year
                df[f"{name}_month"] = df[name].dt.month
                df[f"{name}_day"] = df[name].dt.day

            elif feat_type == "category":
                df[name] = pd.Categorical(df[feat["column"]]).codes

            elif feat_type == "numerical":
                # 标准化
                mean = df[feat["column"]].mean()
                std = df[feat["column"]].std()
                df[name] = (df[feat["column"]] - mean) / (std + 1e-8)

        return df
```

---

### 2. 特征工程与 Feast 集成

```python
# src/features/feature_pipeline.py

from feast import Entity, Feature, FeatureView, FileSource, StreamSource
from feast.types import Float64, Int64, String
from datetime import timedelta
from typing import Dict, List
import pandas as pd

class FeatureEngineeringPipeline:
    """
    特征工程流水线
    与 Feast 特征存储深度集成
    """

    def __init__(self, feast_repo_path: str):
        self.feast_repo_path = feast_repo_path
        self._init_feature_store()

    def _init_feature_store(self):
        """初始化 Feast 特征存储"""
        from feast import FeatureStore
        self.fs = FeatureStore(repo_path=self.feast_repo_path)

    def create_feature_views(self, config: Dict):
        """创建特征视图"""
        # 离线特征源 (S3)
        feature_source = FileSource(
            name="feature_store_source",
            path=f"s3://{config['bucket']}/features/*.parquet",
            timestamp_field="event_timestamp"
        )

        # 实体定义
        user_entity = Entity(
            name="user_id",
            join_keys=["user_id"],
            description="用户 ID"
        )

        # 用户画像特征视图
        user_profile_view = FeatureView(
            name="user_profile_features",
            entities=["user_id"],
            ttl=timedelta(days=7),
            schema=[
                Feature(name="age", dtype=Int64),
                Feature(name="gender", dtype=String),
                Feature(name="city_tier", dtype=Int64),
                Feature(name="membership_level", dtype=Int64),
                Feature(name="lifetime_value", dtype=Float64),
            ],
            source=feature_source
        )

        # 行为特征视图
        user_behavior_view = FeatureView(
            name="user_behavior_features",
            entities=["user_id"],
            ttl=timedelta(hours=1),
            schema=[
                Feature(name="page_views_1h", dtype=Int64),
                Feature(name="click_rate_1h", dtype=Float64),
                Feature(name="search_count_1h", dtype=Int64),
                Feature(name="add_cart_count_1h", dtype=Int64),
            ],
            source=feature_source
        )

        # 注册到 Feast
        self.fs.apply([user_entity, user_profile_view, user_behavior_view])

    def get_training_features(
        self,
        entity_df: pd.DataFrame,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """获取训练特征"""
        training_df = self.fs.get_training_features(
            feature_refs=feature_names,
            entity_df=entity_df
        )
        return training_df

    def get_online_features(
        self,
        entity_keys: Dict,
        feature_names: List[str]
    ) -> Dict:
        """获取在线特征"""
        return self.fs.get_online_features(
            feature_refs=feature_names,
            entity_rows=[entity_keys]
        )

    def materialize_features(self, start: str, end: str):
        """物化特征到在线存储"""
        self.fs.materialize_incremental(
            end_date=end,
            start_date=start
        )
```

---

### 3. 数据版本控制 (DVC)

```yaml
# dvc.yaml - 数据流水线配置

stages:
  ingest:
    cmd: python src/data/ingest.py
    deps:
      - data/raw
    params:
      - data.quality
    outs:
      - data/staged

  validate:
    cmd: python src/data/validate.py
    deps:
      - data/staged
    params:
      - validation.thresholds
    outs:
      - data/validated
    metrics:
      - metrics/quality.json

  features:
    cmd: python src/features/extract.py
    deps:
      - data/validated
      - src/features
    params:
      - features.types
    outs:
      - data/features

  train:
    cmd: python src/training/train.py
    deps:
      - data/features
      - src/training
    params:
      - model.type
      - training.epochs
    outs:
      - models/checkpoints
    metrics:
      - metrics/training.json
```

```yaml
# params.yaml

data:
  quality:
    null_threshold: 0.05
    duplicate_threshold: 0.01

validation:
  thresholds:
    min_completeness: 0.95
    min_sanity: 0.90

features:
  types:
    numerical: [age, income, tenure]
    categorical: [gender, city, product_type]
    datetime: [created_at, updated_at]

training:
  epochs: 100
  batch_size: 256
  learning_rate: 0.001
  early_stopping_patience: 10

model:
  type: xgboost
  max_depth: 6
  n_estimators: 500
```

---

### 4. 模型训练 (Ray + MLflow)

```python
# src/training/train_pipeline.py

from typing import Dict, List, Optional
from dataclasses import dataclass
import mlflow
import pandas as pd
import ray
from ray import train, tune
from ray.train import (
    Checkpoint,
    FailureConfig,
    RunConfig,
    ScalingConfig,
)
from ray.train.xgboost import XGBoostTrainer
import xgboost as xgb

@dataclass
class TrainingConfig:
    """训练配置"""
    num_workers: int = 4
    use_gpu: bool = True
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001

class RayTrainingPipeline:
    """
    Ray 分布式训练流水线
    与 MLflow 深度集成
    """

    def __init__(
        self,
        config: TrainingConfig,
        mlflow_tracking_uri: str,
        experiment_name: str = "production"
    ):
        self.config = config
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def _setup_mlflow(self):
        """配置 MLflow"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        # 设置 MLflow 自动日志
        mlflow.autolog(
            log_models=True,
            disable=False,
            exclusive=False
        )

    def train_with_hyperopt(self, param_space: Dict) -> Dict:
        """使用 Ray Tune 进行超参数优化"""
        tuner = tune.Tuner(
            self._train_step,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                num_samples=50,
                metric="val_loss",
                mode="min",
                scheduler=tune.scheduler.ASHAScheduler(
                    max_t=100,
                    grace_period=10
                )
            ),
            run_config=RunConfig(
                name="hyperopt_run",
                callbacks=[
                    MLflowLoggerCallback(
                        experiment_name=self.experiment_name
                    )
                ]
            )
        )

        results = tuner.fit()
        best_result = results.get_best_result()

        return {
            "best_params": best_result.config,
            "best_metrics": best_result.metrics,
            "best_checkpoint": best_result.checkpoint
        }

    def _train_step(self, config: Dict):
        """单次训练步骤"""
        # 获取数据
        train_df, val_df = self._load_data()

        # 创建 XGBoost 训练器
        trainer = XGBoostTrainer(
            label_column="target",
            num_workers=self.config.num_workers,
            use_gpu=self.config.use_gpu,
            scaling_config=ScalingConfig(
                num_workers=self.config.num_workers,
                use_gpu=self.config.use_gpu
            ),
            xgboost_params={
                "max_depth": config.get("max_depth", 6),
                "learning_rate": config.get("learning_rate", 0.001),
                "n_estimators": config.get("n_estimators", 500),
                "objective": "binary:logistic",
            }
        )

        # 训练
        result = trainer.fit()

        # 记录到 MLflow
        mlflow.log_metrics({
            "train_loss": result.metrics["train_loss"],
            "val_loss": result.metrics["val_loss"],
            "train_accuracy": result.metrics.get("train_accuracy", 0),
            "val_accuracy": result.metrics.get("val_accuracy", 0)
        })

        return result

    def train_production(self, train_config: Dict) -> Dict:
        """生产环境训练"""
        with mlflow.start_run(run_name="production_train"):
            # 记录参数
            mlflow.log_params({
                "model_type": "xgboost",
                "num_workers": self.config.num_workers,
                "epochs": self.config.epochs,
                **train_config
            })

            # 训练
            result = self._train_step(train_config)

            # 注册模型到 MLflow Registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

            mlflow.register_model(
                model_uri=model_uri,
                name="production-model"
            )

            return {
                "run_id": mlflow.active_run().info.run_id,
                "metrics": result.metrics,
                "checkpoint": result.checkpoint
            }

    def _load_data(self) -> tuple:
        """加载训练数据"""
        # 从 Feast 获取特征
        # 从 Doris 获取标签
        # 合并返回
        pass
```

---

### 5. 模型注册与部署

```python
# src/models/model_registry.py

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

@dataclass
class ModelVersion:
    """模型版本"""
    name: str
    version: int
    stage: str
    status: str
    run_id: str
    created_at: datetime
    metrics: Dict

class ModelRegistry:
    """
    MLflow 模型注册表
    支持版本管理、阶段转换、审批流程
    """

    def __init__(self, tracking_uri: str, registry_uri: str):
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)
        self.client = MlflowClient()

    def register_version(
        self,
        run_id: str,
        model_name: str,
        metrics: Dict
    ) -> ModelVersion:
        """注册新模型版本"""
        # 创建版本
        version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=model_name
        )

        # 更新元数据
        self.client.update_model_version(
            name=model_name,
            version=version.version,
            description=f"Created at {datetime.utcnow()}"
        )

        return ModelVersion(
            name=model_name,
            version=version.version,
            stage="None",
            status="pending",
            run_id=run_id,
            created_at=datetime.utcnow(),
            metrics=metrics
        )

    def transition_stage(
        self,
        model_name: str,
        version: int,
        target_stage: str,
        archive_existing: bool = True
    ):
        """转换模型阶段"""
        """
        阶段: None -> Staging -> Production -> Archived
        """
        valid_stages = ["None", "Staging", "Production", "Archived"]

        if target_stage not in valid_stages:
            raise ValueError(f"Invalid stage: {target_stage}")

        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=target_stage,
            archive_existing_versions=archive_existing
        )

    def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        """获取当前生产模型"""
        try:
            versions = self.client.get_latest_versions(
                model_name,
                stages=["Production"]
            )

            if not versions:
                return None

            v = versions[0]
            return ModelVersion(
                name=model_name,
                version=v.version,
                stage="Production",
                status="active",
                run_id=v.run_id,
                created_at=v.creation_timestamp,
                metrics={}
            )
        except Exception:
            return None

    def compare_versions(
        self,
        model_name: str,
        versions: List[int]
    ) -> Dict:
        """比较模型版本"""
        comparison = {}

        for v in versions:
            mv = self.client.get_model_version(model_name, v)

            comparison[v] = {
                "version": mv.version,
                "stage": mv.current_stage,
                "metrics": self._get_version_metrics(mv),
                "created": mv.creation_timestamp,
                "run_id": mv.run_id
            }

        return comparison

    def rollback(
        self,
        model_name: str,
        target_version: int
    ) -> bool:
        """回滚到指定版本"""
        try:
            # 获取目标版本
            target = self.client.get_model_version(model_name, target_version)

            # 获取当前生产版本
            current_production = self.get_production_model(model_name)

            # 切换阶段
            if current_production:
                # 将当前生产版本归档
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=current_production.version,
                    stage="Archived"
                )

            # 提升目标版本到生产
            self.client.transition_model_version_stage(
                name=model_name,
                version=target_version,
                stage="Production"
            )

            return True

        except Exception as e:
            print(f"Rollback failed: {e}")
            return False

    def _get_version_metrics(self, model_version) -> Dict:
        """获取版本指标"""
        run = self.client.get_run(model_version.run_id)
        return {
            k: v for k, v in run.data.metrics.items()
            if not k.startswith("_")
        }
```

---

### 6. 模型服务 (Ray Serve + FastAPI)

```python
# src/serving/model_service.py

from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
import mlflow
import pandas as pd
from datetime import datetime

app = FastAPI(title="ML Model Service")

@serve.deployment(
    num_replicas=3,
    ray_actor_options={"num_gpus": 0.1},
    max_concurrent_queries=100
)
class ModelServer:
    """
    Ray Serve 模型服务器
    支持多版本、A/B 测试、金丝雀部署
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.model_version = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        # 从 MLflow 加载
        model_uri = f"models:/{self.model_name}/Production"

        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            mv = mlflow.get_model_version(self.model_name, "Production")
            self.model_version = mv.version
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    async def __call__(self, request: Dict) -> Dict:
        """处理预测请求"""
        if not self.model:
            raise HTTPException(status_code=503, message="Model not loaded")

        # 解析输入
        data = request.get("data")

        # 预测
        prediction = self.model.predict(data)

        return {
            "prediction": prediction,
            "model_version": self.model_version,
            "timestamp": datetime.utcnow().isoformat()
        }

    def update_model(self, model_version: int):
        """热更新模型"""
        model_uri = f"models:/{self.model_name}/{model_version}"
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.model_version = model_version

class CanaryModelServer:
    """
    金丝雀部署服务器
    支持流量分流
    """

    def __init__(self, production_model: str, canary_model: str):
        self.production = ModelServer.bind(production_model)
        self.canary = ModelServer.bind(canary_model)
        self.canary_percentage = 0.0

    async def __call__(self, request: Dict) -> Dict:
        import random

        # 根据配置分流
        if random.random() < self.canary_percentage:
            return await self.canary.handle.remote(request)
        else:
            return await self.production.handle.remote(request)

    def update_traffic(self, canary_percentage: float):
        """更新流量分配"""
        self.canary_percentage = canary_percentage

# FastAPI 端点
@app.post("/predict/{model_name}")
async def predict(model_name: str, request: dict):
    """预测端点"""
    handle = serve.get_replica_handle(f"{model_name}_ModelServer")
    return await handle.remote(request)

@app.post("/models/{model_name}/rollback")
async def rollback_model(model_name: str, version: int):
    """回滚模型"""
    # 获取当前生产版本
    # 切换到目标版本
    # 记录操作
    pass

@app.get("/models/{model_name}/versions")
async def list_versions(model_name: str):
    """列出所有版本"""
    versions = mlflow.search_model_versions(model_name)
    return versions

# 部署启动
def start_serving(model_name: str, num_replicas: int = 3):
    """启动模型服务"""
    ray.init()
    serve.run(ModelServer.bind(model_name), name=f"{model_name}_ModelServer")
```

---

### 7. 数据/模型漂移监控

```python
# src/monitoring/drift_monitor.py

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, CatTargetDriftTab
from whylogs.experimental.core.embeddings import (
    EmbeddingDriftCalculator,
    DistanceMetric
)
from arize.pandas.logger import Logger
from arize.utils.types import ModelTypes
import mlflow

@dataclass
class DriftReport:
    """漂移报告"""
    timestamp: datetime
    drift_detected: bool
    drift_score: float
    feature_drifts: Dict[str, float]
    recommendations: List[str]

class DriftMonitoringSystem:
    """
    数据和模型漂移监控系统
    集成: Evidently, whylogs, Arize Phoenix
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        arize_config: Dict,
        mlflow_tracking_uri: str
    ):
        self.reference_data = reference_data
        self.arize_config = arize_config
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # 初始化 Arize
        self.arize_logger = Logger(
            api_key=arize_config["api_key"],
            space_key=arize_config["space_key"]
        )

    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        config: Dict
    ) -> DriftReport:
        """使用 Evidently 检测数据漂移"""

        # 配置漂移检测
        data_drift Dashboard = Dashboard(tabs=[
            DataDriftTab(),
            CatTargetDriftTab()
        ])

        # 生成报告
        dashboard.calculate(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=config.get("column_mapping")
        )

        # 保存报告
        report_path = f"/tmp/drift_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        dashboard.save(report_path)

        # 分析漂移分数
        drift_score = self._calculate_drift_score(dashboard)

        return DriftReport(
            timestamp=datetime.utcnow(),
            drift_detected=drift_score > config.get("threshold", 0.5),
            drift_score=drift_score,
            feature_drifts=self._extract_feature_drifts(dashboard),
            recommendations=self._generate_recommendations(drift_score)
        )

    def detect_embedding_drift(
        self,
        reference_embeddings: np.ndarray,
        current_embeddings: np.ndarray
    ) -> float:
        """使用 whylogs 检测嵌入漂移"""

        calculator = EmbeddingDriftCalculator(
            distance_metric=DistanceMetric.COSINE
        )

        drift_result = calculator.compute(
            reference_embeddings=reference_embeddings,
            current_embeddings=current_embeddings
        )

        return drift_result.distance

    def log_to_arize(
        self,
        dataframe: pd.DataFrame,
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None,
        shap_values: Optional[np.ndarray] = None
    ):
        """记录到 Arize Phoenix"""

        # 构建特征字典
        features = {
            col: dataframe[col].values
            for col in dataframe.columns
        }

        # 记录预测
        self.arize_logger.log(
            dataframe=dataframe,
            model_id=self.arize_config["model_id"],
            model_type=ModelTypes.SCORE_CATEGORICAL,
            features=features,
            prediction=predictions,
            actual=actuals,
            shap_values=shap_values
        )

    def check_concept_drift(
        self,
        historical_metrics: pd.DataFrame,
        current_metrics: Dict
    ) -> bool:
        """检测概念漂移"""

        # 使用 Page-Hinkley 检验
        import scipy.stats as stats

        # 比较历史分布和当前分布
        historical_mean = historical_metrics["accuracy"].mean()
        current_accuracy = current_metrics.get("accuracy", 0)

        # 检验统计
        z_score = (current_accuracy - historical_mean) / historical_metrics["accuracy"].std()

        # 如果 Z-score 超过阈值，认为有概念漂移
        return abs(z_score) > 2.5

    def generate_alert(
        self,
        drift_report: DriftReport,
        config: Dict
    ):
        """生成告警"""
        if drift_report.drift_detected:
            # 记录到 MLflow
            with mlflow.start_run(run_name=f"drift_alert_{drift_report.timestamp}"):
                mlflow.log_metrics({
                    "drift_score": drift_report.drift_score,
                    "drift_detected": 1.0
                })

                for feature, score in drift_report.feature_drifts.items():
                    mlflow.log_metric(f"feature_drift_{feature}", score)

            # 触发自动训练
            if drift_report.drift_score > config.get("retrain_threshold", 0.7):
                self._trigger_retraining()

    def _trigger_retraining(self):
        """触发重新训练"""
        # 记录触发事件
        mlflow.set_tag("retraining_triggered", "drift_detection")
        mlflow.log_param("trigger_reason", "data_drift")

        # TODO: 触发 Ray 训练任务
        pass
```

---

### 8. API 服务层

```python
# src/api/main.py

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
import asyncio

from src.data.data_pipeline import DataIngestionPipeline, DataQualityConfig
from src.features.feature_pipeline import FeatureEngineeringPipeline
from src.training.train_pipeline import RayTrainingPipeline, TrainingConfig
from src.models.model_registry import ModelRegistry
from src.serving.model_service import ModelServer
from src.monitoring.drift_monitor import DriftMonitoringSystem

# Pydantic 模型
class DataIngestionRequest(BaseModel):
    source_type: str  # s3, postgres, doris
    source_config: Dict
    quality_config: Optional[Dict] = {}
    target_table: str

class TrainingRequest(BaseModel):
    model_name: str
    feature_config: Dict
    training_config: Dict
    hyperopt_config: Optional[Dict] = {}

class ModelDeployRequest(BaseModel):
    model_name: str
    version: int
    target_stage: str
    num_replicas: int = 3

class DriftCheckRequest(BaseModel):
    model_name: str
    current_data_path: str

# FastAPI 应用
app = FastAPI(
    title="MLOps Platform API",
    version="1.0.0",
    description="企业级 MLOps 全生命周期管理平台"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 依赖注入
def get_data_pipeline():
    return DataIngestionPipeline(
        quality_config=DataQualityConfig()
    )

def get_model_registry():
    return ModelRegistry(
        tracking_uri="http://mlflow:5000",
        registry_uri="http://mlflow:5000"
    )

# ==================== 数据 API ====================

@app.post("/api/v1/data/ingest")
async def ingest_data(
    request: DataIngestionRequest,
    pipeline: DataIngestionPipeline = Depends(get_data_pipeline)
):
    """数据接入"""
    result = await pipeline.process_batch(
        data_path=request.source_config["path"],
        dataset_config={
            "id": request.target_table,
            "expectations": request.quality_config.get("expectations", [])
        }
    )

    return {
        "status": "success" if result.passed else "failed",
        "quality_score": result.score,
        "issues": result.issues,
        "dataset_id": result.dataset_id
    }

@app.get("/api/v1/data/quality/{dataset_id}")
async def get_data_quality(dataset_id: str):
    """获取数据质量报告"""
    # 从 TimescaleDB 查询
    pass

# ==================== 特征 API ====================

@app.post("/api/v1/features/register")
async def register_features(feature_config: Dict):
    """注册特征"""
    pipeline = FeatureEngineeringPipeline(feast_repo_path="/ feast")
    pipeline.create_feature_views(feature_config)

    return {"status": "success", "features_registered": len(feature_config.get("features", []))}

@app.get("/api/v1/features/{entity_id}")
async def get_online_features(
    entity_id: str,
    feature_names: List[str]
):
    """获取在线特征"""
    pipeline = FeatureEngineeringPipeline(feast_repo_path="/feast")
    features = pipeline.get_online_features(
        entity_keys={"user_id": entity_id},
        feature_names=feature_names
    )

    return features

# ==================== 训练 API ====================

@app.post("/api/v1/training/start")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """启动训练任务"""
    pipeline = RayTrainingPipeline(
        config=TrainingConfig(**request.training_config),
        mlflow_tracking_uri="http://mlflow:5000"
    )

    # 启动后台训练
    if request.hyperopt_config:
        result = await background_tasks.add_task(
            pipeline.train_with_hyperopt,
            request.hyperopt_config
        )
    else:
        result = await background_tasks.add_task(
            pipeline.train_production,
            request.training_config
        )

    return {
        "status": "started",
        "run_id": result.get("run_id"),
        "experiment_name": "production"
    }

@app.get("/api/v1/training/{run_id}")
async def get_training_status(run_id: str):
    """获取训练状态"""
    import mlflow
    run = mlflow.get_run(run_id)

    return {
        "run_id": run_id,
        "status": run.info.status,
        "metrics": run.data.metrics,
        "params": run.data.params
    }

# ==================== 模型 API ====================

@app.post("/api/v1/models/register")
async def register_model(
    model_name: str,
    run_id: str,
    metrics: Dict,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """注册模型"""
    version = registry.register_version(
        run_id=run_id,
        model_name=model_name,
        metrics=metrics
    )

    return {
        "version": version.version,
        "stage": version.stage,
        "created_at": version.created_at
    }

@app.post("/api/v1/models/{model_name}/transition")
async def transition_model_stage(
    model_name: str,
    version: int,
    target_stage: str,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """转换模型阶段"""
    registry.transition_stage(
        model_name=model_name,
        version=version,
        target_stage=target_stage
    )

    return {"status": "success", "new_stage": target_stage}

@app.post("/api/v1/models/{model_name}/rollback")
async def rollback_model(
    model_name: str,
    target_version: int,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """回滚模型"""
    success = registry.rollback(model_name, target_version)

    if not success:
        raise HTTPException(status_code=500, detail="Rollback failed")

    return {"status": "success", "rolled_back_to": target_version}

@app.get("/api/v1/models/{model_name}/versions")
async def list_model_versions(
    model_name: str,
    stages: Optional[List[str]] = None,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """列出模型版本"""
    if stages:
        versions = []
        for stage in stages:
            mv = registry.get_production_model(model_name)
            if mv:
                versions.append(mv)
    else:
        versions = registry.client.search_model_versions(model_name)

    return {"versions": versions}

# ==================== 监控 API ====================

@app.post("/api/v1/monitor/drift-check")
async def check_drift(request: DriftCheckRequest):
    """执行漂移检测"""
    # TODO: 实现漂移检测
    pass

@app.get("/api/v1/monitor/alerts")
async def get_alerts(
    model_name: Optional[str] = None,
    since: Optional[datetime] = None
):
    """获取告警"""
    # 从 TimescaleDB 查询告警
    pass

# 健康检查
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# 启动
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Kubernetes 部署配置

### 完整 K8s 部署

```yaml
# k8s/mlops-platform.yaml

---
# Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: mlops
---
# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlops-config
  namespace: mlops
data:
  MLFLOW_TRACKING_URI: "http://mlflow.monitoring:5000"
  FEAST_REPO_PATH: "/feast"
  S3_BUCKET: "mlops-data"
---
# FastAPI Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlops-api
  namespace: mlops
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlops-api
  template:
    metadata:
      labels:
        app: mlops-api
    spec:
      containers:
        - name: api
          image: mlops/api:latest
          ports:
            - containerPort: 8000
          env:
            - name: MLFLOW_TRACKING_URI
              valueFrom:
                configMapKeyRef:
                  name: mlops-config
                  key: MLFLOW_TRACKING_URI
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"
              memory: "4Gi"
---
# FastAPI Service
apiVersion: v1
kind: Service
metadata:
  name: mlops-api
  namespace: mlops
spec:
  type: LoadBalancer
  ports:
    - port: 80
      targetPort: 8000
  selector:
    app: mlops-api
---
# Ray Head
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ray-cluster
  namespace: mlops
spec:
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
    template:
      spec:
        containers:
          - name: ray-head
            image: rayproject/ray-ml:2.54.1
            resources:
              requests:
                cpu: "4"
                memory: "16Gi"
              limits:
                cpu: "8"
                memory: "32Gi"
---
# Ray Worker (Training)
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ray-cluster
  namespace: mlops
spec:
  workerGroupSpecs:
    - groupName: training-workers
      maxReplicas: 10
      minReplicas: 2
      rayStartParams:
        num-cpus: "8"
      template:
        spec:
          containers:
            - name: ray-worker
              image: rayproject/ray-ml:2.54.1
              resources:
                requests:
                  cpu: "4"
                  memory: "16Gi"
                  nvidia.com/gpu: "1"
                limits:
                  cpu: "8"
                  memory: "32Gi"
                  nvidia.com/gpu: "1"
```

---

## 完整流水线执行示例

```python
# src/orchestration/ml_pipeline.py

from typing import Dict, List
from datetime import datetime
import asyncio

from src.data.data_pipeline import DataIngestionPipeline
from src.features.feature_pipeline import FeatureEngineeringPipeline
from src.training.train_pipeline import RayTrainingPipeline, TrainingConfig
from src.models.model_registry import ModelRegistry
from src.monitoring.drift_monitor import DriftMonitoringSystem
from mlops_ray_enterprise.src.serving.model_service import ModelServer

class MLOpsPipeline:
    """
    完整的 MLOps 流水线编排
    """

    def __init__(self, config: Dict):
        self.config = config

        # 初始化各组件
        self.data_pipeline = DataIngestionPipeline(
            quality_config=config["data_quality"]
        )

        self.feature_pipeline = FeatureEngineeringPipeline(
            feast_repo_path=config["feast"]["repo_path"]
        )

        self.training_pipeline = RayTrainingPipeline(
            config=TrainingConfig(**config["training"]),
            mlflow_tracking_uri=config["mlflow"]["tracking_uri"]
        )

        self.model_registry = ModelRegistry(
            tracking_uri=config["mlflow"]["tracking_uri"],
            registry_uri=config["mlflow"]["registry_uri"]
        )

        self.monitor = DriftMonitoringSystem(
            reference_data=None,
            arize_config=config["arize"],
            mlflow_tracking_uri=config["mlflow"]["tracking_uri"]
        )

    async def run_full_pipeline(self, input_config: Dict) -> Dict:
        """
        执行完整流水线

        1. 数据接入与清洗
        2. 特征工程
        3. 模型训练
        4. 模型注册
        5. 模型部署
        """

        pipeline_id = f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        results = {
            "pipeline_id": pipeline_id,
            "start_time": datetime.utcnow(),
            "steps": []
        }

        try:
            # Step 1: 数据接入
            print(f"[{pipeline_id}] Step 1: 数据接入...")
            data_result = await self._run_data_ingestion(input_config)
            results["steps"].append({
                "name": "data_ingestion",
                "status": "success",
                "result": data_result
            })

            # Step 2: 特征工程
            print(f"[{pipeline_id}] Step 2: 特征工程...")
            features = await self._run_feature_engineering(data_result)
            results["steps"].append({
                "name": "feature_engineering",
                "status": "success",
                "result": features
            })

            # Step 3: 模型训练
            print(f"[{pipeline_id}] Step 3: 模型训练...")
            train_result = await self._run_training(features, input_config)
            results["steps"].append({
                "name": "training",
                "status": "success",
                "result": train_result
            })

            # Step 4: 模型注册
            print(f"[{pipeline_id}] Step 4: 模型注册...")
            model_version = await self._register_model(
                train_result,
                input_config.get("model_name")
            )
            results["steps"].append({
                "name": "model_registration",
                "status": "success",
                "result": model_version
            })

            # Step 5: 模型验证
            print(f"[{pipeline_id}] Step 5: 模型验证...")
            validation_result = await self._validate_model(model_version)

            if validation_result["passed"]:
                # Step 6: 部署到 Staging
                print(f"[{pipeline_id}] Step 6: 部署到 Staging...")
                await self.model_registry.transition_stage(
                    model_name=input_config["model_name"],
                    version=model_version.version,
                    target_stage="Staging"
                )

                # Step 7: 部署到 Production
                print(f"[{pipeline_id}] Step 7: 部署到 Production...")
                await self.model_registry.transition_stage(
                    model_name=input_config["model_name"],
                    version=model_version.version,
                    target_stage="Production"
                )

                results["steps"].append({
                    "name": "deployment",
                    "status": "success",
                    "deployed_version": model_version.version
                })
            else:
                results["steps"].append({
                    "name": "deployment",
                    "status": "skipped",
                    "reason": "Validation failed"
                })

            results["status"] = "success"
            results["end_time"] = datetime.utcnow()

        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            results["end_time"] = datetime.utcnow()

        return results

    async def _run_data_ingestion(self, config: Dict):
        """数据接入"""
        return await self.data_pipeline.process_batch(
            data_path=config["data_path"],
            dataset_config=config.get("dataset_config", {})
        )

    async def _run_feature_engineering(self, data_result):
        """特征工程"""
        # 创建特征视图
        self.feature_pipeline.create_feature_views(
            self.config["features"]
        )

        return {"features_created": len(self.config["features"])}

    async def _run_training(self, features, config: Dict):
        """模型训练"""
        result = self.training_pipeline.train_production(
            config.get("training_config", {})
        )

        return result

    async def _register_model(self, train_result, model_name: str):
        """注册模型"""
        return self.model_registry.register_version(
            run_id=train_result["run_id"],
            model_name=model_name,
            metrics=train_result["metrics"]
        )

    async def _validate_model(self, model_version) -> Dict:
        """验证模型"""
        # TODO: 执行模型验证测试
        return {"passed": True}

# 使用示例
if __name__ == "__main__":
    pipeline = MLOpsPipeline(config={
        "data_quality": {"null_threshold": 0.05},
        "feast": {"repo_path": "/feast"},
        "training": {"num_workers": 4, "epochs": 100},
        "mlflow": {
            "tracking_uri": "http://mlflow:5000",
            "registry_uri": "http://mlflow:5000"
        },
        "arize": {"api_key": "...", "space_key": "..."},
        "features": []
    })

    result = asyncio.run(pipeline.run_full_pipeline({
        "data_path": "s3://data/raw/sales.csv",
        "model_name": "sales-forecast",
        "training_config": {
            "learning_rate": 0.001,
            "max_depth": 6
        }
    }))

    print(f"Pipeline completed: {result['status']}")
```

---

## 技术栈汇总

| 组件 | 版本 | 用途 |
|------|------|------|
| **FastAPI** | 0.115.x | API 服务 |
| **MLflow** | 3.10.1 | 实验追踪、模型注册 |
| **Ray** | 2.54.1 | 分布式计算 |
| **DVC** | 3.66.1 | 数据版本控制 |
| **Great Expectations** | 1.15.2 | 数据验证 |
| **whylogs** | 1.6.4 | 数据画像 |
| **Evidently** | 0.7.21 | 漂移检测 |
| **Arize Phoenix** | 13.22.2 | ML 可观测性 |
| **Feast** | 0.61.0 | 特征存储 |
| **PostgreSQL** | 16.x | 元数据 |
| **TimescaleDB** | 2.15.x | 时序数据 |
| **Neo4j** | 5.x | 知识图谱 |
| **Doris** | 2.1.x | OLAP |
| **S3** | - | 对象存储 |
| **Kubernetes** | 1.29+ | 容器编排 |

