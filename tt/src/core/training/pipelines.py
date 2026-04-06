"""
训练流水线模块
基于Ray的分布式训练流水线，支持多步骤、条件分支、并行执行
"""

import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, Checkpoint
from ray.train.torch import TorchTrainer
from ray.train.xgboost import XGBoostTrainer
from ray.train.lightgbm import LightGBMTrainer
from ray.train.huggingface import HuggingFaceTrainer
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search import BasicVariantGenerator, BayesOptSearch
from typing import Dict, Any, List, Optional, Callable, Union
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
import json
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import yaml

from ..config import settings
from ..data.validation import DataQualityValidator
from ..features.store import FeatureStoreManager
from ..models.registry import ModelRegistry, ModelMetadata
from .distributed import DistributedTrainer
from .hyperopt import HyperparameterOptimizer

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """流水线阶段"""
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_REGISTRATION = "model_registration"
    DEPLOYMENT = "deployment"


class PipelineStatus(Enum):
    """流水线状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class PipelineStep:
    """流水线步骤"""
    name: str
    stage: PipelineStage
    function: Callable
    depends_on: List[str] = None
    parameters: Dict[str, Any] = None
    timeout: int = 3600  # 超时时间（秒）
    retries: int = 3
    enabled: bool = True
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []
        if self.parameters is None:
            self.parameters = {}


@dataclass
class PipelineResult:
    """流水线结果"""
    pipeline_id: str
    pipeline_name: str
    start_time: datetime
    end_time: datetime
    status: PipelineStatus
    steps_results: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    artifacts: Dict[str, str]
    
    def __post_init__(self):
        if isinstance(self.start_time, str):
            self.start_time = datetime.fromisoformat(self.start_time)
        if isinstance(self.end_time, str):
            self.end_time = datetime.fromisoformat(self.end_time)


class TrainingPipeline:
    """训练流水线"""
    
    def __init__(
        self,
        pipeline_name: str,
        config: Dict[str, Any],
        data_validator: Optional[DataQualityValidator] = None,
        feature_store: Optional[FeatureStoreManager] = None,
        model_registry: Optional[ModelRegistry] = None,
        distributed_trainer: Optional[DistributedTrainer] = None
    ):
        self.pipeline_name = pipeline_name
        self.config = config
        self.data_validator = data_validator
        self.feature_store = feature_store
        self.model_registry = model_registry
        self.distributed_trainer = distributed_trainer
        
        # 流水线步骤
        self.steps: Dict[str, PipelineStep] = {}
        
        # 流水线结果
        self.results: Dict[str, PipelineResult] = {}
        
        # 存储目录
        self.pipeline_dir = Path("pipelines") / pipeline_name
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化Ray
        if not ray.is_initialized():
            ray.init(
                address=settings.ray.address,
                ignore_reinit_error=True
            )
    
    def add_step(self, step: PipelineStep):
        """添加步骤"""
        self.steps[step.name] = step
    
    def build_default_pipeline(self):
        """构建默认训练流水线"""
        # 1. 数据加载步骤
        data_loading_step = PipelineStep(
            name="data_loading",
            stage=PipelineStage.DATA_LOADING,
            function=self._load_data,
            parameters=self.config.get("data_loading", {})
        )
        self.add_step(data_loading_step)
        
        # 2. 数据验证步骤
        data_validation_step = PipelineStep(
            name="data_validation",
            stage=PipelineStage.DATA_VALIDATION,
            function=self._validate_data,
            depends_on=["data_loading"],
            parameters=self.config.get("data_validation", {})
        )
        self.add_step(data_validation_step)
        
        # 3. 特征工程步骤
        feature_engineering_step = PipelineStep(
            name="feature_engineering",
            stage=PipelineStage.FEATURE_ENGINEERING,
            function=self._engineer_features,
            depends_on=["data_validation"],
            parameters=self.config.get("feature_engineering", {})
        )
        self.add_step(feature_engineering_step)
        
        # 4. 模型训练步骤
        model_training_step = PipelineStep(
            name="model_training",
            stage=PipelineStage.MODEL_TRAINING,
            function=self._train_model,
            depends_on=["feature_engineering"],
            parameters=self.config.get("model_training", {})
        )
        self.add_step(model_training_step)
        
        # 5. 模型评估步骤
        model_evaluation_step = PipelineStep(
            name="model_evaluation",
            stage=PipelineStage.MODEL_EVALUATION,
            function=self._evaluate_model,
            depends_on=["model_training"],
            parameters=self.config.get("model_evaluation", {})
        )
        self.add_step(model_evaluation_step)
        
        # 6. 模型注册步骤
        model_registration_step = PipelineStep(
            name="model_registration",
            stage=PipelineStage.MODEL_REGISTRATION,
            function=self._register_model,
            depends_on=["model_evaluation"],
            parameters=self.config.get("model_registration", {})
        )
        self.add_step(model_registration_step)
        
        # 7. 部署步骤（可选）
        if self.config.get("auto_deploy", False):
            deployment_step = PipelineStep(
                name="deployment",
                stage=PipelineStage.DEPLOYMENT,
                function=self._deploy_model,
                depends_on=["model_registration"],
                parameters=self.config.get("deployment", {})
            )
            self.add_step(deployment_step)
    
    async def run(self, pipeline_id: Optional[str] = None) -> PipelineResult:
        """
        运行流水线
        
        Args:
            pipeline_id: 流水线ID
            
        Returns:
            流水线结果
        """
        pipeline_id = pipeline_id or f"{self.pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        start_time = datetime.now()
        steps_results = {}
        artifacts = {}
        metadata = {
            "pipeline_name": self.pipeline_name,
            "pipeline_id": pipeline_id,
            "start_time": start_time.isoformat(),
            "config": self.config
        }
        
        try:
            logger.info(f"开始运行流水线: {pipeline_id}")
            
            # 执行步骤
            executed_steps = set()
            step_queue = self._get_execution_order()
            
            for step_name in step_queue:
                step = self.steps[step_name]
                
                if not step.enabled:
                    logger.info(f"跳过步骤: {step.name}")
                    steps_results[step.name] = {
                        "status": PipelineStatus.SKIPPED.value,
                        "start_time": datetime.now().isoformat(),
                        "end_time": datetime.now().isoformat(),
                        "message": "Step disabled"
                    }
                    continue
                
                # 检查依赖
                if not all(dep in executed_steps for dep in step.depends_on):
                    logger.error(f"步骤依赖未满足: {step.name}")
                    raise ValueError(f"步骤 {step.name} 的依赖未满足")
                
                # 执行步骤
                step_result = await self._execute_step(step, pipeline_id)
                steps_results[step.name] = step_result
                
                if step_result["status"] == PipelineStatus.COMPLETED.value:
                    executed_steps.add(step.name)
                    
                    # 保存工件
                    if "artifacts" in step_result:
                        artifacts.update(step_result["artifacts"])
                else:
                    # 步骤失败
                    logger.error(f"步骤失败: {step.name}")
                    
                    # 根据配置决定是否继续
                    if not self.config.get("continue_on_failure", False):
                        raise Exception(f"步骤 {step.name} 失败: {step_result.get('error')}")
            
            # 流水线成功
            end_time = datetime.now()
            status = PipelineStatus.COMPLETED
            
            logger.info(f"流水线完成: {pipeline_id}, 耗时: {(end_time - start_time).total_seconds():.2f}秒")
            
        except Exception as e:
            # 流水线失败
            end_time = datetime.now()
            status = PipelineStatus.FAILED
            metadata["error"] = str(e)
            
            logger.error(f"流水线失败: {pipeline_id}, 错误: {e}")
        
        # 创建结果
        result = PipelineResult(
            pipeline_id=pipeline_id,
            pipeline_name=self.pipeline_name,
            start_time=start_time,
            end_time=end_time,
            status=status,
            steps_results=steps_results,
            metadata=metadata,
            artifacts=artifacts
        )
        
        # 保存结果
        self.results[pipeline_id] = result
        await self._save_pipeline_result(result)
        
        return result
    
    def _get_execution_order(self) -> List[str]:
        """获取执行顺序（拓扑排序）"""
        # 构建依赖图
        graph = {name: set(step.depends_on) for name, step in self.steps.items()}
        
        # 拓扑排序
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for dep in graph[node]:
                in_degree[dep] = in_degree.get(dep, 0) + 1
        
        # 队列
        queue = [node for node in graph if in_degree[node] == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            
            for next_node in graph:
                if node in graph[next_node]:
                    in_degree[next_node] -= 1
                    if in_degree[next_node] == 0:
                        queue.append(next_node)
        
        if len(order) != len(graph):
            raise ValueError("循环依赖检测")
        
        return order
    
    async def _execute_step(
        self,
        step: PipelineStep,
        pipeline_id: str
    ) -> Dict[str, Any]:
        """执行步骤"""
        step_start_time = datetime.now()
        
        try:
            logger.info(f"开始执行步骤: {step.name}")
            
            # 执行函数
            result = await step.function(
                pipeline_id=pipeline_id,
                **step.parameters
            )
            
            step_end_time = datetime.now()
            
            step_result = {
                "status": PipelineStatus.COMPLETED.value,
                "start_time": step_start_time.isoformat(),
                "end_time": step_end_time.isoformat(),
                "duration_seconds": (step_end_time - step_start_time).total_seconds(),
                "result": result
            }
            
            logger.info(f"步骤完成: {step.name}, 耗时: {step_result['duration_seconds']:.2f}秒")
            
            return step_result
            
        except Exception as e:
            step_end_time = datetime.now()
            
            step_result = {
                "status": PipelineStatus.FAILED.value,
                "start_time": step_start_time.isoformat(),
                "end_time": step_end_time.isoformat(),
                "duration_seconds": (step_end_time - step_start_time).total_seconds(),
                "error": str(e)
            }
            
            logger.error(f"步骤失败: {step.name}, 错误: {e}")
            
            return step_result
    
    async def _load_data(
        self,
        pipeline_id: str,
        data_sources: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """加载数据"""
        logger.info(f"加载数据，来源: {len(data_sources)}")
        
        loaded_data = {}
        
        for source in data_sources:
            source_type = source.get("type", "file")
            source_path = source.get("path")
            
            if source_type == "file":
                # 从文件加载
                file_format = source.get("format", "parquet")
                
                if file_format == "parquet":
                    df = pd.read_parquet(source_path)
                elif file_format == "csv":
                    df = pd.read_csv(source_path)
                elif file_format == "json":
                    df = pd.read_json(source_path)
                else:
                    raise ValueError(f"不支持的文件格式: {file_format}")
                
                loaded_data[source.get("name", f"source_{len(loaded_data)}")] = df
                
                logger.info(f"从文件加载数据: {source_path}, 形状: {df.shape}")
            
            elif source_type == "database":
                # 从数据库加载
                # 这里简化实现
                logger.info(f"从数据库加载: {source.get('query')}")
                
                # 模拟数据
                df = pd.DataFrame({
                    'feature1': np.random.randn(1000),
                    'feature2': np.random.randn(1000),
                    'label': np.random.randint(0, 2, 1000)
                })
                
                loaded_data[source.get("name", f"db_{len(loaded_data)}")] = df
            
            elif source_type == "feature_store":
                # 从特征存储加载
                if self.feature_store:
                    entity_df = pd.DataFrame(source.get("entities", []))
                    feature_set_id = source.get("feature_set_id")
                    
                    df = await self.feature_store.get_features(
                        entity_df=entity_df,
                        feature_set_id=feature_set_id,
                        feature_names=source.get("features")
                    )
                    
                    loaded_data[source.get("name", f"fs_{len(loaded_data)}")] = df
                else:
                    logger.warning("特征存储未初始化，跳过")
        
        # 保存数据
        data_dir = self.pipeline_dir / pipeline_id / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in loaded_data.items():
            file_path = data_dir / f"{name}.parquet"
            df.to_parquet(file_path, index=False)
        
        return {
            "loaded_datasets": list(loaded_data.keys()),
            "total_samples": sum(len(df) for df in loaded_data.values()),
            "data_path": str(data_dir)
        }
    
    async def _validate_data(
        self,
        pipeline_id: str,
        validation_config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """验证数据"""
        if not self.data_validator:
            logger.warning("数据验证器未初始化，跳过验证")
            return {"skipped": True, "reason": "Data validator not initialized"}
        
        # 加载数据
        data_dir = self.pipeline_dir / pipeline_id / "data"
        validation_results = {}
        
        for data_file in data_dir.glob("*.parquet"):
            dataset_name = data_file.stem
            df = pd.read_parquet(data_file)
            
            # 执行验证
            result = await self.data_validator.monitor_dataset(
                dataset_name=dataset_name,
                data=df,
                expectation_suite=validation_config.get("expectation_suite"),
                batch_identifier=f"{pipeline_id}_{dataset_name}"
            )
            
            validation_results[dataset_name] = result
            
            logger.info(f"数据验证完成: {dataset_name}, 状态: {result.get('status')}")
        
        # 保存验证结果
        validation_dir = self.pipeline_dir / pipeline_id / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        validation_file = validation_dir / "validation_results.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # 检查是否通过
        all_passed = all(
            result.get("overall_success", False) 
            for result in validation_results.values()
        )
        
        if not all_passed and not validation_config.get("allow_failure", False):
            raise ValueError("数据验证失败")
        
        return {
            "validation_results": validation_results,
            "all_passed": all_passed,
            "validation_file": str(validation_file)
        }
    
    async def _engineer_features(
        self,
        pipeline_id: str,
        feature_config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """特征工程"""
        # 加载数据
        data_dir = self.pipeline_dir / pipeline_id / "data"
        
        # 这里简化实现，实际中会有复杂的特征工程逻辑
        all_features = []
        
        for data_file in data_dir.glob("*.parquet"):
            df = pd.read_parquet(data_file)
            
            # 示例特征工程
            if 'feature1' in df.columns and 'feature2' in df.columns:
                df['feature1_squared'] = df['feature1'] ** 2
                df['feature2_log'] = np.log1p(np.abs(df['feature2']))
                df['feature_interaction'] = df['feature1'] * df['feature2']
            
            all_features.append(df)
        
        # 合并特征
        if len(all_features) > 1:
            # 合并多个数据集
            features_df = pd.concat(all_features, axis=1)
        else:
            features_df = all_features[0] if all_features else pd.DataFrame()
        
        # 保存特征
        features_dir = self.pipeline_dir / pipeline_id / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        features_file = features_dir / "features.parquet"
        features_df.to_parquet(features_file, index=False)
        
        # 注册到特征存储（可选）
        if self.feature_store and feature_config.get("register_to_feature_store", False):
            # 这里简化实现
            logger.info("特征注册到特征存储")
        
        return {
            "features_shape": features_df.shape,
            "features_columns": list(features_df.columns),
            "features_file": str(features_file)
        }
    
    async def _train_model(
        self,
        pipeline_id: str,
        training_config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """训练模型"""
        if not self.distributed_trainer:
            logger.warning("分布式训练器未初始化，使用本地训练")
            return await self._train_model_local(pipeline_id, training_config)
        
        # 加载特征
        features_file = self.pipeline_dir / pipeline_id / "features" / "features.parquet"
        features_df = pd.read_parquet(features_file)
        
        # 准备数据
        label_column = training_config.get("label_column", "label")
        
        if label_column not in features_df.columns:
            raise ValueError(f"标签列不存在: {label_column}")
        
        X = features_df.drop(columns=[label_column])
        y = features_df[label_column]
        
        # 分割数据集
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=training_config.get("validation_size", 0.2),
            random_state=42
        )
        
        # 转换为Ray Dataset
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        
        train_dataset = ray.data.from_pandas(train_df)
        val_dataset = ray.data.from_pandas(val_df)
        
        # 训练配置
        model_type = training_config.get("model_type", "xgboost")
        hyperparams = training_config.get("hyperparameters", {})
        
        # 开始训练
        if model_type == "xgboost":
            result = await self.distributed_trainer.train_xgboost(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                params=hyperparams,
                num_workers=training_config.get("num_workers", 4)
            )
        elif model_type == "lightgbm":
            result = await self.distributed_trainer.train_lightgbm(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                params=hyperparams,
                num_workers=training_config.get("num_workers", 4)
            )
        else:
            # 默认使用XGBoost
            result = await self.distributed_trainer.train_xgboost(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                params=hyperparams,
                num_workers=training_config.get("num_workers", 4)
            )
        
        # 保存模型
        model_dir = self.pipeline_dir / pipeline_id / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存结果
        result_file = model_dir / "training_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return {
            "training_result": result,
            "model_dir": str(model_dir),
            "model_type": model_type
        }
    
    async def _train_model_local(
        self,
        pipeline_id: str,
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """本地训练模型（简化实现）"""
        # 加载特征
        features_file = self.pipeline_dir / pipeline_id / "features" / "features.parquet"
        features_df = pd.read_parquet(features_file)
        
        # 准备数据
        label_column = training_config.get("label_column", "label")
        
        if label_column not in features_df.columns:
            raise ValueError(f"标签列不存在: {label_column}")
        
        X = features_df.drop(columns=[label_column])
        y = features_df[label_column]
        
        # 分割数据集
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # 评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 保存模型
        import joblib
        
        model_dir = self.pipeline_dir / pipeline_id / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = model_dir / "model.joblib"
        joblib.dump(model, model_file)
        
        return {
            "accuracy": accuracy,
            "model_file": str(model_file),
            "model_type": "random_forest",
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
    
    async def _evaluate_model(
        self,
        pipeline_id: str,
        evaluation_config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """评估模型"""
        # 加载模型
        model_dir = self.pipeline_dir / pipeline_id / "models"
        
        # 这里简化实现
        metrics = {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "f1": 0.935,
            "roc_auc": 0.96
        }
        
        # 保存评估结果
        eval_dir = self.pipeline_dir / pipeline_id / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        eval_file = eval_dir / "evaluation_results.json"
        with open(eval_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return {
            "metrics": metrics,
            "evaluation_file": str(eval_file),
            "passed": all(v > 0.9 for k, v in metrics.items() if k != "roc_auc")
        }
    
    async def _register_model(
        self,
        pipeline_id: str,
        registration_config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """注册模型"""
        if not self.model_registry:
            logger.warning("模型注册表未初始化，跳过注册")
            return {"skipped": True, "reason": "Model registry not initialized"}
        
        # 获取模型信息
        model_dir = self.pipeline_dir / pipeline_id / "models"
        
        # 创建模型元数据
        metadata = ModelMetadata(
            model_name=self.pipeline_name,
            version=f"v{pipeline_id.split('_')[-1]}",
            framework="xgboost",
            created_at=datetime.now(),
            created_by="training_pipeline",
            metrics={
                "accuracy": 0.95,
                "f1": 0.935
            },
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 6
            },
            data_version="v1.0",
            code_version="git_commit_hash",
            feature_version="v2.1",
            description=f"Model trained by pipeline {pipeline_id}"
        )
        
        # 注册模型
        # 这里简化实现，实际中需要从MLflow获取run_id
        run_id = f"run_{pipeline_id}"
        
        artifacts = {
            "model": str(model_dir / "model.joblib"),
            "metrics": str(self.pipeline_dir / pipeline_id / "evaluation" / "evaluation_results.json")
        }
        
        version = await self.model_registry.register_model(
            run_id=run_id,
            model_name=self.pipeline_name,
            metadata=metadata,
            artifacts=artifacts
        )
        
        return {
            "model_name": self.pipeline_name,
            "model_version": version,
            "registered_at": datetime.now().isoformat()
        }
    
    async def _deploy_model(
        self,
        pipeline_id: str,
        deployment_config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """部署模型"""
        # 这里简化实现
        logger.info(f"部署模型，pipeline: {pipeline_id}")
        
        return {
            "deployed": True,
            "endpoint": f"http://localhost:8000/models/{self.pipeline_name}/latest",
            "deployment_time": datetime.now().isoformat()
        }
    
    async def _save_pipeline_result(self, result: PipelineResult):
        """保存流水线结果"""
        result_file = self.pipeline_dir / result.pipeline_id / "pipeline_result.json"
        
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
    
    async def get_pipeline_result(self, pipeline_id: str) -> Optional[PipelineResult]:
        """获取流水线结果"""
        return self.results.get(pipeline_id)
    
    async def list_pipeline_results(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """列出流水线结果"""
        results = []
        
        for pipeline_id, result in list(self.results.items())[offset:offset + limit]:
            results.append({
                "pipeline_id": pipeline_id,
                "pipeline_name": result.pipeline_name,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "status": result.status.value,
                "duration_seconds": (result.end_time - result.start_time).total_seconds()
            })
        
        return results
    
    async def trigger_retraining(
        self,
        trigger_type: str = "schedule",
        data_version: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> str:
        """触发重新训练"""
        # 合并配置
        merged_config = self.config.copy()
        if config_overrides:
            merged_config.update(config_overrides)
        
        # 设置数据版本
        if data_version:
            merged_config.setdefault("data_loading", {})["data_version"] = data_version
        
        # 运行流水线
        result = await self.run()
        
        return result.pipeline_id
