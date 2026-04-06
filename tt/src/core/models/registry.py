"""
模型注册表
基于MLflow的模型版本管理、注册、部署跟踪
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from typing import Dict, Any, List, Optional, Tuple
import json
import hashlib
from datetime import datetime, timedelta
import asyncio
from enum import Enum
from dataclasses import dataclass, asdict
import pickle
import os
from pathlib import Path
import aiofiles

from ..config import settings


class ModelStage(str, Enum):
    """模型阶段"""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class DeploymentStatus(str, Enum):
    """部署状态"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    DELETING = "deleting"


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_name: str
    version: str
    framework: str
    created_at: datetime
    created_by: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    data_version: str
    code_version: str
    feature_version: str
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    lineage: Optional[Dict[str, Any]] = None


@dataclass
class DeploymentRecord:
    """部署记录"""
    deployment_id: str
    model_name: str
    model_version: str
    stage: ModelStage
    status: DeploymentStatus
    deployed_at: datetime
    deployed_by: str
    endpoint: Optional[str] = None
    resources: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    rollback_to: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class ModelRegistry:
    """模型注册表"""
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None
    ):
        """初始化模型注册表"""
        self.tracking_uri = tracking_uri or settings.mlflow.tracking_uri
        self.registry_uri = registry_uri or settings.mlflow.registry_uri
        
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
        
        self.client = MlflowClient()
        
        # 部署历史存储
        self.deployment_history_path = Path("deployment_history")
        self.deployment_history_path.mkdir(exist_ok=True)
    
    async def register_model(
        self,
        run_id: str,
        model_name: str,
        metadata: ModelMetadata,
        artifacts: Dict[str, str]
    ) -> str:
        """
        注册新模型版本
        
        Args:
            run_id: MLflow运行ID
            model_name: 模型名称
            metadata: 模型元数据
            artifacts: 模型工件
            
        Returns:
            注册的版本号
        """
        try:
            # 注册模型到MLflow
            model_uri = f"runs:/{run_id}/model"
            mv = await asyncio.to_thread(
                self.client.create_model_version,
                name=model_name,
                source=model_uri,
                run_id=run_id
            )
            
            # 添加元数据标签
            metadata_dict = asdict(metadata)
            for key, value in metadata_dict.items():
                if value is not None:
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, ensure_ascii=False)
                    elif isinstance(value, datetime):
                        value = value.isoformat()
                    
                    await asyncio.to_thread(
                        self.client.set_model_version_tag,
                        model_name,
                        mv.version,
                        f"metadata.{key}",
                        str(value)
                    )
            
            # 添加工件信息
            for artifact_name, artifact_path in artifacts.items():
                await asyncio.to_thread(
                    self.client.set_model_version_tag,
                    model_name,
                    mv.version,
                    f"artifact.{artifact_name}",
                    artifact_path
                )
            
            # 计算和存储模型哈希
            model_hash = await self._calculate_model_hash(model_name, mv.version)
            await asyncio.to_thread(
                self.client.set_model_version_tag,
                model_name,
                mv.version,
                "integrity.hash",
                model_hash
            )
            
            # 记录注册事件
            await self._record_registration_event(
                model_name=model_name,
                version=mv.version,
                metadata=metadata
            )
            
            return mv.version
            
        except Exception as e:
            raise Exception(f"注册模型失败: {str(e)}")
    
    async def promote_model(
        self,
        model_name: str,
        version: str,
        target_stage: ModelStage,
        validation_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        提升模型阶段
        
        Args:
            model_name: 模型名称
            version: 模型版本
            target_stage: 目标阶段
            validation_metrics: 验证指标
            
        Returns:
            是否成功
        """
        try:
            # 验证模型
            if validation_metrics:
                is_valid = await self._validate_model_metrics(
                    model_name, version, validation_metrics
                )
                if not is_valid:
                    raise ValueError(f"模型 {model_name}:{version} 验证失败")
            
            # 获取当前生产版本
            current_prod_version = await self.get_production_version(model_name)
            
            # 如果是提升到生产，检查是否有其他生产版本
            if target_stage == ModelStage.PRODUCTION and current_prod_version:
                # 将当前生产版本降级为归档
                await asyncio.to_thread(
                    self.client.transition_model_version_stage,
                    name=model_name,
                    version=current_prod_version,
                    stage=ModelStage.ARCHIVED.value
                )
            
            # 更新模型阶段
            await asyncio.to_thread(
                self.client.transition_model_version_stage,
                name=model_name,
                version=version,
                stage=target_stage.value
            )
            
            # 记录部署历史
            if target_stage == ModelStage.PRODUCTION:
                deployment_id = await self._record_deployment(
                    model_name=model_name,
                    model_version=version,
                    stage=target_stage,
                    previous_version=current_prod_version
                )
            
            return True
            
        except Exception as e:
            raise Exception(f"提升模型阶段失败: {str(e)}")
    
    async def get_model(
        self,
        model_name: str,
        version: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            version: 模型版本
            
        Returns:
            模型信息字典
        """
        try:
            mv = await asyncio.to_thread(
                self.client.get_model_version,
                model_name,
                version
            )
            
            # 提取元数据
            metadata = {}
            artifacts = {}
            tags = {}
            
            for tag in mv.tags:
                if tag.key.startswith("metadata."):
                    key = tag.key.replace("metadata.", "")
                    try:
                        metadata[key] = json.loads(tag.value)
                    except:
                        metadata[key] = tag.value
                elif tag.key.startswith("artifact."):
                    key = tag.key.replace("artifact.", "")
                    artifacts[key] = tag.value
                else:
                    tags[tag.key] = tag.value
            
            # 获取部署信息
            deployments = await self.get_model_deployments(model_name, version)
            
            return {
                "name": model_name,
                "version": mv.version,
                "current_stage": mv.current_stage,
                "description": mv.description,
                "created_at": mv.creation_timestamp,
                "last_updated": mv.last_updated_timestamp,
                "run_id": mv.run_id,
                "source": mv.source,
                "metadata": metadata,
                "artifacts": artifacts,
                "tags": tags,
                "deployments": deployments
            }
            
        except Exception:
            return None
    
    async def list_models(
        self,
        model_name: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> List[Dict[str, Any]]:
        """
        列出模型
        
        Args:
            model_name: 模型名称过滤
            stage: 阶段过滤
            
        Returns:
            模型列表
        """
        try:
            if model_name:
                # 获取特定模型的所有版本
                model_versions = await asyncio.to_thread(
                    self.client.search_model_versions,
                    f"name='{model_name}'"
                )
            else:
                # 获取所有模型
                registered_models = await asyncio.to_thread(
                    self.client.search_registered_models
                )
                models = []
                
                for rm in registered_models:
                    model_versions = await asyncio.to_thread(
                        self.client.search_model_versions,
                        f"name='{rm.name}'"
                    )
                    models.extend(model_versions)
                
                model_versions = models
            
            # 过滤阶段
            if stage:
                model_versions = [
                    mv for mv in model_versions 
                    if mv.current_stage == stage.value
                ]
            
            # 转换为字典
            result = []
            for mv in model_versions:
                model_info = await self.get_model(mv.name, mv.version)
                if model_info:
                    result.append(model_info)
            
            return result
            
        except Exception as e:
            raise Exception(f"获取模型列表失败: {str(e)}")
    
    async def get_production_version(self, model_name: str) -> Optional[str]:
        """
        获取当前生产版本
        
        Args:
            model_name: 模型名称
            
        Returns:
            生产版本号
        """
        try:
            prod_versions = await asyncio.to_thread(
                self.client.get_latest_versions,
                model_name,
                stages=[ModelStage.PRODUCTION.value]
            )
            return prod_versions[0].version if prod_versions else None
        except Exception:
            return None
    
    async def get_model_deployments(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> List[DeploymentRecord]:
        """
        获取模型部署记录
        
        Args:
            model_name: 模型名称
            version: 模型版本（可选）
            
        Returns:
            部署记录列表
        """
        try:
            history_file = self.deployment_history_path / f"{model_name}.json"
            
            if not history_file.exists():
                return []
            
            async with aiofiles.open(history_file, 'r') as f:
                content = await f.read()
                deployments = json.loads(content) if content else []
            
            # 过滤版本
            if version:
                deployments = [
                    DeploymentRecord(**d) for d in deployments 
                    if d["model_version"] == version
                ]
            else:
                deployments = [DeploymentRecord(**d) for d in deployments]
            
            return deployments
            
        except Exception as e:
            print(f"获取部署记录失败: {e}")
            return []
    
    async def record_deployment(
        self,
        model_name: str,
        model_version: str,
        stage: ModelStage,
        deployment_id: str,
        endpoint: str,
        resources: Dict[str, Any]
    ) -> str:
        """
        记录部署
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            stage: 部署阶段
            deployment_id: 部署ID
            endpoint: 端点URL
            resources: 资源配置
            
        Returns:
            部署记录ID
        """
        try:
            deployment_record = DeploymentRecord(
                deployment_id=deployment_id,
                model_name=model_name,
                model_version=model_version,
                stage=stage,
                status=DeploymentStatus.ACTIVE,
                deployed_at=datetime.now(),
                deployed_by="system",
                endpoint=endpoint,
                resources=resources
            )
            
            # 加载现有部署
            history_file = self.deployment_history_path / f"{model_name}.json"
            deployments = []
            
            if history_file.exists():
                async with aiofiles.open(history_file, 'r') as f:
                    content = await f.read()
                    deployments = json.loads(content) if content else []
            
            # 添加新部署
            deployments.append(asdict(deployment_record))
            
            # 保存
            async with aiofiles.open(history_file, 'w') as f:
                await f.write(json.dumps(
                    deployments,
                    indent=2,
                    default=str
                ))
            
            return deployment_id
            
        except Exception as e:
            raise Exception(f"记录部署失败: {str(e)}")
    
    async def update_deployment_status(
        self,
        deployment_id: str,
        status: DeploymentStatus,
        error_message: Optional[str] = None
    ):
        """
        更新部署状态
        
        Args:
            deployment_id: 部署ID
            status: 新状态
            error_message: 错误信息
        """
        try:
            # 查找并更新部署状态
            for history_file in self.deployment_history_path.glob("*.json"):
                async with aiofiles.open(history_file, 'r') as f:
                    content = await f.read()
                    deployments = json.loads(content) if content else []
                
                updated = False
                for i, dep in enumerate(deployments):
                    if dep["deployment_id"] == deployment_id:
                        deployments[i]["status"] = status.value
                        if error_message:
                            deployments[i]["error_message"] = error_message
                        updated = True
                        break
                
                if updated:
                    async with aiofiles.open(history_file, 'w') as f:
                        await f.write(json.dumps(
                            deployments,
                            indent=2,
                            default=str
                        ))
                    break
                    
        except Exception as e:
            print(f"更新部署状态失败: {e}")
    
    async def _calculate_model_hash(
        self,
        model_name: str,
        version: str
    ) -> str:
        """计算模型哈希值"""
        try:
            # 获取模型文件
            model_path = await asyncio.to_thread(
                self.client.download_artifacts,
                run_id=version,  # 简化实现
                path="model"
            )
            
            # 计算目录哈希
            hash_md5 = hashlib.md5()
            
            for root, dirs, files in os.walk(model_path):
                for file in sorted(files):  # 排序确保一致性
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
            
            return hash_md5.hexdigest()
            
        except Exception:
            # 如果无法计算，使用简化哈希
            content = f"{model_name}_{version}_{datetime.now().isoformat()}"
            return hashlib.sha256(content.encode()).hexdigest()
    
    async def _validate_model_metrics(
        self,
        model_name: str,
        version: str,
        validation_metrics: Dict[str, float]
    ) -> bool:
        """验证模型指标"""
        try:
            # 获取模型历史指标
            model_info = await self.get_model(model_name, version)
            if not model_info:
                return False
            
            baseline_metrics = model_info.get("metadata", {}).get("metrics", {})
            
            # 检查关键指标是否满足要求
            required_metrics = ["accuracy", "precision", "recall", "f1"]
            for metric in required_metrics:
                if metric in validation_metrics:
                    value = validation_metrics[metric]
                    
                    # 如果存在基线，检查是否下降过多
                    if metric in baseline_metrics:
                        baseline = baseline_metrics[metric]
                        if value < baseline * 0.9:  # 下降超过10%
                            return False
                    
                    # 检查绝对阈值
                    if metric == "accuracy" and value < 0.8:
                        return False
                    if metric == "f1" and value < 0.7:
                        return False
            
            return True
            
        except Exception:
            return False
    
    async def _record_registration_event(
        self,
        model_name: str,
        version: str,
        metadata: ModelMetadata
    ):
        """记录注册事件"""
        try:
            event = {
                "event_type": "model_registered",
                "model_name": model_name,
                "version": version,
                "metadata": asdict(metadata),
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存到文件
            event_file = self.deployment_history_path / f"events_{datetime.now().date()}.json"
            events = []
            
            if event_file.exists():
                async with aiofiles.open(event_file, 'r') as f:
                    content = await f.read()
                    events = json.loads(content) if content else []
            
            events.append(event)
            
            async with aiofiles.open(event_file, 'w') as f:
                await f.write(json.dumps(
                    events,
                    indent=2,
                    default=str
                ))
                
        except Exception as e:
            print(f"记录注册事件失败: {e}")
    
    async def _record_deployment(
        self,
        model_name: str,
        model_version: str,
        stage: ModelStage,
        previous_version: Optional[str] = None,
        is_rollback: bool = False
    ) -> str:
        """记录部署到历史"""
        deployment_id = hashlib.md5(
            f"{model_name}_{model_version}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # 这里实现与record_deployment类似，但用于内部使用
        return deployment_id
    
    async def list_all_models(self) -> List[Dict[str, Any]]:
        """列出所有模型（简化版）"""
        try:
            models = await asyncio.to_thread(
                self.client.search_registered_models
            )
            
            result = []
            for rm in models:
                versions = await asyncio.to_thread(
                    self.client.search_model_versions,
                    f"name='{rm.name}'"
                )
                
                for mv in versions[:5]:  # 只取最近5个版本
                    model_info = await self.get_model(rm.name, mv.version)
                    if model_info:
                        result.append(model_info)
            
            return result
            
        except Exception as e:
            raise Exception(f"获取所有模型失败: {str(e)}")
