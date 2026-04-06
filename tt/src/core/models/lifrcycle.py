"""
模型生命周期管理
管理模型的注册、部署、验证、回滚、下线等全生命周期
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import hashlib
from enum import Enum
from dataclasses import dataclass, asdict
import aiofiles
from pathlib import Path
import yaml
import mlflow
from mlflow.tracking import MlflowClient
import logging
from functools import wraps

from ..config import settings
from .registry import ModelRegistry, ModelStage, DeploymentStatus
from .serving import ModelServer, ModelType
from ..monitoring.drift import ModelMonitor
from ..orchestration.workflows import WorkflowOrchestrator

logger = logging.getLogger(__name__)


class LifecycleState(Enum):
    """生命周期状态"""
    REGISTERED = "registered"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    MONITORING = "monitoring"
    DRIFT_DETECTED = "drift_detected"
    RETRAINING = "retraining"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    ARCHIVING = "archiving"
    ARCHIVED = "archived"
    DELETING = "deleting"
    DELETED = "deleted"


class ValidationStatus(Enum):
    """验证状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class LifecycleEvent:
    """生命周期事件"""
    event_id: str
    model_name: str
    model_version: str
    from_state: LifecycleState
    to_state: LifecycleState
    timestamp: datetime
    triggered_by: str
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class ValidationResult:
    """验证结果"""
    validation_id: str
    model_name: str
    model_version: str
    status: ValidationStatus
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    passed: List[str]
    failed: List[str]
    warnings: List[str]
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if isinstance(self.started_at, str):
            self.started_at = datetime.fromisoformat(self.started_at)
        if isinstance(self.completed_at, str):
            self.completed_at = datetime.fromisoformat(self.completed_at)


class ModelLifecycleManager:
    """模型生命周期管理器"""
    
    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        model_server: Optional[ModelServer] = None,
        model_monitor: Optional[ModelMonitor] = None,
        workflow_orchestrator: Optional[WorkflowOrchestrator] = None
    ):
        self.model_registry = model_registry or ModelRegistry()
        self.model_server = model_server or ModelServer()
        self.model_monitor = model_monitor
        self.workflow_orchestrator = workflow_orchestrator
        
        # 生命周期状态存储
        self.lifecycle_states: Dict[str, LifecycleState] = {}  # model_version -> state
        self.lifecycle_events: List[LifecycleEvent] = []
        
        # 验证结果存储
        self.validation_results: Dict[str, ValidationResult] = {}  # validation_id -> result
        
        # 初始化存储目录
        self.data_dir = Path("lifecycle_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # 加载现有状态
        self._load_state()
    
    async def register_model(
        self,
        run_id: str,
        model_name: str,
        metadata: Dict[str, Any],
        artifacts: Dict[str, str],
        auto_validate: bool = True
    ) -> Tuple[str, str]:
        """
        注册模型并开始生命周期
        
        Args:
            run_id: MLflow运行ID
            model_name: 模型名称
            metadata: 模型元数据
            artifacts: 模型工件
            auto_validate: 是否自动验证
            
        Returns:
            (模型版本, 生命周期ID)
        """
        try:
            # 注册模型到注册表
            version = await self.model_registry.register_model(
                run_id=run_id,
                model_name=model_name,
                metadata=metadata,
                artifacts=artifacts
            )
            
            # 记录生命周期状态
            lifecycle_id = f"lifecycle_{model_name}_{version}"
            self.lifecycle_states[lifecycle_id] = LifecycleState.REGISTERED
            
            # 记录事件
            await self._record_event(
                lifecycle_id=lifecycle_id,
                model_name=model_name,
                model_version=version,
                from_state=None,
                to_state=LifecycleState.REGISTERED,
                triggered_by="system",
                reason="模型注册完成"
            )
            
            # 自动验证
            if auto_validate:
                await self.validate_model(model_name, version)
            
            return version, lifecycle_id
            
        except Exception as e:
            logger.error(f"模型注册失败: {e}")
            raise
    
    async def validate_model(
        self,
        model_name: str,
        model_version: str,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        验证模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            validation_config: 验证配置
            
        Returns:
            验证ID
        """
        validation_id = f"validation_{model_name}_{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # 更新生命周期状态
            lifecycle_id = f"lifecycle_{model_name}_{model_version}"
            await self._transition_state(
                lifecycle_id=lifecycle_id,
                to_state=LifecycleState.VALIDATING,
                reason="开始模型验证"
            )
            
            # 获取模型信息
            model_info = await self.model_registry.get_model(model_name, model_version)
            if not model_info:
                raise ValueError(f"模型不存在: {model_name}:{model_version}")
            
            # 执行验证
            validation_result = await self._perform_validation(
                model_name=model_name,
                model_version=model_version,
                model_info=model_info,
                config=validation_config
            )
            
            # 保存验证结果
            self.validation_results[validation_id] = validation_result
            
            # 更新状态
            if validation_result.status == ValidationStatus.PASSED:
                await self._transition_state(
                    lifecycle_id=lifecycle_id,
                    to_state=LifecycleState.VALIDATED,
                    reason="模型验证通过"
                )
            else:
                await self._transition_state(
                    lifecycle_id=lifecycle_id,
                    to_state=LifecycleState.REGISTERED,  # 回到注册状态
                    reason=f"模型验证失败: {validation_result.failed}"
                )
            
            # 保存验证结果到文件
            await self._save_validation_result(validation_result)
            
            return validation_id
            
        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            
            # 记录失败事件
            await self._record_event(
                lifecycle_id=lifecycle_id,
                model_name=model_name,
                model_version=model_version,
                from_state=LifecycleState.VALIDATING,
                to_state=LifecycleState.REGISTERED,
                triggered_by="system",
                reason=f"验证过程异常: {str(e)}"
            )
            
            raise
    
    async def _perform_validation(
        self,
        model_name: str,
        model_version: str,
        model_info: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """执行模型验证"""
        started_at = datetime.now()
        
        # 默认配置
        default_config = {
            "data_quality": {
                "required": True,
                "thresholds": {
                    "accuracy": 0.8,
                    "precision": 0.7,
                    "recall": 0.7,
                    "f1": 0.7
                }
            },
            "performance": {
                "required": True,
                "max_latency_ms": 1000,
                "max_memory_mb": 1024
            },
            "fairness": {
                "required": False,
                "thresholds": {}
            }
        }
        
        if config:
            # 合并配置
            for key, value in config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        
        validation_config = default_config
        
        # 执行验证
        passed = []
        failed = []
        warnings = []
        metrics = {}
        
        # 1. 数据质量验证
        if validation_config["data_quality"]["required"]:
            data_quality_result = await self._validate_data_quality(
                model_name, model_version, model_info, validation_config
            )
            metrics.update(data_quality_result["metrics"])
            
            for check, result in data_quality_result["checks"].items():
                if result["status"] == "passed":
                    passed.append(f"data_quality.{check}")
                elif result["status"] == "failed":
                    failed.append(f"data_quality.{check}")
                else:
                    warnings.append(f"data_quality.{check}")
        
        # 2. 性能验证
        if validation_config["performance"]["required"]:
            performance_result = await self._validate_performance(
                model_name, model_version, model_info, validation_config
            )
            metrics.update(performance_result["metrics"])
            
            for check, result in performance_result["checks"].items():
                if result["status"] == "passed":
                    passed.append(f"performance.{check}")
                elif result["status"] == "failed":
                    failed.append(f"performance.{check}")
                else:
                    warnings.append(f"performance.{check}")
        
        # 3. 公平性验证（如果配置）
        if validation_config["fairness"]["required"]:
            fairness_result = await self._validate_fairness(
                model_name, model_version, model_info, validation_config
            )
            metrics.update(fairness_result["metrics"])
            
            for check, result in fairness_result["checks"].items():
                if result["status"] == "passed":
                    passed.append(f"fairness.{check}")
                elif result["status"] == "failed":
                    failed.append(f"fairness.{check}")
                else:
                    warnings.append(f"fairness.{check}")
        
        # 确定总体状态
        if failed:
            status = ValidationStatus.FAILED
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASSED
        
        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()
        
        return ValidationResult(
            validation_id=f"validation_{model_name}_{model_version}_{started_at.strftime('%Y%m%d_%H%M%S')}",
            model_name=model_name,
            model_version=model_version,
            status=status,
            metrics=metrics,
            thresholds=validation_config.get("thresholds", {}),
            passed=passed,
            failed=failed,
            warnings=warnings,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            details={
                "config": validation_config,
                "model_info": model_info
            }
        )
    
    async def deploy_model(
        self,
        model_name: str,
        model_version: str,
        stage: ModelStage = ModelStage.STAGING,
        deployment_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        部署模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            stage: 部署阶段
            deployment_config: 部署配置
            
        Returns:
            部署ID
        """
        try:
            lifecycle_id = f"lifecycle_{model_name}_{model_version}"
            
            # 验证模型状态
            current_state = self.lifecycle_states.get(lifecycle_id)
            if current_state not in [LifecycleState.VALIDATED, LifecycleState.REGISTERED]:
                raise ValueError(f"模型当前状态 {current_state} 不允许部署")
            
            # 更新状态
            await self._transition_state(
                lifecycle_id=lifecycle_id,
                to_state=LifecycleState.DEPLOYING,
                reason=f"开始部署到 {stage.value}"
            )
            
            # 获取模型信息
            model_info = await self.model_registry.get_model(model_name, model_version)
            if not model_info:
                raise ValueError(f"模型不存在: {model_name}:{model_version}")
            
            # 确定模型类型
            model_type = self._infer_model_type(model_info)
            
            # 获取模型路径
            model_path = self._get_model_path(model_info)
            
            # 部署模型
            deployment_id = await self.model_server.deploy_model(
                model_name=model_name,
                model_version=model_version,
                model_path=model_path,
                model_type=model_type,
                config=deployment_config
            )
            
            # 提升模型阶段
            await self.model_registry.promote_model(
                model_name=model_name,
                version=model_version,
                target_stage=stage
            )
            
            # 记录部署
            await self.model_registry.record_deployment(
                model_name=model_name,
                model_version=model_version,
                stage=stage,
                deployment_id=deployment_id,
                endpoint=f"http://{settings.model_server.host}:{settings.model_server.port}/models/{model_name}/{model_version}",
                resources=deployment_config.get("resources", {}) if deployment_config else {}
            )
            
            # 更新状态
            await self._transition_state(
                lifecycle_id=lifecycle_id,
                to_state=LifecycleState.DEPLOYED,
                reason=f"成功部署到 {stage.value}"
            )
            
            # 开始监控
            if stage == ModelStage.PRODUCTION:
                await self._start_monitoring(model_name, model_version)
            
            return deployment_id
            
        except Exception as e:
            logger.error(f"模型部署失败: {e}")
            
            # 回滚状态
            await self._transition_state(
                lifecycle_id=lifecycle_id,
                to_state=LifecycleState.VALIDATED,
                reason=f"部署失败: {str(e)}"
            )
            
            raise
    
    async def rollback_model(
        self,
        model_name: str,
        target_version: str,
        reason: str = "性能下降",
        validate_before_rollback: bool = True
    ) -> Dict[str, Any]:
        """
        回滚到指定版本
        
        Args:
            model_name: 模型名称
            target_version: 目标版本
            reason: 回滚原因
            validate_before_rollback: 回滚前验证
            
        Returns:
            回滚结果
        """
        try:
            # 获取当前生产版本
            current_version = await self.model_registry.get_production_version(model_name)
            if not current_version:
                raise ValueError(f"模型 {model_name} 无生产版本")
            
            if current_version == target_version:
                return {
                    "success": False,
                    "message": f"目标版本 {target_version} 已是当前生产版本"
                }
            
            # 验证目标版本
            target_model = await self.model_registry.get_model(model_name, target_version)
            if not target_model:
                raise ValueError(f"目标版本 {model_name}:{target_version} 不存在")
            
            lifecycle_id = f"lifecycle_{model_name}_{target_version}"
            
            # 更新状态
            await self._transition_state(
                lifecycle_id=lifecycle_id,
                to_state=LifecycleState.ROLLING_BACK,
                reason=f"开始回滚: {reason}"
            )
            
            # 回滚前验证
            if validate_before_rollback:
                validation_id = await self.validate_model(
                    model_name=model_name,
                    model_version=target_version
                )
                
                validation_result = self.validation_results.get(validation_id)
                if validation_result and validation_result.status == ValidationStatus.FAILED:
                    raise ValueError(f"回滚验证失败: {validation_result.failed}")
            
            # 获取当前部署
            current_deployments = await self.model_registry.get_model_deployments(
                model_name, current_version
            )
            
            # 部署目标版本
            deployment_id = await self.deploy_model(
                model_name=model_name,
                model_version=target_version,
                stage=ModelStage.PRODUCTION
            )
            
            # 取消当前版本部署
            for deployment in current_deployments:
                if deployment.status == DeploymentStatus.ACTIVE:
                    await self.model_server.undeploy_model(deployment.deployment_id)
            
            # 记录回滚
            await self._record_rollback_event(
                model_name=model_name,
                from_version=current_version,
                to_version=target_version,
                reason=reason
            )
            
            # 更新状态
            await self._transition_state(
                lifecycle_id=lifecycle_id,
                to_state=LifecycleState.ROLLED_BACK,
                reason=f"回滚完成: {reason}"
            )
            
            return {
                "success": True,
                "from_version": current_version,
                "to_version": target_version,
                "deployment_id": deployment_id,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"模型回滚失败: {e}")
            
            # 回滚失败，回到部署状态
            await self._transition_state(
                lifecycle_id=lifecycle_id,
                to_state=LifecycleState.DEPLOYED,
                reason=f"回滚失败: {str(e)}"
            )
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def archive_model(
        self,
        model_name: str,
        model_version: str,
        reason: str = "模型过时"
    ) -> bool:
        """
        归档模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            reason: 归档原因
            
        Returns:
            是否成功
        """
        try:
            lifecycle_id = f"lifecycle_{model_name}_{model_version}"
            
            # 更新状态
            await self._transition_state(
                lifecycle_id=lifecycle_id,
                to_state=LifecycleState.ARCHIVING,
                reason=f"开始归档: {reason}"
            )
            
            # 取消部署
            deployments = await self.model_registry.get_model_deployments(
                model_name, model_version
            )
            
            for deployment in deployments:
                if deployment.status == DeploymentStatus.ACTIVE:
                    await self.model_server.undeploy_model(deployment.deployment_id)
            
            # 在注册表中归档
            await self.model_registry.promote_model(
                model_name=model_name,
                version=model_version,
                target_stage=ModelStage.ARCHIVED
            )
            
            # 更新状态
            await self._transition_state(
                lifecycle_id=lifecycle_id,
                to_state=LifecycleState.ARCHIVED,
                reason=f"归档完成: {reason}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"模型归档失败: {e}")
            return False
    
    async def delete_model(
        self,
        model_name: str,
        model_version: str,
        force: bool = False
    ) -> bool:
        """
        删除模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            force: 强制删除（即使有部署）
            
        Returns:
            是否成功
        """
        try:
            lifecycle_id = f"lifecycle_{model_name}_{model_version}"
            
            # 检查是否有部署
            deployments = await self.model_registry.get_model_deployments(
                model_name, model_version
            )
            
            if deployments and not force:
                active_deployments = [
                    d for d in deployments 
                    if d.status == DeploymentStatus.ACTIVE
                ]
                if active_deployments:
                    raise ValueError(
                        f"模型有 {len(active_deployments)} 个活跃部署，请先取消部署或使用force=True"
                    )
            
            # 更新状态
            await self._transition_state(
                lifecycle_id=lifecycle_id,
                to_state=LifecycleState.DELETING,
                reason="开始删除模型"
            )
            
            # 取消所有部署
            for deployment in deployments:
                await self.model_server.undeploy_model(deployment.deployment_id)
            
            # TODO: 从模型注册表删除（MLflow可能不支持完全删除）
            # 这里我们只更新状态
            
            # 更新状态
            await self._transition_state(
                lifecycle_id=lifecycle_id,
                to_state=LifecycleState.DELETED,
                reason="模型删除完成"
            )
            
            # 清理本地状态
            if lifecycle_id in self.lifecycle_states:
                del self.lifecycle_states[lifecycle_id]
            
            return True
            
        except Exception as e:
            logger.error(f"模型删除失败: {e}")
            return False
    
    async def trigger_retraining(
        self,
        model_name: str,
        trigger_type: str = "drift",
        data_version: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        触发模型重新训练
        
        Args:
            model_name: 模型名称
            trigger_type: 触发类型 (drift/schedule/manual/performance)
            data_version: 数据版本
            config: 训练配置
            
        Returns:
            工作流ID
        """
        try:
            # 获取当前生产版本
            current_version = await self.model_registry.get_production_version(model_name)
            if not current_version:
                raise ValueError(f"模型 {model_name} 无生产版本")
            
            lifecycle_id = f"lifecycle_{model_name}_{current_version}"
            
            # 更新状态
            await self._transition_state(
                lifecycle_id=lifecycle_id,
                to_state=LifecycleState.RETRAINING,
                reason=f"触发重新训练: {trigger_type}"
            )
            
            # 触发重新训练工作流
            if self.workflow_orchestrator:
                workflow_id = await self.workflow_orchestrator.trigger_retraining(
                    model_name=model_name,
                    current_version=current_version,
                    trigger_type=trigger_type,
                    data_version=data_version,
                    config=config
                )
                
                return workflow_id
            else:
                # 如果没有工作流编排器，记录事件
                await self._record_event(
                    lifecycle_id=lifecycle_id,
                    model_name=model_name,
                    model_version=current_version,
                    from_state=LifecycleState.RETRAINING,
                    to_state=LifecycleState.MONITORING,
                    triggered_by="system",
                    reason=f"重新训练已触发但无工作流编排器: {trigger_type}"
                )
                
                return f"manual_retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        except Exception as e:
            logger.error(f"触发重新训练失败: {e}")
            raise
    
    async def get_lifecycle_state(
        self,
        model_name: str,
        model_version: str
    ) -> Dict[str, Any]:
        """
        获取模型生命周期状态
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            
        Returns:
            生命周期状态信息
        """
        lifecycle_id = f"lifecycle_{model_name}_{model_version}"
        
        state = self.lifecycle_states.get(lifecycle_id, LifecycleState.REGISTERED)
        
        # 获取相关事件
        events = [
            event for event in self.lifecycle_events
            if event.model_name == model_name and event.model_version == model_version
        ]
        
        # 获取验证结果
        validation_results = [
            result for result in self.validation_results.values()
            if result.model_name == model_name and result.model_version == model_version
        ]
        
        # 获取部署信息
        deployments = await self.model_registry.get_model_deployments(
            model_name, model_version
        )
        
        return {
            "model_name": model_name,
            "model_version": model_version,
            "lifecycle_id": lifecycle_id,
            "current_state": state.value,
            "events": [asdict(event) for event in events[-10:]],  # 最近10个事件
            "validation_results": [
                {
                    "validation_id": result.validation_id,
                    "status": result.status.value,
                    "passed": result.passed,
                    "failed": result.failed,
                    "warnings": result.warnings,
                    "completed_at": result.completed_at.isoformat()
                }
                for result in validation_results[-5:]  # 最近5个验证结果
            ],
            "deployments": [
                {
                    "deployment_id": dep.deployment_id,
                    "stage": dep.stage.value,
                    "status": dep.status.value,
                    "endpoint": dep.endpoint
                }
                for dep in deployments
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _transition_state(
        self,
        lifecycle_id: str,
        to_state: LifecycleState,
        reason: str,
        triggered_by: str = "system"
    ):
        """转换生命周期状态"""
        from_state = self.lifecycle_states.get(lifecycle_id)
        
        # 更新状态
        self.lifecycle_states[lifecycle_id] = to_state
        
        # 记录事件
        event = LifecycleEvent(
            event_id=f"event_{lifecycle_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_name=lifecycle_id.split('_')[1],
            model_version=lifecycle_id.split('_')[2],
            from_state=from_state,
            to_state=to_state,
            timestamp=datetime.now(),
            triggered_by=triggered_by,
            reason=reason
        )
        
        self.lifecycle_events.append(event)
        
        # 保存状态
        await self._save_state()
        
        logger.info(f"生命周期状态转换: {lifecycle_id} {from_state} -> {to_state}: {reason}")
    
    async def _record_event(
        self,
        lifecycle_id: str,
        model_name: str,
        model_version: str,
        from_state: Optional[LifecycleState],
        to_state: LifecycleState,
        triggered_by: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """记录生命周期事件"""
        event = LifecycleEvent(
            event_id=f"event_{lifecycle_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_name=model_name,
            model_version=model_version,
            from_state=from_state,
            to_state=to_state,
            timestamp=datetime.now(),
            triggered_by=triggered_by,
            reason=reason,
            metadata=metadata
        )
        
        self.lifecycle_events.append(event)
        
        # 保存到文件
        await self._save_event(event)
    
    async def _record_rollback_event(
        self,
        model_name: str,
        from_version: str,
        to_version: str,
        reason: str
    ):
        """记录回滚事件"""
        event = LifecycleEvent(
            event_id=f"rollback_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_name=model_name,
            model_version=to_version,
            from_state=LifecycleState.DEPLOYED,
            to_state=LifecycleState.ROLLED_BACK,
            timestamp=datetime.now(),
            triggered_by="system",
            reason=f"从版本 {from_version} 回滚到 {to_version}: {reason}",
            metadata={
                "from_version": from_version,
                "to_version": to_version,
                "reason": reason
            }
        )
        
        self.lifecycle_events.append(event)
        await self._save_event(event)
    
    async def _start_monitoring(self, model_name: str, model_version: str):
        """开始监控模型"""
        if self.model_monitor:
            await self.model_monitor.start_monitoring(
                model_name=model_name,
                model_version=model_version
            )
    
    def _infer_model_type(self, model_info: Dict[str, Any]) -> ModelType:
        """推断模型类型"""
        framework = model_info.get("metadata", {}).get("framework", "").lower()
        
        if "pytorch" in framework or "torch" in framework:
            return ModelType.PYTORCH
        elif "tensorflow" in framework or "tf" in framework:
            return ModelType.TENSORFLOW
        elif "xgboost" in framework:
            return ModelType.XGBOOST
        elif "lightgbm" in framework or "lgb" in framework:
            return ModelType.LIGHTGBM
        elif "sklearn" in framework or "scikit" in framework:
            return ModelType.SKLEARN
        elif "transformers" in framework or "huggingface" in framework:
            return ModelType.HUGGINGFACE
        else:
            return ModelType.CUSTOM
    
    def _get_model_path(self, model_info: Dict[str, Any]) -> str:
        """获取模型路径"""
        # 从工件中获取模型路径
        artifacts = model_info.get("artifacts", {})
        
        for artifact_name, artifact_path in artifacts.items():
            if "model" in artifact_name.lower():
                return artifact_path
        
        # 如果找不到，使用MLflow路径
        return model_info.get("source", "")
    
    async def _validate_data_quality(
        self,
        model_name: str,
        model_version: str,
        model_info: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证数据质量"""
        # 简化实现
        return {
            "metrics": {
                "data_quality_score": 0.95,
                "missing_values": 0.01,
                "outliers": 0.02
            },
            "checks": {
                "missing_values": {"status": "passed", "value": 0.01, "threshold": 0.05},
                "outliers": {"status": "passed", "value": 0.02, "threshold": 0.1},
                "data_distribution": {"status": "warning", "value": 0.15, "threshold": 0.1}
            }
        }
    
    async def _validate_performance(
        self,
        model_name: str,
        model_version: str,
        model_info: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证性能"""
        # 简化实现
        return {
            "metrics": {
                "inference_latency_ms": 150.5,
                "throughput_qps": 65.2,
                "memory_usage_mb": 245.3
            },
            "checks": {
                "latency": {"status": "passed", "value": 150.5, "threshold": 1000},
                "memory": {"status": "passed", "value": 245.3, "threshold": 1024}
            }
        }
    
    async def _validate_fairness(
        self,
        model_name: str,
        model_version: str,
        model_info: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证公平性"""
        # 简化实现
        return {
            "metrics": {
                "disparate_impact": 0.85,
                "equal_opportunity": 0.92
            },
            "checks": {
                "disparate_impact": {"status": "warning", "value": 0.85, "threshold": 0.8},
                "equal_opportunity": {"status": "passed", "value": 0.92, "threshold": 0.8}
            }
        }
    
    async def _save_state(self):
        """保存状态到文件"""
        state_file = self.data_dir / "lifecycle_state.json"
        
        state_data = {
            "states": {
                k: v.value for k, v in self.lifecycle_states.items()
            },
            "last_updated": datetime.now().isoformat()
        }
        
        async with aiofiles.open(state_file, 'w') as f:
            await f.write(json.dumps(state_data, indent=2, default=str))
    
    async def _save_event(self, event: LifecycleEvent):
        """保存事件到文件"""
        event_file = self.data_dir / f"events_{datetime.now().date()}.json"
        
        events = []
        if event_file.exists():
            async with aiofiles.open(event_file, 'r') as f:
                content = await f.read()
                if content:
                    events = json.loads(content)
        
        events.append(asdict(event))
        
        async with aiofiles.open(event_file, 'w') as f:
            await f.write(json.dumps(events, indent=2, default=str))
    
    async def _save_validation_result(self, result: ValidationResult):
        """保存验证结果到文件"""
        validation_file = self.data_dir / f"validation_{result.validation_id}.json"
        
        async with aiofiles.open(validation_file, 'w') as f:
            await f.write(json.dumps(asdict(result), indent=2, default=str))
    
    def _load_state(self):
        """从文件加载状态"""
        state_file = self.data_dir / "lifecycle_state.json"
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                
                for k, v in state_data.get("states", {}).items():
                    self.lifecycle_states[k] = LifecycleState(v)
        
        # 加载事件
        for event_file in self.data_dir.glob("events_*.json"):
            with open(event_file, 'r') as f:
                events_data = json.load(f)
                
                for event_data in events_data:
                    event = LifecycleEvent(**event_data)
                    self.lifecycle_events.append(event)
        
        # 加载验证结果
        for validation_file in self.data_dir.glob("validation_*.json"):
            with open(validation_file, 'r') as f:
                validation_data = json.load(f)
                result = ValidationResult(**validation_data)
                self.validation_results[result.validation_id] = result
