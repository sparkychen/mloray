"""
模型管理API路由
提供模型部署、预测、版本切换、回滚、重训练等功能
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional, Union
import mlflow
from mlflow.tracking import MlflowClient
import ray
from ray import serve
import json
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import aiohttp
import uuid
from pydantic import BaseModel, Field, validator
from enum import Enum

from ...core.config import settings
from ...core.models.registry import ModelRegistry, ModelStage, DeploymentStatus
from ...core.models.serving import ModelServer
from ...core.models.deployment import DeploymentManager
from ...core.models.lifecycle import ModelLifecycleManager
from ...core.monitoring.drift import ModelMonitor
from ...core.orchestration.workflows import WorkflowOrchestrator
from ..dependencies import get_model_registry, get_model_server, get_deployment_manager
from ..dependencies import get_model_monitor, get_lifecycle_manager, get_workflow_orchestrator
from ..schemas.models import (
    ModelDeployRequest,
    ModelDeployResponse,
    ModelPredictRequest,
    ModelPredictResponse,
    ModelSwitchRequest,
    ModelSwitchResponse,
    ModelRollbackRequest,
    ModelRollbackResponse,
    ModelRetrainRequest,
    ModelRetrainResponse,
    ModelStatusResponse,
    ModelListResponse,
    ModelVersionInfo,
    DeploymentInfo
)

router = APIRouter(prefix="/models", tags=["模型管理"])


class SwitchStrategy(str, Enum):
    """模型切换策略"""
    INSTANT = "instant"
    CANARY = "canary"
    SHADOW = "shadow"


# 缓存生产版本信息
_model_cache = {}


@router.post("/deploy", response_model=ModelDeployResponse, status_code=status.HTTP_202_ACCEPTED)
async def deploy_model(
    request: ModelDeployRequest,
    background_tasks: BackgroundTasks,
    model_registry: ModelRegistry = Depends(get_model_registry),
    deployment_manager: DeploymentManager = Depends(get_deployment_manager)
):
    """
    部署模型到指定环境
    
    - **model_name**: 模型名称
    - **model_version**: 模型版本
    - **stage**: 部署阶段 (Staging/Production)
    - **resources**: 部署资源配置
    - **config**: 额外配置
    """
    try:
        # 验证模型存在
        model_info = await model_registry.get_model(request.model_name, request.model_version)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"模型 {request.model_name}:{request.model_version} 不存在"
            )
        
        # 检查是否已有相同环境的部署
        existing_deployments = await deployment_manager.get_model_deployments(
            request.model_name, 
            request.stage
        )
        
        if existing_deployments and request.stage == "Production":
            # 生产环境只能有一个活跃部署
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"模型 {request.model_name} 在生产环境已有部署，请先切换或下线"
            )
        
        # 执行部署
        deployment_info = await deployment_manager.deploy_model(
            model_name=request.model_name,
            model_version=request.model_version,
            stage=request.stage,
            resources=request.resources,
            config=request.config
        )
        
        # 后台任务：预热模型
        if request.stage in ["Staging", "Production"]:
            background_tasks.add_task(
                warmup_model_deployment,
                deployment_info["endpoint"],
                model_info["framework"],
                request.config.get("warmup_samples", 10)
            )
        
        # 后台任务：记录部署历史
        background_tasks.add_task(
            record_deployment_history,
            model_registry,
            request.model_name,
            request.model_version,
            request.stage,
            deployment_info
        )
        
        return ModelDeployResponse(
            status="deploying",
            deployment_id=deployment_info["deployment_id"],
            endpoint=deployment_info["endpoint"],
            model_name=request.model_name,
            model_version=request.model_version,
            stage=request.stage,
            deployed_at=datetime.now().isoformat(),
            message="模型部署已启动"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型部署失败: {str(e)}"
        )


@router.post("/predict", response_model=ModelPredictResponse)
async def predict(
    request: ModelPredictRequest,
    background_tasks: BackgroundTasks,
    model_server: ModelServer = Depends(get_model_server),
    model_monitor: ModelMonitor = Depends(get_model_monitor)
):
    """
    模型预测接口
    
    - **model_name**: 模型名称
    - **model_version**: 模型版本（可选，默认使用生产版本）
    - **data**: 预测数据
    - **request_id**: 请求ID（可选）
    """
    start_time = datetime.now()
    
    try:
        # 确定模型版本
        model_version = request.model_version
        if not model_version:
            # 获取生产版本
            model_version = await _get_production_version(request.model_name)
            if not model_version:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"模型 {request.model_name} 无生产版本"
                )
        
        # 生成请求ID
        request_id = request.request_id or f"req_{uuid.uuid4().hex[:8]}"
        
        # 验证输入数据
        if not request.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="预测数据不能为空"
            )
        
        # 执行预测
        predictions = await model_server.predict(
            model_name=request.model_name,
            model_version=model_version,
            data=request.data
        )
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # 构建响应
        response = ModelPredictResponse(
            request_id=request_id,
            model_name=request.model_name,
            model_version=model_version,
            predictions=predictions.get("predictions", []),
            probabilities=predictions.get("probabilities"),
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat()
        )
        
        # 后台任务：监控预测
        background_tasks.add_task(
            _monitor_prediction,
            model_monitor,
            request.model_name,
            model_version,
            request.data,
            predictions,
            request_id,
            latency_ms
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预测失败: {str(e)}"
        )


@router.post("/switch", response_model=ModelSwitchResponse)
async def switch_model_version(
    request: ModelSwitchRequest,
    background_tasks: BackgroundTasks,
    model_registry: ModelRegistry = Depends(get_model_registry),
    deployment_manager: DeploymentManager = Depends(get_deployment_manager)
):
    """
    切换模型版本
    
    - **model_name**: 模型名称
    - **from_version**: 源版本
    - **to_version**: 目标版本
    - **strategy**: 切换策略 (instant/canary/shadow)
    - **traffic_percentage**: 流量百分比（canary模式使用）
    - **validation_required**: 是否需要验证
    """
    try:
        # 验证模型版本
        source_model = await model_registry.get_model(request.model_name, request.from_version)
        target_model = await model_registry.get_model(request.model_name, request.to_version)
        
        if not source_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"源模型 {request.model_name}:{request.from_version} 不存在"
            )
        
        if not target_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"目标模型 {request.model_name}:{request.to_version} 不存在"
            )
        
        # 检查目标模型是否已部署
        target_deployments = await deployment_manager.get_model_deployments(
            request.model_name, 
            "Production"
        )
        
        target_already_deployed = any(
            dep["model_version"] == request.to_version 
            for dep in target_deployments
        )
        
        if not target_already_deployed:
            # 先部署目标模型
            await deployment_manager.deploy_model(
                model_name=request.model_name,
                model_version=request.to_version,
                stage="Staging",  # 先部署到Staging
                resources={"num_replicas": 2},
                config={"warmup_samples": 20}
            )
        
        # 执行切换
        switch_result = await deployment_manager.switch_versions(
            model_name=request.model_name,
            from_version=request.from_version,
            to_version=request.to_version,
            strategy=request.strategy,
            traffic_percentage=request.traffic_percentage,
            validation_required=request.validation_required
        )
        
        # 后台任务：监控切换效果
        if request.strategy in [SwitchStrategy.CANARY, SwitchStrategy.SHADOW]:
            background_tasks.add_task(
                monitor_switch_performance,
                model_registry,
                request.model_name,
                request.from_version,
                request.to_version,
                request.strategy
            )
        
        return ModelSwitchResponse(
            switch_id=switch_result["switch_id"],
            model_name=request.model_name,
            from_version=request.from_version,
            to_version=request.to_version,
            strategy=request.strategy,
            switch_time=datetime.now().isoformat(),
            status=switch_result["status"],
            message=switch_result.get("message", "版本切换已启动")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"版本切换失败: {str(e)}"
        )


@router.post("/rollback/{model_name}/{version}", response_model=ModelRollbackResponse)
async def rollback_model(
    model_name: str,
    version: str,
    request: ModelRollbackRequest,
    background_tasks: BackgroundTasks,
    model_registry: ModelRegistry = Depends(get_model_registry),
    lifecycle_manager: ModelLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    回滚到指定模型版本
    
    - **model_name**: 模型名称
    - **version**: 目标回滚版本
    - **reason**: 回滚原因
    - **validate_before_rollback**: 回滚前验证
    """
    try:
        # 获取当前生产版本
        current_version = await _get_production_version(model_name)
        if not current_version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"模型 {model_name} 无生产版本"
            )
        
        if current_version == version:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"目标版本 {version} 已是当前生产版本"
            )
        
        # 验证目标版本
        target_model = await model_registry.get_model(model_name, version)
        if not target_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"目标版本 {model_name}:{version} 不存在"
            )
        
        # 执行回滚
        rollback_result = await lifecycle_manager.rollback_model(
            model_name=model_name,
            target_version=version,
            reason=request.reason,
            validate_before_rollback=request.validate_before_rollback
        )
        
        if not rollback_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"回滚失败: {rollback_result.get('message')}"
            )
        
        # 后台任务：清理旧版本部署
        background_tasks.add_task(
            cleanup_old_deployments,
            lifecycle_manager,
            model_name,
            current_version
        )
        
        return ModelRollbackResponse(
            success=True,
            model_name=model_name,
            from_version=current_version,
            to_version=version,
            reason=request.reason,
            rollback_time=datetime.now().isoformat(),
            message="模型回滚成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"回滚失败: {str(e)}"
        )


@router.post("/retrain/{model_name}", response_model=ModelRetrainResponse, status_code=status.HTTP_202_ACCEPTED)
async def retrain_model(
    model_name: str,
    request: ModelRetrainRequest,
    background_tasks: BackgroundTasks,
    workflow_orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator),
    model_registry: ModelRegistry = Depends(get_model_registry)
):
    """
    触发模型重新训练
    
    - **model_name**: 模型名称
    - **trigger**: 触发原因 (drift/schedule/manual/performance)
    - **data_version**: 数据版本
    - **config**: 训练配置
    """
    try:
        # 获取当前模型信息
        current_version = await _get_production_version(model_name)
        if not current_version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"模型 {model_name} 无生产版本"
            )
        
        # 触发重新训练工作流
        workflow_id = await workflow_orchestrator.trigger_retraining(
            model_name=model_name,
            current_version=current_version,
            trigger_type=request.trigger,
            data_version=request.data_version,
            config=request.config
        )
        
        # 后台任务：监控训练进度
        background_tasks.add_task(
            monitor_retraining_progress,
            workflow_orchestrator,
            workflow_id,
            model_name
        )
        
        return ModelRetrainResponse(
            status="retraining_triggered",
            workflow_id=workflow_id,
            model_name=model_name,
            current_version=current_version,
            trigger=request.trigger,
            triggered_at=datetime.now().isoformat(),
            message="模型重新训练工作流已触发"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"触发重新训练失败: {str(e)}"
        )


@router.get("/status/{model_name}", response_model=ModelStatusResponse)
async def get_model_status(
    model_name: str,
    version: Optional[str] = Query(None, description="模型版本，为空时返回所有版本状态"),
    model_registry: ModelRegistry = Depends(get_model_registry),
    deployment_manager: DeploymentManager = Depends(get_deployment_manager)
):
    """
    获取模型状态信息
    """
    try:
        if version:
            # 获取特定版本状态
            model_info = await model_registry.get_model(model_name, version)
            if not model_info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"模型 {model_name}:{version} 不存在"
                )
            
            deployments = await deployment_manager.get_model_deployments(model_name, version=version)
            
            return ModelStatusResponse(
                model_name=model_name,
                version=version,
                stage=model_info.get("current_stage", "None"),
                status="deployed" if deployments else "registered",
                deployments=deployments,
                metrics=model_info.get("metrics", {}),
                metadata=model_info.get("metadata", {}),
                last_updated=model_info.get("last_updated")
            )
        else:
            # 获取所有版本状态
            all_models = await model_registry.list_models(model_name)
            versions_status = []
            
            for model in all_models:
                version = model["version"]
                deployments = await deployment_manager.get_model_deployments(model_name, version=version)
                
                version_info = ModelVersionInfo(
                    version=version,
                    stage=model.get("current_stage", "None"),
                    status="deployed" if deployments else "registered",
                    deployments=deployments,
                    metrics=model.get("metrics", {}),
                    created_at=model.get("created_at")
                )
                versions_status.append(version_info)
            
            return ModelStatusResponse(
                model_name=model_name,
                versions=versions_status,
                production_version=await _get_production_version(model_name)
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型状态失败: {str(e)}"
        )


@router.get("/", response_model=ModelListResponse)
async def list_models(
    stage: Optional[str] = Query(None, description="过滤阶段"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
    model_registry: ModelRegistry = Depends(get_model_registry)
):
    """
    列出所有模型
    """
    try:
        models = await model_registry.list_all_models()
        
        # 按阶段过滤
        if stage:
            models = [m for m in models if m.get("current_stage") == stage]
        
        # 分页
        total = len(models)
        paginated_models = models[offset:offset + limit]
        
        return ModelListResponse(
            total=total,
            limit=limit,
            offset=offset,
            models=paginated_models
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型列表失败: {str(e)}"
        )


@router.delete("/{model_name}/{version}")
async def delete_model_version(
    model_name: str,
    version: str,
    force: bool = Query(False, description="强制删除，即使有部署"),
    background_tasks: BackgroundTasks,
    lifecycle_manager: ModelLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    删除模型版本
    """
    try:
        # 检查是否有部署
        deployments = await lifecycle_manager.get_model_deployments(model_name, version)
        if deployments and not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"模型版本 {model_name}:{version} 有活跃部署，请先下线或使用force=true"
            )
        
        # 执行删除
        result = await lifecycle_manager.delete_model_version(
            model_name=model_name,
            version=version,
            force=force
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "删除失败")
            )
        
        # 后台任务：清理相关资源
        if deployments and force:
            background_tasks.add_task(
                cleanup_deployment_resources,
                deployments
            )
        
        return {
            "status": "success",
            "model_name": model_name,
            "version": version,
            "deleted_at": datetime.now().isoformat(),
            "message": "模型版本已删除"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除模型版本失败: {str(e)}"
        )


# 辅助函数
async def _get_production_version(model_name: str) -> Optional[str]:
    """获取生产版本"""
    # 检查缓存
    cache_key = f"{model_name}_production"
    if cache_key in _model_cache:
        cached = _model_cache[cache_key]
        if datetime.now() - cached["timestamp"] < timedelta(minutes=5):
            return cached["version"]
    
    # 从MLflow获取
    client = MlflowClient()
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            version = prod_versions[0].version
            # 更新缓存
            _model_cache[cache_key] = {
                "version": version,
                "timestamp": datetime.now()
            }
            return version
    except Exception:
        pass
    
    return None


async def warmup_model_deployment(endpoint: str, framework: str, num_samples: int = 10):
    """预热模型部署"""
    try:
        # 根据框架生成预热数据
        warmup_data = _generate_warmup_data(framework, num_samples)
        
        # 发送预热请求
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/predict",
                json={"data": warmup_data},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    print(f"✅ 模型预热成功: {endpoint}")
                else:
                    print(f"⚠️ 模型预热失败: {response.status}")
                    
    except Exception as e:
        print(f"❌ 模型预热异常: {e}")


def _generate_warmup_data(framework: str, num_samples: int) -> list:
    """生成预热数据"""
    if framework in ["pytorch", "tensorflow"]:
        # 神经网络模型通常需要浮点数组
        return [[0.1] * 10 for _ in range(num_samples)]
    elif framework == "xgboost":
        # XGBoost特征
        return [[0.5] * 20 for _ in range(num_samples)]
    elif framework == "sklearn":
        # Scikit-learn特征
        return [[0.0] * 15 for _ in range(num_samples)]
    else:
        # 默认
        return [[0.0] * 10 for _ in range(num_samples)]


async def record_deployment_history(
    model_registry: ModelRegistry,
    model_name: str,
    model_version: str,
    stage: str,
    deployment_info: dict
):
    """记录部署历史"""
    try:
        await model_registry.record_deployment(
            model_name=model_name,
            model_version=model_version,
            stage=stage,
            deployment_id=deployment_info["deployment_id"],
            endpoint=deployment_info["endpoint"],
            resources=deployment_info.get("resources", {})
        )
    except Exception as e:
        print(f"记录部署历史失败: {e}")


async def _monitor_prediction(
    model_monitor: ModelMonitor,
    model_name: str,
    model_version: str,
    input_data: list,
    predictions: dict,
    request_id: str,
    latency_ms: float
):
    """监控预测数据"""
    try:
        await model_monitor.record_prediction(
            model_name=model_name,
            model_version=model_version,
            input_data=input_data,
            predictions=predictions.get("predictions"),
            probabilities=predictions.get("probabilities"),
            request_id=request_id,
            latency_ms=latency_ms
        )
    except Exception as e:
        print(f"预测监控失败: {e}")


async def monitor_switch_performance(
    model_registry: ModelRegistry,
    model_name: str,
    from_version: str,
    to_version: str,
    strategy: str
):
    """监控切换性能"""
    try:
        # 这里可以实现切换性能监控逻辑
        # 例如：比较两个版本的预测延迟、准确性等
        await asyncio.sleep(300)  # 监控5分钟
        
        print(f"切换监控完成: {model_name} {from_version} -> {to_version} ({strategy})")
        
    except Exception as e:
        print(f"切换监控失败: {e}")


async def cleanup_old_deployments(
    lifecycle_manager: ModelLifecycleManager,
    model_name: str,
    old_version: str
):
    """清理旧版本部署"""
    try:
        await lifecycle_manager.cleanup_old_deployments(model_name, old_version)
    except Exception as e:
        print(f"清理旧部署失败: {e}")


async def monitor_retraining_progress(
    workflow_orchestrator: WorkflowOrchestrator,
    workflow_id: str,
    model_name: str
):
    """监控重新训练进度"""
    try:
        # 轮询训练进度
        max_checks = 60  # 最多检查60次
        check_interval = 30  # 每30秒检查一次
        
        for i in range(max_checks):
            status = await workflow_orchestrator.get_workflow_status(workflow_id)
            
            if status in ["completed", "failed", "cancelled"]:
                print(f"重新训练完成: {model_name} - 状态: {status}")
                break
            
            await asyncio.sleep(check_interval)
        
    except Exception as e:
        print(f"训练进度监控失败: {e}")


async def cleanup_deployment_resources(deployments: list):
    """清理部署资源"""
    try:
        # 这里可以实现清理逻辑
        # 例如：删除Ray Serve部署、清理GPU内存等
        for deployment in deployments:
            print(f"清理部署资源: {deployment['deployment_id']}")
            
    except Exception as e:
        print(f"清理部署资源失败: {e}")
