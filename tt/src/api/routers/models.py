from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import asyncio
import json
import uuid
from enum import Enum
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
import pandas as pd
import numpy as np
import aiohttp
import aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
import logging

from ...schemas.api import (
    ModelDeployRequest, ModelPredictRequest, ModelPredictResponse,
    ModelSwitchRequest, ModelVersionInfo, ModelStatusResponse,
    ModelEvaluationRequest, ModelEvaluationResponse
)
from ...core.config import Settings, get_settings
from ...core.models.registry import ModelRegistry, ModelMetadata, ModelStage
from ...core.models.serving import ModelServer, PredictionRequest, PredictionResponse
from ...core.models.lifecycle import ModelLifecycleManager
from ...core.monitoring.drift import ModelMonitor
from ...utils.metrics import record_metric
from ..dependencies import (
    get_db_session, get_redis_client, get_mlflow_client,
    get_current_user, get_current_active_superuser
)
from ...schemas.models import User

router = APIRouter(prefix="/models", tags=["models"])
logger = logging.getLogger(__name__)

# Models
class ModelDeployResponse(BaseModel):
    status: str
    deployment_id: str
    endpoint: str
    model_name: str
    model_version: str
    stage: str
    deployed_at: str
    message: Optional[str] = None

class ModelRollbackRequest(BaseModel):
    target_version: str
    reason: str = Field(default="Performance degradation")
    validate_before_rollback: bool = Field(default=True)

class ModelRetrainRequest(BaseModel):
    trigger: str = Field(default="drift", description="drift, schedule, manual, error")
    data_version: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = Field(default=0.2, ge=0.0, le=1.0)
    notify_on_complete: bool = Field(default=True)

class BatchPredictRequest(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    data: List[Dict[str, Any]]
    batch_size: int = Field(default=100, ge=1, le=1000)
    return_probabilities: bool = Field(default=False)
    request_id: Optional[str] = None

class BatchPredictResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    predictions: List[Any]
    probabilities: Optional[List[Any]] = None
    batch_count: int
    total_records: int
    avg_latency_ms: float
    timestamp: str

# Initialize services
model_registry = None
model_server = None
model_monitor = None
model_lifecycle = None

async def get_model_registry() -> ModelRegistry:
    """Dependency to get model registry"""
    global model_registry
    if model_registry is None:
        settings = get_settings()
        model_registry = ModelRegistry(
            tracking_uri=settings.mlflow.tracking_uri,
            registry_uri=settings.mlflow.registry_uri
        )
    return model_registry

async def get_model_server() -> ModelServer:
    """Dependency to get model server"""
    global model_server
    if model_server is None:
        settings = get_settings()
        model_server = ModelServer(
            host="0.0.0.0",
            port=8001,
            max_concurrent_queries=100,
            model_registry=await get_model_registry(),
            observability_system=None  # Will be initialized separately
        )
        await model_server.initialize()
    return model_server

async def get_model_monitor() -> ModelMonitor:
    """Dependency to get model monitor"""
    global model_monitor
    if model_monitor is None:
        settings = get_settings()
        model_monitor = ModelMonitor(
            drift_threshold=settings.monitoring.drift_threshold,
            performance_threshold=settings.monitoring.performance_threshold,
            phoenix_endpoint=settings.monitoring.phoenix_collector_endpoint
        )
    return model_monitor

async def get_model_lifecycle() -> ModelLifecycleManager:
    """Dependency to get model lifecycle manager"""
    global model_lifecycle
    if model_lifecycle is None:
        model_lifecycle = ModelLifecycleManager(
            model_registry=await get_model_registry(),
            model_server=await get_model_server(),
            model_monitor=await get_model_monitor()
        )
    return model_lifecycle

# Routes
@router.get("/")
async def list_models(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    stage: Optional[str] = None,
    search: Optional[str] = None,
    registry: ModelRegistry = Depends(get_model_registry),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all registered models with pagination
    """
    try:
        models = await registry.list_models(
            skip=skip,
            limit=limit,
            stage=stage,
            search=search
        )
        
        return {
            "models": models,
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total": len(models),
                "has_more": len(models) == limit
            }
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_name}")
async def get_model_info(
    model_name: str,
    registry: ModelRegistry = Depends(get_model_registry),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific model
    """
    try:
        model_info = await registry.get_model_info(model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_name}/versions")
async def list_model_versions(
    model_name: str,
    stage: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
    registry: ModelRegistry = Depends(get_model_registry),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all versions of a specific model
    """
    try:
        versions = await registry.list_model_versions(
            model_name=model_name,
            stage=stage,
            limit=limit
        )
        
        return {
            "model_name": model_name,
            "versions": versions,
            "total_versions": len(versions)
        }
    except Exception as e:
        logger.error(f"Error listing model versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_name}/versions/{version}")
async def get_model_version(
    model_name: str,
    version: str,
    registry: ModelRegistry = Depends(get_model_registry),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific model version
    """
    try:
        version_info = await registry.get_model_version_info(model_name, version)
        if not version_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} version {version} not found"
            )
        
        return version_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_name}/versions/{version}/deploy")
async def deploy_model(
    model_name: str,
    version: str,
    request: ModelDeployRequest,
    background_tasks: BackgroundTasks,
    registry: ModelRegistry = Depends(get_model_registry),
    server: ModelServer = Depends(get_model_server),
    current_user: User = Depends(get_current_active_superuser)
) -> ModelDeployResponse:
    """
    Deploy a specific model version
    """
    try:
        # Verify model exists
        model_info = await registry.get_model_version_info(model_name, version)
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} version {version} not found"
            )
        
        # Get deployment resources
        resources = request.resources or {
            "num_replicas": 2,
            "num_cpus": 1,
            "num_gpus": 0.5 if model_info.get("framework") == "pytorch" else 0
        }
        
        # Deploy model
        deployment_info = await registry.deploy_model(
            model_name=model_name,
            model_version=version,
            stage=request.stage,
            resources=resources,
            config=request.config
        )
        
        # Update model stage in registry
        await registry.transition_model_stage(
            model_name=model_name,
            version=version,
            stage=request.stage
        )
        
        # Warm up model in background
        background_tasks.add_task(
            warmup_model_deployment,
            deployment_info["endpoint"],
            model_info["framework"],
            model_info.get("input_schema", {})
        )
        
        # Record deployment in audit log
        background_tasks.add_task(
            record_deployment_audit,
            model_name=model_name,
            version=version,
            stage=request.stage,
            deployed_by=current_user.username,
            deployment_info=deployment_info
        )
        
        return ModelDeployResponse(
            status="success",
            deployment_id=deployment_info["deployment_id"],
            endpoint=deployment_info["endpoint"],
            model_name=model_name,
            model_version=version,
            stage=request.stage,
            deployed_at=datetime.now().isoformat(),
            message=f"Model {model_name}:{version} deployed successfully to {request.stage}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def predict(
    request: ModelPredictRequest,
    background_tasks: BackgroundTasks,
    server: ModelServer = Depends(get_model_server),
    monitor: ModelMonitor = Depends(get_model_monitor),
    redis: aioredis.Redis = Depends(get_redis_client),
    current_user: User = Depends(get_current_user)
) -> ModelPredictResponse:
    """
    Make predictions using a deployed model
    """
    start_time = datetime.now()
    request_id = request.request_id or f"req_{uuid.uuid4().hex[:8]}"
    
    try:
        # Rate limiting
        rate_limit_key = f"rate_limit:{current_user.id}:{datetime.now().minute}"
        current_requests = await redis.incr(rate_limit_key)
        if current_requests == 1:
            await redis.expire(rate_limit_key, 60)
        
        if current_requests > 100:  # 100 requests per minute
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 100 requests per minute."
            )
        
        # Determine model version
        model_version = request.model_version
        if not model_version:
            # Get production version
            registry = await get_model_registry()
            model_version = await registry.get_production_version(request.model_name)
            if not model_version:
                raise HTTPException(
                    status_code=404,
                    detail=f"No production version found for model {request.model_name}"
                )
        
        # Make prediction
        prediction = await server.predict(
            model_name=request.model_name,
            model_version=model_version,
            data=request.data
        )
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Record metrics
        background_tasks.add_task(
            record_prediction_metrics,
            request_id=request_id,
            model_name=request.model_name,
            model_version=model_version,
            latency_ms=latency_ms,
            input_size=len(request.data)
        )
        
        # Monitor for drift
        background_tasks.add_task(
            monitor_prediction_drift,
            monitor=monitor,
            model_name=request.model_name,
            model_version=model_version,
            input_data=request.data,
            predictions=prediction["predictions"]
        )
        
        # Cache prediction if needed
        if len(request.data) == 1:  # Only cache single predictions
            background_tasks.add_task(
                cache_prediction,
                redis=redis,
                model_name=request.model_name,
                model_version=model_version,
                input_data=request.data[0],
                prediction=prediction
            )
        
        return ModelPredictResponse(
            request_id=request_id,
            model_name=request.model_name,
            model_version=model_version,
            predictions=prediction["predictions"],
            probabilities=prediction.get("probabilities"),
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch_predict")
async def batch_predict(
    request: BatchPredictRequest,
    background_tasks: BackgroundTasks,
    server: ModelServer = Depends(get_model_server),
    current_user: User = Depends(get_current_user)
) -> BatchPredictResponse:
    """
    Make batch predictions
    """
    start_time = datetime.now()
    request_id = request.request_id or f"batch_{uuid.uuid4().hex[:8]}"
    
    try:
        # Determine model version
        model_version = request.model_version
        if not model_version:
            registry = await get_model_registry()
            model_version = await registry.get_production_version(request.model_name)
            if not model_version:
                raise HTTPException(
                    status_code=404,
                    detail=f"No production version found for model {request.model_name}"
                )
        
        # Split data into batches
        batches = [
            request.data[i:i + request.batch_size]
            for i in range(0, len(request.data), request.batch_size)
        ]
        
        all_predictions = []
        all_probabilities = [] if request.return_probabilities else None
        batch_latencies = []
        
        # Process batches
        for batch_idx, batch in enumerate(batches):
            batch_start = datetime.now()
            
            try:
                result = await server.predict(
                    model_name=request.model_name,
                    model_version=model_version,
                    data=batch
                )
                
                all_predictions.extend(result["predictions"])
                if request.return_probabilities and "probabilities" in result:
                    all_probabilities.extend(result["probabilities"])
                
                batch_latency = (datetime.now() - batch_start).total_seconds() * 1000
                batch_latencies.append(batch_latency)
                
                logger.info(f"Processed batch {batch_idx + 1}/{len(batches)}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                # Fill with None for failed predictions
                all_predictions.extend([None] * len(batch))
                if request.return_probabilities:
                    all_probabilities.extend([None] * len(batch))
        
        total_latency = (datetime.now() - start_time).total_seconds() * 1000
        avg_latency = sum(batch_latencies) / len(batch_latencies) if batch_latencies else 0
        
        return BatchPredictResponse(
            request_id=request_id,
            model_name=request.model_name,
            model_version=model_version,
            predictions=all_predictions,
            probabilities=all_probabilities,
            batch_count=len(batches),
            total_records=len(request.data),
            avg_latency_ms=avg_latency,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_name}/switch")
async def switch_model_version(
    model_name: str,
    request: ModelSwitchRequest,
    background_tasks: BackgroundTasks,
    registry: ModelRegistry = Depends(get_model_registry),
    current_user: User = Depends(get_current_active_superuser)
) -> Dict[str, Any]:
    """
    Switch between model versions
    """
    try:
        # Verify both versions exist
        from_version_info = await registry.get_model_version_info(model_name, request.from_version)
        to_version_info = await registry.get_model_version_info(model_name, request.to_version)
        
        if not from_version_info or not to_version_info:
            raise HTTPException(
                status_code=404,
                detail="One or both model versions not found"
            )
        
        # Perform switch based on strategy
        if request.strategy == "instant":
            result = await registry.switch_model_version_instant(
                model_name=model_name,
                from_version=request.from_version,
                to_version=request.to_version
            )
        elif request.strategy == "canary":
            if not request.traffic_percentage:
                raise HTTPException(
                    status_code=400,
                    detail="traffic_percentage required for canary deployment"
                )
            result = await registry.switch_model_version_canary(
                model_name=model_name,
                from_version=request.from_version,
                to_version=request.to_version,
                traffic_percentage=request.traffic_percentage
            )
        elif request.strategy == "shadow":
            result = await registry.switch_model_version_shadow(
                model_name=model_name,
                from_version=request.from_version,
                to_version=request.to_version
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown switch strategy: {request.strategy}"
            )
        
        # Validate if requested
        if request.validation_required:
            background_tasks.add_task(
                validate_model_switch,
                model_name=model_name,
                from_version=request.from_version,
                to_version=request.to_version,
                strategy=request.strategy
            )
        
        # Record switch in audit log
        background_tasks.add_task(
            record_switch_audit,
            model_name=model_name,
            from_version=request.from_version,
            to_version=request.to_version,
            strategy=request.strategy,
            switched_by=current_user.username,
            switch_info=result
        )
        
        return {
            "status": "success",
            "switch_id": result.get("switch_id"),
            "model_name": model_name,
            "from_version": request.from_version,
            "to_version": request.to_version,
            "strategy": request.strategy,
            "switch_time": datetime.now().isoformat(),
            "details": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_name}/rollback")
async def rollback_model(
    model_name: str,
    request: ModelRollbackRequest,
    background_tasks: BackgroundTasks,
    registry: ModelRegistry = Depends(get_model_registry),
    lifecycle: ModelLifecycleManager = Depends(get_model_lifecycle),
    current_user: User = Depends(get_current_active_superuser)
) -> Dict[str, Any]:
    """
    Rollback to a previous model version
    """
    try:
        # Perform rollback
        rollback_result = await lifecycle.rollback_model(
            model_name=model_name,
            target_version=request.target_version,
            reason=request.reason,
            validate_before_rollback=request.validate_before_rollback
        )
        
        if not rollback_result["success"]:
            raise HTTPException(
                status_code=400,
                detail=rollback_result["message"]
            )
        
        # Record rollback in audit log
        background_tasks.add_task(
            record_rollback_audit,
            model_name=model_name,
            target_version=request.target_version,
            reason=request.reason,
            rolled_back_by=current_user.username,
            rollback_info=rollback_result
        )
        
        return {
            "status": "success",
            "model_name": model_name,
            "target_version": request.target_version,
            "reason": request.reason,
            "rollback_time": datetime.now().isoformat(),
            "details": rollback_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rolling back model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_name}/retrain")
async def retrain_model(
    model_name: str,
    request: ModelRetrainRequest,
    background_tasks: BackgroundTasks,
    registry: ModelRegistry = Depends(get_model_registry),
    lifecycle: ModelLifecycleManager = Depends(get_model_lifecycle),
    current_user: User = Depends(get_current_active_superuser)
) -> Dict[str, Any]:
    """
    Trigger model retraining
    """
    try:
        # Get current production model info
        current_version = await registry.get_production_version(model_name)
        if not current_version:
            raise HTTPException(
                status_code=404,
                detail=f"No production version found for model {model_name}"
            )
        
        # Trigger retraining
        retrain_id = f"retrain_{uuid.uuid4().hex[:8]}"
        
        background_tasks.add_task(
            execute_retraining_pipeline,
            retrain_id=retrain_id,
            model_name=model_name,
            current_version=current_version,
            trigger=request.trigger,
            data_version=request.data_version,
            hyperparameters=request.hyperparameters,
            validation_split=request.validation_split,
            requested_by=current_user.username,
            notify_on_complete=request.notify_on_complete
        )
        
        return {
            "status": "retraining_triggered",
            "retrain_id": retrain_id,
            "model_name": model_name,
            "current_version": current_version,
            "trigger": request.trigger,
            "triggered_at": datetime.now().isoformat(),
            "message": "Retraining pipeline has been queued"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise
