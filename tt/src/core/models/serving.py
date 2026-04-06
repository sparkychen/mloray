"""
模型服务模块
基于Ray Serve的高性能、可扩展模型服务
支持多模型部署、动态负载均衡、自动扩缩容
"""

import ray
from ray import serve
from ray.serve import Application, Deployment
from ray.serve.handle import DeploymentHandle
from ray.serve.http_adapters import json_request
from typing import Dict, Any, List, Optional, Union
import json
import numpy as np
import pandas as pd
from datetime import datetime
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
from enum import Enum
import time
import hashlib
import pickle
from prometheus_client import Counter, Histogram, Gauge
import logging
from functools import wraps

from ..config import settings

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """模型类型枚举"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SKLEARN = "sklearn"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    version: str
    type: ModelType
    path: str
    framework: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    metadata: Dict[str, Any]
    loaded_at: datetime
    memory_mb: float
    is_loaded: bool = False


@dataclass
class ServingConfig:
    """服务配置"""
    num_replicas: int = 2
    max_concurrent_queries: int = 100
    ray_actor_options: Dict[str, Any] = None
    autoscaling_config: Optional[Dict[str, Any]] = None
    health_check_period_s: float = 10.0
    health_check_timeout_s: float = 30.0
    graceful_shutdown_timeout_s: float = 20.0
    
    def __post_init__(self):
        if self.ray_actor_options is None:
            self.ray_actor_options = {
                "num_cpus": 2,
                "num_gpus": 0.5,
                "memory": 4 * 1024 * 1024 * 1024,  # 4GB
            }
        if self.autoscaling_config is None:
            self.autoscaling_config = {
                "min_replicas": 1,
                "max_replicas": 10,
                "target_num_ongoing_requests_per_replica": 10,
                "upscale_delay_s": 30.0,
                "downscale_delay_s": 300.0,
            }


class ModelServer:
    """模型服务器"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.models: Dict[str, ModelInfo] = {}
        self.deployments: Dict[str, DeploymentHandle] = {}
        self._init_metrics()
        self._init_serve()
    
    def _init_metrics(self):
        """初始化性能指标"""
        # 请求指标
        self.request_counter = Counter(
            'model_server_requests_total',
            'Total number of model server requests',
            ['model_name', 'version', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'model_server_request_duration_seconds',
            'Model server request duration',
            ['model_name', 'version', 'endpoint'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
        )
        
        self.request_queue_size = Gauge(
            'model_server_request_queue_size',
            'Model server request queue size',
            ['model_name', 'version']
        )
        
        # 模型指标
        self.model_load_time = Histogram(
            'model_server_model_load_time_seconds',
            'Model load time in seconds',
            ['model_name', 'version', 'model_type']
        )
        
        self.model_memory_usage = Gauge(
            'model_server_model_memory_bytes',
            'Model memory usage in bytes',
            ['model_name', 'version']
        )
        
        self.model_cache_hits = Counter(
            'model_server_cache_hits_total',
            'Total number of model cache hits',
            ['model_name', 'version']
        )
        
        self.model_cache_misses = Counter(
            'model_server_cache_misses_total',
            'Total number of model cache misses',
            ['model_name', 'version']
        )
    
    def _init_serve(self):
        """初始化Ray Serve"""
        if not serve.context._internal_client:
            serve.start(
                http_options={"host": self.host, "port": self.port},
                dedicated_cpu=False
            )
    
    async def deploy_model(
        self,
        model_name: str,
        model_version: str,
        model_path: str,
        model_type: ModelType,
        config: ServingConfig = None
    ) -> str:
        """
        部署模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            model_path: 模型路径
            model_type: 模型类型
            config: 服务配置
            
        Returns:
            部署ID
        """
        try:
            config = config or ServingConfig()
            
            # 创建部署
            deployment = self._create_deployment(
                model_name=model_name,
                model_version=model_version,
                model_path=model_path,
                model_type=model_type,
                config=config
            )
            
            # 部署到Serve
            handle = serve.run(
                deployment.bind(),
                name=f"{model_name}_{model_version}",
                route_prefix=f"/models/{model_name}/{model_version}",
                host=self.host,
                port=self.port
            )
            
            # 记录部署
            deployment_id = f"{model_name}_{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.deployments[deployment_id] = handle
            
            # 记录模型信息
            model_info = ModelInfo(
                name=model_name,
                version=model_version,
                type=model_type,
                path=model_path,
                framework=model_type.value,
                input_schema=self._infer_input_schema(model_type, model_path),
                output_schema=self._infer_output_schema(model_type, model_path),
                metadata={
                    "deployed_at": datetime.now().isoformat(),
                    "config": asdict(config)
                },
                loaded_at=datetime.now(),
                memory_mb=await self._estimate_model_memory(model_path, model_type)
            )
            
            self.models[deployment_id] = model_info
            
            logger.info(f"模型部署成功: {model_name}:{model_version}, 部署ID: {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"模型部署失败: {e}")
            raise
    
    def _create_deployment(
        self,
        model_name: str,
        model_version: str,
        model_path: str,
        model_type: ModelType,
        config: ServingConfig
    ) -> Deployment:
        """创建部署定义"""
        
        class ModelDeployment:
            """模型部署类"""
            
            def __init__(self):
                self.model = None
                self.model_name = model_name
                self.model_version = model_version
                self.model_type = model_type
                self.model_path = model_path
                self.load_time = None
            
            async def __init__(self):
                """异步初始化"""
                await self.load_model()
            
            async def load_model(self):
                """加载模型"""
                start_time = time.time()
                
                try:
                    if model_type == ModelType.PYTORCH:
                        import torch
                        self.model = torch.load(model_path, map_location=torch.device('cpu'))
                    elif model_type == ModelType.XGBOOST:
                        import xgboost as xgb
                        self.model = xgb.Booster(model_file=model_path)
                    elif model_type == ModelType.SKLEARN:
                        with open(model_path, 'rb') as f:
                            self.model = pickle.load(f)
                    elif model_type == ModelType.HUGGINGFACE:
                        from transformers import AutoModel, AutoTokenizer
                        self.model = AutoModel.from_pretrained(model_path)
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    else:
                        raise ValueError(f"不支持的模型类型: {model_type}")
                    
                    self.load_time = time.time() - start_time
                    logger.info(f"模型加载完成: {model_name}:{model_version}, 耗时: {self.load_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"模型加载失败: {e}")
                    raise
            
            async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """预测接口"""
                start_time = time.time()
                
                try:
                    # 根据模型类型调用不同的预测方法
                    if model_type == ModelType.PYTORCH:
                        result = await self._predict_pytorch(data)
                    elif model_type == ModelType.XGBOOST:
                        result = await self._predict_xgboost(data)
                    elif model_type == ModelType.SKLEARN:
                        result = await self._predict_sklearn(data)
                    elif model_type == ModelType.HUGGINGFACE:
                        result = await self._predict_huggingface(data)
                    else:
                        result = {"predictions": [], "error": f"Unsupported model type: {model_type}"}
                    
                    latency = time.time() - start_time
                    result["latency_ms"] = latency * 1000
                    result["model_name"] = model_name
                    result["model_version"] = model_version
                    result["timestamp"] = datetime.now().isoformat()
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"预测失败: {e}")
                    return {
                        "error": str(e),
                        "model_name": model_name,
                        "model_version": model_version
                    }
            
            async def batch_predict(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """批量预测接口"""
                results = []
                for data in batch_data:
                    result = await self.predict(data)
                    results.append(result)
                return results
            
            async def _predict_pytorch(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """PyTorch模型预测"""
                import torch
                
                # 准备输入
                if "tensor" in data:
                    input_tensor = torch.tensor(data["tensor"], dtype=torch.float32)
                elif "features" in data:
                    input_tensor = torch.tensor(data["features"], dtype=torch.float32)
                else:
                    raise ValueError("输入数据需要包含'tensor'或'features'字段")
                
                # 预测
                self.model.eval()
                with torch.no_grad():
                    output = self.model(input_tensor)
                
                # 处理输出
                if output.shape[1] == 1:
                    # 二分类
                    predictions = (output > 0.5).int().numpy().tolist()
                    probabilities = torch.sigmoid(output).numpy().tolist()
                else:
                    # 多分类
                    _, predicted = torch.max(output, 1)
                    predictions = predicted.numpy().tolist()
                    probabilities = torch.softmax(output, dim=1).numpy().tolist()
                
                return {
                    "predictions": predictions,
                    "probabilities": probabilities
                }
            
            async def _predict_xgboost(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """XGBoost模型预测"""
                import xgboost as xgb
                
                # 准备输入
                if "features" in data:
                    import numpy as np
                    features = np.array(data["features"])
                    dmatrix = xgb.DMatrix(features)
                else:
                    raise ValueError("输入数据需要包含'features'字段")
                
                # 预测
                predictions = self.model.predict(dmatrix)
                
                # 获取概率（如果可用）
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(dmatrix)
                    return {
                        "predictions": predictions.tolist(),
                        "probabilities": probabilities.tolist()
                    }
                else:
                    return {
                        "predictions": predictions.tolist()
                    }
            
            async def _predict_sklearn(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """Scikit-learn模型预测"""
                import numpy as np
                
                # 准备输入
                if "features" in data:
                    features = np.array(data["features"])
                else:
                    raise ValueError("输入数据需要包含'features'字段")
                
                # 预测
                predictions = self.model.predict(features)
                
                # 获取概率（如果可用）
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(features)
                    return {
                        "predictions": predictions.tolist(),
                        "probabilities": probabilities.tolist()
                    }
                else:
                    return {
                        "predictions": predictions.tolist()
                    }
            
            async def _predict_huggingface(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """HuggingFace模型预测"""
                from transformers import pipeline
                
                # 准备输入
                if "text" in data:
                    text = data["text"]
                else:
                    raise ValueError("输入数据需要包含'text'字段")
                
                # 创建推理管道
                task = data.get("task", "text-classification")
                pipe = pipeline(task, model=self.model, tokenizer=self.tokenizer)
                
                # 预测
                result = pipe(text)
                
                return {
                    "predictions": result
                }
            
            async def health_check(self) -> Dict[str, Any]:
                """健康检查"""
                return {
                    "status": "healthy" if self.model is not None else "unhealthy",
                    "model_name": model_name,
                    "model_version": model_version,
                    "load_time": self.load_time,
                    "timestamp": datetime.now().isoformat()
                }
            
            async def get_model_info(self) -> Dict[str, Any]:
                """获取模型信息"""
                return {
                    "name": model_name,
                    "version": model_version,
                    "type": model_type.value,
                    "loaded": self.model is not None,
                    "load_time": self.load_time,
                    "input_schema": self._infer_input_schema(),
                    "output_schema": self._infer_output_schema()
                }
        
        # 创建部署配置
        deployment_config = {
            "num_replicas": config.num_replicas,
            "max_concurrent_queries": config.max_concurrent_queries,
            "ray_actor_options": config.ray_actor_options,
            "health_check_period_s": config.health_check_period_s,
            "health_check_timeout_s": config.health_check_timeout_s,
            "graceful_shutdown_timeout_s": config.graceful_shutdown_timeout_s
        }
        
        if config.autoscaling_config:
            deployment_config["autoscaling_config"] = config.autoscaling_config
        
        return Deployment(
            ModelDeployment,
            name=f"{model_name}_{model_version}",
            **deployment_config
        )
    
    async def predict(
        self,
        model_name: str,
        model_version: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        执行预测
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            data: 输入数据
            
        Returns:
            预测结果
        """
        start_time = time.time()
        
        try:
            # 查找部署
            deployment_id = f"{model_name}_{model_version}"
            deployment = self.deployments.get(deployment_id)
            
            if not deployment:
                # 尝试查找最近部署
                for dep_id, dep in self.deployments.items():
                    if dep_id.startswith(f"{model_name}_"):
                        deployment = dep
                        break
            
            if not deployment:
                raise ValueError(f"模型 {model_name}:{model_version} 未部署")
            
            # 执行预测
            if isinstance(data, list):
                result = await deployment.batch_predict.remote(data)
            else:
                result = await deployment.predict.remote(data)
            
            # 记录指标
            latency = time.time() - start_time
            self.request_duration.labels(
                model_name=model_name,
                version=model_version,
                endpoint="predict"
            ).observe(latency)
            
            self.request_counter.labels(
                model_name=model_name,
                version=model_version,
                endpoint="predict",
                status="success"
            ).inc()
            
            return result
            
        except Exception as e:
            # 记录错误
            self.request_counter.labels(
                model_name=model_name,
                version=model_version or "unknown",
                endpoint="predict",
                status="error"
            ).inc()
            
            logger.error(f"预测失败: {e}")
            raise
    
    async def undeploy_model(self, deployment_id: str) -> bool:
        """
        取消部署模型
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            是否成功
        """
        try:
            if deployment_id in self.deployments:
                # 从Serve中删除部署
                serve.delete(deployment_id)
                
                # 清理本地记录
                del self.deployments[deployment_id]
                if deployment_id in self.models:
                    del self.models[deployment_id]
                
                logger.info(f"模型取消部署成功: {deployment_id}")
                return True
            else:
                logger.warning(f"部署不存在: {deployment_id}")
                return False
                
        except Exception as e:
            logger.error(f"取消部署失败: {e}")
            return False
    
    async def list_deployments(self) -> List[Dict[str, Any]]:
        """列出所有部署"""
        deployments = []
        
        for deployment_id, handle in self.deployments.items():
            if deployment_id in self.models:
                model_info = self.models[deployment_id]
                
                # 获取运行状态
                try:
                    health = await handle.health_check.remote()
                    status = health.get("status", "unknown")
                except:
                    status = "unreachable"
                
                deployments.append({
                    "deployment_id": deployment_id,
                    "model_name": model_info.name,
                    "model_version": model_info.version,
                    "type": model_info.type.value,
                    "status": status,
                    "deployed_at": model_info.metadata.get("deployed_at"),
                    "memory_mb": model_info.memory_mb,
                    "config": model_info.metadata.get("config", {})
                })
        
        return deployments
    
    async def scale_deployment(
        self,
        deployment_id: str,
        num_replicas: int
    ) -> bool:
        """
        扩缩容部署
        
        Args:
            deployment_id: 部署ID
            num_replicas: 副本数
            
        Returns:
            是否成功
        """
        try:
            if deployment_id in self.deployments:
                # 更新部署配置
                deployment = self.deployments[deployment_id]
                
                # 这里需要调用Serve的API来更新副本数
                # 注意：Ray Serve 2.x 的API有所变化
                serve.update_deployment_options(
                    deployment_id,
                    num_replicas=num_replicas
                )
                
                logger.info(f"部署扩缩容成功: {deployment_id} -> {num_replicas} replicas")
                return True
            else:
                logger.warning(f"部署不存在: {deployment_id}")
                return False
                
        except Exception as e:
            logger.error(f"扩缩容失败: {e}")
            return False
    
    def _infer_input_schema(
        self,
        model_type: ModelType,
        model_path: str
    ) -> Dict[str, Any]:
        """推断输入模式"""
        # 简化实现，实际中需要根据模型类型和文件推断
        schema = {
            "type": "object",
            "properties": {}
        }
        
        if model_type in [ModelType.PYTORCH, ModelType.TENSORFLOW]:
            schema["properties"]["features"] = {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "description": "特征矩阵"
            }
        elif model_type == ModelType.XGBOOST:
            schema["properties"]["features"] = {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "description": "特征矩阵"
            }
        elif model_type == ModelType.SKLEARN:
            schema["properties"]["features"] = {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "description": "特征矩阵"
            }
        elif model_type == ModelType.HUGGINGFACE:
            schema["properties"]["text"] = {
                "type": "string",
                "description": "输入文本"
            }
            schema["properties"]["task"] = {
                "type": "string",
                "enum": ["text-classification", "token-classification", "question-answering"],
                "default": "text-classification"
            }
        
        return schema
    
    def _infer_output_schema(
        self,
        model_type: ModelType,
        model_path: str
    ) -> Dict[str, Any]:
        """推断输出模式"""
        # 简化实现
        schema = {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "预测结果"
                },
                "probabilities": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "预测概率"
                },
                "latency_ms": {
                    "type": "number",
                    "description": "预测延迟(毫秒)"
                },
                "model_name": {
                    "type": "string",
                    "description": "模型名称"
                },
                "model_version": {
                    "type": "string",
                    "description": "模型版本"
                },
                "timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "预测时间"
                }
            }
        }
        
        return schema
    
    async def _estimate_model_memory(
        self,
        model_path: str,
        model_type: ModelType
    ) -> float:
        """估计模型内存使用"""
        import os
        
        try:
            if os.path.isdir(model_path):
                # 计算目录大小
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(model_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total_size += os.path.getsize(fp)
            else:
                total_size = os.path.getsize(model_path)
            
            # 转换为MB
            memory_mb = total_size / (1024 * 1024)
            
            # 根据模型类型调整估计
            if model_type == ModelType.PYTORCH:
                # PyTorch模型加载后通常占用更多内存
                memory_mb *= 1.5
            elif model_type == ModelType.HUGGINGFACE:
                # HuggingFace模型通常较大
                memory_mb *= 2.0
            
            return memory_mb
            
        except Exception:
            # 默认估计
            if model_type == ModelType.PYTORCH:
                return 500.0  # 500MB
            elif model_type == ModelType.HUGGINGFACE:
                return 1000.0  # 1GB
            else:
                return 100.0  # 100MB
    
    async def warmup_model(
        self,
        model_name: str,
        model_version: str,
        num_requests: int = 10
    ):
        """预热模型"""
        try:
            deployment_id = f"{model_name}_{model_version}"
            if deployment_id not in self.deployments:
                logger.warning(f"部署不存在: {deployment_id}")
                return
            
            deployment = self.deployments[deployment_id]
            
            # 生成预热数据
            warmup_data = self._generate_warmup_data(
                model_name, model_version, num_requests
            )
            
            # 发送预热请求
            for data in warmup_data:
                try:
                    await deployment.predict.remote(data)
                except Exception as e:
                    logger.debug(f"预热请求失败（预期中）: {e}")
            
            logger.info(f"模型预热完成: {model_name}:{model_version}")
            
        except Exception as e:
            logger.error(f"模型预热失败: {e}")
    
    def _generate_warmup_data(
        self,
        model_name: str,
        model_version: str,
        num_requests: int
    ) -> List[Dict[str, Any]]:
        """生成预热数据"""
        warmup_data = []
        
        # 获取模型信息
        deployment_id = f"{model_name}_{model_version}"
        if deployment_id in self.models:
            model_info = self.models[deployment_id]
            model_type = model_info.type
        else:
            model_type = ModelType.SKLEARN  # 默认
        
        # 根据模型类型生成数据
        for i in range(num_requests):
            if model_type in [ModelType.PYTORCH, ModelType.TENSORFLOW, ModelType.SKLEARN, ModelType.XGBOOST]:
                data = {
                    "features": [[0.1] * 10 for _ in range(5)]  # 5个样本，每个10个特征
                }
            elif model_type == ModelType.HUGGINGFACE:
                data = {
                    "text": "This is a warmup request.",
                    "task": "text-classification"
                }
            else:
                data = {"data": [0.1] * 10}
            
            warmup_data.append(data)
        
        return warmup_data
    
    async def shutdown(self):
        """关闭服务"""
        try:
            serve.shutdown()
            logger.info("模型服务已关闭")
        except Exception as e:
            logger.error(f"关闭服务失败: {e}")


# 装饰器：监控函数执行
def monitor_model_call(func):
    """监控模型调用的装饰器"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(self, *args, **kwargs)
            
            # 记录成功
            duration = time.time() - start_time
            if hasattr(self, 'request_duration'):
                # 这里需要模型名称和版本，可能需要从参数中提取
                pass
            
            return result
            
        except Exception as e:
            # 记录失败
            if hasattr(self, 'request_counter'):
                # 这里需要模型名称和版本
                pass
            
            raise
    
    return wrapper
