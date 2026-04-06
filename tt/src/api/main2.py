"""
API主应用模块
FastAPI应用入口，集成所有路由和中间件
"""

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GzipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import logging
from datetime import datetime
import uvicorn
import asyncio
import os
from pathlib import Path
import yaml
import json

from .routers import data, training, models, monitoring, feature_store
from .dependencies import (
    get_settings,
    get_data_validator,
    get_feature_store,
    get_model_registry,
    get_model_server,
    get_deployment_manager,
    get_lifecycle_manager,
    get_workflow_orchestrator
)
from .middlewares import LoggingMiddleware, MetricsMiddleware, AuthenticationMiddleware
from ..core.config import settings
from ..utils.logging import setup_logging
from ..utils.metrics import setup_metrics

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    - 启动时：初始化组件
    - 关闭时：清理资源
    """
    # 启动
    logger.info("启动MLOps API服务...")
    
    # 初始化组件
    await initialize_components()
    
    # 启动后台任务
    background_tasks = []
    
    # 健康检查任务
    health_task = asyncio.create_task(health_check_background())
    background_tasks.append(health_task)
    
    yield
    
    # 关闭
    logger.info("关闭MLOps API服务...")
    
    # 取消后台任务
    for task in background_tasks:
        task.cancel()
    
    # 清理组件
    await cleanup_components()


async def initialize_components():
    """初始化组件"""
    logger.info("初始化组件...")
    
    # 创建必要的目录
    directories = [
        "data",
        "models",
        "logs",
        "pipelines",
        "feature_store",
        "monitoring"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # 初始化监控指标
    setup_metrics()
    
    logger.info("组件初始化完成")


async def cleanup_components():
    """清理组件"""
    logger.info("清理组件...")


async def health_check_background():
    """后台健康检查"""
    while True:
        try:
            # 这里可以添加各种健康检查逻辑
            await asyncio.sleep(60)  # 每分钟检查一次
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"健康检查失败: {e}")


# 创建FastAPI应用
app = FastAPI(
    title="企业级MLOps平台API",
    description="基于FastAPI + MLflow + Ray + DVC的全生命周期MLOps平台",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# 添加中间件
# 1. CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 可信主机中间件
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else settings.allowed_hosts
)

# 3. Gzip压缩中间件
app.add_middleware(GzipMiddleware, minimum_size=1000)

# 4. 日志中间件
app.add_middleware(LoggingMiddleware)

# 5. 指标中间件
app.add_middleware(MetricsMiddleware)

# 6. 认证中间件（如果启用认证）
if not settings.debug:
    app.add_middleware(AuthenticationMiddleware)


# 自定义OpenAPI文档
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # 添加安全定义
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # 添加全局安全
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    # 添加标签
    openapi_schema["tags"] = [
        {
            "name": "数据管理",
            "description": "数据加载、验证、版本控制等操作"
        },
        {
            "name": "模型管理",
            "description": "模型训练、注册、部署、版本管理等操作"
        },
        {
            "name": "监控管理",
            "description": "数据质量、漂移检测、性能监控等操作"
        },
        {
            "name": "特征存储",
            "description": "特征注册、检索、物化等操作"
        },
        {
            "name": "系统管理",
            "description": "健康检查、配置管理、日志查看等操作"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# 自定义文档界面
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"
    )


# 包含路由
# 数据管理路由
app.include_router(
    data.router,
    prefix="/api/v1",
    tags=["数据管理"]
)

# 训练管理路由
app.include_router(
    training.router,
    prefix="/api/v1",
    tags=["模型管理"]
)

# 模型管理路由
app.include_router(
    models.router,
    prefix="/api/v1",
    tags=["模型管理"]
)

# 监控管理路由
app.include_router(
    monitoring.router,
    prefix="/api/v1",
    tags=["监控管理"]
)

# 特征存储路由
app.include_router(
    feature_store.router,
    prefix="/api/v1",
    tags=["特征存储"]
)


# 根路由
@app.get("/")
async def root():
    """根端点"""
    return {
        "service": "企业级MLOps平台API",
        "version": "1.0.0",
        "status": "运行中",
        "timestamp": datetime.now().isoformat(),
        "documentation": "/docs",
        "health": "/health"
    }


# 健康检查
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "database": "connected",  # 实际中应该检查数据库连接
            "mlflow": "connected",    # 实际中应该检查MLflow连接
            "ray": "connected"        # 实际中应该检查Ray连接
        }
    }


# 配置端点
@app.get("/config")
async def get_config():
    """获取配置信息（不包含敏感信息）"""
    config_safe = {
        "environment": settings.environment,
        "debug": settings.debug,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
        "mlflow": {
            "tracking_uri": settings.mlflow.tracking_uri,
            "experiment_name": settings.mlflow.experiment_name
        },
        "ray": {
            "address": settings.ray.address,
            "num_cpus": settings.ray.num_cpus,
            "num_gpus": settings.ray.num_gpus
        },
        "monitoring": {
            "drift_threshold": settings.monitoring.drift_threshold,
            "performance_threshold": settings.monitoring.performance_threshold
        }
    }
    
    return config_safe


# 系统信息
@app.get("/system/info")
async def system_info():
    """获取系统信息"""
    import sys
    import platform
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory": {},  # 实际中可以添加内存信息
        "disk": {}     # 实际中可以添加磁盘信息
    }


# 日志端点
@app.get("/logs")
async def get_logs(
    level: str = "INFO",
    lines: int = 100,
    service: str = "api"
):
    """获取日志"""
    log_file = f"logs/{service}.log"
    
    if not Path(log_file).exists():
        raise HTTPException(
            status_code=404,
            detail=f"日志文件不存在: {log_file}"
        )
    
    try:
        with open(log_file, 'r') as f:
            log_lines = f.readlines()[-lines:]
        
        # 过滤日志级别
        filtered_logs = [
            line for line in log_lines
            if f" {level} " in line
        ]
        
        return {
            "service": service,
            "level": level,
            "total_lines": len(log_lines),
            "filtered_lines": len(filtered_logs),
            "logs": filtered_logs[-lines:]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"读取日志失败: {str(e)}"
        )


# 指标端点
@app.get("/metrics")
async def get_metrics():
    """获取性能指标"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path,
            "method": request.method,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "内部服务器错误",
            "detail": str(exc) if settings.debug else "请查看服务器日志",
            "path": request.url.path,
            "method": request.method,
            "timestamp": datetime.now().isoformat()
        }
    )


# 启动脚本
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.api_workers if not settings.debug else 1,
        log_level="info" if settings.debug else "warning"
    )
