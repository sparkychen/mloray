from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_ui_html
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import logging
import time
from typing import Dict, Any, Optional
import asyncio

from .dependencies import get_settings, get_db_session
from .middlewares import LoggingMiddleware, MetricsMiddleware
from .routers import data, training, models, monitoring, feature_store
from src.core.config import Settings
from src.utils.logging import setup_logging
from src.utils.metrics import setup_metrics, get_metrics_registry

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management
    """
    # Startup
    logger.info("Starting up MLOps Platform...")
    
    # Initialize connections
    settings = get_settings()
    
    # Initialize Ray
    try:
        import ray
        if not ray.is_initialized():
            ray.init(
                address=settings.ray.address,
                num_cpus=settings.ray.num_cpus,
                num_gpus=settings.ray.num_gpus,
                object_store_memory=settings.ray.object_store_memory,
                dashboard_host=settings.ray.dashboard_host,
                dashboard_port=settings.ray.dashboard_port,
                ignore_reinit_error=True
            )
            logger.info(f"Ray initialized: {ray.cluster_resources()}")
    except Exception as e:
        logger.warning(f"Failed to initialize Ray: {e}")
    
    # Initialize MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_registry_uri(settings.mlflow.registry_uri)
        mlflow.set_experiment(settings.mlflow.experiment_name)
        logger.info(f"MLflow initialized: {settings.mlflow.tracking_uri}")
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow: {e}")
    
    # Setup metrics
    setup_metrics()
    
    yield
    
    # Shutdown
    logger.info("Shutting down MLOps Platform...")
    
    # Cleanup Ray
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown completed")
    except:
        pass

# Create FastAPI application
app = FastAPI(
    title="Enterprise MLOps Platform",
    description="Enterprise-grade MLOps platform with full lifecycle management",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan
)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    # Add tags metadata
    openapi_schema["tags"] = [
        {
            "name": "data",
            "description": "Data management operations"
        },
        {
            "name": "training",
            "description": "Model training and hyperparameter optimization"
        },
        {
            "name": "models",
            "description": "Model management, deployment, and serving"
        },
        {
            "name": "monitoring",
            "description": "Model and data monitoring"
        },
        {
            "name": "feature-store",
            "description": "Feature store operations"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted hosts middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # Configure properly in production
)

# Custom middlewares
app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)

# Include routers
app.include_router(data.router, prefix="/api/v1", tags=["data"])
app.include_router(training.router, prefix="/api/v1", tags=["training"])
app.include_router(models.router, prefix="/api/v1", tags=["models"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])
app.include_router(feature_store.router, prefix="/api/v1", tags=["feature-store"])

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check(
    request: Request,
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Health check endpoint
    """
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": app.version,
        "environment": settings.environment,
        "services": {}
    }
    
    # Check database connection
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession
        
        session: AsyncSession = await anext(get_db_session())
        await session.execute(text("SELECT 1"))
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check MLflow connection
    try:
        import mlflow
        experiments = mlflow.search_experiments()
        health_status["services"]["mlflow"] = "healthy"
    except Exception as e:
        health_status["services"]["mlflow"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Ray connection
    try:
        import ray
        if ray.is_initialized():
            resources = ray.cluster_resources()
            health_status["services"]["ray"] = {
                "status": "healthy",
                "resources": resources
            }
        else:
            health_status["services"]["ray"] = "unhealthy: not initialized"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["ray"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

# Metrics endpoint
@app.get("/metrics", tags=["monitoring"])
async def metrics_endpoint():
    """
    Prometheus metrics endpoint
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    metrics = generate_latest(get_metrics_registry())
    return Response(
        metrics,
        media_type=CONTENT_TYPE_LATEST,
        headers={"Cache-Control": "no-cache"}
    )

# Documentation endpoints
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
    return get_redoc_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"
    )

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": time.time(),
                "path": request.url.path
            }
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": time.time(),
                "path": request.url.path
            }
        }
    )

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Enterprise MLOps Platform API",
        "version": app.version,
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/api/v1/openapi.json",
        "health": "/health",
        "metrics": "/metrics"
    }
