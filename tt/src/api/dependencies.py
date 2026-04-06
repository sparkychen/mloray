from typing import AsyncGenerator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from jose import JWTError, jwt
from pydantic import ValidationError
import redis.asyncio as redis
from neo4j import AsyncGraphDatabase
import aiobotocore.session
import mlflow
import ray
from contextlib import asynccontextmanager
import logging

from src.core.config import Settings, get_settings
from src.schemas.api import TokenPayload, User
from src.utils.security import verify_password, get_password_hash

logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()

# Dependency: Get settings
def get_settings() -> Settings:
    return get_settings()

# Dependency: Get database session
@asynccontextmanager
async def get_db_session(settings: Settings = Depends(get_settings)) -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session
    """
    engine = create_async_engine(
        settings.database.async_postgres_url,
        echo=settings.debug,
        pool_pre_ping=True,
        pool_size=20,
        max_overflow=30,
        pool_recycle=3600
    )
    
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Dependency: Get Redis connection
@asynccontextmanager
async def get_redis_client(settings: Settings = Depends(get_settings)) -> AsyncGenerator[redis.Redis, None]:
    """
    Dependency to get Redis client
    """
    client = redis.Redis(
        host="redis" if settings.environment == "prod" else "localhost",
        port=6379,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True
    )
    
    try:
        yield client
    finally:
        await client.close()

# Dependency: Get Neo4j driver
@asynccontextmanager
async def get_neo4j_driver(settings: Settings = Depends(get_settings)) -> AsyncGenerator:
    """
    Dependency to get Neo4j driver
    """
    driver = AsyncGraphDatabase.driver(
        settings.neo4j.uri,
        auth=(settings.neo4j.user, settings.neo4j.password)
    )
    
    try:
        yield driver
    finally:
        await driver.close()

# Dependency: Get S3 client
@asynccontextmanager
async def get_s3_client(settings: Settings = Depends(get_settings)) -> AsyncGenerator:
    """
    Dependency to get S3 client
    """
    session = aiobotocore.session.get_session()
    
    async with session.create_client(
        's3',
        endpoint_url=settings.storage.minio_endpoint if settings.storage.use_minio else None,
        aws_access_key_id=settings.storage.s3_access_key,
        aws_secret_access_key=settings.storage.s3_secret_key,
        region_name=settings.storage.s3_region
    ) as client:
        yield client

# Dependency: Get MLflow client
def get_mlflow_client(settings: Settings = Depends(get_settings)):
    """
    Dependency to get MLflow client
    """
    from mlflow.tracking import MlflowClient
    return MlflowClient(settings.mlflow.tracking_uri)

# Dependency: Get Ray client
def get_ray_client():
    """
    Dependency to get Ray client
    """
    if not ray.is_initialized():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ray cluster not initialized"
        )
    return ray

# Security
security = HTTPBearer(
    scheme_name="JWT",
    description="Enter JWT token in format: Bearer <token>"
)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings: Settings = Depends(get_settings)
) -> User:
    """
    Dependency to get current authenticated user
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    try:
        payload = jwt.decode(
            token,
            settings.security.secret_key,
            algorithms=[settings.security.algorithm]
        )
        token_data = TokenPayload(**payload)
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    
    # In a real application, you would fetch the user from database
    # This is a simplified example
    user = User(
        id=token_data.sub,
        username=token_data.username,
        email=token_data.email,
        is_active=token_data.is_active,
        is_superuser=token_data.is_superuser
    )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user

def get_current_active_superuser(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to get current active superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

# Rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}
    
    async def __call__(self, request: Request) -> bool:
        client_ip = request.client.host
        current_time = time.time()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < 60
        ]
        
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests"
            )
        
        self.requests[client_ip].append(current_time)
        return True

# Feature store dependency
def get_feature_store(settings: Settings = Depends(get_settings)):
    """
    Dependency to get Feast feature store
    """
    from feast import FeatureStore
    
    store = FeatureStore(
        repo_path=settings.feature_store.feast_registry_path
    )
    
    return store
