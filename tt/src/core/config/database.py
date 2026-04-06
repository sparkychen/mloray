"""
数据库连接管理模块
支持 PostgreSQL/TimescaleDB 同步和异步连接
"""

import asyncio
from typing import AsyncGenerator, Generator, Optional
from contextlib import asynccontextmanager, contextmanager
from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import sessionmaker, Session
import logging
from ..config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self._sync_engine: Optional[Engine] = None
        self._async_engine: Optional[AsyncEngine] = None
        self._sync_session_factory = None
        self._async_session_factory = None
        self._initialized = False
    
    def init_sync_engine(self) -> Engine:
        """
        初始化同步数据库引擎
        
        Returns:
            Engine: SQLAlchemy 同步引擎
        """
        if self._sync_engine is None:
            db_config = settings.get_database_config()
            
            self._sync_engine = create_engine(
                url=db_config["url"],
                pool_size=db_config["pool_size"],
                max_overflow=db_config["max_overflow"],
                pool_timeout=db_config["pool_timeout"],
                pool_recycle=db_config["pool_recycle"],
                pool_pre_ping=db_config["pool_pre_ping"],
                echo=db_config["echo"],
                echo_pool=False,
                connect_args={
                    "connect_timeout": 10,
                    "application_name": "mlops_sync"
                }
            )
            
            # 创建会话工厂
            self._sync_session_factory = sessionmaker(
                bind=self._sync_engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
                class_=Session
            )
            
            logger.info("Sync database engine initialized")
        
        return self._sync_engine
    
    def init_async_engine(self) -> AsyncEngine:
        """
        初始化异步数据库引擎
        
        Returns:
            AsyncEngine: SQLAlchemy 异步引擎
        """
        if self._async_engine is None:
            db_config = settings.get_database_config()
            
            self._async_engine = create_async_engine(
                url=db_config["async_url"],
                pool_size=db_config["pool_size"],
                max_overflow=db_config["max_overflow"],
                pool_timeout=db_config["pool_timeout"],
                pool_recycle=db_config["pool_recycle"],
                pool_pre_ping=db_config["pool_pre_ping"],
                echo=db_config["echo"],
                echo_pool=False,
                connect_args={
                    "command_timeout": 60,
                    "server_settings": {"application_name": "mlops_async"}
                },
                future=True
            )
            
            # 创建异步会话工厂
            self._async_session_factory = async_sessionmaker(
                bind=self._async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False
            )
            
            logger.info("Async database engine initialized")
        
        return self._async_engine
    
    def initialize(self):
        """初始化所有数据库连接"""
        if not self._initialized:
            self.init_sync_engine()
            self.init_async_engine()
            self._initialized = True
            logger.info("Database manager fully initialized")
    
    @contextmanager
    def get_sync_session(self) -> Generator[Session, None, None]:
        """
        获取同步数据库会话的上下文管理器
        
        Yields:
            Session: SQLAlchemy 同步会话
        """
        if self._sync_session_factory is None:
            self.init_sync_engine()
        
        session: Session = self._sync_session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Sync session error: {e}")
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        获取异步数据库会话的上下文管理器
        
        Yields:
            AsyncSession: SQLAlchemy 异步会话
        """
        if self._async_session_factory is None:
            self.init_async_engine()
        
        session: AsyncSession = self._async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async session error: {e}")
            raise
        finally:
            await session.close()
    
    def get_sync_session_dependency(self) -> Generator[Session, None, None]:
        """
        用于 FastAPI 依赖注入的同步会话生成器
        
        Yields:
            Session: SQLAlchemy 同步会话
        """
        with self.get_sync_session() as session:
            yield session
    
    async def get_async_session_dependency(self) -> AsyncGenerator[AsyncSession, None]:
        """
        用于 FastAPI 依赖注入的异步会话生成器
        
        Yields:
            AsyncSession: SQLAlchemy 异步会话
        """
        async with self.get_async_session() as session:
            yield session
    
    def close(self):
        """关闭所有数据库连接"""
        if self._sync_engine:
            self._sync_engine.dispose()
            self._sync_engine = None
            logger.info("Sync database engine closed")
        
        if self._async_engine:
            # 异步引擎需要在事件循环中关闭
            async def _close_async():
                await self._async_engine.dispose()
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，创建任务关闭
                    asyncio.create_task(_close_async())
                else:
                    # 否则直接运行
                    loop.run_until_complete(_close_async())
            except RuntimeError:
                # 没有事件循环，同步关闭
                pass
            
            self._async_engine = None
            logger.info("Async database engine closed")
        
        self._initialized = False
    
    def ping_sync(self) -> bool:
        """
        测试同步数据库连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            with self.get_sync_session() as session:
                result = session.execute("SELECT 1").scalar()
                return result == 1
        except Exception as e:
            logger.error(f"Sync database ping failed: {e}")
            return False
    
    async def ping_async(self) -> bool:
        """
        测试异步数据库连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            async with self.get_async_session() as session:
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Async database ping failed: {e}")
            return False
    
    def get_engine_info(self) -> dict:
        """
        获取引擎信息
        
        Returns:
            dict: 引擎信息
        """
        sync_info = {
            "type": "sync",
            "initialized": self._sync_engine is not None,
            "pool_size": self._sync_engine.pool.size() if self._sync_engine else 0,
            "checked_out": self._sync_engine.pool.checkedout() if self._sync_engine else 0,
            "overflow": self._sync_engine.pool.overflow() if self._sync_engine else 0,
        } if self._sync_engine else {"type": "sync", "initialized": False}
        
        async_info = {
            "type": "async",
            "initialized": self._async_engine is not None,
        } if self._async_engine else {"type": "async", "initialized": False}
        
        return {
            "sync": sync_info,
            "async": async_info,
            "fully_initialized": self._initialized
        }


# 全局数据库管理器实例
db_manager = DatabaseManager()


def get_sync_session():
    """
    获取同步会话的依赖函数
    
    Returns:
        Generator: 同步会话生成器
    """
    return db_manager.get_sync_session_dependency()


async def get_async_session():
    """
    获取异步会话的依赖函数
    
    Returns:
        AsyncGenerator: 异步会话生成器
    """
    async for session in db_manager.get_async_session_dependency():
        yield session


def init_database():
    """
    初始化数据库连接
    用于应用启动时调用
    """
    db_manager.initialize()
    logger.info("Database initialized")


async def close_database():
    """
    关闭数据库连接
    用于应用关闭时调用
    """
    db_manager.close()
    logger.info("Database closed")


# 健康检查端点使用的函数
async def check_database_health() -> dict:
    """
    检查数据库健康状态
    
    Returns:
        dict: 健康状态信息
    """
    sync_healthy = db_manager.ping_sync()
    
    # 异步检查在协程中执行
    async_healthy = await db_manager.ping_async()
    
    engine_info = db_manager.get_engine_info()
    
    return {
        "healthy": sync_healthy and async_healthy,
        "sync": {"healthy": sync_healthy, **engine_info["sync"]},
        "async": {"healthy": async_healthy, **engine_info["async"]},
        "timestamp": datetime.now().isoformat()
    }
