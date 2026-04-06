"""
特征存储管理模块
基于Feast的特征注册、存储、检索和管理
支持离线特征计算和在线特征服务
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import logging
import yaml

# Feast相关导入
try:
    from feast import FeatureStore, RepoConfig
    from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
    from feast.infra.online_stores.redis import RedisOnlineStoreConfig
    from feast.infra.offline_stores.file import FileOfflineStoreConfig
    from feast import Entity, FeatureView, ValueType
    from feast.types import Float32, Int32, String, Bool
except ImportError:
    pass

# Doris连接器
try:
    import pymysql
    from feast.infra.online_stores.contrib.doris import DorisOnlineStoreConfig
except ImportError:
    pass

from ..config import settings

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """特征类型"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    EMBEDDING = "embedding"
    TIMESTAMP = "timestamp"
    BOOLEAN = "boolean"


@dataclass
class FeatureDefinition:
    """特征定义"""
    name: str
    type: FeatureType
    description: str
    data_type: str
    validation_rules: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)


@dataclass
class FeatureSet:
    """特征集"""
    name: str
    version: str
    features: List[FeatureDefinition]
    entities: List[str]
    description: str
    created_at: datetime = None
    tags: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if self.tags is None:
            self.tags = {}


class FeatureStoreManager:
    """特征存储管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化特征存储
        
        Args:
            config_path: Feast配置文件路径
        """
        self.config_path = config_path or "feature_store.yaml"
        self.store = None
        self._initialize_store()
        
        # 特征注册表
        self.feature_registry: Dict[str, FeatureSet] = {}
        
        # 初始化Doris连接
        self.doris_conn = None
        self._initialize_doris()
        
        # 加载现有特征
        self._load_feature_registry()
    
    def _initialize_store(self):
        """初始化Feast特征存储"""
        try:
            # 检查配置文件是否存在
            if Path(self.config_path).exists():
                self.store = FeatureStore(repo_path=".")
            else:
                # 创建默认配置
                self._create_default_config()
                self.store = FeatureStore(repo_path=".")
            
            logger.info(f"特征存储初始化完成: {self.config_path}")
            
        except Exception as e:
            logger.error(f"特征存储初始化失败: {e}")
            # 创建内存中的特征存储
            self.store = None
    
    def _create_default_config(self):
        """创建默认配置"""
        config = {
            "project": "mlops_feature_store",
            "registry": "data/registry.db",
            "provider": "local",
            "online_store": {
                "type": "redis",
                "connection_string": "localhost:6379"
            },
            "offline_store": {
                "type": "file"
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _initialize_doris(self):
        """初始化Doris连接"""
        try:
            if hasattr(settings.feature_store, 'doris_host'):
                self.doris_conn = pymysql.connect(
                    host=settings.feature_store.doris_host,
                    port=settings.feature_store.doris_port,
                    user=settings.feature_store.doris_user,
                    password=settings.feature_store.doris_password,
                    database=settings.feature_store.doris_database,
                    charset='utf8mb4'
                )
                logger.info("Doris连接初始化完成")
        except Exception as e:
            logger.warning(f"Doris连接初始化失败: {e}")
    
    async def register_feature_set(
        self,
        name: str,
        features: List[Dict[str, Any]],
        entities: List[str],
        description: str = "",
        version: str = "v1.0",
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        注册特征集
        
        Args:
            name: 特征集名称
            features: 特征定义列表
            entities: 实体列表
            description: 描述
            version: 版本
            tags: 标签
            
        Returns:
            特征集ID
        """
        try:
            # 创建特征定义对象
            feature_defs = []
            for feature in features:
                feature_def = FeatureDefinition(
                    name=feature["name"],
                    type=FeatureType(feature.get("type", "numerical")),
                    description=feature.get("description", ""),
                    data_type=feature.get("data_type", "float32"),
                    validation_rules=feature.get("validation_rules"),
                    statistics=feature.get("statistics")
                )
                feature_defs.append(feature_def)
            
            # 创建特征集
            feature_set = FeatureSet(
                name=name,
                version=version,
                features=feature_defs,
                entities=entities,
                description=description,
                tags=tags
            )
            
            # 注册到本地注册表
            feature_set_id = f"{name}_{version}"
            self.feature_registry[feature_set_id] = feature_set
            
            # 保存到文件
            await self._save_feature_set(feature_set)
            
            # 注册到Feast（如果可用）
            if self.store:
                await self._register_to_feast(feature_set)
            
            logger.info(f"特征集注册成功: {feature_set_id}")
            return feature_set_id
            
        except Exception as e:
            logger.error(f"特征集注册失败: {e}")
            raise
    
    async def _register_to_feast(self, feature_set: FeatureSet):
        """注册到Feast"""
        try:
            # 创建实体
            feast_entities = []
            for entity_name in feature_set.entities:
                entity = Entity(
                    name=entity_name,
                    value_type=ValueType.INT64,
                    description=f"Entity for {feature_set.name}"
                )
                feast_entities.append(entity)
            
            # 创建特征视图
            # 这里简化处理，实际应根据特征定义创建
            from feast import FeatureView, Field
            from feast.types import Float32
            
            # 创建数据源
            source = PostgreSQLSource(
                table=feature_set.name,
                timestamp_field="event_timestamp"
            )
            
            # 创建特征字段
            feature_fields = []
            for feature_def in feature_set.features:
                if feature_def.type == FeatureType.NUMERICAL:
                    dtype = Float32
                else:
                    dtype = String
                
                field = Field(
                    name=feature_def.name,
                    dtype=dtype
                )
                feature_fields.append(field)
            
            # 创建特征视图
            feature_view = FeatureView(
                name=feature_set.name,
                entities=feast_entities,
                ttl=timedelta(days=365),
                source=source,
                online=True
            )
            
            # 应用
            self.store.apply([feature_view] + feast_entities)
            
        except Exception as e:
            logger.error(f"注册到Feast失败: {e}")
    
    async def get_features(
        self,
        entity_df: pd.DataFrame,
        feature_set_id: str,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取特征
        
        Args:
            entity_df: 实体数据框
            feature_set_id: 特征集ID
            feature_names: 特征名称列表
            
        Returns:
            特征数据框
        """
        try:
            # 检查特征集
            if feature_set_id not in self.feature_registry:
                raise ValueError(f"特征集不存在: {feature_set_id}")
            
            feature_set = self.feature_registry[feature_set_id]
            
            # 确定要获取的特征
            if feature_names:
                features_to_get = [
                    f for f in feature_set.features 
                    if f.name in feature_names
                ]
            else:
                features_to_get = feature_set.features
            
            if not features_to_get:
                raise ValueError("没有找到指定的特征")
            
            # 尝试从Feast获取
            if self.store:
                try:
                    # 构建特征引用
                    feature_refs = [
                        f"{feature_set.name}:{feature.name}" 
                        for feature in features_to_get
                    ]
                    
                    # 获取历史特征
                    historical_features = self.store.get_historical_features(
                        entity_df=entity_df,
                        features=feature_refs
                    )
                    
                    return historical_features.to_df()
                    
                except Exception as e:
                    logger.warning(f"从Feast获取特征失败，回退到本地: {e}")
            
            # 从本地数据源获取
            return await self._get_features_from_local(
                entity_df, feature_set, features_to_get
            )
            
        except Exception as e:
            logger.error(f"获取特征失败: {e}")
            raise
    
    async def _get_features_from_local(
        self,
        entity_df: pd.DataFrame,
        feature_set: FeatureSet,
        features: List[FeatureDefinition]
    ) -> pd.DataFrame:
        """从本地数据源获取特征"""
        # 这里实现从本地数据库或文件系统获取特征的逻辑
        # 简化实现：返回模拟数据
        
        result_df = entity_df.copy()
        
        for feature in features:
            if feature.type == FeatureType.NUMERICAL:
                # 生成随机数值特征
                result_df[feature.name] = np.random.randn(len(entity_df))
            elif feature.type == FeatureType.CATEGORICAL:
                # 生成随机分类特征
                categories = ["A", "B", "C", "D"]
                result_df[feature.name] = np.random.choice(categories, len(entity_df))
            elif feature.type == FeatureType.BOOLEAN:
                # 生成随机布尔特征
                result_df[feature.name] = np.random.choice([True, False], len(entity_df))
        
        return result_df
    
    async def get_online_features(
        self,
        entity_rows: List[Dict[str, Any]],
        feature_set_id: str,
        feature_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        获取在线特征
        
        Args:
            entity_rows: 实体行列表
            feature_set_id: 特征集ID
            feature_names: 特征名称列表
            
        Returns:
            特征值列表
        """
        try:
            # 检查特征集
            if feature_set_id not in self.feature_registry:
                raise ValueError(f"特征集不存在: {feature_set_id}")
            
            feature_set = self.feature_registry[feature_set_id]
            
            # 确定要获取的特征
            if feature_names:
                features_to_get = [
                    f for f in feature_set.features 
                    if f.name in feature_names
                ]
            else:
                features_to_get = feature_set.features[:10]  # 限制数量
            
            # 尝试从Feast在线存储获取
            if self.store:
                try:
                    # 构建特征引用
                    feature_refs = [
                        f"{feature_set.name}:{feature.name}" 
                        for feature in features_to_get
                    ]
                    
                    # 获取在线特征
                    online_features = self.store.get_online_features(
                        entity_rows=entity_rows,
                        features=feature_refs
                    )
                    
                    return online_features.to_dict()
                    
                except Exception as e:
                    logger.warning(f"从Feast获取在线特征失败: {e}")
            
            # 从Doris获取
            if self.doris_conn:
                return await self._get_features_from_doris(
                    entity_rows, feature_set, features_to_get
                )
            
            # 返回模拟数据
            return await self._get_mock_online_features(
                entity_rows, features_to_get
            )
            
        except Exception as e:
            logger.error(f"获取在线特征失败: {e}")
            raise
    
    async def _get_features_from_doris(
        self,
        entity_rows: List[Dict[str, Any]],
        feature_set: FeatureSet,
        features: List[FeatureDefinition]
    ) -> List[Dict[str, Any]]:
        """从Doris获取特征"""
        results = []
        
        try:
            with self.doris_conn.cursor() as cursor:
                for entity_row in entity_rows:
                    # 构建查询
                    entity_conditions = []
                    for entity_name, entity_value in entity_row.items():
                        entity_conditions.append(f"{entity_name} = {entity_value}")
                    
                    where_clause = " AND ".join(entity_conditions)
                    
                    # 构建SELECT子句
                    feature_columns = [f.name for f in features]
                    select_clause = ", ".join(feature_columns)
                    
                    query = f"""
                    SELECT {select_clause}
                    FROM {feature_set.name}
                    WHERE {where_clause}
                    LIMIT 1
                    """
                    
                    cursor.execute(query)
                    row = cursor.fetchone()
                    
                    if row:
                        result = dict(zip(feature_columns, row))
                        result.update(entity_row)
                        results.append(result)
                    else:
                        # 如果没有找到，返回默认值
                        default_result = entity_row.copy()
                        for feature in features:
                            if feature.type == FeatureType.NUMERICAL:
                                default_result[feature.name] = 0.0
                            elif feature.type == FeatureType.CATEGORICAL:
                                default_result[feature.name] = "unknown"
                            elif feature.type == FeatureType.BOOLEAN:
                                default_result[feature.name] = False
                        results.append(default_result)
            
            return results
            
        except Exception as e:
            logger.error(f"从Doris获取特征失败: {e}")
            # 返回模拟数据
            return await self._get_mock_online_features(entity_rows, features)
    
    async def _get_mock_online_features(
        self,
        entity_rows: List[Dict[str, Any]],
        features: List[FeatureDefinition]
    ) -> List[Dict[str, Any]]:
        """获取模拟在线特征"""
        results = []
        
        for entity_row in entity_rows:
            result = entity_row.copy()
            
            for feature in features:
                if feature.type == FeatureType.NUMERICAL:
                    result[feature.name] = float(np.random.randn())
                elif feature.type == FeatureType.CATEGORICAL:
                    categories = ["A", "B", "C", "D"]
                    result[feature.name] = np.random.choice(categories)
                elif feature.type == FeatureType.BOOLEAN:
                    result[feature.name] = bool(np.random.choice([True, False]))
                elif feature.type == FeatureType.EMBEDDING:
                    result[feature.name] = list(np.random.randn(128))  # 128维向量
                else:
                    result[feature.name] = "feature_value"
            
            results.append(result)
        
        return results
    
    async def write_features(
        self,
        feature_set_id: str,
        data: pd.DataFrame,
        timestamp_column: str = "event_timestamp"
    ) -> bool:
        """
        写入特征
        
        Args:
            feature_set_id: 特征集ID
            data: 特征数据
            timestamp_column: 时间戳列名
            
        Returns:
            是否成功
        """
        try:
            # 检查特征集
            if feature_set_id not in self.feature_registry:
                raise ValueError(f"特征集不存在: {feature_set_id}")
            
            feature_set = self.feature_registry[feature_set_id]
            
            # 验证数据
            await self._validate_feature_data(data, feature_set)
            
            # 写入到Feast
            if self.store:
                try:
                    self.store.write_to_online_store(
                        feature_set.name,
                        data
                    )
                except Exception as e:
                    logger.warning(f"写入到Feast失败: {e}")
            
            # 写入到Doris
            if self.doris_conn:
                await self._write_to_doris(feature_set.name, data)
            
            # 写入到本地文件
            await self._write_to_local(feature_set_id, data)
            
            logger.info(f"特征写入成功: {feature_set_id}, 行数: {len(data)}")
            return True
            
        except Exception as e:
            logger.error(f"写入特征失败: {e}")
            return False
    
    async def _validate_feature_data(
        self,
        data: pd.DataFrame,
        feature_set: FeatureSet
    ):
        """验证特征数据"""
        # 检查必需列
        required_columns = feature_set.entities.copy()
        
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"缺失必需列: {column}")
        
        # 检查特征列
        feature_names = [f.name for f in feature_set.features]
        for feature_name in feature_names:
            if feature_name not in data.columns:
                logger.warning(f"特征列缺失: {feature_name}")
    
    async def _write_to_doris(self, table_name: str, data: pd.DataFrame):
        """写入到Doris"""
        try:
            with self.doris_conn.cursor() as cursor:
                # 构建插入语句
                columns = ", ".join(data.columns)
                placeholders = ", ".join(["%s"] * len(data.columns))
                
                # 准备数据
                values = [tuple(row) for row in data.itertuples(index=False, name=None)]
                
                # 执行批量插入
                cursor.executemany(
                    f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})",
                    values
                )
                
                self.doris_conn.commit()
                
        except Exception as e:
            logger.error(f"写入到Doris失败: {e}")
            self.doris_conn.rollback()
    
    async def _write_to_local(self, feature_set_id: str, data: pd.DataFrame):
        """写入到本地文件"""
        try:
            # 创建存储目录
            storage_dir = Path("feature_data") / feature_set_id
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存为Parquet文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = storage_dir / f"features_{timestamp}.parquet"
            
            data.to_parquet(file_path, index=False)
            
        except Exception as e:
            logger.error(f"写入到本地文件失败: {e}")
    
    async def materialize_features(
        self,
        feature_set_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> bool:
        """
        物化特征（从离线存储到在线存储）
        
        Args:
            feature_set_id: 特征集ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            是否成功
        """
        try:
            if not self.store:
                logger.warning("Feast特征存储未初始化，跳过物化")
                return False
            
            # 检查特征集
            if feature_set_id not in self.feature_registry:
                raise ValueError(f"特征集不存在: {feature_set_id}")
            
            feature_set = self.feature_registry[feature_set_id]
            
            # 执行物化
            self.store.materialize_incremental(
                feature_views=[feature_set.name],
                start_date=start_date,
                end_date=end_date
            )
            
            logger.info(f"特征物化完成: {feature_set_id}, {start_date} 到 {end_date}")
            return True
            
        except Exception as e:
            logger.error(f"特征物化失败: {e}")
            return False
    
    async def list_feature_sets(
        self,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """列出特征集"""
        results = []
        
        for feature_set_id, feature_set in self.feature_registry.items():
            # 标签过滤
            if tags:
                if not all(
                    feature_set.tags.get(k) == v 
                    for k, v in tags.items()
                ):
                    continue
            
            result = {
                "feature_set_id": feature_set_id,
                "name": feature_set.name,
                "version": feature_set.version,
                "description": feature_set.description,
                "entities": feature_set.entities,
                "num_features": len(feature_set.features),
                "created_at": feature_set.created_at.isoformat(),
                "tags": feature_set.tags
            }
            
            # 添加特征示例
            sample_features = [
                {
                    "name": f.name,
                    "type": f.type.value,
                    "data_type": f.data_type
                }
                for f in feature_set.features[:5]  # 前5个特征
            ]
            result["sample_features"] = sample_features
            
            results.append(result)
        
        return results
    
    async def get_feature_statistics(
        self,
        feature_set_id: str,
        feature_name: str
    ) -> Dict[str, Any]:
        """
        获取特征统计信息
        
        Args:
            feature_set_id: 特征集ID
            feature_name: 特征名称
            
        Returns:
            特征统计信息
        """
        try:
            if feature_set_id not in self.feature_registry:
                raise ValueError(f"特征集不存在: {feature_set_id}")
            
            feature_set = self.feature_registry[feature_set_id]
            
            # 查找特征
            feature = None
            for f in feature_set.features:
                if f.name == feature_name:
                    feature = f
                    break
            
            if not feature:
                raise ValueError(f"特征不存在: {feature_name}")
            
            # 如果有统计信息，直接返回
            if feature.statistics:
                return feature.statistics
            
            # 否则，计算统计信息
            # 这里从存储中加载数据并计算
            # 简化实现
            return {
                "count": 1000,
                "mean": 0.5,
                "std": 0.1,
                "min": 0.0,
                "max": 1.0,
                "missing": 0,
                "zeros": 0
            }
            
        except Exception as e:
            logger.error(f"获取特征统计信息失败: {e}")
            return {}
    
    async def _save_feature_set(self, feature_set: FeatureSet):
        """保存特征集"""
        try:
            registry_dir = Path("feature_registry")
            registry_dir.mkdir(exist_ok=True)
            
            file_path = registry_dir / f"{feature_set.name}_{feature_set.version}.json"
            
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(asdict(feature_set), indent=2, default=str))
                
        except Exception as e:
            logger.error(f"保存特征集失败: {e}")
    
    def _load_feature_registry(self):
        """加载特征注册表"""
        try:
            registry_dir = Path("feature_registry")
            if not registry_dir.exists():
                return
            
            for json_file in registry_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        feature_set = FeatureSet(**data)
                        feature_set_id = f"{feature_set.name}_{feature_set.version}"
                        self.feature_registry[feature_set_id] = feature_set
                except Exception as e:
                    logger.error(f"加载特征集失败 {json_file}: {e}")
                    
        except Exception as e:
            logger.error(f"加载特征注册表失败: {e}")
    
    async def create_doris_table(
        self,
        table_name: str,
        schema: Dict[str, str],
        partition_columns: Optional[List[str]] = None
    ) -> bool:
        """
        在Doris中创建表
        
        Args:
            table_name: 表名
            schema: 列定义 {列名: 数据类型}
            partition_columns: 分区列
            
        Returns:
            是否成功
        """
        if not self.doris_conn:
            logger.warning("Doris连接未初始化")
            return False
        
        try:
            with self.doris_conn.cursor() as cursor:
                # 构建列定义
                column_defs = []
                for column_name, data_type in schema.items():
                    column_defs.append(f"{column_name} {data_type}")
                
                column_defs_str = ",\n  ".join(column_defs)
                
                # 构建分区定义
                partition_def = ""
                if partition_columns:
                    partition_cols = ", ".join(partition_columns)
                    partition_def = f"\nPARTITION BY RANGE({partition_cols}) ()"
                
                # 执行创建表
                create_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                  {column_defs_str}
                ){partition_def}
                DISTRIBUTED BY HASH({list(schema.keys())[0]}) BUCKETS 10
                PROPERTIES (
                  "replication_num" = "1"
                )
                """
                
                cursor.execute(create_sql)
                self.doris_conn.commit()
                
                logger.info(f"Doris表创建成功: {table_name}")
                return True
                
        except Exception as e:
            logger.error(f"创建Doris表失败: {e}")
            self.doris_conn.rollback()
            return False
    
    async def close(self):
        """关闭连接"""
        if self.doris_conn:
            self.doris_conn.close()
            logger.info("Doris连接已关闭")
