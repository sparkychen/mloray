"""
企业级数据质量监控模块
基于Great Expectations、whylogs、Evidently的完整数据质量管理系统
支持多源数据验证、实时监控、趋势分析和自动告警
"""

import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint import SimpleCheckpoint
from great_expectations.data_context import FileDataContext
import whylogs as why
from whylogs.core import DatasetProfile
from whylogs.api.writer.whylabs import WhyLabsWriter
from whylogs.api.writer.whylabs import WhyLabsWriter
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path
import aiofiles
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import warnings
from collections import defaultdict
import yaml
import pickle
import hashlib
import sys
import traceback

# 导入Evidently用于数据质量分析
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataQualityPreset
    from evidently.test_suite import TestSuite
    from evidently.tests import *
except ImportError:
    pass

from ..config import settings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DataQualityStatus(Enum):
    """数据质量状态枚举"""
    EXCELLENT = "excellent"    # 90-100分，所有检查通过
    GOOD = "good"             # 80-89分，轻微问题
    FAIR = "fair"             # 70-79分，需要注意
    WARNING = "warning"       # 60-69分，需要关注
    POOR = "poor"             # 50-59分，质量较差
    ERROR = "error"           # 30-49分，严重问题
    FAILED = "failed"         # 0-29分，检查失败
    UNKNOWN = "unknown"       # 未知状态


class QualityMetricType(Enum):
    """质量指标类型"""
    COMPLETENESS = "completeness"      # 完整性
    ACCURACY = "accuracy"              # 准确性
    CONSISTENCY = "consistency"        # 一致性
    TIMELINESS = "timeliness"          # 及时性
    VALIDITY = "validity"              # 有效性
    UNIQUENESS = "uniqueness"          # 唯一性
    INTEGRITY = "integrity"            # 完整性
    FRESHNESS = "freshness"            # 新鲜度
    VOLUME = "volume"                  # 数据量
    DISTRIBUTION = "distribution"      # 分布特征


@dataclass
class QualityMetric:
    """质量指标数据类"""
    name: str
    value: float
    metric_type: QualityMetricType
    threshold: Optional[float] = None
    status: str = "unknown"
    description: Optional[str] = None
    unit: str = ""
    dimensions: Optional[Dict[str, str]] = None
    calculated_at: datetime = None
    
    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = {}
        if self.calculated_at is None:
            self.calculated_at = datetime.now()
        elif isinstance(self.calculated_at, str):
            self.calculated_at = datetime.fromisoformat(self.calculated_at)
    
    def is_passed(self) -> bool:
        """检查指标是否通过"""
        if self.threshold is None:
            return True
        
        # 根据指标类型决定通过逻辑
        if self.metric_type in [
            QualityMetricType.COMPLETENESS,
            QualityMetricType.ACCURACY,
            QualityMetricType.CONSISTENCY
        ]:
            return self.value >= self.threshold
        else:
            # 对于其他指标，值越小越好
            return self.value <= self.threshold


@dataclass
class DataQualityResult:
    """数据质量结果数据类"""
    check_id: str
    dataset_name: str
    dataset_version: str
    timestamp: datetime
    status: DataQualityStatus
    overall_score: float
    metrics: List[QualityMetric]
    summary: Dict[str, Any]
    alerts: List[str]
    details: Optional[Dict[str, Any]] = None
    lineage: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []
        if self.details is None:
            self.details = {}
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        
        # 计算总体得分
        if not hasattr(self, 'overall_score'):
            self.overall_score = self._calculate_overall_score()
    
    def _calculate_overall_score(self) -> float:
        """计算总体质量得分"""
        if not self.metrics:
            return 0.0
        
        # 加权平均计算得分
        weights = {
            QualityMetricType.COMPLETENESS: 0.25,
            QualityMetricType.ACCURACY: 0.25,
            QualityMetricType.CONSISTENCY: 0.20,
            QualityMetricType.VALIDITY: 0.15,
            QualityMetricType.UNIQUENESS: 0.10,
            QualityMetricType.TIMELINESS: 0.05
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for metric in self.metrics:
            weight = weights.get(metric.metric_type, 0.05)
            
            # 归一化指标值到0-100分
            if metric.threshold is not None:
                if metric.metric_type in [
                    QualityMetricType.COMPLETENESS,
                    QualityMetricType.ACCURACY
                ]:
                    normalized_value = min(100, max(0, metric.value * 100))
                else:
                    normalized_value = min(100, max(0, 100 - metric.value * 100))
            else:
                normalized_value = 50  # 默认值
            
            weighted_sum += normalized_value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def get_failed_metrics(self) -> List[QualityMetric]:
        """获取失败的指标"""
        return [m for m in self.metrics if not m.is_passed()]
    
    def get_warning_metrics(self) -> List[QualityMetric]:
        """获取警告的指标"""
        warning_metrics = []
        for metric in self.metrics:
            if metric.threshold is not None:
                if metric.metric_type in [
                    QualityMetricType.COMPLETENESS,
                    QualityMetricType.ACCURACY
                ]:
                    if 0.8 <= metric.value < metric.threshold:  # 接近阈值
                        warning_metrics.append(metric)
                else:
                    if metric.threshold < metric.value <= 1.2:  # 接近阈值
                        warning_metrics.append(metric)
        return warning_metrics


@dataclass
class QualityThreshold:
    """质量阈值配置"""
    metric_type: QualityMetricType
    metric_name: str
    warning_threshold: float
    error_threshold: float
    description: str
    check_condition: str  # "greater", "less", "equal", "not_equal"


@dataclass
class DatasetSchema:
    """数据集模式定义"""
    dataset_name: str
    version: str
    columns: Dict[str, Dict[str, Any]]  # 列名 -> 列定义
    primary_keys: List[str]
    required_columns: List[str]
    constraints: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)


class DataQualityMonitor:
    """企业级数据质量监控器"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        enable_whylabs: bool = True,
        enable_evidently: bool = True
    ):
        """
        初始化数据质量监控器
        
        Args:
            config_path: 配置文件路径
            enable_whylabs: 是否启用whylabs
            enable_evidently: 是否启用evidently
        """
        # 初始化存储目录
        self.data_dir = Path("data_quality")
        self.data_dir.mkdir(exist_ok=True)
        
        # 初始化Great Expectations
        self.ge_context = self._initialize_great_expectations()
        
        # 初始化whylogs writer
        self.whylabs_writer = None
        if enable_whylabs and hasattr(settings.monitoring, 'whylogs_whylabs_key'):
            try:
                self.whylabs_writer = WhyLabsWriter(
                    org_id=settings.monitoring.whylogs_whylabs_org_id,
                    api_key=settings.monitoring.whylogs_whylabs_key
                )
                logger.info("WhyLabs writer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize WhyLabs writer: {e}")
        
        # 结果缓存
        self.results: Dict[str, DataQualityResult] = {}
        self.history_cache: Dict[str, List[DataQualityResult]] = defaultdict(list)
        
        # 阈值配置
        self.thresholds: Dict[str, QualityThreshold] = self._load_thresholds()
        
        # 数据集模式
        self.schemas: Dict[str, DatasetSchema] = self._load_schemas()
        
        # 告警配置
        self.alert_config = self._load_alert_config()
        
        # 加载历史结果
        self._load_history()
        
        logger.info(f"DataQualityMonitor initialized with config: {config_path}")
    
    def _initialize_great_expectations(self) -> FileDataContext:
        """初始化Great Expectations上下文"""
        try:
            # 创建Great Expectations项目结构
            context = FileDataContext(project_root_dir=".")
            
            # 配置数据源
            if "pandas_datasource" not in [ds["name"] for ds in context.list_datasources()]:
                context.add_datasource(
                    name="pandas_datasource",
                    module_name="great_expectations.datasource",
                    class_name="PandasDatasource"
                )
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to initialize Great Expectations: {e}")
            # 创建基本的上下文
            return FileDataContext(project_root_dir=".")
    
    def _load_thresholds(self) -> Dict[str, QualityThreshold]:
        """加载阈值配置"""
        thresholds_path = self.data_dir / "thresholds.yaml"
        
        if not thresholds_path.exists():
            # 创建默认阈值配置
            default_thresholds = [
                QualityThreshold(
                    metric_type=QualityMetricType.COMPLETENESS,
                    metric_name="missing_values_rate",
                    warning_threshold=0.05,
                    error_threshold=0.10,
                    description="缺失值比例阈值",
                    check_condition="less"
                ),
                QualityThreshold(
                    metric_type=QualityMetricType.ACCURACY,
                    metric_name="accuracy_score",
                    warning_threshold=0.90,
                    error_threshold=0.80,
                    description="准确率阈值",
                    check_condition="greater"
                ),
                QualityThreshold(
                    metric_type=QualityMetricType.CONSISTENCY,
                    metric_name="inconsistency_rate",
                    warning_threshold=0.05,
                    error_threshold=0.10,
                    description="不一致性比例阈值",
                    check_condition="less"
                ),
                QualityThreshold(
                    metric_type=QualityMetricType.VALIDITY,
                    metric_name="invalid_values_rate",
                    warning_threshold=0.05,
                    error_threshold=0.10,
                    description="无效值比例阈值",
                    check_condition="less"
                ),
                QualityThreshold(
                    metric_type=QualityMetricType.UNIQUENESS,
                    metric_name="duplicate_rate",
                    warning_threshold=0.05,
                    error_threshold=0.10,
                    description="重复值比例阈值",
                    check_condition="less"
                )
            ]
            
            thresholds_dict = {t.metric_name: t for t in default_thresholds}
            self._save_thresholds(thresholds_dict)
            return thresholds_dict
        
        try:
            with open(thresholds_path, 'r') as f:
                thresholds_data = yaml.safe_load(f)
            
            thresholds = {}
            for metric_name, threshold_data in thresholds_data.items():
                thresholds[metric_name] = QualityThreshold(**threshold_data)
            
            return thresholds
            
        except Exception as e:
            logger.error(f"Failed to load thresholds: {e}")
            return {}
    
    def _save_thresholds(self, thresholds: Dict[str, QualityThreshold]):
        """保存阈值配置"""
        try:
            thresholds_path = self.data_dir / "thresholds.yaml"
            
            thresholds_data = {}
            for metric_name, threshold in thresholds.items():
                thresholds_data[metric_name] = asdict(threshold)
            
            with open(thresholds_path, 'w') as f:
                yaml.dump(thresholds_data, f, default_flow_style=False)
                
        except Exception as e:
            logger.error(f"Failed to save thresholds: {e}")
    
    def _load_schemas(self) -> Dict[str, DatasetSchema]:
        """加载数据集模式"""
        schemas_path = self.data_dir / "schemas.yaml"
        
        if not schemas_path.exists():
            return {}
        
        try:
            with open(schemas_path, 'r') as f:
                schemas_data = yaml.safe_load(f)
            
            schemas = {}
            for dataset_name, schema_data in schemas_data.items():
                schemas[dataset_name] = DatasetSchema(**schema_data)
            
            return schemas
            
        except Exception as e:
            logger.error(f"Failed to load schemas: {e}")
            return {}
    
    def _save_schema(self, schema: DatasetSchema):
        """保存数据集模式"""
        try:
            schemas_path = self.data_dir / "schemas.yaml"
            
            schemas_data = {}
            if schemas_path.exists():
                with open(schemas_path, 'r') as f:
                    schemas_data = yaml.safe_load(f) or {}
            
            schemas_data[schema.dataset_name] = asdict(schema)
            
            with open(schemas_path, 'w') as f:
                yaml.dump(schemas_data, f, default_flow_style=False)
                
        except Exception as e:
            logger.error(f"Failed to save schema: {e}")
    
    def _load_alert_config(self) -> Dict[str, Any]:
        """加载告警配置"""
        config_path = self.data_dir / "alert_config.yaml"
        
        if not config_path.exists():
            return {
                "webhook_url": settings.monitoring.get("alert_webhook_url"),
                "email_enabled": False,
                "slack_enabled": False,
                "alert_levels": {
                    "critical": ["email", "slack", "webhook"],
                    "high": ["slack", "webhook"],
                    "medium": ["webhook"],
                    "low": []
                },
                "quiet_hours": {
                    "start": "22:00",
                    "end": "08:00"
                }
            }
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load alert config: {e}")
            return {}
    
    def _load_history(self):
        """加载历史结果"""
        history_dir = self.data_dir / "history"
        history_dir.mkdir(exist_ok=True)
        
        for result_file in history_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                # 转换metrics
                metrics = []
                for metric_data in result_data.get("metrics", []):
                    metric_data["metric_type"] = QualityMetricType(metric_data["metric_type"])
                    metrics.append(QualityMetric(**metric_data))
                
                result_data["status"] = DataQualityStatus(result_data["status"])
                result_data["metrics"] = metrics
                
                result = DataQualityResult(**result_data)
                
                dataset_name = result.dataset_name
                self.history_cache[dataset_name].append(result)
                
            except Exception as e:
                logger.error(f"Failed to load history result {result_file}: {e}")
        
        # 按时间排序
        for dataset_name in self.history_cache:
            self.history_cache[dataset_name].sort(key=lambda x: x.timestamp)
    
    async def monitor_dataset(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        dataset_version: str = "latest",
        expectation_suite_name: Optional[str] = None,
        batch_identifier: Optional[str] = None,
        validation_mode: str = "full"  # quick, standard, full
    ) -> DataQualityResult:
        """
        监控数据集质量
        
        Args:
            dataset_name: 数据集名称
            data: 数据集
            dataset_version: 数据集版本
            expectation_suite_name: 期望套件名称
            batch_identifier: 批处理标识符
            validation_mode: 验证模式
            
        Returns:
            数据质量结果
        """
        check_id = f"check_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"开始数据质量监控: {dataset_name}, 版本: {dataset_version}, 行数: {len(data)}")
            
            # 1. 验证数据格式
            self._validate_data_format(data, dataset_name)
            
            # 2. 使用Great Expectations进行规则验证
            ge_results = await self._validate_with_great_expectations(
                data=data,
                dataset_name=dataset_name,
                expectation_suite=expectation_suite_name,
                batch_identifier=batch_identifier,
                mode=validation_mode
            )
            
            # 3. 使用whylogs进行数据画像
            profile_results = await self._profile_with_whylogs(
                data=data,
                dataset_name=dataset_name,
                dataset_version=dataset_version
            )
            
            # 4. 计算自定义质量指标
            custom_metrics = await self._calculate_custom_metrics(
                data=data,
                dataset_name=dataset_name,
                dataset_version=dataset_version
            )
            
            # 5. 使用Evidently进行数据质量分析
            evidently_results = {}
            try:
                from evidently.report import Report
                from evidently.metric_preset import DataQualityPreset
                
                if len(data) > 0:  # 确保有数据
                    report = Report(metrics=[DataQualityPreset()])
                    report.run(
                        reference_data=data.head(min(1000, len(data))),  # 限制数据量
                        current_data=data.head(min(1000, len(data)))
                    )
                    evidently_results = report.as_dict()
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Evidently analysis failed: {e}")
            
            # 合并所有指标
            all_metrics = []
            
            # 添加Great Expectations指标
            for metric_name, metric_value in ge_results.get("metrics", {}).items():
                metric_type = self._infer_metric_type(metric_name)
                threshold = self._get_threshold(metric_name)
                
                metric = QualityMetric(
                    name=metric_name,
                    value=metric_value,
                    metric_type=metric_type,
                    threshold=threshold,
                    status="passed" if ge_results.get("success", False) else "failed",
                    description=f"Great Expectations指标: {metric_name}",
                    unit="ratio"
                )
                all_metrics.append(metric)
            
            # 添加whylogs指标
            for metric_name, metric_value in profile_results.get("metrics", {}).items():
                metric_type = self._infer_metric_type(metric_name)
                threshold = self._get_threshold(metric_name)
                
                metric = QualityMetric(
                    name=f"whylogs_{metric_name}",
                    value=metric_value,
                    metric_type=metric_type,
                    threshold=threshold,
                    status="measured",
                    description=f"whylogs指标: {metric_name}",
                    unit="ratio"
                )
                all_metrics.append(metric)
            
            # 添加自定义指标
            all_metrics.extend(custom_metrics)
            
            # 添加Evidently指标
            for metric_name, metric_value in evidently_results.get("metrics", {}).items():
                if isinstance(metric_value, (int, float)):
                    metric_type = self._infer_metric_type(metric_name)
                    threshold = self._get_threshold(metric_name)
                    
                    metric = QualityMetric(
                        name=f"evidently_{metric_name}",
                        value=float(metric_value),
                        metric_type=metric_type,
                        threshold=threshold,
                        status="measured",
                        description=f"Evidently指标: {metric_name}",
                        unit="ratio"
                    )
                    all_metrics.append(metric)
            
            # 确定总体状态
            overall_score = self._calculate_overall_score(all_metrics)
            status = self._determine_quality_status(overall_score, all_metrics)
            
            # 检查告警
            alerts = await self._check_alerts(all_metrics, dataset_name)
            
            # 创建结果
            result = DataQualityResult(
                check_id=check_id,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                timestamp=datetime.now(),
                status=status,
                overall_score=overall_score,
                metrics=all_metrics,
                summary={
                    "total_checks": len(all_metrics),
                    "passed_checks": sum(1 for m in all_metrics if m.is_passed()),
                    "failed_checks": sum(1 for m in all_metrics if not m.is_passed()),
                    "warning_checks": sum(1 for m in all_metrics if 0.8 <= (m.value if m.threshold else 1.0) < (m.threshold or 1.0)),
                    "ge_success": ge_results.get("success", False),
                    "ge_statistics": ge_results.get("statistics", {}),
                    "whylogs_profile": profile_results.get("profile_id"),
                    "validation_mode": validation_mode,
                    "data_shape": data.shape,
                    "data_memory_mb": data.memory_usage(deep=True).sum() / 1024**2
                },
                alerts=alerts,
                details={
                    "ge_results": ge_results,
                    "profile_results": profile_results,
                    "evidently_results": evidently_results,
                    "custom_metrics": [asdict(m) for m in custom_metrics],
                    "data_sample": data.head(10).to_dict('records') if len(data) > 0 else []
                },
                lineage={
                    "source": "data_quality_monitor",
                    "check_id": check_id,
                    "validated_at": datetime.now().isoformat()
                }
            )
            
            # 缓存结果
            self.results[check_id] = result
            self.history_cache[dataset_name].append(result)
            
            # 保存结果
            await self._save_result(result)
            
            # 发送告警
            if alerts:
                await self._send_alerts(dataset_name, alerts, result)
            
            logger.info(f"数据质量监控完成: {dataset_name}, 得分: {overall_score:.2f}, 状态: {status.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"数据质量监控失败: {e}", exc_info=True)
            
            # 返回错误结果
            return DataQualityResult(
                check_id=check_id,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                timestamp=datetime.now(),
                status=DataQualityStatus.FAILED,
                overall_score=0.0,
                metrics=[
                    QualityMetric(
                        name="monitoring_error",
                        value=0.0,
                        metric_type=QualityMetricType.ACCURACY,
                        status="failed",
                        description=f"监控失败: {str(e)[:200]}"
                    )
                ],
                summary={
                    "error": str(e),
                    "total_checks": 0,
                    "passed_checks": 0,
                    "failed_checks": 1
                },
                alerts=[f"数据质量监控失败: {str(e)[:200]}"]
            )
    
    def _validate_data_format(self, data: pd.DataFrame, dataset_name: str):
        """验证数据格式"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"数据必须是pandas DataFrame, 实际类型: {type(data)}")
        
        if len(data) == 0:
            logger.warning(f"数据集 {dataset_name} 为空")
        
        # 检查列名是否有重复
        if len(set(data.columns)) != len(data.columns):
            duplicate_columns = [col for col in data.columns if list(data.columns).count(col) > 1]
            raise ValueError(f"数据集 {dataset_name} 存在重复列名: {duplicate_columns}")
    
    async def _validate_with_great_expectations(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        expectation_suite: Optional[str] = None,
        batch_identifier: Optional[str] = None,
        mode: str = "standard"
    ) -> Dict[str, Any]:
        """
        使用Great Expectations验证数据
        
        Args:
            data: 数据集
            dataset_name: 数据集名称
            expectation_suite: 期望套件名称
            batch_identifier: 批处理标识符
            mode: 验证模式
            
        Returns:
            验证结果
        """
        try:
            if expectation_suite is None:
                # 使用默认期望套件
                expectation_suite = f"{dataset_name}_suite"
                
                # 如果套件不存在，创建基本套件
                if expectation_suite not in self.ge_context.list_expectation_suite_names():
                    await self._create_basic_expectation_suite(
                        dataset_name, data.columns.tolist()
                    )
            
            # 创建批处理请求
            batch_request = RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="default_runtime_data_connector",
                data_asset_name=dataset_name,
                runtime_parameters={"batch_data": data},
                batch_identifiers={"default_identifier_name": batch_identifier or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"},
            )
            
            # 创建检查点
            checkpoint_name = f"{dataset_name}_checkpoint_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            checkpoint = SimpleCheckpoint(
                name=checkpoint_name,
                data_context=self.ge_context,
                config_version=1,
                run_name_template=f"%Y%m%d-%H%M%S-{dataset_name}",
                expectation_suite_name=expectation_suite,
            )
            
            # 运行验证
            results = checkpoint.run(batch_request=batch_request)
            
            # 处理结果
            validation_result = {
                "success": results["success"],
                "statistics": results["statistics"],
                "metrics": {},
                "results": []
            }
            
            # 提取关键指标
            for result in results["results"]:
                expectation_type = result.expectation_config.expectation_type
                
                if hasattr(result.result, 'get'):
                    observed_value = result.result.get("observed_value")
                    
                    if observed_value is not None:
                        # 根据期望类型映射到质量指标
                        if expectation_type == "expect_column_values_to_not_be_null":
                            validation_result["metrics"]["completeness"] = 1.0 - (observed_value / len(data))
                        elif expectation_type == "expect_column_unique_value_count_to_be_between":
                            validation_result["metrics"]["uniqueness"] = observed_value / len(data)
                        elif expectation_type == "expect_column_values_to_be_in_set":
                            validation_result["metrics"]["validity"] = observed_value / len(data)
                
                validation_result["results"].append({
                    "expectation_type": expectation_type,
                    "success": result.success,
                    "observed_value": observed_value if 'observed_value' in locals() else None,
                    "exception_info": result.exception_info
                })
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Great Expectations验证失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "metrics": {},
                "results": []
            }
    
    async def _create_basic_expectation_suite(
        self,
        dataset_name: str,
        columns: List[str]
    ):
        """创建基本期望套件"""
        try:
            suite = self.ge_context.add_or_update_expectation_suite(
                expectation_suite_name=f"{dataset_name}_suite"
            )
            
            # 为每个列添加基本期望
            for column in columns:
                # 检查列是否为空
                suite.add_expectation(
                    gx.core.ExpectationConfiguration(
                        expectation_type="expect_column_values_to_not_be_null",
                        kwargs={"column": column}
                    )
                )
                
                # 检查唯一性
                suite.add_expectation(
                    gx.core.ExpectationConfiguration(
                        expectation_type="expect_column_unique_value_count_to_be_between",
                        kwargs={"column": column, "min_value": 1, "max_value": None}
                    )
                )
            
            # 保存套件
            self.ge_context.save_expectation_suite(suite)
            
        except Exception as e:
            logger.error(f"创建期望套件失败: {e}")
    
    async def _profile_with_whylogs(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        dataset_version: str
    ) -> Dict[str, Any]:
        """
        使用whylogs进行数据画像
        
        Args:
            data: 数据集
            dataset_name: 数据集名称
            dataset_version: 数据集版本
            
        Returns:
            画像结果
        """
        try:
            if len(data) == 0:
                return {
                    "profile_id": None,
                    "metrics": {},
                    "error": "数据为空"
                }
            
            # 创建数据画像
            profile = why.log(data).profile()
            
            # 计算关键指标
            metrics = {}
            
            # 1. 完整性指标
            for column in data.columns:
                missing_count = data[column].isna().sum()
                total_count = len(data)
                completeness = 1.0 - (missing_count / total_count) if total_count > 0 else 0.0
                metrics[f"completeness_{column}"] = completeness
            
            # 2. 唯一性指标
            for column in data.columns:
                unique_count = data[column].nunique()
                total_count = len(data)
                uniqueness = unique_count / total_count if total_count > 0 else 0.0
                metrics[f"uniqueness_{column}"] = uniqueness
            
            # 3. 有效性指标（基于数据类型）
            for column in data.columns:
                if pd.api.types.is_numeric_dtype(data[column]):
                    # 数值型：检查是否为有限数
                    valid_count = data[column].apply(lambda x: np.isfinite(x) if isinstance(x, (int, float, np.number)) else True).sum()
                    total_count = len(data)
                    validity = valid_count / total_count if total_count > 0 else 0.0
                    metrics[f"validity_{column}"] = validity
            
            # 写入WhyLabs
            if self.whylabs_writer:
                try:
                    self.whylabs_writer.write(
                        profile=profile,
                        dataset_id=dataset_name
                    )
                except Exception as e:
                    logger.warning(f"写入WhyLabs失败: {e}")
            
            return {
                "profile_id": profile._profile_id,
                "metrics": metrics,
                "column_profiles": profile._columns
            }
            
        except Exception as e:
            logger.error(f"whylogs画像失败: {e}", exc_info=True)
            return {
                "profile_id": None,
                "metrics": {},
                "error": str(e)
            }
    
    async def _calculate_custom_metrics(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        dataset_version: str
    ) -> List[QualityMetric]:
        """
        计算自定义质量指标
        
        Args:
            data: 数据集
            dataset_name: 数据集名称
            dataset_version: 数据集版本
            
        Returns:
            质量指标列表
        """
        metrics = []
        
        try:
            if len(data) == 0:
                return metrics
            
            # 1. 总体完整性
            total_cells = data.size
            missing_cells = data.isna().sum().sum()
            completeness_rate = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0
            
            metrics.append(QualityMetric(
                name="overall_completeness",
                value=completeness_rate,
                metric_type=QualityMetricType.COMPLETENESS,
                threshold=0.95,
                status="passed" if completeness_rate >= 0.95 else "failed",
                description="总体数据完整性",
                unit="ratio"
            ))
            
            # 2. 行级重复率
            duplicate_rows = data.duplicated().sum()
            total_rows = len(data)
            duplicate_rate = duplicate_rows / total_rows if total_rows > 0 else 0.0
            
            metrics.append(QualityMetric(
                name="duplicate_rate",
                value=duplicate_rate,
                metric_type=QualityMetricType.UNIQUENESS,
                threshold=0.05,
                status="passed" if duplicate_rate <= 0.05 else "failed",
                description="行级重复率",
                unit="ratio"
            ))
            
            # 3. 列级统计
            for column in data.columns:
                # 缺失率
                missing_rate = data[column].isna().sum() / len(data)
                
                metrics.append(QualityMetric(
                    name=f"missing_rate_{column}",
                    value=missing_rate,
                    metric_type=QualityMetricType.COMPLETENESS,
                    threshold=0.10,
                    status="passed" if missing_rate <= 0.10 else "failed",
                    description=f"列 {column} 缺失率",
                    unit="ratio",
                    dimensions={"column": column}
                ))
                
                # 如果是数值型列，计算统计信息
                if pd.api.types.is_numeric_dtype(data[column]):
                    col_data = data[column].dropna()
                    
                    if len(col_data) > 0:
                        # 零值比例
                        zero_rate = (col_data == 0).sum() / len(col_data)
                        
                        metrics.append(QualityMetric(
                            name=f"zero_rate_{column}",
                            value=zero_rate,
                            metric_type=QualityMetricType.VALIDITY,
                            threshold=0.50,  # 允许最多50%的零值
                            status="passed" if zero_rate <= 0.50 else "failed",
                            description=f"列 {column} 零值比例",
                            unit="ratio",
                            dimensions={"column": column}
                        ))
                        
                        # 异常值检测（使用IQR方法）
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                        outlier_rate = len(outliers) / len(col_data)
                        
                        metrics.append(QualityMetric(
                            name=f"outlier_rate_{column}",
                            value=outlier_rate,
                            metric_type=QualityMetricType.VALIDITY,
                            threshold=0.05,  # 允许最多5%的异常值
                            status="passed" if outlier_rate <= 0.05 else "failed",
                            description=f"列 {column} 异常值比例",
                            unit="ratio",
                            dimensions={"column": column}
                        ))
            
            # 4. 数据新鲜度（如果有时间戳列）
            timestamp_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            for ts_col in timestamp_columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(data[ts_col]):
                        latest_timestamp = data[ts_col].max()
                        if pd.notna(latest_timestamp):
                            now = pd.Timestamp.now()
                            age_days = (now - latest_timestamp).days
                            
                            metrics.append(QualityMetric(
                                name=f"data_freshness_{ts_col}",
                                value=age_days,
                                metric_type=QualityMetricType.FRESHNESS,
                                threshold=7,  # 数据应该在一周内
                                status="passed" if age_days <= 7 else "failed",
                                description=f"数据新鲜度（基于列 {ts_col}）",
                                unit="days",
                                dimensions={"column": ts_col}
                            ))
                except:
                    continue
            
            return metrics
            
        except Exception as e:
            logger.error(f"计算自定义指标失败: {e}", exc_info=True)
            return metrics
    
    def _infer_metric_type(self, metric_name: str) -> QualityMetricType:
        """根据指标名称推断指标类型"""
        metric_name_lower = metric_name.lower()
        
        if any(word in metric_name_lower for word in ['completeness', 'missing', 'null']):
            return QualityMetricType.COMPLETENESS
        elif any(word in metric_name_lower for word in ['accuracy', 'correctness', 'precision']):
            return QualityMetricType.ACCURACY
        elif any(word in metric_name_lower for word in ['consistency', 'coherence']):
            return QualityMetricType.CONSISTENCY
        elif any(word in metric_name_lower for word in ['validity', 'valid', 'outlier']):
            return QualityMetricType.VALIDITY
        elif any(word in metric_name_lower for word in ['uniqueness', 'duplicate', 'unique']):
            return QualityMetricType.UNIQUENESS
        elif any(word in metric_name_lower for word in ['freshness', 'timeliness', 'age']):
            return QualityMetricType.TIMELINESS
        elif any(word in metric_name_lower for word in ['volume', 'count', 'size']):
            return QualityMetricType.VOLUME
        elif any(word in metric_name_lower for word in ['distribution', 'skew', 'kurtosis']):
            return QualityMetricType.DISTRIBUTION
        else:
            return QualityMetricType.COMPLETENESS  # 默认
    
    def _get_threshold(self, metric_name: str) -> Optional[float]:
        """获取指标阈值"""
        threshold_obj = self.thresholds.get(metric_name)
        if threshold_obj:
            return threshold_obj.warning_threshold
        return None
    
    def _calculate_overall_score(self, metrics: List[QualityMetric]) -> float:
        """计算总体质量得分"""
        if not metrics:
            return 0.0
        
        # 为不同类型的指标分配不同权重
        weights = {
            QualityMetricType.COMPLETENESS: 0.30,
            QualityMetricType.ACCURACY: 0.25,
            QualityMetricType.VALIDITY: 0.20,
            QualityMetricType.CONSISTENCY: 0.15,
            QualityMetricType.UNIQUENESS: 0.10
        }
        
        total_score = 0
        total_weight = 0
        
        for metric in metrics:
            weight = weights.get(metric.metric_type, 0.05)
            
            # 计算指标得分
            if metric.threshold is not None:
                if metric.metric_type in [
                    QualityMetricType.COMPLETENESS,
                    QualityMetricType.ACCURACY
                ]:
                    # 值越大越好
                    if metric.value >= metric.threshold:
                        score = 100
                    else:
                        score = (metric.value / metric.threshold) * 100
                else:
                    # 值越小越好
                    if metric.value <= metric.threshold:
                        score = 100
                    else:
                        score = max(0, 100 - ((metric.value - metric.threshold) / metric.threshold) * 100)
            else:
                # 没有阈值，给默认分
                score = 50
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_status(self, overall_score: float, metrics: List[QualityMetric]) -> DataQualityStatus:
        """根据总体得分和指标确定质量状态"""
        # 检查是否有严重失败
        critical_failures = 0
        for metric in metrics:
            if metric.threshold is not None and not metric.is_passed():
                # 检查是否严重失败（低于阈值的50%）
                if metric.metric_type in [QualityMetricType.COMPLETENESS, QualityMetricType.ACCURACY]:
                    if metric.value < metric.threshold * 0.5:
                        critical_failures += 1
                else:
                    if metric.value > metric.threshold * 2.0:
                        critical_failures += 1
        
        if critical_failures >= 3:
            return DataQualityStatus.FAILED
        
        # 根据得分确定状态
        if overall_score >= 90:
            return DataQualityStatus.EXCELLENT
        elif overall_score >= 80:
            return DataQualityStatus.GOOD
        elif overall_score >= 70:
            return DataQualityStatus.FAIR
        elif overall_score >= 60:
            return DataQualityStatus.WARNING
        elif overall_score >= 50:
            return DataQualityStatus.POOR
        elif overall_score >= 30:
            return DataQualityStatus.ERROR
        else:
            return DataQualityStatus.FAILED
    
    async def _check_alerts(
        self,
        metrics: List[QualityMetric],
        dataset_name: str
    ) -> List[str]:
        """检查告警"""
        alerts = []
        
        for metric in metrics:
            if metric.threshold is not None and not metric.is_passed():
                # 构建告警消息
                alert_msg = f"[{dataset_name}] 指标 {metric.name} 超出阈值: 值={metric.value:.4f}, 阈值={metric.threshold:.4f}"
                
                if metric.description:
                    alert_msg += f", 描述: {metric.description}"
                
                alerts.append(alert_msg)
        
        return alerts
    
    async def _send_alerts(
        self,
        dataset_name: str,
        alerts: List[str],
        result: DataQualityResult
    ):
        """发送告警"""
        if not alerts:
            return
        
        alert_data = {
            "alert_type": "data_quality_alert",
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "overall_score": result.overall_score,
            "status": result.status.value,
            "alerts": alerts,
            "failed_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "metric_type": m.metric_type.value
                }
                for m in result.get_failed_metrics()
            ],
            "result_url": f"/data-quality/results/{result.check_id}"
        }
        
        # 保存告警
        await self._save_alert(alert_data)
        
        # 发送Webhook告警
        if self.alert_config.get("webhook_url"):
            await self._send_webhook_alert(alert_data)
        
        logger.warning(f"数据质量告警: {dataset_name}, 得分: {result.overall_score:.2f}, 告警数: {len(alerts)}")
    
    async def _save_result(self, result: DataQualityResult):
        """保存结果到文件"""
        try:
            result_dir = self.data_dir / "results"
            result_dir.mkdir(exist_ok=True)
            
            result_file = result_dir / f"result_{result.check_id}.json"
            
            # 准备序列化数据
            result_data = asdict(result)
            
            # 处理特殊类型
            result_data["status"] = result.status.value
            result_data["metrics"] = []
            
            for metric in result.metrics:
                metric_data = asdict(metric)
                metric_data["metric_type"] = metric.metric_type.value
                result_data["metrics"].append(metric_data)
            
            async with aiofiles.open(result_file, 'w') as f:
                await f.write(json.dumps(result_data, indent=2, default=str))
            
            # 同时保存到历史目录
            history_dir = self.data_dir / "history"
            history_dir.mkdir(exist_ok=True)
            
            history_file = history_dir / f"result_{result.check_id}.json"
            async with aiofiles.open(history_file, 'w') as f:
                await f.write(json.dumps(result_data, indent=2, default=str))
                
        except Exception as e:
            logger.error(f"保存结果失败: {e}", exc_info=True)
    
    async def _save_alert(self, alert_data: Dict[str, Any]):
        """保存告警"""
        try:
            alert_dir = self.data_dir / "alerts"
            alert_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = alert_dir / f"alert_{timestamp}.json"
            
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(alert_data, indent=2, default=str))
                
        except Exception as e:
            logger.error(f"保存告警失败: {e}")
    
    async def _send_webhook_alert(self, alert_data: Dict[str, Any]):
        """发送Webhook告警"""
        webhook_url = self.alert_config.get("webhook_url")
        if not webhook_url:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=alert_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Webhook告警发送失败: {response.status}")
        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")
    
    async def validate_data_pipeline(
        self,
        pipeline_name: str,
        datasets: Dict[str, pd.DataFrame],
        pipeline_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        验证数据流水线
        
        Args:
            pipeline_name: 流水线名称
            datasets: 数据集字典
            pipeline_config: 流水线配置
            
        Returns:
            流水线验证结果
        """
        results = {}
        pipeline_start_time = datetime.now()
        
        try:
            logger.info(f"开始验证数据流水线: {pipeline_name}, 数据集数: {len(datasets)}")
            
            for dataset_name, data in datasets.items():
                # 获取数据集配置
                dataset_config = pipeline_config.get("datasets", {}).get(dataset_name, {})
                
                # 执行验证
                result = await self.monitor_dataset(
                    dataset_name=dataset_name,
                    data=data,
                    dataset_version=dataset_config.get("version", "1.0"),
                    expectation_suite_name=dataset_config.get("expectation_suite"),
                    batch_identifier=f"{pipeline_name}_{dataset_name}",
                    validation_mode=dataset_config.get("validation_mode", "standard")
                )
                
                results[dataset_name] = result
            
            # 计算流水线总体状态
            pipeline_status = self._calculate_pipeline_status(results)
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            
            # 创建流水线验证结果
            pipeline_result = DataQualityResult(
                check_id=f"pipeline_{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                dataset_name=pipeline_name,
                dataset_version="1.0",
                timestamp=datetime.now(),
                status=pipeline_status,
                overall_score=self._calculate_pipeline_score(results),
                metrics=self._aggregate_pipeline_metrics(results),
                summary={
                    "total_datasets": len(results),
                    "datasets_status": {name: result.status.value for name, result in results.items()},
                    "overall_status": pipeline_status.value,
                    "pipeline_duration_seconds": pipeline_duration,
                    "success_rate": self._calculate_success_rate(results)
                },
                alerts=self._aggregate_pipeline_alerts(results)
            )
            
            # 保存流水线结果
            await self._save_result(pipeline_result)
            
            return {
                "pipeline": pipeline_result,
                "datasets": results,
                "success": pipeline_status in [DataQualityStatus.EXCELLENT, DataQualityStatus.GOOD, DataQualityStatus.FAIR]
            }
            
        except Exception as e:
            logger.error(f"验证数据流水线失败: {e}", exc_info=True)
            
            pipeline_result = DataQualityResult(
                check_id=f"pipeline_{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                dataset_name=pipeline_name,
                dataset_version="1.0",
                timestamp=datetime.now(),
                status=DataQualityStatus.FAILED,
                overall_score=0.0,
                metrics=[],
                summary={
                    "error": str(e),
                    "total_datasets": len(datasets),
                    "success": False
                },
                alerts=[f"流水线验证失败: {str(e)[:200]}"]
            )
            
            return {
                "pipeline": pipeline_result,
                "datasets": {},
                "success": False
            }
    
    def _calculate_pipeline_status(self, results: Dict[str, DataQualityResult]) -> DataQualityStatus:
        """计算流水线总体状态"""
        if not results:
            return DataQualityStatus.UNKNOWN
        
        # 获取最差的状态
        status_priority = {
            DataQualityStatus.FAILED: 0,
            DataQualityStatus.ERROR: 1,
            DataQualityStatus.POOR: 2,
            DataQualityStatus.WARNING: 3,
            DataQualityStatus.FAIR: 4,
            DataQualityStatus.GOOD: 5,
            DataQualityStatus.EXCELLENT: 6,
            DataQualityStatus.UNKNOWN: 7
        }
        
        worst_status = DataQualityStatus.EXCELLENT
        for result in results.values():
            if status_priority[result.status] < status_priority[worst_status]:
                worst_status = result.status
        
        return worst_status
    
    def _calculate_pipeline_score(self, results: Dict[str, DataQualityResult]) -> float:
        """计算流水线总体得分"""
        if not results:
            return 0.0
        
        total_score = 0
        for result in results.values():
            total_score += result.overall_score
        
        return total_score / len(results)
    
    def _calculate_success_rate(self, results: Dict[str, DataQualityResult]) -> float:
        """计算成功率"""
        if not results:
            return 0.0
        
        successful_datasets = 0
        for result in results.values():
            if result.status in [DataQualityStatus.EXCELLENT, DataQualityStatus.GOOD, DataQualityStatus.FAIR]:
                successful_datasets += 1
        
        return successful_datasets / len(results)
    
    def _aggregate_pipeline_metrics(self, results: Dict[str, DataQualityResult]) -> List[QualityMetric]:
        """聚合流水线指标"""
        aggregated_metrics = []
        
        # 按指标类型聚合
        metric_by_type = defaultdict(list)
        
        for dataset_name, result in results.items():
            for metric in result.metrics:
                key = f"{metric.metric_type.value}_{metric.name}"
                metric_by_type[key].append({
                    "value": metric.value,
                    "dataset": dataset_name
                })
        
        # 创建聚合指标
        for metric_key, metric_values in metric_by_type.items():
            if metric_values:
                values = [m["value"] for m in metric_values]
                avg_value = sum(values) / len(values)
                
                # 推断指标类型
                parts = metric_key.split('_', 1)
                if len(parts) == 2:
                    metric_type_str, metric_name = parts
                    try:
                        metric_type = QualityMetricType(metric_type_str)
                    except:
                        metric_type = QualityMetricType.COMPLETENESS
                else:
                    metric_type = QualityMetricType.COMPLETENESS
                    metric_name = metric_key
                
                aggregated_metrics.append(QualityMetric(
                    name=f"pipeline_avg_{metric_name}",
                    value=avg_value,
                    metric_type=metric_type,
                    description=f"流水线平均指标: {metric_name}",
                    unit="ratio"
                ))
        
        return aggregated_metrics
    
    def _aggregate_pipeline_alerts(self, results: Dict[str, DataQualityResult]) -> List[str]:
        """聚合流水线告警"""
        all_alerts = []
        
        for dataset_name, result in results.items():
            for alert in result.alerts:
                all_alerts.append(f"[{dataset_name}] {alert}")
        
        return all_alerts
    
    async def track_data_quality_over_time(
        self,
        dataset_name: str,
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """
        跟踪数据质量随时间变化
        
        Args:
            dataset_name: 数据集名称
            lookback_days: 回溯天数
            
        Returns:
            质量趋势
        """
        # 加载历史结果
        historical_results = await self._load_historical_results(dataset_name, lookback_days)
        
        if not historical_results:
            return {
                "dataset_name": dataset_name,
                "has_data": False,
                "message": f"最近 {lookback_days} 天内没有找到 {dataset_name} 的历史数据"
            }
        
        # 分析趋势
        trend_analysis = self._analyze_quality_trend(historical_results)
        
        # 检测异常
        anomalies = self._detect_quality_anomalies(historical_results)
        
        # 计算统计信息
        scores = [r.overall_score for r in historical_results]
        statuses = [r.status.value for r in historical_results]
        
        return {
            "dataset_name": dataset_name,
            "has_data": True,
            "period": f"最近 {lookback_days} 天",
            "total_checks": len(historical_results),
            "score_statistics": {
                "mean": np.mean(scores) if scores else 0,
                "std": np.std(scores) if len(scores) > 1 else 0,
                "min": np.min(scores) if scores else 0,
                "max": np.max(scores) if scores else 0,
                "latest": scores[-1] if scores else 0
            },
            "status_distribution": {
                status: statuses.count(status) for status in set(statuses)
            },
            "trend_analysis": trend_analysis,
            "anomalies": anomalies,
            "historical_summary": [
                {
                    "check_id": result.check_id,
                    "timestamp": result.timestamp.isoformat(),
                    "score": result.overall_score,
                    "status": result.status.value,
                    "passed_checks": result.summary.get("passed_checks", 0),
                    "failed_checks": result.summary.get("failed_checks", 0)
                }
                for result in historical_results[-10:]  # 最近10次结果
            ]
        }
    
    async def _load_historical_results(
        self,
        dataset_name: str,
        lookback_days: int
    ) -> List[DataQualityResult]:
        """加载历史结果"""
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        # 从缓存中获取
        if dataset_name in self.history_cache:
            cached_results = self.history_cache[dataset_name]
            return [r for r in cached_results if r.timestamp >= cutoff_date]
        
        # 从文件系统加载
        history_dir = self.data_dir / "history"
        if not history_dir.exists():
            return []
        
        historical_results = []
        
        for result_file in history_dir.glob(f"*{dataset_name}*.json"):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                # 检查时间
                result_time = datetime.fromisoformat(result_data["timestamp"])
                if result_time < cutoff_date:
                    continue
                
                # 转换metrics
                metrics = []
                for metric_data in result_data.get("metrics", []):
                    metric_data["metric_type"] = QualityMetricType(metric_data["metric_type"])
                    metrics.append(QualityMetric(**metric_data))
                
                result_data["status"] = DataQualityStatus(result_data["status"])
                result_data["metrics"] = metrics
                
                result = DataQualityResult(**result_data)
                historical_results.append(result)
                
            except Exception as e:
                continue
        
        # 按时间排序
        historical_results.sort(key=lambda x: x.timestamp)
        
        return historical_results
    
    def _analyze_quality_trend(self, historical_results: List[DataQualityResult]) -> Dict[str, Any]:
        """分析质量趋势"""
        if len(historical_results) < 2:
            return {"has_trend": False, "message": "数据不足，无法分析趋势"}
        
        scores = [r.overall_score for r in historical_results]
        timestamps = [r.timestamp for r in historical_results]
        
        # 计算趋势线
        try:
            # 将时间转换为数值
            time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
            
            # 线性回归
            slope, intercept = np.polyfit(time_numeric, scores, 1)
            
            # 趋势判断
            if slope > 0.1:
                trend = "improving"
            elif slope < -0.1:
                trend = "deteriorating"
            else:
                trend = "stable"
            
            return {
                "has_trend": True,
                "trend": trend,
                "slope": float(slope),
                "latest_score": scores[-1],
                "score_change": scores[-1] - scores[0],
                "period_days": (timestamps[-1] - timestamps[0]).days
            }
            
        except Exception as e:
            return {
                "has_trend": False,
                "error": str(e),
                "latest_score": scores[-1] if scores else 0
            }
    
    def _detect_quality_anomalies(self, historical_results: List[DataQualityResult]) -> List[Dict[str, Any]]:
        """检测质量异常"""
        if len(historical_results) < 3:
            return []
        
        anomalies = []
        scores = [r.overall_score for r in historical_results]
        timestamps = [r.timestamp for r in historical_results]
        
        # 使用简单的标准差方法检测异常
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        for i, (score, timestamp, result) in enumerate(zip(scores, timestamps, historical_results)):
            # 检查是否显著偏离均值
            if abs(score - mean_score) > 2 * std_score:
                anomalies.append({
                    "timestamp": timestamp.isoformat(),
                    "score": score,
                    "mean_score": mean_score,
                    "deviation": score - mean_score,
                    "check_id": result.check_id,
                    "reason": f"得分显著偏离平均值: {score:.2f} vs {mean_score:.2f} (标准差: {std_score:.2f})"
                })
            
            # 检查相邻点之间的剧烈变化
            if i > 0:
                prev_score = scores[i-1]
                score_change = abs(score - prev_score)
                
                if score_change > 10:  # 得分变化超过10分
                    anomalies.append({
                        "timestamp": timestamp.isoformat(),
                        "score": score,
                        "prev_score": prev_score,
                        "change": score - prev_score,
                        "check_id": result.check_id,
                        "reason": f"得分剧烈变化: {prev_score:.2f} -> {score:.2f} (变化: {score - prev_score:.2f})"
                    })
        
        return anomalies
    
    async def register_dataset_schema(
        self,
        dataset_name: str,
        version: str,
        columns: Dict[str, Dict[str, Any]],
        primary_keys: List[str],
        required_columns: List[str],
        constraints: List[Dict[str, Any]]
    ) -> bool:
        """
        注册数据集模式
        
        Args:
            dataset_name: 数据集名称
            version: 版本
            columns: 列定义
            primary_keys: 主键列
            required_columns: 必需列
            constraints: 约束条件
            
        Returns:
            是否成功
        """
        try:
            schema = DatasetSchema(
                dataset_name=dataset_name,
                version=version,
                columns=columns,
                primary_keys=primary_keys,
                required_columns=required_columns,
                constraints=constraints,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.schemas[dataset_name] = schema
            self._save_schema(schema)
            
            logger.info(f"数据集模式注册成功: {dataset_name} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"注册数据集模式失败: {e}")
            return False
    
    async def validate_against_schema(
        self,
        dataset_name: str,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        根据模式验证数据
        
        Args:
            dataset_name: 数据集名称
            data: 要验证的数据
            
        Returns:
            验证结果
        """
        if dataset_name not in self.schemas:
            return {
                "valid": False,
                "error": f"数据集 {dataset_name} 的模式未注册"
            }
        
        schema = self.schemas[dataset_name]
        violations = []
        
        # 检查必需列
        for required_column in schema.required_columns:
            if required_column not in data.columns:
                violations.append(f"缺失必需列: {required_column}")
        
        # 检查列类型
        for column_name, column_def in schema.columns.items():
            if column_name in data.columns:
                expected_type = column_def.get("type")
                if expected_type:
                    actual_type = str(data[column_name].dtype)
                    # 简化类型检查
                    if not self._check_type_compatibility(actual_type, expected_type):
                        violations.append(f"列 {column_name} 类型不匹配: 期望 {expected_type}, 实际 {actual_type}")
        
        # 检查主键唯一性
        if schema.primary_keys:
            pk_columns = [col for col in schema.primary_keys if col in data.columns]
            if pk_columns:
                duplicate_rows = data.duplicated(subset=pk_columns).sum()
                if duplicate_rows > 0:
                    violations.append(f"主键重复: {duplicate_rows} 行重复")
        
        # 检查约束
        for constraint in schema.constraints:
            constraint_type = constraint.get("type")
            column = constraint.get("column")
            
            if column in data.columns:
                if constraint_type == "not_null":
                    null_count = data[column].isna().sum()
                    if null_count > 0:
                        violations.append(f"列 {column} 有 {null_count} 个空值，违反非空约束")
                
                elif constraint_type == "unique":
                    duplicate_count = data.duplicated(subset=[column]).sum()
                    if duplicate_count > 0:
                        violations.append(f"列 {column} 有 {duplicate_count} 个重复值，违反唯一约束")
                
                elif constraint_type == "range":
                    min_val = constraint.get("min")
                    max_val = constraint.get("max")
                    
                    if min_val is not None:
                        below_min = (data[column] < min_val).sum()
                        if below_min > 0:
                            violations.append(f"列 {column} 有 {below_min} 个值小于最小值 {min_val}")
                    
                    if max_val is not None:
                        above_max = (data[column] > max_val).sum()
                        if above_max > 0:
                            violations.append(f"列 {column} 有 {above_max} 个值大于最大值 {max_val}")
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "schema_version": schema.version,
            "total_violations": len(violations)
        }
    
    def _check_type_compatibility(self, actual_type: str, expected_type: str) -> bool:
        """检查类型兼容性"""
        type_mapping = {
            "int": ["int64", "int32", "int16", "int8"],
            "float": ["float64", "float32"],
            "str": ["object", "string"],
            "bool": ["bool"],
            "datetime": ["datetime64[ns]", "datetime64"],
            "category": ["category"]
        }
        
        for compatible_types in type_mapping.values():
            if expected_type in compatible_types and actual_type in compatible_types:
                return True
        
        return False
    
    async def get_dataset_report(
        self,
        dataset_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        获取数据集质量报告
        
        Args:
            dataset_name: 数据集名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            质量报告
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # 获取历史结果
        historical_results = []
        if dataset_name in self.history_cache:
            for result in self.history_cache[dataset_name]:
                if start_date <= result.timestamp <= end_date:
                    historical_results.append(result)
        
        if not historical_results:
            return {
                "dataset_name": dataset_name,
                "has_data": False,
                "period": f"{start_date.date()} 到 {end_date.date()}",
                "message": f"在指定时间段内没有找到 {dataset_name} 的质量数据"
            }
        
        # 计算统计信息
        scores = [r.overall_score for r in historical_results]
        statuses = [r.status.value for r in historical_results]
        
        # 按天聚合
        daily_metrics = defaultdict(list)
        for result in historical_results:
            date_key = result.timestamp.date().isoformat()
            daily_metrics[date_key].append(result.overall_score)
        
        daily_stats = {}
        for date_key, date_scores in daily_metrics.items():
            daily_stats[date_key] = {
                "mean": np.mean(date_scores),
                "min": np.min(date_scores),
                "max": np.max(date_scores),
                "count": len(date_scores)
            }
        
        # 获取最新结果
        latest_result = historical_results[-1]
        
        return {
            "dataset_name": dataset_name,
            "has_data": True,
            "period": f"{start_date.date()} 到 {end_date.date()}",
            "total_checks": len(historical_results),
            "score_statistics": {
                "mean": np.mean(scores),
                "std": np.std(scores) if len(scores) > 1 else 0,
                "min": np.min(scores),
                "max": np.max(scores),
                "latest": scores[-1]
            },
            "status_distribution": {
                status: statuses.count(status) for status in set(statuses)
            },
            "latest_check": {
                "timestamp": latest_result.timestamp.isoformat(),
                "score": latest_result.overall_score,
                "status": latest_result.status.value,
                "passed_checks": latest_result.summary.get("passed_checks", 0),
                "failed_checks": latest_result.summary.get("failed_checks", 0)
            },
            "daily_statistics": daily_stats,
            "recommendations": self._generate_recommendations(historical_results)
        }
    
    def _generate_recommendations(self, historical_results: List[DataQualityResult]) -> List[str]:
        """生成改进建议"""
        if not historical_results:
            return []
        
        recommendations = []
        latest_result = historical_results[-1]
        
        # 检查失败的指标
        failed_metrics = latest_result.get_failed_metrics()
        for metric in failed_metrics:
            if metric.metric_type == QualityMetricType.COMPLETENESS:
                recommendations.append(f"提高数据完整性: {metric.name} 当前值为 {metric.value:.2%}，低于阈值 {metric.threshold:.2%}")
            elif metric.metric_type == QualityMetricType.ACCURACY:
                recommendations.append(f"提高数据准确性: {metric.name} 当前值为 {metric.value:.2%}，低于阈值 {metric.threshold:.2%}")
            elif metric.metric_type == QualityMetricType.VALIDITY:
                recommendations.append(f"改善数据有效性: {metric.name} 当前值为 {metric.value:.2%}，高于阈值 {metric.threshold:.2%}")
        
        # 检查趋势
        if len(historical_results) >= 5:
            recent_scores = [r.overall_score for r in historical_results[-5:]]
            if all(recent_scores[i] < recent_scores[i-1] for i in range(1, 5)):
                recommendations.append("数据质量持续下降，建议立即检查数据处理流程")
        
        return recommendations
    
    async def cleanup_old_results(self, days_to_keep: int = 90):
        """
        清理旧的结果数据
        
        Args:
            days_to_keep: 保留天数
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # 清理结果文件
        result_dir = self.data_dir / "results"
        if result_dir.exists():
            for result_file in result_dir.glob("*.json"):
                try:
                    file_time = datetime.fromtimestamp(result_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        result_file.unlink()
                except:
                    pass
        
        # 清理历史文件
        history_dir = self.data_dir / "history"
        if history_dir.exists():
            for history_file in history_dir.glob("*.json"):
                try:
                    file_time = datetime.fromtimestamp(history_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        history_file.unlink()
                except:
                    pass
        
        # 清理告警文件
        alert_dir = self.data_dir / "alerts"
        if alert_dir.exists():
            for alert_file in alert_dir.glob("*.json"):
                try:
                    file_time = datetime.fromtimestamp(alert_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        alert_file.unlink()
                except:
                    pass
        
        # 更新缓存
        for dataset_name in list(self.history_cache.keys()):
            self.history_cache[dataset_name] = [
                r for r in self.history_cache[dataset_name]
                if r.timestamp >= cutoff_date
            ]
            
            if not self.history_cache[dataset_name]:
                del self.history_cache[dataset_name]
        
        logger.info(f"清理完成: 删除了 {days_to_keep} 天前的数据质量结果")
    
    async def export_report(
        self,
        dataset_name: str,
        format: str = "json",  # json, html, pdf, csv
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        导出质量报告
        
        Args:
            dataset_name: 数据集名称
            format: 导出格式
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            导出结果
        """
        try:
            # 获取报告数据
            report_data = await self.get_dataset_report(dataset_name, start_date, end_date)
            
            if not report_data.get("has_data", False):
                return {
                    "success": False,
                    "error": "没有可用的报告数据"
                }
            
            # 生成导出文件
            export_dir = self.data_dir / "exports"
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{dataset_name}_quality_report_{timestamp}"
            
            if format == "json":
                filepath = export_dir / f"{filename}.json"
                with open(filepath, 'w') as f:
                    json.dump(report_data, f, indent=2)
