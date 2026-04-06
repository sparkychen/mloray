"""
完整的数据质量监控模块
基于Great Expectations、whylogs、自定义规则的数据质量验证、监控和告警
支持实时监控、批处理验证、趋势分析和异常检测
"""

import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint import SimpleCheckpoint
from great_expectations.data_context import FileDataContext
import whylogs as why
from whylogs.core import DatasetProfile
from whylogs.api.writer.whylabs import WhyLabsWriter
from whylogs.api.writer.local import LocalWriter
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path
import aiofiles
import logging
from dataclasses import dataclass, asdict, field
from enum import Enum
import aiohttp
import hashlib
import yaml
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from ..config import settings

logger = logging.getLogger(__name__)


class DataQualityStatus(Enum):
    """数据质量状态"""
    EXCELLENT = "excellent"    # 所有检查通过，指标优秀
    GOOD = "good"             # 所有检查通过，指标良好
    WARNING = "warning"       # 有警告但无错误
    ERROR = "error"          # 有错误但不严重
    CRITICAL = "critical"    # 严重错误
    FAILED = "failed"        # 检查失败


class QualityCheckType(Enum):
    """质量检查类型"""
    COMPLETENESS = "completeness"    # 完整性检查
    ACCURACY = "accuracy"           # 准确性检查
    CONSISTENCY = "consistency"     # 一致性检查
    VALIDITY = "validity"           # 有效性检查
    UNIQUENESS = "uniqueness"       # 唯一性检查
    TIMELINESS = "timeliness"       # 及时性检查
    DISTRIBUTION = "distribution"   # 分布检查
    CUSTOM = "custom"              # 自定义检查


@dataclass
class QualityMetric:
    """质量指标"""
    name: str
    value: float
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    status: str = "unknown"
    description: Optional[str] = None
    check_type: QualityCheckType = QualityCheckType.CUSTOM
    feature_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_passed(self) -> bool:
        """检查是否通过"""
        if self.status in ["passed", "excellent", "good"]:
            return True
        
        if self.threshold_min is not None and self.value < self.threshold_min:
            return False
        if self.threshold_max is not None and self.value > self.threshold_max:
            return False
        
        return self.status in ["passed", "excellent", "good"]
    
    def is_warning(self) -> bool:
        """检查是否为警告"""
        return self.status in ["warning", "needs_attention"]


@dataclass
class QualityCheckResult:
    """质量检查结果"""
    check_id: str
    check_name: str
    check_type: QualityCheckType
    status: str
    message: str
    metrics: List[QualityMetric]
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    dataset_name: str
    batch_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityResult:
    """数据质量结果"""
    result_id: str
    dataset_name: str
    timestamp: datetime
    overall_status: DataQualityStatus
    total_checks: int
    passed_checks: int
    warning_checks: int
    failed_checks: int
    error_checks: int
    checks: List[QualityCheckResult]
    metrics_summary: Dict[str, Any]
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityRule:
    """质量规则"""
    rule_id: str
    name: str
    description: str
    rule_type: QualityCheckType
    condition: Dict[str, Any]  # 规则条件
    severity: str  # low, medium, high, critical
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class DataProfile:
    """数据画像"""
    profile_id: str
    dataset_name: str
    timestamp: datetime
    row_count: int
    column_count: int
    missing_values: int
    duplicate_rows: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    text_columns: List[str]
    column_stats: Dict[str, Dict[str, Any]]
    distribution_info: Dict[str, Any]
    whylogs_profile_id: Optional[str] = None
    profile_hash: Optional[str] = None


class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化数据质量监控器
        
        Args:
            config_path: 配置文件路径
        """
        # 初始化Great Expectations上下文
        self.ge_context = self._initialize_ge_context()
        
        # 初始化whylogs writer
        self.whylabs_writer = self._initialize_whylabs_writer()
        self.local_writer = LocalWriter()
        
        # 质量规则存储
        self.quality_rules: Dict[str, QualityRule] = {}
        
        # 结果缓存
        self.results_cache: Dict[str, DataQualityResult] = {}
        self.profiles_cache: Dict[str, DataProfile] = {}
        
        # 告警历史
        self.alert_history: List[Dict[str, Any]] = []
        
        # 存储配置
        self.data_dir = Path("data_quality")
        self.data_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.data_dir / "results").mkdir(exist_ok=True)
        (self.data_dir / "profiles").mkdir(exist_ok=True)
        (self.data_dir / "alerts").mkdir(exist_ok=True)
        (self.data_dir / "rules").mkdir(exist_ok=True)
        
        # 加载配置和规则
        self.config = self._load_config(config_path)
        self._load_quality_rules()
        
        # 初始化指标追踪
        self._init_metrics_tracking()
        
        logger.info("数据质量监控器初始化完成")
    
    def _initialize_ge_context(self) -> FileDataContext:
        """初始化Great Expectations上下文"""
        try:
            context = FileDataContext(project_root_dir=".")
            
            # 检查是否已配置数据源
            if "pandas_datasource" not in context.list_datasources():
                # 添加Pandas数据源
                context.add_datasource(
                    name="pandas_datasource",
                    module_name="great_expectations.datasource",
                    class_name="PandasDatasource"
                )
            
            return context
            
        except Exception as e:
            logger.error(f"初始化Great Expectations失败: {e}")
            raise
    
    def _initialize_whylabs_writer(self) -> Optional[WhyLabsWriter]:
        """初始化whylogs writer"""
        try:
            if (settings.monitoring.whylogs_whylabs_key and 
                settings.monitoring.whylogs_whylabs_org_id):
                writer = WhyLabsWriter(
                    org_id=settings.monitoring.whylogs_whylabs_org_id,
                    api_key=settings.monitoring.whylogs_whylabs_key
                )
                logger.info("WhyLabs writer初始化完成")
                return writer
        except Exception as e:
            logger.warning(f"初始化WhyLabs writer失败: {e}")
        
        return None
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            "monitoring": {
                "enabled": True,
                "batch_size": 10000,
                "check_interval_minutes": 60,
                "retention_days": 30
            },
            "alerts": {
                "enabled": True,
                "webhook_url": None,
                "email_recipients": [],
                "slack_webhook": None
            },
            "thresholds": {
                "completeness": 0.95,      # 完整性阈值
                "accuracy": 0.98,          # 准确性阈值
                "validity": 0.99,          # 有效性阈值
                "uniqueness": 0.95,        # 唯一性阈值
                "timeliness_hours": 24,    # 及时性阈值（小时）
                "distribution_ks_threshold": 0.05,  # 分布KS检验阈值
                "outlier_threshold": 0.01   # 异常值阈值
            },
            "profiling": {
                "enabled": True,
                "sample_size": 10000,
                "generate_reports": True,
                "store_profiles": True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                # 合并配置
                self._merge_configs(default_config, user_config)
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
        
        return default_config
    
    def _merge_configs(self, base: Dict[str, Any], update: Dict[str, Any]):
        """递归合并配置"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def _load_quality_rules(self):
        """加载质量规则"""
        rules_dir = self.data_dir / "rules"
        
        if not rules_dir.exists():
            # 创建默认规则
            self._create_default_rules()
            return
        
        for rule_file in rules_dir.glob("*.json"):
            try:
                with open(rule_file, 'r') as f:
                    rule_data = json.load(f)
                    rule = QualityRule(**rule_data)
                    self.quality_rules[rule.rule_id] = rule
            except Exception as e:
                logger.error(f"加载规则文件失败 {rule_file}: {e}")
    
    def _create_default_rules(self):
        """创建默认质量规则"""
        default_rules = [
            {
                "rule_id": "completeness_rule_001",
                "name": "数据完整性检查",
                "description": "检查数据缺失值比例",
                "rule_type": QualityCheckType.COMPLETENESS,
                "condition": {
                    "metric": "missing_ratio",
                    "threshold": 0.05,
                    "operator": "<="
                },
                "severity": "high",
                "enabled": True,
                "tags": ["completeness", "mandatory"]
            },
            {
                "rule_id": "uniqueness_rule_001",
                "name": "主键唯一性检查",
                "description": "检查主键列的唯一性",
                "rule_type": QualityCheckType.UNIQUENESS,
                "condition": {
                    "metric": "duplicate_ratio",
                    "threshold": 0.001,
                    "operator": "<="
                },
                "severity": "high",
                "enabled": True,
                "tags": ["uniqueness", "mandatory"]
            },
            {
                "rule_id": "validity_rule_001",
                "name": "数据有效性检查",
                "description": "检查数据是否在有效范围内",
                "rule_type": QualityCheckType.VALIDITY,
                "condition": {
                    "metric": "invalid_ratio",
                    "threshold": 0.01,
                    "operator": "<="
                },
                "severity": "medium",
                "enabled": True,
                "tags": ["validity", "business"]
            },
            {
                "rule_id": "consistency_rule_001",
                "name": "数据一致性检查",
                "description": "检查跨表数据一致性",
                "rule_type": QualityCheckType.CONSISTENCY,
                "condition": {
                    "metric": "inconsistency_ratio",
                    "threshold": 0.02,
                    "operator": "<="
                },
                "severity": "medium",
                "enabled": True,
                "tags": ["consistency", "business"]
            },
            {
                "rule_id": "distribution_rule_001",
                "name": "数据分布检查",
                "description": "检查数据分布是否异常",
                "rule_type": QualityCheckType.DISTRIBUTION,
                "condition": {
                    "metric": "distribution_ks_stat",
                    "threshold": 0.1,
                    "operator": "<="
                },
                "severity": "low",
                "enabled": True,
                "tags": ["distribution", "monitoring"]
            }
        ]
        
        for rule_data in default_rules:
            rule = QualityRule(**rule_data)
            self.quality_rules[rule.rule_id] = rule
            self._save_quality_rule(rule)
    
    def _save_quality_rule(self, rule: QualityRule):
        """保存质量规则"""
        rules_dir = self.data_dir / "rules"
        rules_dir.mkdir(exist_ok=True)
        
        file_path = rules_dir / f"{rule.rule_id}.json"
        
        with open(file_path, 'w') as f:
            json.dump(asdict(rule), f, indent=2, default=str)
    
    def _init_metrics_tracking(self):
        """初始化指标追踪"""
        # 这里可以初始化Prometheus或其他监控指标
        self.metrics_history = defaultdict(list)
    
    async def monitor_dataset(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        expectation_suite_name: Optional[str] = None,
        batch_identifier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataQualityResult:
        """
        监控数据集质量
        
        Args:
            dataset_name: 数据集名称
            data: 数据集
            expectation_suite_name: Great Expectations期望套件名称
            batch_identifier: 批处理标识符
            metadata: 元数据
            
        Returns:
            数据质量结果
        """
        result_id = f"dq_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        try:
            logger.info(f"开始数据质量监控: {dataset_name}, 数据形状: {data.shape}")
            
            # 1. 数据画像
            profile = await self._create_data_profile(dataset_name, data)
            
            # 2. Great Expectations验证
            ge_checks = []
            if expectation_suite_name:
                ge_checks = await self._validate_with_great_expectations(
                    data, dataset_name, expectation_suite_name, batch_identifier
                )
            
            # 3. 规则检查
            rule_checks = await self._check_quality_rules(dataset_name, data, profile)
            
            # 4. 自定义检查
            custom_checks = await self._perform_custom_checks(dataset_name, data, profile)
            
            # 5. 合并所有检查结果
            all_checks = ge_checks + rule_checks + custom_checks
            
            # 6. 计算总体状态
            overall_status = self._calculate_overall_status(all_checks)
            
            # 7. 汇总指标
            metrics_summary = self._summarize_metrics(all_checks, profile)
            
            # 8. 创建结果
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = DataQualityResult(
                result_id=result_id,
                dataset_name=dataset_name,
                timestamp=end_time,
                overall_status=overall_status,
                total_checks=len(all_checks),
                passed_checks=sum(1 for c in all_checks if c.status in ["passed", "excellent", "good"]),
                warning_checks=sum(1 for c in all_checks if c.status == "warning"),
                failed_checks=sum(1 for c in all_checks if c.status in ["failed", "error"]),
                error_checks=sum(1 for c in all_checks if c.status == "error"),
                checks=all_checks,
                metrics_summary=metrics_summary,
                metadata=metadata or {}
            )
            
            # 9. 添加建议
            result.recommendations = self._generate_recommendations(result, profile)
            
            # 10. 检查告警
            alerts = await self._check_alerts(result)
            result.alerts = alerts
            
            # 11. 保存结果
            self.results_cache[result_id] = result
            await self._save_result(result)
            
            # 12. 发送告警
            if alerts:
                await self._send_alerts(dataset_name, alerts, result)
            
            # 13. 记录到whylogs
            if self.whylabs_writer:
                await self._log_to_whylabs(dataset_name, data, result)
            
            logger.info(f"数据质量监控完成: {dataset_name}, 状态: {overall_status.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"数据质量监控失败: {e}")
            
            # 创建错误结果
            return DataQualityResult(
                result_id=result_id,
                dataset_name=dataset_name,
                timestamp=datetime.now(),
                overall_status=DataQualityStatus.FAILED,
                total_checks=0,
                passed_checks=0,
                warning_checks=0,
                failed_checks=1,
                error_checks=1,
                checks=[],
                metrics_summary={"error": str(e)},
                alerts=[f"监控过程失败: {str(e)}"]
            )
    
    async def _create_data_profile(
        self,
        dataset_name: str,
        data: pd.DataFrame
    ) -> DataProfile:
        """创建数据画像"""
        try:
            # 生成唯一ID
            profile_id = f"profile_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 基本统计
            row_count = len(data)
            column_count = len(data.columns)
            
            # 计算缺失值
            missing_values = data.isnull().sum().sum()
            
            # 计算重复行
            duplicate_rows = data.duplicated().sum()
            
            # 列类型分类
            numeric_columns = []
            categorical_columns = []
            datetime_columns = []
            text_columns = []
            
            column_stats = {}
            
            for column in data.columns:
                col_data = data[column]
                dtype = str(col_data.dtype)
                
                # 列统计
                stats = {
                    "dtype": dtype,
                    "non_null_count": col_data.count(),
                    "null_count": col_data.isnull().sum(),
                    "null_ratio": col_data.isnull().sum() / row_count if row_count > 0 else 0
                }
                
                # 根据类型添加额外统计
                if np.issubdtype(col_data.dtype, np.number):
                    numeric_columns.append(column)
                    stats.update({
                        "min": float(col_data.min()) if not col_data.empty else None,
                        "max": float(col_data.max()) if not col_data.empty else None,
                        "mean": float(col_data.mean()) if not col_data.empty else None,
                        "std": float(col_data.std()) if not col_data.empty else None,
                        "median": float(col_data.median()) if not col_data.empty else None
                    })
                elif np.issubdtype(col_data.dtype, np.datetime64):
                    datetime_columns.append(column)
                    if not col_data.empty:
                        stats.update({
                            "min": col_data.min().isoformat(),
                            "max": col_data.max().isoformat()
                        })
                elif col_data.dtype == 'object' or col_data.dtype == 'string':
                    # 检查是否为分类数据
                    unique_count = col_data.nunique()
                    if unique_count < 50 and unique_count < row_count * 0.5:
                        categorical_columns.append(column)
                        stats.update({
                            "unique_count": unique_count,
                            "top_values": col_data.value_counts().head(5).to_dict()
                        })
                    else:
                        text_columns.append(column)
                        # 文本统计
                        if not col_data.empty:
                            avg_length = col_data.dropna().apply(lambda x: len(str(x))).mean()
                            stats["avg_length"] = avg_length
                
                column_stats[column] = stats
            
            # 创建whylogs画像
            whylogs_profile = None
            whylogs_profile_id = None
            
            try:
                whylogs_profile = why.log(data).profile()
                whylogs_profile_id = f"whylogs_{dataset_name}_{datetime.now().strftime('%Y%m%d')}"
                
                # 保存到本地
                self.local_writer.write(profile=whylogs_profile, file=whylogs_profile_id)
            except Exception as e:
                logger.warning(f"创建whylogs画像失败: {e}")
            
            # 计算数据哈希
            profile_hash = self._calculate_data_hash(data)
            
            # 创建数据画像
            profile = DataProfile(
                profile_id=profile_id,
                dataset_name=dataset_name,
                timestamp=datetime.now(),
                row_count=row_count,
                column_count=column_count,
                missing_values=missing_values,
                duplicate_rows=duplicate_rows,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                datetime_columns=datetime_columns,
                text_columns=text_columns,
                column_stats=column_stats,
                distribution_info={
                    "skewness": self._calculate_skewness(data) if numeric_columns else {},
                    "kurtosis": self._calculate_kurtosis(data) if numeric_columns else {}
                },
                whylogs_profile_id=whylogs_profile_id,
                profile_hash=profile_hash
            )
            
            # 缓存画像
            self.profiles_cache[profile_id] = profile
            
            # 保存画像
            await self._save_profile(profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"创建数据画像失败: {e}")
            raise
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """计算数据哈希"""
        try:
            # 将数据转换为字符串并计算哈希
            data_str = data.to_csv(index=False)
            return hashlib.md5(data_str.encode()).hexdigest()
        except:
            return "unknown"
    
    def _calculate_skewness(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算偏度"""
        skewness = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            try:
                skewness[col] = float(data[col].skew())
            except:
                skewness[col] = 0.0
        return skewness
    
    def _calculate_kurtosis(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算峰度"""
        kurtosis = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            try:
                kurtosis[col] = float(data[col].kurtosis())
            except:
                kurtosis[col] = 0.0
        return kurtosis
    
    async def _validate_with_great_expectations(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        expectation_suite_name: str,
        batch_identifier: Optional[str] = None
    ) -> List[QualityCheckResult]:
        """使用Great Expectations验证数据"""
        checks = []
        
        try:
            # 创建批处理请求
            batch_request = RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="default_runtime_data_connector",
                data_asset_name=dataset_name,
                runtime_parameters={"batch_data": data},
                batch_identifiers={"batch_id": batch_identifier or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')"},
            )
            
            # 运行验证
            checkpoint = SimpleCheckpoint(
                name=f"{dataset_name}_checkpoint",
                data_context=self.ge_context,
                config_version=1,
                run_name_template=f"%Y%m%d-%H%M%S-{dataset_name}",
                expectation_suite_name=expectation_suite_name,
            )
            
            results = checkpoint.run(batch_request=batch_request)
            
            # 处理结果
            for i, expectation_result in enumerate(results["results"]):
                check_id = f"ge_check_{dataset_name}_{i}"
                check_name = expectation_result.expectation_config.expectation_type
                
                # 创建指标
                metrics = []
                
                # 期望结果指标
                if expectation_result.success:
                    status = "passed"
                else:
                    status = "failed"
                
                metrics.append(QualityMetric(
                    name="expectation_success",
                    value=1.0 if expectation_result.success else 0.0,
                    status=status,
                    description=f"Expectation: {check_name}"
                ))
                
                # 添加观察值（如果存在）
                if "observed_value" in expectation_result.result:
                    metrics.append(QualityMetric(
                        name="observed_value",
                        value=float(expectation_result.result["observed_value"]),
                        description="Observed value from expectation"
                    ))
                
                # 创建检查结果
                check_result = QualityCheckResult(
                    check_id=check_id,
                    check_name=check_name,
                    check_type=QualityCheckType.VALIDITY,
                    status=status,
                    message=expectation_result.expectation_config.kwargs.get("meta", {}).get("notes", ""),
                    metrics=metrics,
                    start_time=datetime.now() - timedelta(seconds=0.1),  # 模拟
                    end_time=datetime.now(),
                    duration_seconds=0.1,
                    dataset_name=dataset_name,
                    metadata={
                        "expectation_config": expectation_result.expectation_config.to_json_dict(),
                        "result": expectation_result.result
                    }
                )
                
                checks.append(check_result)
            
            logger.info(f"Great Expectations验证完成: {dataset_name}, 检查数: {len(checks)}")
            
        except Exception as e:
            logger.error(f"Great Expectations验证失败: {e}")
            
            # 添加错误检查
            error_check = QualityCheckResult(
                check_id=f"ge_error_{dataset_name}",
                check_name="GE Validation Error",
                check_type=QualityCheckType.CUSTOM,
                status="error",
                message=f"Great Expectations验证失败: {str(e)}",
                metrics=[],
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0,
                dataset_name=dataset_name
            )
            checks.append(error_check)
        
        return checks
    
    async def _check_quality_rules(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        profile: DataProfile
    ) -> List[QualityCheckResult]:
        """检查质量规则"""
        checks = []
        
        for rule_id, rule in self.quality_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # 根据规则类型执行检查
                if rule.rule_type == QualityCheckType.COMPLETENESS:
                    check_result = await self._check_completeness_rule(
                        dataset_name, data, profile, rule
                    )
                elif rule.rule_type == QualityCheckType.UNIQUENESS:
                    check_result = await self._check_uniqueness_rule(
                        dataset_name, data, profile, rule
                    )
                elif rule.rule_type == QualityCheckType.VALIDITY:
                    check_result = await self._check_validity_rule(
                        dataset_name, data, profile, rule
                    )
                elif rule.rule_type == QualityCheckType.CONSISTENCY:
                    check_result = await self._check_consistency_rule(
                        dataset_name, data, profile, rule
                    )
                elif rule.rule_type == QualityCheckType.DISTRIBUTION:
                    check_result = await self._check_distribution_rule(
                        dataset_name, data, profile, rule
                    )
                else:
                    # 自定义规则
                    check_result = await self._check_custom_rule(
                        dataset_name, data, profile, rule
                    )
                
                if check_result:
                    checks.append(check_result)
                    
            except Exception as e:
                logger.error(f"规则检查失败 {rule_id}: {e}")
        
        return checks
    
    async def _check_completeness_rule(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        profile: DataProfile,
        rule: QualityRule
    ) -> Optional[QualityCheckResult]:
        """检查完整性规则"""
        try:
            # 计算缺失值比例
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
            
            # 提取阈值
            threshold = rule.condition.get("threshold", 0.05)
            operator = rule.condition.get("operator", "<=")
            
            # 检查条件
            is_passed = self._evaluate_condition(missing_ratio, threshold, operator)
            
            # 创建指标
            metrics = [
                QualityMetric(
                    name="missing_ratio",
                    value=missing_ratio,
                    threshold_max=threshold,
                    status="passed" if is_passed else "failed",
                    description="缺失值比例",
                    check_type=QualityCheckType.COMPLETENESS
                ),
                QualityMetric(
                    name="missing_count",
                    value=float(missing_cells),
                    description="缺失值数量"
                )
            ]
            
            # 创建检查结果
            return QualityCheckResult(
                check_id=f"completeness_{rule.rule_id}",
                check_name=rule.name,
                check_type=QualityCheckType.COMPLETENESS,
                status="passed" if is_passed else "failed",
                message=f"缺失值比例: {missing_ratio:.4f}, 阈值: {threshold}",
                metrics=metrics,
                start_time=datetime.now() - timedelta(seconds=0.5),
                end_time=datetime.now(),
                duration_seconds=0.5,
                dataset_name=dataset_name,
                metadata={
                    "rule_id": rule.rule_id,
                    "severity": rule.severity
                }
            )
            
        except Exception as e:
            logger.error(f"完整性检查失败: {e}")
            return None
    
    async def _check_uniqueness_rule(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        profile: DataProfile,
        rule: QualityRule
    ) -> Optional[QualityCheckResult]:
        """检查唯一性规则"""
        try:
            # 计算重复行比例
            total_rows = len(data)
            duplicate_rows = data.duplicated().sum()
            duplicate_ratio = duplicate_rows / total_rows if total_rows > 0 else 0
            
            # 提取阈值
            threshold = rule.condition.get("threshold", 0.001)
            operator = rule.condition.get("operator", "<=")
            
            # 检查条件
            is_passed = self._evaluate_condition(duplicate_ratio, threshold, operator)
            
            # 创建指标
            metrics = [
                QualityMetric(
                    name="duplicate_ratio",
                    value=duplicate_ratio,
                    threshold_max=threshold,
                    status="passed" if is_passed else "failed",
                    description="重复行比例",
                    check_type=QualityCheckType.UNIQUENESS
                ),
                QualityMetric(
                    name="duplicate_count",
                    value=float(duplicate_rows),
                    description="重复行数量"
                )
            ]
            
            # 创建检查结果
            return QualityCheckResult(
                check_id=f"uniqueness_{rule.rule_id}",
                check_name=rule.name,
                check_type=QualityCheckType.UNIQUENESS,
                status="passed" if is_passed else "failed",
                message=f"重复行比例: {duplicate_ratio:.4f}, 阈值: {threshold}",
                metrics=metrics,
                start_time=datetime.now() - timedelta(seconds=0.5),
                end_time=datetime.now(),
                duration_seconds=0.5,
                dataset_name=dataset_name,
                metadata={
                    "rule_id": rule.rule_id,
                    "severity": rule.severity
                }
            )
            
        except Exception as e:
            logger.error(f"唯一性检查失败: {e}")
            return None
    
    async def _check_validity_rule(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        profile: DataProfile,
        rule: QualityRule
    ) -> Optional[QualityCheckResult]:
        """检查有效性规则"""
        # 这里实现具体的有效性检查逻辑
        # 简化实现
        return None
    
    async def _check_consistency_rule(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        profile: DataProfile,
        rule: QualityRule
    ) -> Optional[QualityCheckResult]:
        """检查一致性规则"""
        # 这里实现具体的一致性检查逻辑
        # 简化实现
        return None
    
    async def _check_distribution_rule(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        profile: DataProfile,
        rule: QualityRule
    ) -> Optional[QualityCheckResult]:
        """检查分布规则"""
        # 这里实现具体的分布检查逻辑
        # 简化实现
        return None
    
    async def _check_custom_rule(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        profile: DataProfile,
        rule: QualityRule
    ) -> Optional[QualityCheckResult]:
        """检查自定义规则"""
        # 这里实现自定义规则检查逻辑
        # 简化实现
        return None
    
    def _evaluate_condition(
        self,
        value: float,
        threshold: float,
        operator: str
    ) -> bool:
        """评估条件"""
        if operator == "<=":
            return value <= threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == ">":
            return value > threshold
        elif operator == "==":
            return abs(value - threshold) < 1e-10
        elif operator == "!=":
            return abs(value - threshold) >= 1e-10
        else:
            return False
    
    async def _perform_custom_checks(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        profile: DataProfile
    ) -> List[QualityCheckResult]:
        """执行自定义检查"""
        checks = []
        
        try:
            # 1. 检查列数变化
            if len(data.columns) < profile.column_count * 0.8:
                check = QualityCheckResult(
                    check_id=f"custom_column_count_{dataset_name}",
                    check_name="列数检查",
                    check_type=QualityCheckType.CONSISTENCY,
                    status="warning",
                    message=f"列数变化较大: 当前{len(data.columns)}列, 基准{profile.column_count}列",
                    metrics=[],
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0,
                    dataset_name=dataset_name
                )
                checks.append(check)
            
            # 2. 检查行数异常
            expected_size = profile.row_count
            current_size = len(data)
            size_ratio = current_size / expected_size if expected_size > 0 else 1
            
            if size_ratio < 0.5 or size_ratio > 2.0:
                check = QualityCheckResult(
                    check_id=f"custom_row_count_{dataset_name}",
                    check_name="行数检查",
                    check_type=QualityCheckType.CONSISTENCY,
                    status="warning",
                    message=f"行数异常: 当前{current_size}行, 基准{expected_size}行, 比例{size_ratio:.2f}",
                    metrics=[
                        QualityMetric(
                            name="size_ratio",
                            value=size_ratio,
                            threshold_min=0.5,
                            threshold_max=2.0,
                            status="warning" if size_ratio < 0.5 or size_ratio > 2.0 else "passed",
                            description="行数比例"
                        )
                    ],
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0,
                    dataset_name=dataset_name
                )
                checks.append(check)
            
            # 3. 检查数据类型一致性
            for col in data.columns:
                if col in profile.column_stats:
                    expected_dtype = profile.column_stats[col].get("dtype", "")
                    current_dtype = str(data[col].dtype)
                    
                    if expected_dtype != current_dtype:
                        check = QualityCheckResult(
                            check_id=f"custom_dtype_{dataset_name}_{col}",
                            check_name="数据类型检查",
                            check_type=QualityCheckType.CONSISTENCY,
                            status="error",
                            message=f"列'{col}'数据类型变化: 预期{expected_dtype}, 实际{current_dtype}",
                            metrics=[],
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            duration_seconds=0,
                            dataset_name=dataset_name
                        )
                        checks.append(check)
        
        except Exception as e:
            logger.error(f"自定义检查失败: {e}")
        
        return checks
    
    def _calculate_overall_status(
        self,
        checks: List[QualityCheckResult]
    ) -> DataQualityStatus:
        """计算总体状态"""
        if not checks:
            return DataQualityStatus.EXCELLENT
        
        # 检查是否有错误
        has_critical = any(
            c.status in ["error", "failed"] 
            for c in checks
        )
        
        has_warning = any(
            c.status == "warning" 
            for c in checks
        )
        
        passed_count = sum(
            1 for c in checks 
            if c.status in ["passed", "excellent", "good"]
        )
        pass_rate = passed_count / len(checks) if checks else 1.0
        
        if has_critical:
            return DataQualityStatus.CRITICAL
        elif has_warning:
            return DataQualityStatus.WARNING
        elif pass_rate >= 0.95:
            return DataQualityStatus.EXCELLENT
        elif pass_rate >= 0.85:
            return DataQualityStatus.GOOD
        else:
            return DataQualityStatus.ERROR
    
    def _summarize_metrics(
        self,
        checks: List[QualityCheckResult],
        profile: DataProfile
    ) -> Dict[str, Any]:
        """汇总指标"""
        all_metrics = []
        for check in checks:
            all_metrics.extend(check.metrics)
        
        # 计算基本统计
        passed_metrics = [m for m in all_metrics if m.is_passed()]
        warning_metrics = [m for m in all_metrics if m.is_warning()]
        
        metric_summary = {
            "total_metrics": len(all_metrics),
            "passed_metrics": len(passed_metrics),
            "warning_metrics": len(warning_metrics),
            "failed_metrics": len(all_metrics) - len(passed_metrics) - len(warning_metrics),
            "pass_rate": len(passed_metrics) / len(all_metrics) if all_metrics else 1.0,
            "dataset_stats": {
                "row_count": profile.row_count,
                "column_count": profile.column_count,
                "missing_ratio": profile.missing_values / (profile.row_count * profile.column_count) 
                               if profile.row_count * profile.column_count > 0 else 0,
                "duplicate_ratio": profile.duplicate_rows / profile.row_count if profile.row_count > 0 else 0
            }
        }
        
        return metric_summary
    
    def _generate_recommendations(
        self,
        result: DataQualityResult,
        profile: DataProfile
    ) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于总体状态
        if result.overall_status == DataQualityStatus.CRITICAL:
            recommendations.append("数据质量严重问题，建议立即调查和修复")
        elif result.overall_status == DataQualityStatus.ERROR:
            recommendations.append("数据质量存在问题，建议优先处理")
        elif result.overall_status == DataQualityStatus.WARNING:
            recommendations.append("数据质量有警告，建议关注")
        
        # 基于具体指标
        if result.metrics_summary["dataset_stats"]["missing_ratio"] > 0.1:
            recommendations.append("缺失值比例较高，建议检查数据源或进行填充")
        
        if result.metrics_summary["dataset_stats"]["duplicate_ratio"] > 0.05:
            recommendations.append("重复数据较多，建议去重处理")
        
        # 基于检查结果
        failed_checks = [c for c in result.checks if c.status in ["failed", "error"]]
        for check in failed_checks[:3]:  # 前3个失败的检查
            recommendations.append(f"检查失败: {check.check_name} - {check.message}")
        
        return recommendations
    
    async def _check_alerts(
        self,
        result: DataQualityResult
    ) -> List[str]:
        """检查告警"""
        alerts = []
        
        # 检查总体状态
        if result.overall_status in [DataQualityStatus.CRITICAL, DataQualityStatus.ERROR]:
            alerts.append(f"数据质量{result.overall_status.value}: {result.dataset_name}")
        
        # 检查特定指标
        if result.metrics_summary["pass_rate"] < 0.8:
            alerts.append(f"检查通过率低: {result.metrics_summary['pass_rate']:.2%}")
        
        # 检查失败的重要规则
        for rule_id, rule in self.quality_rules.items():
            if rule.severity in ["high", "critical"] and rule.enabled:
                # 查找对应的检查结果
                related_checks = [
                    c for c in result.checks 
                    if c.metadata.get("rule_id") == rule_id and c.status in ["failed", "error"]
                ]
                if related_checks:
                    alerts.append(f"重要规则失败: {rule.name}")
        
        return alerts
    
    async def _send_alerts(
        self,
        dataset_name: str,
        alerts: List[str],
        result: DataQualityResult
    ):
        """发送告警"""
        alert_data = {
            "alert_type": "data_quality",
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "overall_status": result.overall_status.value,
            "alerts": alerts,
            "result_id": result.result_id,
            "pass_rate": result.metrics_summary.get("pass_rate", 0)
        }
        
        # 保存告警
        await self._save_alert(alert_data)
        
        # 发送到webhook
        if self.config["alerts"].get("webhook_url"):
            await self._send_webhook_alert(alert_data)
        
        # 发送到Slack
        if self.config["alerts"].get("slack_webhook"):
            await self._send_slack_alert(alert_data)
        
        logger.warning(f"数据质量告警: {dataset_name}, 告警数: {len(alerts)}")
    
    async def _log_to_whylabs(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        result: DataQualityResult
    ):
        """记录到whylogs"""
        if not self.whylabs_writer:
            return
        
        try:
            # 创建包含质量指标的标签
            tags = {
                "dataset": dataset_name,
                "quality_status": result.overall_status.value,
                "pass_rate": str(result.metrics_summary.get("pass_rate", 0))
            }
            
            # 记录数据
            profile = why.log(data, tags=tags).profile()
            
            # 写入到WhyLabs
            self.whylabs_writer.write(
                profile=profile,
                dataset_id=dataset_name
            )
            
        except Exception as e:
            logger.error(f"记录到WhyLabs失败: {e}")
    
    async def _save_result(self, result: DataQualityResult):
        """保存结果"""
        result_dir = self.data_dir / "results"
        result_dir.mkdir(exist_ok=True)
        
        file_path = result_dir / f"{result.result_id}.json"
        
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(asdict(result), indent=2, default=str))
    
    async def _save_profile(self, profile: DataProfile):
        """保存画像"""
        profile_dir = self.data_dir / "profiles"
        profile_dir.mkdir(exist_ok=True)
        
        file_path = profile_dir / f"{profile.profile_id}.json"
        
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(asdict(profile), indent=2, default=str))
    
    async def _save_alert(self, alert_data: Dict[str, Any]):
        """保存告警"""
        alert_dir = self.data_dir / "alerts"
        alert_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = alert_dir / f"alert_{timestamp}.json"
        
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(alert_data, indent=2, default=str))
    
    async def _send_webhook_alert(self, alert_data: Dict[str, Any]):
        """发送Webhook告警"""
        try:
            webhook_url = self.config["alerts"]["webhook_url"]
            if not webhook_url:
                return
            
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
    
    async def _send_slack_alert(self, alert_data: Dict[str, Any]):
        """发送Slack告警"""
        try:
            slack_webhook = self.config["alerts"]["slack_webhook"]
            if not slack_webhook:
                return
            
            # 格式化Slack消息
            message = {
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"🚨 数据质量告警: {alert_data['dataset_name']}"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*状态:* {alert_data['overall_status']}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*通过率:* {alert_data['pass_rate']:.2%}"
                            }
                        ]
                    }
                ]
            }
            
            # 添加告警详情
            for alert in alert_data["alerts"][:5]:  # 最多5个告警
                message["blocks"].append({
                    "type": "section",
                    "text": {
                        "type": "plain_text",
                        "text": f"• {alert}"
                    }
                })
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    slack_webhook,
                    json=message,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Slack告警发送失败: {response.status}")
                        
        except Exception as e:
            logger.error(f"发送Slack告警失败: {e}")
    
    async def get_quality_history(
        self,
        dataset_name: str,
        lookback_days: int = 7
    ) -> List[DataQualityResult]:
        """获取质量历史"""
        results = []
        result_dir = self.data_dir / "results"
        
        if not result_dir.exists():
            return results
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        for result_file in result_dir.glob(f"dq_{dataset_name}_*.json"):
            try:
                async with aiofiles.open(result_file, 'r') as f:
                    content = await f.read()
                    result_data = json.loads(content)
                    
                    # 转换时间戳
                    if isinstance(result_data["timestamp"], str):
                        result_data["timestamp"] = datetime.fromisoformat(
                            result_data["timestamp"].replace('Z', '+00:00')
                        )
                    
                    # 检查时间
                    if result_data["timestamp"] >= cutoff_date:
                        result = DataQualityResult(**result_data)
                        results.append(result)
                        
            except Exception as e:
                logger.error(f"加载结果文件失败 {result_file}: {e}")
        
        # 按时间排序
        results.sort(key=lambda x: x.timestamp)
        
        return results
    
    async def get_quality_summary(
        self,
        dataset_name: str,
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """获取质量摘要"""
        history = await self.get_quality_history(dataset_name, lookback_days)
        
        if not history:
            return {
                "dataset_name": dataset_name,
                "has_history": False,
                "message": f"No quality history found for {dataset_name} in last {lookback_days} days"
            }
        
        # 计算统计
        total_runs = len(history)
        success_runs = sum(1 for r in history 
                          if r.overall_status in [DataQualityStatus.EXCELLENT, DataQualityStatus.GOOD])
        warning_runs = sum(1 for r in history 
                          if r.overall_status == DataQualityStatus.WARNING)
        error_runs = sum(1 for r in history 
                        if r.overall_status in [DataQualityStatus.ERROR, DataQualityStatus.CRITICAL])
        
        # 计算平均通过率
        avg_pass_rate = np.mean([r.metrics_summary.get("pass_rate", 0) for r in history])
        
        # 最近结果
        recent_result = history[-1] if history else None
        
        return {
            "dataset_name": dataset_name,
            "has_history": True,
            "period": f"last_{lookback_days}_days",
            "total_runs": total_runs,
            "success_runs": success_runs,
            "warning_runs": warning_runs,
            "error_runs": error_runs,
            "success_rate": success_runs / total_runs if total_runs > 0 else 0,
            "avg_pass_rate": avg_pass_rate,
            "recent_status": recent_result.overall_status.value if recent_result else "unknown",
            "recent_pass_rate": recent_result.metrics_summary.get("pass_rate", 0) if recent_result else 0,
            "last_check": recent_result.timestamp.isoformat() if recent_result else None
        }
    
    async def add_quality_rule(
        self,
        name: str,
        description: str,
        rule_type: QualityCheckType,
        condition: Dict[str, Any],
        severity: str = "medium",
        tags: Optional[List[str]] = None
    ) -> str:
        """添加质量规则"""
        rule_id = f"rule_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        rule = QualityRule(
            rule_id=rule_id,
            name=name,
            description=description,
            rule_type=rule_type,
            condition=condition,
            severity=severity,
            enabled=True,
            tags=tags or []
        )
        
        self.quality_rules[rule_id] = rule
        self._save_quality_rule(rule)
        
        logger.info(f"质量规则添加成功: {rule_id} - {name}")
        return rule_id
    
    async def update_quality_rule(
        self,
        rule_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """更新质量规则"""
        if rule_id not in self.quality_rules:
            return False
        
        rule = self.quality_rules[rule_id]
        
        # 更新字段
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        rule.updated_at = datetime.now()
        
        # 保存
        self._save_quality_rule(rule)
        
        logger.info(f"质量规则更新成功: {rule_id}")
        return True
    
    async def delete_quality_rule(self, rule_id: str) -> bool:
        """删除质量规则"""
        if rule_id not in self.quality_rules:
            return False
        
        # 从内存中删除
        del self.quality_rules[rule_id]
        
        # 删除文件
        rule_file = self.data_dir / "rules" / f"{rule_id}.json"
        if rule_file.exists():
            rule_file.unlink()
        
        logger.info(f"质量规则删除成功: {rule_id}")
        return True
    
    async def get_data_profile(
        self,
        dataset_name: str
    ) -> Optional[DataProfile]:
        """获取数据画像"""
        # 首先检查缓存
        for profile in self.profiles_cache.values():
            if profile.dataset_name == dataset_name:
                return profile
        
        # 从文件加载
        profile_dir = self.data_dir / "profiles"
        for profile_file in profile_dir.glob(f"profile_{dataset_name}_*.json"):
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                    profile = DataProfile(**profile_data)
                    self.profiles_cache[profile.profile_id] = profile
                    return profile
            except Exception as e:
                logger.error(f"加载画像文件失败 {profile_file}: {e}")
        
        return None
    
    async def monitor_data_pipeline(
        self,
        pipeline_name: str,
        datasets: Dict[str, pd.DataFrame],
        pipeline_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        监控数据流水线
        
        Args:
            pipeline_name: 流水线名称
            datasets: 数据集字典
            pipeline_config: 流水线配置
            
        Returns:
            监控结果
        """
        results = {}
        
        for dataset_name, data in datasets.items():
            # 获取数据集配置
            dataset_config = pipeline_config.get("datasets", {}).get(dataset_name, {})
            
            # 执行监控
            result = await self.monitor_dataset(
                dataset_name=dataset_name,
                data=data,
                expectation_suite_name=dataset_config.get("expectation_suite"),
                batch_identifier=f"{pipeline_name}_{dataset_name}",
                metadata={"pipeline": pipeline_name}
            )
            
            results[dataset_name] = result
        
        # 计算流水线总体状态
        pipeline_status = self._calculate_pipeline_status(results)
        
        # 创建流水线结果
        pipeline_result = DataQualityResult(
            result_id=f"pipeline_{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataset_name=pipeline_name,
            timestamp=datetime.now(),
            overall_status=pipeline_status,
            total_checks=0,
            passed_checks=0,
            warning_checks=0,
            failed_checks=0,
            error_checks=0,
            checks=[],
            metrics_summary={
                "total_datasets": len(results),
                "datasets_status": {name: result.overall_status.value for name, result in results.items()},
                "overall_status": pipeline_status.value
            }
        )
        
        await self._save_result(pipeline_result)
        
        return {
            "pipeline": pipeline_result,
            "datasets": results
        }
    
    def _calculate_pipeline_status(
        self,
        results: Dict[str, DataQualityResult]
    ) -> DataQualityStatus:
        """计算流水线状态"""
        if not results:
            return DataQualityStatus.EXCELLENT
        
        # 检查是否有严重错误
        has_critical = any(
            r.overall_status in [DataQualityStatus.CRITICAL, DataQualityStatus.ERROR]
            for r in results.values()
        )
        
        has_warning = any(
            r.overall_status == DataQualityStatus.WARNING
            for r in results.values()
        )
        
        if has_critical:
            return DataQualityStatus.CRITICAL
        elif has_warning:
            return DataQualityStatus.WARNING
        
        # 计算平均通过率
        avg_pass_rate = np.mean([
            r.metrics_summary.get("pass_rate", 0) 
            for r in results.values()
        ])
        
        if avg_pass_rate >= 0.95:
            return DataQualityStatus.EXCELLENT
        elif avg_pass_rate >= 0.85:
            return DataQualityStatus.GOOD
        else:
            return DataQualityStatus.ERROR
    
    async def track_quality_trend(
        self,
        dataset_name: str,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """跟踪质量趋势"""
        history = await self.get_quality_history(dataset_name, lookback_days)
        
        if not history or len(history) < 2:
            return {
                "dataset_name": dataset_name,
                "has_trend": False,
                "message": "Not enough data for trend analysis"
            }
        
        # 提取趋势数据
        timestamps = [r.timestamp for r in history]
        pass_rates = [r.metrics_summary.get("pass_rate", 0) for r in history]
        statuses = [r.overall_status.value for r in history]
        
        # 计算趋势
        if len(pass_rates) >= 2:
            from scipy import stats
            x = np.arange(len(pass_rates))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, pass_rates)
            
            trend = "improving" if slope > 0.001 else "declining" if slope < -0.001 else "stable"
        else:
            slope = 0
            trend = "stable"
        
        # 检测异常点
        anomalies = self._detect_anomalies(pass_rates)
        
        return {
            "dataset_name": dataset_name,
            "has_trend": True,
            "period": f"last_{lookback_days}_days",
            "data_points": len(history),
            "trend": trend,
            "trend_slope": slope,
            "current_pass_rate": pass_rates[-1] if pass_rates else 0,
            "avg_pass_rate": np.mean(pass_rates) if pass_rates else 0,
            "std_pass_rate": np.std(pass_rates) if len(pass_rates) > 1 else 0,
            "anomalies": anomalies,
            "timeline": [
                {
                    "timestamp": ts.isoformat(),
                    "pass_rate": rate,
                    "status": status
                }
                for ts, rate, status in zip(timestamps, pass_rates, statuses)
            ]
        }
    
    def _detect_anomalies(self, values: List[float]) -> List[int]:
        """检测异常点"""
        if len(values) < 3:
            return []
        
        # 使用IQR方法检测异常
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        anomalies = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                anomalies.append(i)
        
        return anomalies
    
    async def generate_quality_report(
        self,
        dataset_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """生成质量报告"""
        # 获取历史数据
        all_history = await self.get_quality_history(dataset_name, 365)  # 获取一年数据
        period_history = [
            r for r in all_history
            if start_date <= r.timestamp <= end_date
        ]
        
        if not period_history:
            return {
                "dataset_name": dataset_name,
                "has_data": False,
                "period": f"{start_date.date()} to {end_date.date()}",
                "message": "No data available for the specified period"
            }
        
        # 计算统计
        pass_rates = [r.metrics_summary.get("pass_rate", 0) for r in period_history]
        status_counts = defaultdict(int)
        for r in period_history:
            status_counts[r.overall_status.value] += 1
        
        # 计算最常见的问题
        all_checks = []
        for r in period_history:
            all_checks.extend(r.checks)
        
        failed_checks = [c for c in all_checks if c.status in ["failed", "error"]]
        
        # 按检查类型分组
        failed_by_type = defaultdict(int)
        for check in failed_checks:
            failed_by_type[check.check_type.value] += 1
        
        # 按严重程度分组
        failed_by_severity = defaultdict(int)
        for check in failed_checks:
            severity = check.metadata.get("severity", "unknown")
            failed_by_severity[severity] += 1
        
        return {
            "dataset_name": dataset_name,
            "has_data": True,
            "period": f"{start_date.date()} to {end_date.date()}",
            "total_checks": len(period_history),
            "avg_pass_rate": np.mean(pass_rates) if pass_rates else 0,
            "min_pass_rate": np.min(pass_rates) if pass_rates else 0,
            "max_pass_rate": np.max(pass_rates) if pass_rates else 0,
            "status_distribution": dict(status_counts),
            "failed_check_analysis": {
                "total_failed": len(failed_checks),
                "by_type": dict(failed_by_type),
                "by_severity": dict(failed_by_severity),
                "top_failed_checks": [
                    {
                        "check_name": check.check_name,
                        "check_type": check.check_type.value,
                        "message": check.message,
                        "severity": check.metadata.get("severity", "unknown")
                    }
                    for check in failed_checks[:10]  # 前10个失败的检查
                ]
            },
            "recommendations": self._generate_report_recommendations(
                period_history, failed_checks, failed_by_type, failed_by_severity
            )
        }
    
    def _generate_report_recommendations(
        self,
        history: List[DataQualityResult],
        failed_checks: List[QualityCheckResult],
        failed_by_type: Dict[str, int],
        failed_by_severity: Dict[str, int]
    ) -> List[str]:
        """生成报告建议"""
        recommendations = []
        
        # 总体状态建议
        avg_pass_rate = np.mean([r.metrics_summary.get("pass_rate", 0) for r in history])
        if avg_pass_rate < 0.8:
            recommendations.append(f"平均通过率较低 ({avg_pass_rate:.2%})，建议优先改善数据质量")
        
        # 按检查类型建议
        if failed_by_type.get("completeness", 0) > len(failed_checks) * 0.3:
            recommendations.append("完整性检查失败较多，建议检查数据源和ETL流程")
        
        if failed_by_type.get("validity", 0) > len(failed_checks) * 0.3:
            recommendations.append("有效性检查失败较多，建议验证业务规则和数据验证逻辑")
        
        # 按严重程度建议
        if failed_by_severity.get("critical", 0) > 0:
            recommendations.append("存在严重级别检查失败，建议立即处理")
        
        if failed_by_severity.get("high", 0) > 5:
            recommendations.append("高级别检查失败较多，建议优先处理")
        
        return recommendations
    
    async def cleanup_old_data(self, retention_days: int = 30):
        """清理旧数据"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # 清理结果
            result_dir = self.data_dir / "results"
            if result_dir.exists():
                for result_file in result_dir.glob("*.json"):
                    file_time = datetime.fromtimestamp(result_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        result_file.unlink()
            
            # 清理告警
            alert_dir = self.data_dir / "alerts"
            if alert_dir.exists():
                for alert_file in alert_dir.glob("*.json"):
                    file_time = datetime.fromtimestamp(alert_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        alert_file.unlink()
            
            logger.info(f"清理完成，保留 {retention_days} 天数据")
            
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
