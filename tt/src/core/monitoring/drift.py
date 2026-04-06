"""
数据漂移检测模块
基于Evidently、KS检验、PSI、MMD等方法检测数据漂移和模型漂移
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import aiofiles
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# 导入监控工具
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import (
        DataDriftPreset, 
        DataQualityPreset,
        TargetDriftPreset,
        RegressionPreset,
        ClassificationPreset
    )
    from evidently.metrics import *
    from evidently.test_suite import TestSuite
    from evidently.tests import *
except ImportError:
    pass

try:
    import arize
    from arize.api import Client
    from arize.utils.types import ModelTypes
except ImportError:
    pass

from ..config import settings

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """漂移类型"""
    DATA_DRIFT = "data_drift"          # 数据漂移
    CONCEPT_DRIFT = "concept_drift"    # 概念漂移
    MODEL_DRIFT = "model_drift"       # 模型漂移
    LABEL_DRIFT = "label_drift"       # 标签漂移
    DISTRIBUTION_DRIFT = "distribution_drift"  # 分布漂移


class DetectionMethod(Enum):
    """检测方法"""
    KS_TEST = "ks_test"          # Kolmogorov-Smirnov检验
    PSI = "psi"                  # 群体稳定性指数
    MMD = "mmd"                  # 最大均值差异
    CLASSIFIATOR = "classifier"  # 分类器方法
    CHI_SQUARE = "chi_square"    # 卡方检验
    EVIDENTLY = "evidently"      # Evidently库
    ARIZE = "arize"              # Arize Phoenix


@dataclass
class DriftResult:
    """漂移检测结果"""
    drift_id: str
    model_name: str
    model_version: str
    drift_type: DriftType
    detection_method: DetectionMethod
    timestamp: datetime
    drift_score: float
    threshold: float
    is_drifted: bool
    p_value: Optional[float] = None
    confidence: float = 0.95
    metrics: Dict[str, Any] = None
    features_drifted: List[str] = None
    severity: str = "none"  # none, low, medium, high, critical
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.features_drifted is None:
            self.features_drifted = []
        if self.recommendations is None:
            self.recommendations = []
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class ReferenceData:
    """参考数据"""
    data: Union[pd.DataFrame, np.ndarray]
    timestamp: datetime
    data_version: str
    metadata: Dict[str, Any]
    features: List[str]
    sample_size: int
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


class DriftDetector:
    """漂移检测器"""
    
    def __init__(self):
        self.reference_data: Dict[str, ReferenceData] = {}  # model_name -> ReferenceData
        self.drift_history: Dict[str, List[DriftResult]] = {}  # model_name -> List[DriftResult]
        self.detection_methods = {
            DetectionMethod.KS_TEST: self._ks_test_detector,
            DetectionMethod.PSI: self._psi_detector,
            DetectionMethod.MMD: self._mmd_detector,
            DetectionMethod.CLASSIFIATOR: self._classifier_detector,
            DetectionMethod.CHI_SQUARE: self._chi_square_detector,
            DetectionMethod.EVIDENTLY: self._evidently_detector
        }
        
        # 初始化Arize Phoenix客户端
        self.arize_client = None
        if hasattr(settings.monitoring, 'arize_api_key') and settings.monitoring.arize_api_key:
            try:
                self.arize_client = Client(
                    api_key=settings.monitoring.arize_api_key,
                    space_key=settings.monitoring.arize_space_key
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Arize client: {e}")
        
        # 存储目录
        self.data_dir = Path("drift_detection")
        self.data_dir.mkdir(exist_ok=True)
        
        # 加载历史数据
        self._load_history()
    
    async def set_reference_data(
        self,
        model_name: str,
        data: Union[pd.DataFrame, np.ndarray],
        data_version: str = "v1.0",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        设置参考数据
        
        Args:
            model_name: 模型名称
            data: 参考数据
            data_version: 数据版本
            metadata: 元数据
        """
        if isinstance(data, np.ndarray):
            # 转换为DataFrame
            if data.ndim == 1:
                data = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[0])])
            else:
                data = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[1])])
        
        features = list(data.columns)
        sample_size = len(data)
        
        reference = ReferenceData(
            data=data,
            timestamp=datetime.now(),
            data_version=data_version,
            metadata=metadata or {},
            features=features,
            sample_size=sample_size
        )
        
        self.reference_data[model_name] = reference
        
        # 保存参考数据
        await self._save_reference_data(model_name, reference)
        
        logger.info(f"设置参考数据完成: {model_name}, 版本: {data_version}, 样本数: {sample_size}")
    
    async def detect_drift(
        self,
        model_name: str,
        current_data: Union[pd.DataFrame, np.ndarray],
        drift_type: DriftType = DriftType.DATA_DRIFT,
        detection_method: DetectionMethod = DetectionMethod.EVIDENTLY,
        features: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> DriftResult:
        """
        检测漂移
        
        Args:
            model_name: 模型名称
            current_data: 当前数据
            drift_type: 漂移类型
            detection_method: 检测方法
            features: 要检测的特征列表
            threshold: 漂移阈值
            
        Returns:
            漂移检测结果
        """
        # 检查参考数据
        if model_name not in self.reference_data:
            raise ValueError(f"未找到模型 {model_name} 的参考数据")
        
        reference = self.reference_data[model_name]
        
        # 准备数据
        if isinstance(current_data, np.ndarray):
            if current_data.ndim == 1:
                current_df = pd.DataFrame(
                    current_data, 
                    columns=[f"feature_{i}" for i in range(current_data.shape[0])]
                )
            else:
                current_df = pd.DataFrame(
                    current_data, 
                    columns=[f"feature_{i}" for i in range(current_data.shape[1])]
                )
        else:
            current_df = current_data.copy()
        
        # 特征对齐
        if features:
            ref_features = [f for f in reference.features if f in features]
            cur_features = [f for f in current_df.columns if f in features]
            common_features = list(set(ref_features) & set(cur_features))
        else:
            common_features = list(set(reference.features) & set(current_df.columns))
        
        if not common_features:
            raise ValueError("参考数据和当前数据没有共同特征")
        
        # 提取共同特征的数据
        ref_data = reference.data[common_features]
        cur_data = current_df[common_features]
        
        # 设置阈值
        if threshold is None:
            threshold = self._get_default_threshold(detection_method)
        
        # 执行检测
        detector = self.detection_methods.get(detection_method)
        if not detector:
            raise ValueError(f"不支持的检测方法: {detection_method}")
        
        try:
            result = await detector(
                reference_data=ref_data,
                current_data=cur_data,
                drift_type=drift_type,
                threshold=threshold,
                model_name=model_name
            )
            
            # 创建漂移结果
            drift_id = f"drift_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            drift_result = DriftResult(
                drift_id=drift_id,
                model_name=model_name,
                model_version=reference.data_version,
                drift_type=drift_type,
                detection_method=detection_method,
                timestamp=datetime.now(),
                drift_score=result["score"],
                threshold=threshold,
                is_drifted=result["drifted"],
                p_value=result.get("p_value"),
                metrics=result.get("metrics", {}),
                features_drifted=result.get("features_drifted", []),
                severity=self._calculate_severity(result["score"], threshold)
            )
            
            # 添加建议
            drift_result.recommendations = self._generate_recommendations(drift_result)
            
            # 记录历史
            if model_name not in self.drift_history:
                self.drift_history[model_name] = []
            self.drift_history[model_name].append(drift_result)
            
            # 保存结果
            await self._save_drift_result(drift_result)
            
            # 触发告警
            if drift_result.is_drifted:
                await self._trigger_drift_alert(drift_result)
            
            return drift_result
            
        except Exception as e:
            logger.error(f"漂移检测失败: {e}")
            raise
    
    async def _ks_test_detector(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        drift_type: DriftType,
        threshold: float,
        **kwargs
    ) -> Dict[str, Any]:
        """KS检验漂移检测器"""
        ks_scores = []
        p_values = []
        features_drifted = []
        
        for feature in reference_data.columns:
            ref_feature = reference_data[feature].dropna()
            cur_feature = current_data[feature].dropna()
            
            # 执行KS检验
            ks_stat, p_value = stats.ks_2samp(ref_feature, cur_feature)
            ks_scores.append(ks_stat)
            p_values.append(p_value)
            
            if p_value < threshold:
                features_drifted.append(feature)
        
        # 计算平均KS统计量
        avg_ks_score = np.mean(ks_scores)
        min_p_value = np.min(p_values) if p_values else 1.0
        
        drifted = min_p_value < threshold
        
        return {
            "score": avg_ks_score,
            "p_value": min_p_value,
            "drifted": drifted,
            "features_drifted": features_drifted,
            "metrics": {
                "ks_scores": ks_scores,
                "p_values": p_values,
                "num_features": len(reference_data.columns),
                "num_drifted_features": len(features_drifted)
            }
        }
    
    async def _psi_detector(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        drift_type: DriftType,
        threshold: float,
        n_bins: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """PSI（群体稳定性指数）漂移检测器"""
        psi_scores = []
        features_drifted = []
        
        for feature in reference_data.columns:
            ref_feature = reference_data[feature].dropna()
            cur_feature = current_data[feature].dropna()
            
            # 计算分箱边界
            min_val = min(ref_feature.min(), cur_feature.min())
            max_val = max(ref_feature.max(), cur_feature.max())
            
            if min_val == max_val:
                # 所有值相同
                psi_scores.append(0.0)
                continue
            
            bins = np.linspace(min_val, max_val, n_bins + 1)
            
            # 计算分布
            ref_dist, _ = np.histogram(ref_feature, bins=bins)
            cur_dist, _ = np.histogram(cur_feature, bins=bins)
            
            # 归一化
            ref_dist = ref_dist / len(ref_feature)
            cur_dist = cur_dist / len(cur_feature)
            
            # 避免零除
            ref_dist = np.where(ref_dist == 0, 0.0001, ref_dist)
            cur_dist = np.where(cur_dist == 0, 0.0001, cur_dist)
            
            # 计算PSI
            psi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
            psi_scores.append(psi)
            
            if psi > threshold:
                features_drifted.append(feature)
        
        avg_psi = np.mean(psi_scores)
        drifted = avg_psi > threshold
        
        return {
            "score": avg_psi,
            "drifted": drifted,
            "features_drifted": features_drifted,
            "metrics": {
                "psi_scores": psi_scores,
                "num_features": len(reference_data.columns),
                "num_drifted_features": len(features_drifted)
            }
        }
    
    async def _mmd_detector(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        drift_type: DriftType,
        threshold: float,
        **kwargs
    ) -> Dict[str, Any]:
        """MMD（最大均值差异）漂移检测器"""
        try:
            from sklearn.metrics.pairwise import rbf_kernel
            
            # 合并数据
            X = np.vstack([reference_data.values, current_data.values])
            
            # 计算RBF核矩阵
            gamma = 1.0 / reference_data.shape[1]  # 默认gamma
            K = rbf_kernel(X, gamma=gamma)
            
            n_ref = len(reference_data)
            n_cur = len(current_data)
            
            # 计算MMD统计量
            K_ref_ref = K[:n_ref, :n_ref]
            K_cur_cur = K[n_ref:, n_ref:]
            K_ref_cur = K[:n_ref, n_ref:]
            
            mmd = (K_ref_ref.mean() + K_cur_cur.mean() - 2 * K_ref_cur.mean())
            
            drifted = mmd > threshold
            
            return {
                "score": mmd,
                "drifted": drifted,
                "features_drifted": [],  # MMD是整体度量
                "metrics": {
                    "mmd_score": mmd,
                    "reference_size": n_ref,
                    "current_size": n_cur
                }
            }
        except ImportError:
            raise ImportError("scikit-learn is required for MMD detector")
    
    async def _classifier_detector(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        drift_type: DriftType,
        threshold: float,
        **kwargs
    ) -> Dict[str, Any]:
        """分类器漂移检测器"""
        try:
            # 创建标签：0表示参考数据，1表示当前数据
            X = np.vstack([reference_data.values, current_data.values])
            y = np.array([0] * len(reference_data) + [1] * len(current_data))
            
            # 训练随机森林分类器
            clf = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                n_jobs=-1
            )
            
            # 使用交叉验证评估分类器性能
            scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
            avg_score = np.mean(scores)
            
            # 如果分类器能很好地区分两组数据，说明存在漂移
            drifted = avg_score > threshold
            
            return {
                "score": avg_score,
                "drifted": drifted,
                "features_drifted": [],  # 分类器是整体度量
                "metrics": {
                    "roc_auc": avg_score,
                    "roc_auc_scores": scores.tolist(),
                    "reference_size": len(reference_data),
                    "current_size": len(current_data)
                }
            }
        except ImportError:
            raise ImportError("scikit-learn is required for classifier detector")
    
    async def _chi_square_detector(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        drift_type: DriftType,
        threshold: float,
        n_bins: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """卡方检验漂移检测器"""
        chi_scores = []
        p_values = []
        features_drifted = []
        
        for feature in reference_data.columns:
            ref_feature = reference_data[feature].dropna()
            cur_feature = current_data[feature].dropna()
            
            # 计算分箱
            min_val = min(ref_feature.min(), cur_feature.min())
            max_val = max(ref_feature.max(), cur_feature.max())
            
            if min_val == max_val:
                chi_scores.append(0.0)
                p_values.append(1.0)
                continue
            
            bins = np.linspace(min_val, max_val, n_bins + 1)
            
            # 计算频数
            ref_hist, _ = np.histogram(ref_feature, bins=bins)
            cur_hist, _ = np.histogram(cur_feature, bins=bins)
            
            # 执行卡方检验
            chi2, p_value, _, _ = stats.chi2_contingency([ref_hist, cur_hist])
            
            chi_scores.append(chi2)
            p_values.append(p_value)
            
            if p_value < threshold:
                features_drifted.append(feature)
        
        avg_chi2 = np.mean(chi_scores)
        min_p_value = np.min(p_values) if p_values else 1.0
        
        drifted = min_p_value < threshold
        
        return {
            "score": avg_chi2,
            "p_value": min_p_value,
            "drifted": drifted,
            "features_drifted": features_drifted,
            "metrics": {
                "chi2_scores": chi_scores,
                "p_values": p_values,
                "num_features": len(reference_data.columns),
                "num_drifted_features": len(features_drifted)
            }
        }
    
    async def _evidently_detector(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        drift_type: DriftType,
        threshold: float,
        model_name: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evidently库漂移检测器"""
        try:
            # 根据漂移类型选择适当的preset
            if drift_type == DriftType.DATA_DRIFT:
                report = Report(metrics=[DataDriftPreset()])
            elif drift_type == DriftType.DATA_QUALITY:
                report = Report(metrics=[DataQualityPreset()])
            elif drift_type == DriftType.TARGET_DRIFT:
                report = Report(metrics=[TargetDriftPreset()])
            elif drift_type == DriftType.REGRESSION:
                report = Report(metrics=[RegressionPreset()])
            elif drift_type == DriftType.CLASSIFICATION:
                report = Report(metrics=[ClassificationPreset()])
            else:
                report = Report(metrics=[DataDriftPreset()])
            
            # 运行报告
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=ColumnMapping()
            )
            
            # 获取结果
            result = report.as_dict()
            
            # 解析漂移信息
            metrics = result.get("metrics", [])
            drift_metric = None
            for metric in metrics:
                if metric.get("metric") == "DataDriftTable":
                    drift_metric = metric
                    break
            
            if not drift_metric:
                return {
                    "score": 0.0,
                    "drifted": False,
                    "features_drifted": [],
                    "metrics": result
                }
            
            drift_result = drift_metric.get("result", {})
            drift_score = drift_result.get("drift_score", 0.0)
            dataset_drift = drift_result.get("dataset_drift", False)
            drifted_features = []
            
            # 检查每个特征的漂移
            drift_by_columns = drift_result.get("drift_by_columns", {})
            for feature, feature_drift in drift_by_columns.items():
                if feature_drift.get("detected", False):
                    drifted_features.append(feature)
            
            drifted = dataset_drift or (drift_score > threshold)
            
            return {
                "score": drift_score,
                "drifted": drifted,
                "features_drifted": drifted_features,
                "metrics": result
            }
            
        except ImportError:
            raise ImportError("evidently is required for evidently detector")
        except Exception as e:
            logger.error(f"Evidently检测失败: {e}")
            return {
                "score": 0.0,
                "drifted": False,
                "features_drifted": [],
                "error": str(e)
            }
    
    def _get_default_threshold(self, detection_method: DetectionMethod) -> float:
        """获取默认阈值"""
        thresholds = {
            DetectionMethod.KS_TEST: 0.05,      # p-value阈值
            DetectionMethod.PSI: 0.1,          # PSI阈值
            DetectionMethod.MMD: 0.05,         # MMD阈值
            DetectionMethod.CLASSIFIATOR: 0.7, # ROC AUC阈值
            DetectionMethod.CHI_SQUARE: 0.05,  # p-value阈值
            DetectionMethod.EVIDENTLY: 0.1     # 漂移分数阈值
        }
        return thresholds.get(detection_method, 0.05)
    
    def _calculate_severity(self, drift_score: float, threshold: float) -> str:
        """计算漂移严重程度"""
        if drift_score <= threshold:
            return "none"
        elif drift_score <= threshold * 1.5:
            return "low"
        elif drift_score <= threshold * 2.0:
            return "medium"
        elif drift_score <= threshold * 3.0:
            return "high"
        else:
            return "critical"
    
    def _generate_recommendations(self, drift_result: DriftResult) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if drift_result.is_drifted:
            recommendations.append("检测到数据漂移，建议重新评估模型性能")
            
            if drift_result.severity in ["high", "critical"]:
                recommendations.append("严重漂移检测，建议立即重新训练模型")
            
            if drift_result.features_drifted:
                drifted_features = ", ".join(drift_result.features_drifted[:5])
                if len(drift_result.features_drifted) > 5:
                    drifted_features += f" 等 {len(drift_result.features_drifted)} 个特征"
                recommendations.append(f"漂移特征: {drifted_features}")
        
        return recommendations
    
    async def monitor_model_drift(
        self,
        model_name: str,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        reference_labels: Optional[np.ndarray] = None,
        current_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        监控模型漂移（预测分布变化）
        """
        # 计算预测分布的KS检验
        ks_stat, p_value = stats.ks_2samp(reference_predictions, current_predictions)
        
        result = {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "is_drifted": p_value < 0.05,
            "reference_mean": float(np.mean(reference_predictions)),
            "current_mean": float(np.mean(current_predictions)),
            "reference_std": float(np.std(reference_predictions)),
            "current_std": float(np.std(current_predictions))
        }
        
        # 如果有标签，计算性能漂移
        if reference_labels is not None and current_labels is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # 计算参考集性能
            ref_accuracy = accuracy_score(reference_labels, reference_predictions > 0.5)
            ref_precision = precision_score(reference_labels, reference_predictions > 0.5, zero_division=0)
            ref_recall = recall_score(reference_labels, reference_predictions > 0.5, zero_division=0)
            ref_f1 = f1_score(reference_labels, reference_predictions > 0.5, zero_division=0)
            
            # 计算当前集性能
            cur_accuracy = accuracy_score(current_labels, current_predictions > 0.5)
            cur_precision = precision_score(current_labels, current_predictions > 0.5, zero_division=0)
            cur_recall = recall_score(current_labels, current_predictions > 0.5, zero_division=0)
            cur_f1 = f1_score(current_labels, current_predictions > 0.5, zero_division=0)
            
            # 计算性能下降
            accuracy_drop = ref_accuracy - cur_accuracy
            precision_drop = ref_precision - cur_precision
            recall_drop = ref_recall - cur_recall
            f1_drop = ref_f1 - cur_f1
            
            result.update({
                "reference_accuracy": ref_accuracy,
                "current_accuracy": cur_accuracy,
                "accuracy_drop": accuracy_drop,
                "reference_precision": ref_precision,
                "current_precision": cur_precision,
                "precision_drop": precision_drop,
                "reference_recall": ref_recall,
                "current_recall": cur_recall,
                "recall_drop": recall_drop,
                "reference_f1": ref_f1,
                "current_f1": cur_f1,
                "f1_drop": f1_drop,
                "performance_drifted": any([
                    accuracy_drop > 0.1,
                    precision_drop > 0.1,
                    recall_drop > 0.1,
                    f1_drop > 0.1
                ])
            })
        
        return result
    
    async def _trigger_drift_alert(self, drift_result: DriftResult):
        """触发漂移告警"""
        alert_data = {
            "alert_type": "drift_detected",
            "model_name": drift_result.model_name,
            "model_version": drift_result.model_version,
            "drift_type": drift_result.drift_type.value,
            "detection_method": drift_result.detection_method.value,
            "drift_score": drift_result.drift_score,
            "threshold": drift_result.threshold,
            "severity": drift_result.severity,
            "is_drifted": drift_result.is_drifted,
            "features_drifted": drift_result.features_drifted,
            "timestamp": drift_result.timestamp.isoformat(),
            "recommendations": drift_result.recommendations
        }
        
        # 保存告警
        await self._save_alert(alert_data)
        
        # 发送通知（示例：Webhook）
        if settings.monitoring.alert_webhook_url:
            await self._send_webhook_alert(alert_data)
        
        logger.warning(f"漂移告警: {drift_result.model_name} - 严重度: {drift_result.severity}")
    
    async def _save_reference_data(self, model_name: str, reference: ReferenceData):
        """保存参考数据"""
        file_path = self.data_dir / f"reference_{model_name}.json"
        
        data_to_save = {
            "model_name": model_name,
            "data_version": reference.data_version,
            "timestamp": reference.timestamp.isoformat(),
            "metadata": reference.metadata,
            "features": reference.features,
            "sample_size": reference.sample_size,
            # 注意：不保存实际数据，只保存元数据
        }
        
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(data_to_save, indent=2, default=str))
    
    async def _save_drift_result(self, drift_result: DriftResult):
        """保存漂移结果"""
        file_path = self.data_dir / f"drift_{drift_result.drift_id}.json"
        
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(asdict(drift_result), indent=2, default=str))
    
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
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    settings.monitoring.alert_webhook_url,
                    json=alert_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Webhook告警发送失败: {response.status}")
        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")
    
    def _load_history(self):
        """加载历史数据"""
        # 加载漂移结果
        for drift_file in self.data_dir.glob("drift_*.json"):
            try:
                with open(drift_file, 'r') as f:
                    data = json.load(f)
                    drift_result = DriftResult(**data)
                    
                    model_name = drift_result.model_name
                    if model_name not in self.drift_history:
                        self.drift_history[model_name] = []
                    self.drift_history[model_name].append(drift_result)
            except Exception as e:
                logger.error(f"加载漂移结果失败 {drift_file}: {e}")
        
        # 按时间排序
        for model_name in self.drift_history:
            self.drift_history[model_name].sort(key=lambda x: x.timestamp)
    
    async def get_drift_history(
        self,
        model_name: str,
        lookback_days: int = 30
    ) -> List[DriftResult]:
        """获取漂移历史"""
        if model_name not in self.drift_history:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        history = [
            result for result in self.drift_history[model_name]
            if result.timestamp >= cutoff_date
        ]
        
        return history
    
    async def get_drift_summary(
        self,
        model_name: str,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """获取漂移摘要"""
        history = await self.get_drift_history(model_name, lookback_days)
        
        if not history:
            return {
                "model_name": model_name,
                "has_history": False,
                "message": f"No drift history found for {model_name} in last {lookback_days} days"
            }
        
        # 统计
        total_drifts = len(history)
        drifted_count = sum(1 for h in history if h.is_drifted)
        drift_rate = drifted_count / total_drifts if total_drifts > 0 else 0
        
        # 按严重程度统计
        severity_counts = {}
        for result in history:
            if result.is_drifted:
                severity = result.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # 最近漂移
        recent_drifts = [h for h in history if h.is_drifted][-5:]  # 最近5次
        
        return {
            "model_name": model_name,
            "has_history": True,
            "period": f"last_{lookback_days}_days",
            "total_checks": total_drifts,
            "drifted_checks": drifted_count,
            "drift_rate": drift_rate,
            "severity_counts": severity_counts,
            "recent_drifts": [
                {
                    "drift_id": d.drift_id,
                    "timestamp": d.timestamp.isoformat(),
                    "drift_score": d.drift_score,
                    "severity": d.severity,
                    "features_drifted": d.features_drifted[:5]  # 最多5个特征
                }
                for d in recent_drifts
            ],
            "last_check": history[-1].timestamp.isoformat() if history else None
        }
