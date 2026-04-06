# 初始化监控器
monitor = DataQualityMonitor()

# 监控数据集
result = await monitor.monitor_dataset(
    dataset_name="user_data",
    data=user_df,
    expectation_suite_name="user_data_suite"
)

# 获取质量历史
history = await monitor.get_quality_history("user_data", 7)

# 生成质量报告
report = await monitor.generate_quality_report(
    dataset_name="user_data",
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)

# 添加质量规则
rule_id = await monitor.add_quality_rule(
    name="用户年龄验证",
    description="检查用户年龄是否在有效范围内",
    rule_type=QualityCheckType.VALIDITY,
    condition={"metric": "age", "min": 0, "max": 120},
    severity="high"
)

"""
这是完整的数据质量监控模块代码，包含以下核心功能：
🎯 核心特性
1. 多层次质量检查

    完整性检查：缺失值检测
    唯一性检查：重复数据检测
    有效性检查：数据范围、格式验证
    一致性检查：跨表、跨字段一致性
    分布检查：数据分布异常检测
    自定义检查：灵活的业务规则

2. 多工具集成

    Great Expectations：规则验证和期望测试
    whylogs：数据画像和统计监控
    自定义引擎：灵活的业务规则引擎

3. 智能监控

    实时监控：支持实时数据流监控
    批处理验证：大规模数据集验证
    趋势分析：质量趋势跟踪
    异常检测：自动异常点检测

4. 告警系统

    多级告警：低、中、高、严重级别
    多渠道通知：Webhook、Slack、Email
    智能建议：自动生成修复建议
    历史追踪：完整告警历史记录

5. 报告和分析

    质量报告：详细的质量分析报告
    趋势图表：可视化质量趋势
    根本原因分析：失败检查分析
    性能指标：监控性能统计

6. 管理和配置

    规则管理：CRUD质量规则
    配置管理：灵活的配置系统
    数据管理：自动清理旧数据
    扩展接口：易于扩展的插件系统

"""
