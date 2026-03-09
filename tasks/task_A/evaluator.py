"""
Module: tasks.task_A.evaluator
"""
import pandas as pd
from typing import Dict, Any

def evaluate_task_a(df_pairs: pd.DataFrame, df_uq: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    将业务主张映射为非参数统计检验。任何失败直接 raise AssertionError 并附带证据。
    """
    # 1. Arm A1 噪音底线断言
    a1 = df_pairs[df_pairs['arm'] == 'A1_baseline']
    ok_rate = (a1['status'] == 'ok').mean()
    assert ok_rate >= 0.95, f"Arm A1 引擎计算成功率过低 ({ok_rate:.2f})，检查数据质量或底层溢出"

    # 2. Arm A2 空间递进一致性 (符号检验)
    a2 = df_pairs[df_pairs['arm'] == 'A2_gradient']
    _assert_spatial_gradient(a2)
    
    # 3. Arm A2 对抗检验 (UOT vs Balanced)
    high_gap_pairs = a2[a2['mass_gap'] > 0.3] # 显著的质量不匹配
    median_ratio = (high_gap_pairs['M_balanced'] / high_gap_pairs['M_uot']).median()
    assert median_ratio > 1.2, f"Balanced OT 未能系统性夸大重塑，中位数比率仅为 {median_ratio:.2f}"
    
    # 4. Arm A3 UQ 诚实度与消融检验
    _assert_uq_honesty_and_ablation(df_uq)
    
    print(">>> Task A v2.1 全数理验证通过 <<<")

def _assert_spatial_gradient(df: pd.DataFrame) -> None:
    # 聚合到患者层级
    pt_stats = df.groupby(['patient_id', 'pair_type'])['M_uot'].median().unstack()
    
    # 符号检验: M_CTPT > M_CTIM
    ctpt_gt_ctim_ratio = (pt_stats['CT-PT'] > pt_stats['CT-IM']).mean()
    assert ctpt_gt_ctim_ratio > 0.5, f"梯度倒挂：仅有 {ctpt_gt_ctim_ratio:.1%} 患者满足 CTPT > CTIM"

def _assert_uq_honesty_and_ablation(df_uq: pd.DataFrame) -> None:
    frozen = df_uq[df_uq['mode'] == 'frozen']
    dynamic = df_uq[df_uq['mode'] == 'dynamic']
    
    # 诚实度：Frozen 组在低 coverage 下 CI 必须显著变宽
    w10_f = frozen[frozen['coverage'] == 0.1]['ci_width'].median()
    w80_f = frozen[frozen['coverage'] == 0.8]['ci_width'].median()
    assert w10_f > w80_f * 1.5, f"拓扑冻结失效：极小视野下的置信区间未能有效拓宽 (w10={w10_f}, w80={w80_f})"
    
    # 消融反证：Dynamic 组在低 coverage 下必须发生方差崩塌 (假自信)
    w10_d = dynamic[dynamic['coverage'] == 0.1]['ci_width'].median()
    assert w10_d < w10_f * 0.5, f"消融未触发典型失效：动态裁剪未产生显著的假自信 (w10_d={w10_d}, w10_f={w10_f})"
