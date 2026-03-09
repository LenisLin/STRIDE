"""
Module: tasks.task_A.arm2_spatial_gradient
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Sequence
from anndata import AnnData
from slotar.uot import UOTSolveConfig, calibrate_lambdas  # library 原语
from .common import assemble_tensors, run_uot_batch_safe, run_balanced_ot_batch

def run_arm2(
    adata: AnnData, 
    config: Dict[str, Any], 
    uot_cfg: UOTSolveConfig, 
    kernels: Sequence[np.ndarray]
) -> pd.DataFrame:
    """
    执行跨区划对比 (CT-IM, IM-PT, CT-PT)，并实施 Pair-specific Joint Calibration。
    """
    pair_meta = _generate_cross_compartment_pairs(adata)
    results = []
    
    # 业务循环：按 pair_type 分别执行联合校准与推断 (遵循 D004)
    for pair_type, group_df in pair_meta.groupby('pair_type'):
        # 1. 张量组装
        A, B, mass_gap = assemble_tensors(adata, group_df, config['data']['k_full'], config['data']['mass_mode'])
        
        # 2. 调用 library 原语进行该 Pair Type 的联合校准
        # 这里传入 joint pool (A 和 B 的全集) 让库层纯粹寻找最佳 lambda
        best_lambda = calibrate_lambdas(A, B, candidates=np.logspace(-2, 2, 10))
        lambda_pl = np.full(len(group_df), best_lambda)
        
        # 3. 运行 UOT
        df_uot = run_uot_batch_safe(A, B, lambda_pl, kernels, uot_cfg, group_df)
        
        # 4. 运行 Balanced OT (强制使用配置中定义的物理尺度对齐方式)
        M_bal, m_star = run_balanced_ot_batch(
            A, B, 
            cost_matrix=_get_cost_matrix(), 
            scale_mode=config['baselines']['balanced_mass_scale_mode'],
            eps=config['baselines']['balanced_eps']
        )
        
        # 5. 合并并追加审计字段
        df_uot['M_balanced'] = M_bal
        df_uot['m_star_audit'] = m_star
        df_uot['lambda_pl'] = lambda_pl
        df_uot['mass_gap'] = mass_gap
        results.append(df_uot)
        
    df_final = pd.concat(results, ignore_index=True)
    df_final['arm'] = 'A2_gradient'
    return df_final

def _generate_cross_compartment_pairs(adata: AnnData) -> pd.DataFrame:
    pass

def _get_cost_matrix() -> np.ndarray:
    pass
