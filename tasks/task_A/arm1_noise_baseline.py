"""
Module: tasks.task_A.arm1_noise_baseline
"""
import pandas as pd
from typing import Dict, Any, Sequence
from anndata import AnnData
from slotar.uot import UOTSolveConfig
from .common import assemble_tensors, run_uot_batch_safe

def run_arm1(
    adata: AnnData, 
    config: Dict[str, Any], 
    uot_cfg: UOTSolveConfig, 
    kernels: Sequence[np.ndarray]
) -> pd.DataFrame:
    """
    生成同患者、同区划 (CT-CT, IM-IM, PT-PT) 的配对并计算底噪。
    """
    # 1. 业务逻辑：生成 within-patient, within-compartment 的配对表 pair_meta
    pair_meta = _generate_within_compartment_pairs(adata)
    
    # 2. 调度 common 进行张量组装 [N, K_full]
    A, B, mass_gap = assemble_tensors(adata, pair_meta, config['data']['k_full'], config['data']['mass_mode'])
    
    # 3. 提取全局/基线 lambda (Arm1 不做复杂的联合校准，使用全局中值即可)
    lambda_pl = _get_baseline_lambdas(len(pair_meta))
    
    # 4. 安全调用底层的 Batched UOT
    df_result = run_uot_batch_safe(A, B, lambda_pl, kernels, uot_cfg, pair_meta)
    df_result['arm'] = 'A1_baseline'
    
    return df_result

def _generate_within_compartment_pairs(adata: AnnData) -> pd.DataFrame:
    # 仅作结构展示：按 patient_id 和 compartment 聚合，生成 C(n, 2) 的 ROI 对
    pass

def _get_baseline_lambdas(n: int) -> np.ndarray:
    pass
