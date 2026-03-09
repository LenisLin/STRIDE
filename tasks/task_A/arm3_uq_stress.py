"""
Module: tasks.task_A.arm3_uq_stress
"""
import copy
import pandas as pd
from typing import Dict, Any, Sequence
from anndata import AnnData
from slotar.uot import UOTSolveConfig
from .common import assemble_tensors, run_uot_batch_safe

def run_arm3(
    adata: AnnData, 
    config: Dict[str, Any], 
    uot_cfg_base: UOTSolveConfig, 
    kernels: Sequence[np.ndarray]
) -> pd.DataFrame:
    """
    执行 CT-PT 的多覆盖率 Bootstrap，利用 n_min_proto 进行 Frozen/Dynamic 消融。
    """
    pair_meta = _get_ct_pt_pairs(adata)
    uq_results = []
    
    n_bootstraps = 100
    coverages = config['arm3_uq']['coverage_list']
    modes = ['frozen', 'dynamic']
    
    for coverage in coverages:
        # 生成重采样元数据 (Block Bootstrap)
        boot_meta = _generate_block_bootstraps(pair_meta, coverage, n_bootstraps)
        
        for mode in modes:
            # 核心消融逻辑：复制配置并修改保留阈值，严禁修改底层 solver 签名
            cfg_ablation = copy.deepcopy(uot_cfg_base)
            if mode == 'frozen':
                cfg_ablation.n_min_proto = config['arm3_uq']['n_min_proto_frozen']
            else:
                cfg_ablation.n_min_proto = config['arm3_uq']['n_min_proto_dynamic']
                
            # 张量维度 K_full 保持不变
            A, B, _ = assemble_tensors(adata, boot_meta, config['data']['k_full'], config['data']['mass_mode'])
            lambda_pl = _get_precalibrated_lambdas(boot_meta)
            
            df_boot = run_uot_batch_safe(A, B, lambda_pl, kernels, cfg_ablation, boot_meta)
            
            # 聚合计算 CI
            df_ci = _compute_ci_from_replicates(df_boot)
            df_ci['coverage'] = coverage
            df_ci['mode'] = mode
            uq_results.append(df_ci)
            
    df_final = pd.concat(uq_results, ignore_index=True)
    return df_final

def _get_ct_pt_pairs(adata: AnnData) -> pd.DataFrame:
    pass
def _generate_block_bootstraps(pairs: pd.DataFrame, cov: float, b: int) -> pd.DataFrame:
    pass
def _get_precalibrated_lambdas(df: pd.DataFrame) -> np.ndarray:
    pass
def _compute_ci_from_replicates(df_boot: pd.DataFrame) -> pd.DataFrame:
    # 按 pair_id 聚合，计算 2.5% 和 97.5% 分位数，求得 ci_width
    pass
