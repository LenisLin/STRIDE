"""
Module: tasks.task_A.common
"""
from typing import Sequence, Tuple, Dict, Any
import numpy as np
import pandas as pd
from anndata import AnnData

# 注意：这里只导入 Library 的契约和纯函数，不实现循环控制
from slotar.uot import UOTSolveConfig, batched_uot_solve

def assemble_tensors(
    adata: AnnData, 
    roi_pairs: pd.DataFrame, 
    k_full: int,
    mass_mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据 ROI 配对表组装批处理张量，保证输出尺寸严格为 [N, K_full]。
    
    Args:
        adata: 包含 'spatial', 'proto_id' 的 AnnData。
        roi_pairs: 包含 'roi_a', 'roi_b' 等列的 DataFrame。
        k_full: 全局特征维度。
        mass_mode: "counts", "density", "proportion" (用于校验统一尺度)。
        
    Returns:
        A: shape [N, K_full] 的源质量矩阵。
        B: shape [N, K_full] 的目标质量矩阵。
        mass_gap: shape [N] 的质量差异比率 (|sum(A)-sum(B)| / max(sumA, sumB))。
    """
    pass

def run_uot_batch_safe(
    A: np.ndarray, 
    B: np.ndarray, 
    lambda_pl: np.ndarray, 
    kernels: Sequence[np.ndarray], 
    uot_cfg: UOTSolveConfig,
    pair_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    安全调用 batched_uot_solve，将 status!="ok" 拦截并对指标进行 NaN padding。
    
    Args:
        A, B, lambda_pl: [N, K_full] 和 [N] 的批处理输入。
        kernels: 预计算的 log-kernels 列表。
        uot_cfg: 包含 eps_schedule, n_min_proto (用于消融) 的配置对象。
        pair_meta: 原始的配对元数据表。
        
    Returns:
        pd.DataFrame: 包含 pair_meta 的原始列，并追加计算结果及审计字段：
            - 数学指标: 'M_uot', 'U', 'T', 'D_pos', 'B_pos' (U = B_pos + D_pos)
            - 状态审计: 'status', 'mass_pruned_ratio'
            - 配置审计: 'n_min_proto_used'
    """
    pass

def run_balanced_ot_batch(
    A: np.ndarray, 
    B: np.ndarray, 
    cost_matrix: np.ndarray, 
    scale_mode: str,
    eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在绝对质量对齐的前提下，运行 Balanced OT 基线。
    
    Args:
        scale_mode: 'project_B_to_A' 等，确保 A/B 处于等价物理尺度。
        
    Returns:
        M_balanced: shape [N] 的等价尺度平衡传输成本。
        m_star: shape [N] 的对齐质量基准 (用于审计)。
    """
    pass
