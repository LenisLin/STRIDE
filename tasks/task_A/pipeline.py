"""
Module: tasks.task_A.pipeline
"""
import yaml
import anndata as ad
import pandas as pd
from pathlib import Path
from slotar.contracts import validate_adata_inputs
from slotar.uot import UOTSolveConfig, precompute_logKernels
from .arm1_noise_baseline import run_arm1
from .arm2_spatial_gradient import run_arm2
from .arm3_uq_stress import run_arm3
from .evaluator import evaluate_task_a

def main(config_path: str, data_path: str, output_dir: str) -> None:
    # 1. 业务解析与契约验证
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    adata = ad.read_h5ad(data_path)
    validate_adata_inputs(
        adata,
        require_representation=True,
        require_cost_scale=True,
        require_cost_matrix=True,
    )
    
    # 2. 组装全局 UOT 配置 (传递给各 Arm)
    uot_cfg = UOTSolveConfig(
        eps_schedule=config['uot_params']['eps_schedule'],
        eta_floor=float(config['uot_params']['eta_floor']),
        tau_q=config['uot_params']['tau_q'],
        tau_mode=config['uot_params']['tau_mode']
    )
    # 预计算全局核矩阵，避免各 Arm 重复计算
    kernels = precompute_logKernels(adata.uns['cost_matrix'], uot_cfg.eps_schedule)
    
    # 3. 按序调度
    df_arm1 = run_arm1(adata, config, uot_cfg, kernels)
    df_arm2 = run_arm2(adata, config, uot_cfg, kernels)
    df_arm3 = run_arm3(adata, config, uot_cfg, kernels)
    
    # 4. 组装与落盘审计文件
    df_pairs = pd.concat([df_arm1, df_arm2], ignore_index=True)
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_pairs.to_parquet(out_dir / "task_A_pairs.parquet")
    df_arm3.to_parquet(out_dir / "task_A_uq.parquet")
    
    # 5. 最终统计验收
    evaluate_task_a(df_pairs, df_arm3, config)

if __name__ == "__main__":
    main("config.yaml", "data/cohort.h5ad", "outputs/")
