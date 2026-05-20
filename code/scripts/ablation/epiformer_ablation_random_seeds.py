#!/usr/bin/env python3
"""
Focused Ablation Study Script - 37 Specific Experiments with 3 Random Seeds
Runs each ablation configuration with 3 different random seeds for statistical significance.
Uses base config from test.sh and overrides specific parameters.
"""

import os, sys, time, argparse
import subprocess, torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess as sp
import warnings

warnings.filterwarnings("ignore")

# ──────── FOCUSED ABLATION CONFIGURATIONS ──────────────────────────
# Based on the 37-experiment table - exact configurations only

ABLATION_CONFIGS = {
    # Backbone GNN ablations (6 experiments)
    'backbone_gnn': {
        'egnn': {'model.epiformer.ag_resmp_type': 'egnn', 'model.epiformer.ab_resmp_type': 'egnn'},
        'regnn': {'model.epiformer.ag_resmp_type': 'regnn', 'model.epiformer.ab_resmp_type': 'regnn'},
        'gcn': {'model.epiformer.ag_resmp_type': 'gcn', 'model.epiformer.ab_resmp_type': 'gcn'},
        'rgcn': {'model.epiformer.ag_resmp_type': 'rgcn', 'model.epiformer.ab_resmp_type': 'rgcn'},
        'gat': {'model.epiformer.ag_resmp_type': 'gat', 'model.epiformer.ab_resmp_type': 'gat'},
        'gin': {'model.epiformer.ag_resmp_type': 'gin', 'model.epiformer.ab_resmp_type': 'gin'}
    },
    
    # Data split ablations (3 experiments)
    'split': {
        'random': {'dataset.split.method': 'random'},
        'epitope_ratio': {'dataset.split.method': 'epitope_ratio'},
        'epitope_group': {'dataset.split.method': 'epitope_group'}
    },
    
    # Loss function ablations (13 experiments) - sparsity = count_regularizer
    'loss': {
        'bce': {
            'loss.node_prediction.enabled':'true',
            'loss.node_prediction.count_regularizer_enabled': 'false',  # Disable sparsity
            'loss.node_prediction.dice_enabled': 'false',    # Disable dice
            'loss.contrastive.enabled': 'false',             # Disable infonce  
            'loss.edge_prediction.enabled': 'false',          # Disable edge
            'model.decoder.predict_distances': 'false',
            "loss.auxiliary_distance.enabled": "false"
        },
        'bce_edge': {
            'loss.node_prediction.enabled':'true',
            'loss.node_prediction.count_regularizer_enabled': 'false',  # Disable sparsity
            'loss.node_prediction.dice_enabled': 'false',    # Disable dice
            'loss.contrastive.enabled': 'false',             # Disable infonce  
            'loss.edge_prediction.enabled': 'true',          # Disable edge
            'model.decoder.predict_distances': 'false',
            "loss.auxiliary_distance.enabled": "false"
        },
        'edge': {
            'loss.node_prediction.enabled':'false',
            'loss.node_prediction.count_regularizer_enabled': 'false',  # Disable sparsity
            'loss.node_prediction.dice_enabled': 'false',    # Disable dice
            'loss.contrastive.enabled': 'false',             # Disable infonce  
            'loss.edge_prediction.enabled': 'true',          # Disable edge
            'model.decoder.predict_distances': 'false',
            "loss.auxiliary_distance.enabled": "false"
        },
        'bce_edge_dist': {
            'loss.node_prediction.enabled':'true',
            'loss.node_prediction.count_regularizer_enabled': 'false',  # Disable sparsity
            'loss.node_prediction.dice_enabled': 'false',    # Disable dice
            'loss.contrastive.enabled': 'false',             # Disable infonce  
            'loss.edge_prediction.enabled': 'true',          # Disable edge
            'model.decoder.predict_distances': 'true',
            "loss.auxiliary_distance.enabled": "true"
        },
        'bce_dist': {
            'loss.node_prediction.enabled':'true',
            'loss.node_prediction.count_regularizer_enabled': 'false',  # Disable sparsity
            'loss.node_prediction.dice_enabled': 'false',    # Disable dice
            'loss.contrastive.enabled': 'false',             # Disable infonce  
            'loss.edge_prediction.enabled': 'false',          # Disable edge
            'model.decoder.predict_distances': 'true',
            "loss.auxiliary_distance.enabled": "true"
        },
        'bce_edge_sparsity': {
            'loss.node_prediction.enabled':'true',
            'loss.node_prediction.count_regularizer_enabled': 'true',  # enable sparsity
            'loss.node_prediction.dice_enabled': 'false',
            'loss.contrastive.enabled': 'false', 
            'loss.edge_prediction.enabled': 'true',
            'model.decoder.predict_distances': 'false',
            "loss.auxiliary_distance.enabled": "false"
        },
        'bce_edge_dice': {
            'loss.node_prediction.enabled':'true',
            'loss.node_prediction.count_regularizer_enabled': 'false',  
            'loss.node_prediction.dice_enabled': 'true',       # Enable dice (from base)
            'loss.contrastive.enabled': 'false',
            'loss.edge_prediction.enabled': 'true',
            'model.decoder.predict_distances': 'false',
            "loss.auxiliary_distance.enabled": "false"
        },
        'bce_edge_dist_sparsity_dice': {  # This is the base config
            'loss.node_prediction.enabled':'true',
            'loss.node_prediction.count_regularizer_enabled': 'true',  # enable sparsity
            'loss.node_prediction.dice_enabled': 'true',
            'loss.contrastive.enabled': 'false',
            'loss.edge_prediction.enabled': 'true',
            'model.decoder.predict_distances': 'true',
            "loss.auxiliary_distance.enabled": "true"
        },
        'bce_edge_sparsity_dice_infonce': {
            'loss.node_prediction.enabled':'true',
            'loss.node_prediction.count_regularizer_enabled': 'true',  # enable sparsity
            'loss.node_prediction.dice_enabled': 'true', 
            'loss.contrastive.enabled': 'true',             # Enable infonce
            'loss.contrastive.name': 'infonce',
            'loss.edge_prediction.enabled': 'true',
            'model.decoder.predict_distances': 'false',
            "loss.auxiliary_distance.enabled": "false"
        },
        'bce_edge_sparsity_dice': {
            'loss.node_prediction.enabled':'true',
            'loss.node_prediction.count_regularizer_enabled': 'true',  # enable sparsity
            'loss.node_prediction.dice_enabled': 'true',
            'loss.contrastive.enabled': 'false',
            'loss.edge_prediction.enabled': 'true',         # Enable edge
            'model.decoder.predict_distances': 'false',
            "loss.auxiliary_distance.enabled": "false"
        },
        'bce_edge_sparsity_dice_infonce': {
            'loss.node_prediction.enabled':'true',
            'loss.node_prediction.count_regularizer_enabled': 'true',  # enable sparsity
            'loss.node_prediction.dice_enabled': 'true',
            'loss.contrastive.enabled': 'true',
            'loss.contrastive.name': 'infonce',
            'loss.edge_prediction.enabled': 'true',
            'model.decoder.predict_distances': 'false',
            "loss.auxiliary_distance.enabled": "false"
        },
        'bce_edge_infonce': {
            'loss.node_prediction.enabled':'true',
            'loss.node_prediction.count_regularizer_enabled': 'true',  # disable sparsity
            'loss.node_prediction.dice_enabled': 'false',
            'loss.contrastive.enabled': 'true',
            'loss.contrastive.name': 'infonce',
            'loss.edge_prediction.enabled': 'true',
            'model.decoder.predict_distances': 'false',
            "loss.auxiliary_distance.enabled": "false"
        },
        'bce_edge_dist_sparsity_dice_infonce': {
            'loss.node_prediction.enabled':'true',
            'loss.node_prediction.count_regularizer_enabled': 'true',  # enable sparsity
            'loss.node_prediction.dice_enabled': 'true',
            'loss.contrastive.enabled': 'true',
            'loss.contrastive.name': 'infonce',
            'loss.edge_prediction.enabled': 'true',
            'model.decoder.predict_distances': 'true',
            "loss.auxiliary_distance.enabled": "true"
        },
    },
    
    # Decoder ablations (3 experiments)
    'decoder': {
        'dot_product': {'model.decoder.type': 'dot_product'},  # Base config
        'cross_attn': {'model.decoder.type': 'cross_attention'},
        'dual': {'model.decoder.type': 'dual'},
        'bilinear': {'model.decoder.type': 'enhanced_bilinear'}
    },
    
    # Sampling strategy ablations (7 experiments) 
    'sampling': {
        'max': {'model.decoder.sampling_strat': 'max_row'},
        'mean': {'model.decoder.sampling_strat': 'mean_row'},
        'mean_top_k_2': {'model.decoder.sampling_strat': 'top_k_mean_2'},  # Base config
        'mean_top_k_3': {'model.decoder.sampling_strat': 'top_k_mean_3'},
        'mean_top_k_4': {'model.decoder.sampling_strat': 'top_k_mean_4'},
        'softmax_attn': {'model.decoder.sampling_strat': 'softmax_attention'},
        'hierarchical_pooling': {'model.decoder.sampling_strat': 'hierarchical_pooling'}
    },
    
    # Mixed encoder ablations (2 experiments)
    'mixed_encoders': {
        'ag_egnn_ab_regnn': {'model.epiformer.ag_resmp_type': 'egnn', 'model.epiformer.ab_resmp_type': 'regnn'},
        'ag_regnn_ab_egnn': {'model.epiformer.ag_resmp_type': 'regnn', 'model.epiformer.ab_resmp_type': 'egnn'}
    },
    
    # Graph type ablations (4 experiments)
    'graph_type': {
        'raad_egnn': {'dataset.graph_type': 'raad-plm', 'dataset.plm_type':"esm2_650m", 'model.epiformer.ag_resmp_type': 'egnn', 'model.epiformer.ab_resmp_type': 'egnn'},
        'gearnet_egnn': {'dataset.graph_type': 'gearnet', 'model.epiformer.ag_resmp_type': 'egnn', 'model.epiformer.ab_resmp_type': 'egnn'},
        'base_egnn': {'dataset.graph_type': 'base', 'model.epiformer.ag_resmp_type': 'egnn', 'model.epiformer.ab_resmp_type': 'egnn'},
        'base_gcn': {'dataset.graph_type': 'base', 'model.epiformer.ag_resmp_type': 'gcn', 'model.epiformer.ab_resmp_type': 'gcn'}
    }
}

# Define random seeds to use
RANDOM_SEEDS = [42, 123, 456]  # Three different random seeds

# ──────── RESULTS SETUP ────────────────────────────────────────
RESULTS_DIR = os.path.join(os.getcwd(), "../../../../results/hgraphepi/m3epi/ablation")
summary_dir = os.path.join(RESULTS_DIR, "summary")
logs_dir = os.path.join(RESULTS_DIR, "logs")

Path(summary_dir).mkdir(parents=True, exist_ok=True)
Path(logs_dir).mkdir(parents=True, exist_ok=True)

CODE_DIR = Path(__file__).parent.parent.parent.resolve()  # Go up to scripts/ level
env = os.environ.copy()
env["PYTHONPATH"] = str(CODE_DIR)
PYTHON = sys.executable

# ──────── ARGUMENT PARSER ──────────────────────────────────────
parser = argparse.ArgumentParser(description="Run focused ablation experiments with 3 random seeds")
parser.add_argument("--multi_gpu", action="store_true", help="Run experiments in parallel across GPUs")
parser.add_argument("--gpu_id", type=int, default=0, help="CUDA GPU ID for sequential mode")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--output_csv", type=str, default="focused_ablation_results.csv", help="Output CSV filename")
args = parser.parse_args()

# ──────── CREATE RESULTS FILENAMES ─────────────────────────────
ts = datetime.now().strftime("%Y%m%dT%H%M%S")
try:
    sha = sp.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
except Exception:
    sha = "nogit"

raw_path = os.path.join(summary_dir, f"{ts}_{sha}_focused_ablation_raw.csv")
agg_path = os.path.join(summary_dir, f"{ts}_{sha}_{args.output_csv}")

# ──────── BUILD BASE CONFIG FROM test.sh ──────────────────────
def get_base_overrides(seed=42):
    """Complete base configuration from test.sh - ALL parameters"""
    return [
        "mode=val",
        f"seed={seed}",  # Use the provided seed
        "model.name=epiformer",
        f"gpu_id={args.gpu_id}",
        "wandb.project=m3epi_v3_dev",
        "dataset.split.method=epitope_ratio",
        "dataset.graph_type=raad-plm",
        "dataset.plm_type=esm2_650m",
        "dataset.graph_num_relations=4",
        "dataset.tensor=hierarchical_dataset.pkl",
        f"hparams.train.num_epochs={args.epochs}",
        f"hparams.train.batch_size={args.batch_size}",
        "hparams.pretrain.num_epochs=5",
        "hparams.pretrain.lr=0.00005",
        "hparams.train.learning_rate=0.00009095",
        "hparams.train.weight_decay=0.00001179",
        "hparams.train.kfolds=2",
        "hparams.train.regularization.use_l2_reg=false",
        "hparams.train.scheduler=reduce_lr_on_plateau",
        "num_threads=3",
        "resume=false",
        "model.enable_pretraining=false",
        "model.epiformer.ag_resmp_type=egnn",
        "model.epiformer.ab_resmp_type=egnn",
        "model.epiformer.residue_layers=4",
        "model.epiformer.residue_dim=128",
        "model.epiformer.residue_hidden_dim=128",
        "model.epiformer.plm_dim=128",
        "model.epiformer.n_heads=8",
        "model.epiformer.use_layer_norm=true",
        "model.epiformer.use_pair_repr=false",
        "model.epiformer.use_gradient_checkpointing=false",
        "model.epiformer.ag_feature_fusion_type=concat",
        "model.epiformer.ab_feature_fusion_type=gated",
        "model.epiformer.activation=silu",
        "model.epiformer.dropout=0.1324459",
        "model.dropout_rates.decoder=0.072024237",
        "model.dropout_rates.projections=0.1",
        "model.decoder.type=cross_attention",
        "model.decoder.num_rbf=16",
        "model.decoder.d_k=64",
        "model.decoder.d_ff=128",
        "model.decoder.d_model=128",
        "model.decoder.n_heads=8",
        "model.decoder.decoder_layers=2",
        "model.decoder.sampling_strat=top_k_mean_2",
        "model.epi_threshold=0.3",
        "model.para_threshold=0.3",
        "model.use_layer_norm=true",
        "callbacks.early_stopping.patience=10",
        "callbacks.checkpoint_interval=2",
        "loss.node_prediction.enabled=true",
        "loss.node_prediction.weight=0.481570678",
        "loss.node_prediction.name=bce",
        "loss.node_prediction.task=epi_only",
        "loss.node_prediction.bce_weight=9.324872",
        "loss.node_prediction.dice_weight=2.2965577",
        "loss.node_prediction.smoothness_weight=0.01",
        "loss.node_prediction.consistency_weight=0.1",
        "loss.node_prediction.dice_enabled=true",
        "loss.node_prediction.count_regularizer_enabled=true",
        "loss.node_prediction.smoothness_enabled=false",
        "loss.node_prediction.edge_node_consistency_enabled=false",
        "loss.node_prediction.epi_pos_weight=15.28555",
        "loss.node_prediction.para_pos_weight=3",
        "loss.count_regularizer.per_graph_matching=true",
        "loss.count_regularizer.epitope_weight=0.306774",
        "loss.count_regularizer.paratope_weight=0.1",
        "loss.count_regularizer.dataset_prior=false",
        "loss.count_regularizer.epitope_prior_mean=14.6",
        "loss.count_regularizer.prior_weight=0.05",
        "loss.count_regularizer.anneal_epochs=10",
        "loss.label_smoothing=0.1",
        "loss.class_balance.beta=0.9999",
        "loss.edge_prediction.enabled=true",
        "loss.edge_prediction.weight=1.0",
        "loss.edge_prediction.pos_weight=58.7076536",
        "loss.edge_count_regularizer.enabled=false",
        "loss.edge_count_regularizer.weight=0.1",
        "loss.contrastive.enabled=false",
        "loss.contrastive.name=infonce",
        "loss.contrastive.weight=0.1",
        "loss.contrastive.temperature=0.4",
        "loss.contrastive.inter_weight=0.5",
        "loss.contrastive.intra_weight=0.5",
        "loss.gwnce.weight=0.1",
        "loss.gwnce.cut_way=2",
        "loss.gwnce.cut_rate=0.5",
        "model.decoder.predict_distances=true",
        "loss.auxiliary_distance.enabled=true",
        "loss.auxiliary_distance.weight=0.0513977",
        "loss.auxiliary_distance.distance_weighting=true",
        "loss.auxiliary_distance.class_balancing=true",
        "loss.auxiliary_distance.max_distance=32.0",
        "loss.force.enabled=false",
        "loss.force.weight=0.01",
        "loss.force.bond_weight=1",
        "loss.force.angle_weight=0.5",
        "loss.force.smooth_alpha=1",
        "loss.force.smooth_weight=0.1",
        "loss.force.bond_tolerance=0.1",
        "loss.force.angle_tolerance=0.1",
        "loss.walle.enabled=false"
    ]

# ──────── BUILD ALL TASKS ─────────────────────────────────────────
print("Building focused ablation task list with 3 random seeds per experiment...")

all_tasks = []
task_id = 0

for category, configs in ABLATION_CONFIGS.items():
    for name, overrides in configs.items():
        for seed_idx, seed in enumerate(RANDOM_SEEDS):
            exp_name = f"{category}_{name}_seed{seed}"
            
            # Start with base config for this seed
            task_overrides = get_base_overrides(seed)
            
            # Add experiment-specific overrides
            for key, value in overrides.items():
                task_overrides.append(f"{key}={value}")
            
            # Add experiment metadata
            task_overrides.extend([
                f"run_id=epiformer_ablation_{task_id}_{category}_{name}_seed{seed}",
                f"wandb.notes=epiformer_ablation_{category}_{name}_seed{seed}",
                f"wandb.tags=['epiformer_ablation', '{category}', '{name}', 'seed{seed}']"
            ])
            
            cmd = [PYTHON, "trainer.py"] + task_overrides
            log_file = os.path.join(logs_dir, f"ablation_{task_id}_{category}_{name}_seed{seed}.log")
            
            task_info = {
                'task_id': task_id,
                'category': category,
                'name': name,
                'experiment': exp_name,
                'seed': seed,
                'seed_idx': seed_idx
            }
            
            all_tasks.append((cmd, log_file, task_info))
            task_id += 1

total_experiments = len(ABLATION_CONFIGS) * sum(len(configs) for configs in ABLATION_CONFIGS.values()) * len(RANDOM_SEEDS)
print(f"Total focused ablation experiments: {len(all_tasks)} ({total_experiments} expected: 37 configs × 3 seeds = 111 experiments)")

# ──────── EXECUTION ────────────────────────────────────────────────
records = []
gpu_count = torch.cuda.device_count() if (args.multi_gpu and torch.cuda.is_available()) else 0

if not args.multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Sequential mode: pinned to GPU {args.gpu_id}")

def parse_metrics_from_lines(lines):
    """Parse test metrics from trainer output - same as comprehensive ablation"""
    in_test = False
    metas = {}
    for line in lines:
        if line.strip().startswith("===") and "Test" in line:
            in_test = True
            continue
        if in_test and ":" in line and not line.strip().startswith("==="):
            try:
                k, v = line.split(":", 1)
                metas[k.strip()] = float(v.strip())
            except ValueError:
                continue
    return metas

# ──────── EXECUTE TASKS ────────────────────────────────────────────
if args.multi_gpu and gpu_count > 0:
    print(f"Multi-GPU mode: running up to {gpu_count} jobs in parallel")
    tasks = []
    for idx, (cmd, log_file, task_info) in enumerate(all_tasks):
        gpu_id = idx % gpu_count
        e = env.copy()
        e["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        tasks.append((cmd, log_file, task_info, e))

    active = []
    while tasks or active:
        while len(active) < gpu_count and tasks:
            cmd, log_file, task_info, e = tasks.pop(0)
            print(f"▶︎ Launch Task {task_info['task_id']}: {task_info['experiment']}")
            
            f = open(log_file, "w")
            p = subprocess.Popen(cmd, cwd=str(CODE_DIR), env=e, stdout=f, stderr=subprocess.STDOUT, text=True)
            active.append((p, log_file, f, cmd, task_info))
        
        time.sleep(2)
        for (p, log_file, f, cmd, task_info) in list(active):
            if p.poll() is not None:
                f.close()
                active.remove((p, log_file, f, cmd, task_info))
                
                try:
                    with open(log_file, 'r') as file:
                        lines = file.readlines()
                    metrics = parse_metrics_from_lines(lines)
                    
                    if metrics:
                        rec = {
                            "experiment": task_info['experiment'],
                            "category": task_info['category'],
                            "name": task_info['name'],
                            "seed": task_info['seed'],
                            "seed_idx": task_info['seed_idx'],
                            **metrics
                        }
                        records.append(rec)
                        print(f"✅ Task {task_info['task_id']} ({task_info['experiment']}) completed")
                        pd.DataFrame(records).to_csv(raw_path, index=False)
                    else:
                        print(f"❌ Task {task_info['task_id']} failed to parse metrics")
                except Exception as e:
                    print(f"❌ Task {task_info['task_id']} failed: {e}")

else:
    print("Sequential execution of all tasks")
    for cmd, log_file, task_info in all_tasks:
        print(f"▶︎ Running Task {task_info['task_id']}: {task_info['experiment']}")
        
        start = time.time()
        proc = subprocess.run(cmd, cwd=str(CODE_DIR), env=env, capture_output=True, text=True)
        duration = time.time() - start
        
        if proc.returncode != 0:
            print(f"   ❌ Task {task_info['task_id']} failed: {proc.stderr.splitlines()[-1] if proc.stderr.splitlines() else 'Unknown error'}")
            continue
            
        metrics = parse_metrics_from_lines(proc.stdout.splitlines())
        if metrics:
            rec = {
                "experiment": task_info['experiment'],
                "category": task_info['category'], 
                "name": task_info['name'],
                "seed": task_info['seed'],
                "seed_idx": task_info['seed_idx'],
                "duration_s": duration,
                **metrics
            }
            records.append(rec)
            print(f"✅ Task {task_info['task_id']} ({task_info['experiment']}) completed")
            pd.DataFrame(records).to_csv(raw_path, index=False)
        else:
            print(f"❌ Task {task_info['task_id']} failed to parse metrics")

# ──────── SAVE FINAL RESULTS ───────────────────────────────────────
print(f"\nProcessing results from {len(records)} successful experiments...")

df = pd.DataFrame(records)
if not df.empty:
    print(f"Available columns in results: {df.columns.tolist()}")
    
    # Look for epitope metrics (similar to comprehensive ablation)
    possible_metrics = [
        "epitope_mcc", "epitope_auc", "epitope_auprc", "epitope_precision", "epitope_recall", "epitope_f1",
        "paratope_mcc", "paratope_auc", "paratope_auprc", "paratope_precision", "paratope_recall", "paratope_f1",
        "mcc", "auc", "auroc", "auprc", "precision", "recall", "f1"  # Fallback names
    ]
    available_metrics = [col for col in possible_metrics if col in df.columns]
    
    if available_metrics:
        # Create aggregated results with mean and std for each experiment across seeds
        agg_results = []
        for (category, name), group in df.groupby(['category', 'name']):
            if len(group) == len(RANDOM_SEEDS):  # Only include experiments with all seeds
                agg_row = {
                    'category': category,
                    'name': name,
                    'experiment': f"{category}_{name}"
                }
                
                # Calculate mean and std for each metric
                for metric in available_metrics:
                    if metric in group.columns:
                        agg_row[f'{metric}_mean'] = group[metric].mean()
                        agg_row[f'{metric}_std'] = group[metric].std()
                        agg_row[f'{metric}_min'] = group[metric].min()
                        agg_row[f'{metric}_max'] = group[metric].max()
                
                agg_results.append(agg_row)
        
        # Create final results DataFrame
        result_df = pd.DataFrame(agg_results)
        
        if not result_df.empty:
            # Format metrics to match your table precision
            for col in result_df.columns:
                if any(metric in col for metric in available_metrics):
                    result_df[col] = result_df[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
            
            # Sort by category then name for organized output
            result_df = result_df.sort_values(['category', 'name'])
            
            # Save results
            result_df.to_csv(agg_path, index=False)
            print(f"✅ Aggregated results saved to {agg_path}")
            
            # Print summary grouped by category
            print(f"\n{'='*80}")
            print("FOCUSED ABLATION RESULTS SUMMARY (Mean ± Std across 3 seeds)")
            print(f"{'='*80}")
            
            for category in result_df['category'].unique():
                cat_df = result_df[result_df['category'] == category]
                print(f"\n{category.upper()}:")
                # Show only mean values for cleaner output
                mean_cols = [col for col in cat_df.columns if '_mean' in col]
                print(cat_df[['name'] + mean_cols].to_string(index=False))
        
        else:
            print("⚠️ No complete experiment runs (all 3 seeds) to aggregate")
            df.to_csv(agg_path, index=False)
            print(f"✅ Raw results saved to {agg_path}")
    
    else:
        print("⚠️ No recognized metrics found in results")
        df.to_csv(agg_path, index=False)
        print(f"✅ Raw results saved to {agg_path}")
        print(f"Available columns were: {df.columns.tolist()}")
else:
    print("⚠️ No successful runs to process.")

print(f"\n{'='*80}")
print("FOCUSED ABLATION STUDY WITH 3 SEEDS COMPLETED")
print(f"{'='*80}")
print(f"✅ Successful experiments: {len(records)}/{len(all_tasks)}")
print(f"📊 Results: {agg_path}")
print(f"📁 Logs: {logs_dir}")



"""
nohup python scripts/ablation/epiformer_ablation_random_seeds.py --multi_gpu --epochs 130 --batch_size 8 > logs/ablation_output_random_seeds.log 2>&1 &
"""