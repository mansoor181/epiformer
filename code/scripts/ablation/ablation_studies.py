#!/usr/bin/env python3
"""
Focused Ablation Study Script - 37 Specific Experiments
Runs only the selected ablation configurations from the results table.
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
        'egnn': {'model.ag_encoder.resmp_type': 'egnn', 'model.ab_encoder.resmp_type': 'egnn'},
        'regnn': {'model.ag_encoder.resmp_type': 'regnn', 'model.ab_encoder.resmp_type': 'regnn'},
        'gcn': {'model.ag_encoder.resmp_type': 'gcn', 'model.ab_encoder.resmp_type': 'gcn'},
        'rgcn': {'model.ag_encoder.resmp_type': 'rgcn', 'model.ab_encoder.resmp_type': 'rgcn'},
        'gat': {'model.ag_encoder.resmp_type': 'gat', 'model.ab_encoder.resmp_type': 'gat'},
        'gin': {'model.ag_encoder.resmp_type': 'gin', 'model.ab_encoder.resmp_type': 'gin'}
    },
    
    # Data split ablations (3 experiments)
    'split': {
        'random': {'dataset.split.method': 'random'},
        'epitope_ratio': {'dataset.split.method': 'epitope_ratio'},
        'epitope_group': {'dataset.split.method': 'epitope_group'}
    },
    
    # Loss function ablations (8 experiments) - sparsity = count_regularizer
    'loss': {
        'bce': {
            'loss.node_prediction.count_regularizer_enabled': 'false',  # Disable sparsity
            'loss.node_prediction.dice_enabled': 'false',    # Disable dice
            'loss.contrastive.enabled': 'false',             # Disable infonce  
            'loss.edge_prediction.enabled': 'false'          # Disable edge
        },
        'bce_sparsity': {
            'loss.count_regularizer.epitope_weight': '0.1122', # Enable sparsity (from base)
            'loss.node_prediction.dice_enabled': 'false',
            'loss.contrastive.enabled': 'false', 
            'loss.edge_prediction.enabled': 'false'
        },
        'bce_dice': {
            'loss.node_prediction.count_regularizer_enabled': 'false',  # Disable sparsity
            'loss.node_prediction.dice_enabled': 'true',       # Enable dice (from base)
            'loss.contrastive.enabled': 'false',
            'loss.edge_prediction.enabled': 'false'
        },
        'bce_sparsity_dice': {  # This is the base config
            'loss.node_prediction.count_regularizer_enabled': 'true',  # enable sparsity
            'loss.node_prediction.dice_enabled': 'true',
            'loss.contrastive.enabled': 'false',
            'loss.edge_prediction.enabled': 'false'
        },
        'bce_sparsity_dice_infonce': {
            'loss.node_prediction.count_regularizer_enabled': 'true',  # enable sparsity
            'loss.node_prediction.dice_enabled': 'true', 
            'loss.contrastive.enabled': 'true',             # Enable infonce
            'loss.contrastive.name': 'infonce',
            'loss.edge_prediction.enabled': 'false'
        },
        'bce_sparsity_dice_edge': {
            'loss.node_prediction.count_regularizer_enabled': 'true',  # enable sparsity
            'loss.node_prediction.dice_enabled': 'true',
            'loss.contrastive.enabled': 'false',
            'loss.edge_prediction.enabled': 'true'         # Enable edge
        },
        'bce_sparsity_dice_infonce_edge': {
            'loss.node_prediction.count_regularizer_enabled': 'true',  # enable sparsity
            'loss.node_prediction.dice_enabled': 'true',
            'loss.contrastive.enabled': 'true',
            'loss.contrastive.name': 'infonce',
            'loss.edge_prediction.enabled': 'true'
        },
        'bce_edge_infonce': {
            'loss.node_prediction.count_regularizer_enabled': 'false',  # disable sparsity
            'loss.node_prediction.dice_enabled': 'false',
            'loss.contrastive.enabled': 'true',
            'loss.contrastive.name': 'infonce',
            'loss.edge_prediction.enabled': 'true'
        }
    },
    
    # Decoder ablations (3 experiments)
    'decoder': {
        'dot_product': {'model.decoder.type': 'dot_product'},  # Base config
        'cross_attn': {'model.decoder.type': 'cross_attention'},
        'dual': {'model.decoder.type': 'dual'}
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
        'ag_egnn_ab_regnn': {'model.ag_encoder.resmp_type': 'egnn', 'model.ab_encoder.resmp_type': 'regnn'},
        'ag_regnn_ab_egnn': {'model.ag_encoder.resmp_type': 'regnn', 'model.ab_encoder.resmp_type': 'egnn'}
    },
    
    # Layer count ablations (3 experiments)
    'layers': {
        'ag6_ab3': {'model.ag_encoder.residue_layers': '6', 'model.ab_encoder.residue_layers': '3'},
        'ag5_ab3': {'model.ag_encoder.residue_layers': '5', 'model.ab_encoder.residue_layers': '3'},
        'ag4_ab3': {'model.ag_encoder.residue_layers': '4', 'model.ab_encoder.residue_layers': '3'}
    },
    
    # Graph type ablations (4 experiments)
    'graph_type': {
        'raad_egnn': {'model.graph_type': 'raad', 'model.ag_encoder.resmp_type': 'egnn', 'model.ab_encoder.resmp_type': 'egnn'},
        'gearnet_egnn': {'model.graph_type': 'gearnet', 'model.ag_encoder.resmp_type': 'egnn', 'model.ab_encoder.resmp_type': 'egnn'},
        'base_egnn': {'model.graph_type': 'base', 'model.ag_encoder.resmp_type': 'egnn', 'model.ab_encoder.resmp_type': 'egnn'},
        'base_gcn': {'model.graph_type': 'base', 'model.ag_encoder.resmp_type': 'gcn', 'model.ab_encoder.resmp_type': 'gcn'}
    }
}

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
parser = argparse.ArgumentParser(description="Run focused ablation experiments - 37 specific configs")
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
def get_base_overrides():
    """Complete base configuration from test.sh - ALL parameters"""
    return [
        "mode=val",
        f"gpu_id={args.gpu_id}",
        "wandb.project=m3epi_v3",
        "dataset.split.method=epitope_ratio",
        "dataset.tensor=hierarchical_dataset.pkl",
        f"hparams.train.num_epochs={args.epochs}",
        f"hparams.train.batch_size={args.batch_size}",
        "hparams.pretrain.num_epochs=5",  # From test.sh default
        "hparams.pretrain.lr=0.0002",
        "hparams.train.learning_rate=0.0004",
        "hparams.train.weight_decay=0.0001013",
        "hparams.train.kfolds=2",
        "hparams.train.regularization.use_l2_reg=false",
        "num_threads=3",
        "resume=false",
        
        # Model configuration - complete from test.sh
        "model.enable_pretraining=false",
        "model.graph_type=raad",
        "model.ab_encoder.resmp_enabled=true",
        "model.ab_encoder.resmp_type=egnn",
        "model.ab_encoder.edgemp_enabled=false",
        "model.ab_encoder.edgemp_type=egnn",
        "model.ab_encoder.atommp_enabled=false",
        "model.ab_encoder.atom_mp_type=egnn",
        "model.ab_encoder.residue_layers=4",
        "model.ab_encoder.atom_layers=4",
        "model.ab_encoder.edge_layers=3",
        "model.ab_encoder.atom2res_inj=ca_only",
        "model.ab_encoder.feature_fusion_type=gated",
        "model.ag_encoder.resmp_enabled=true",
        "model.ag_encoder.resmp_type=egnn",
        "model.ag_encoder.edgemp_enabled=false",
        "model.ag_encoder.edgemp_type=egnn",  
        "model.ag_encoder.atommp_enabled=false",
        "model.ag_encoder.atom_mp_type=et",  
        "model.ag_encoder.residue_layers=4",
        "model.ag_encoder.atom_layers=4",
        "model.ag_encoder.edge_layers=3",
        "model.ag_encoder.atom2res_inj=ca_only",
        "model.ag_encoder.feature_fusion_type=concat",
        
        # Dropout rates - complete from test.sh
        "model.dropout_rates.atom_mp=0.25",
        "model.dropout_rates.edge_mp=0.25",
        "model.dropout_rates.res_mp=0.2572",
        "model.dropout_rates.decoder=0.0247",
        "model.dropout_rates.projections=0.0262",
        "model.dropout=0.2395",
        
        # Decoder configuration - complete from test.sh
        "model.decoder.type=dot_product",
        "model.decoder.d_k=64",
        "model.decoder.d_ff=256",
        "model.decoder.d_model=128",
        "model.decoder.n_heads=8",
        "model.decoder.decoder_layers=3",
        "model.decoder.sampling_strat=top_k_mean_2",
        "model.decoder.predict_distances=false",
        
        # Model thresholds and settings - complete from test.sh
        "model.epi_threshold=0.3",
        "model.para_threshold=0.3",
        "model.use_layer_norm=true",
        
        # Callbacks - complete from test.sh
        "callbacks.early_stopping.patience=10",
        "callbacks.checkpoint_interval=2",
        
        # Loss configuration - complete from test.sh
        "loss.node_prediction.enabled=true",
        "loss.node_prediction.weight=0.9902",
        "loss.node_prediction.name=bce",
        "loss.node_prediction.task=epi_only",
        "loss.node_prediction.bce_weight=4.5030",
        "loss.node_prediction.dice_weight=0.6610",
        "loss.node_prediction.smoothness_weight=0.3521",
        "loss.node_prediction.consistency_weight=0.5807",
        "loss.node_prediction.dice_enabled=true",
        "loss.node_prediction.count_regularizer_enabled=true",
        "loss.node_prediction.smoothness_enabled=false",
        "loss.node_prediction.edge_node_consistency_enabled=false",
        "loss.node_prediction.epi_pos_weight=18",
        "loss.node_prediction.para_pos_weight=3",
        
        # Count regularizer (sparsity) - complete from test.sh
        "loss.count_regularizer.per_graph_matching=true",
        "loss.count_regularizer.epitope_weight=0.1122",
        "loss.count_regularizer.paratope_weight=0.2258",
        "loss.count_regularizer.dataset_prior=false",
        "loss.count_regularizer.epitope_prior_mean=14.6",
        "loss.count_regularizer.prior_weight=0.05",
        "loss.count_regularizer.anneal_epochs=10",
        
        # Other loss settings - complete from test.sh
        "loss.label_smoothing=0.1",
        "loss.class_balance.beta=0.9999",
        
        # Edge prediction loss - complete from test.sh
        "loss.edge_prediction.enabled=false",
        "loss.edge_prediction.weight=0.3",
        "loss.edge_prediction.pos_weight=28.2386",
        
        # Contrastive loss - complete from test.sh
        "loss.contrastive.enabled=false",
        "loss.contrastive.name=gwnce",
        "loss.contrastive.weight=0.4",
        "loss.contrastive.temperature=0.395",
        "loss.contrastive.inter_weight=0.456",
        "loss.contrastive.intra_weight=0.456",
        
        # GWNCE loss - complete from test.sh
        "loss.gwnce.weight=0.1",
        "loss.gwnce.cut_way=2",
        "loss.gwnce.cut_rate=0.5",
        
        # Force loss - complete from test.sh
        "loss.force.enabled=false",
        "loss.force.weight=0.01",
        "loss.force.bond_weight=1",
        "loss.force.angle_weight=0.5",
        "loss.force.smooth_alpha=1",
        "loss.force.smooth_weight=0.1",
        "loss.force.bond_tolerance=0.1",
        "loss.force.angle_tolerance=0.1",
        
        # WALLE loss - complete from test.sh
        "loss.walle.enabled=false",
        
        # Fixed seed for reproducibility
        f"seed=42"
    ]

# ──────── BUILD ALL TASKS ─────────────────────────────────────────
print("Building focused ablation task list (37 experiments)...")

all_tasks = []
task_id = 0

for category, configs in ABLATION_CONFIGS.items():
    for name, overrides in configs.items():
        exp_name = f"{category}_{name}"
        
        # Start with base config
        task_overrides = get_base_overrides()
        
        # Add experiment-specific overrides
        for key, value in overrides.items():
            task_overrides.append(f"{key}={value}")
        
        # Add experiment metadata
        task_overrides.extend([
            f"run_id=focused_ablation_{task_id}_{exp_name}",
            f"wandb.notes=focused_ablation_{exp_name}",
            f"wandb.tags=['focused_ablation', '{category}', '{name}']"
        ])
        
        cmd = [PYTHON, "trainer.py"] + task_overrides
        log_file = os.path.join(logs_dir, f"ablation_{task_id}_{exp_name}.log")
        
        task_info = {
            'task_id': task_id,
            'category': category,
            'name': name,
            'experiment': exp_name
        }
        
        all_tasks.append((cmd, log_file, task_info))
        task_id += 1

print(f"Total focused ablation experiments: {len(all_tasks)} (37 expected)")
assert len(all_tasks) == 36, f"Expected 37 experiments, got {len(all_tasks)}"

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
        # Create final results with rounded metrics
        final_cols = ['experiment', 'category', 'name'] + available_metrics
        result_df = df[final_cols].copy()
        
        # Format metrics to match your table precision
        for metric in available_metrics:
            result_df[metric] = result_df[metric].apply(lambda x: f"{x:.6f}")
        
        # Sort by category then name for organized output
        result_df = result_df.sort_values(['category', 'name'])
        
        # Save results
        result_df.to_csv(agg_path, index=False)
        print(f"✅ Results saved to {agg_path}")
        
        # Print summary grouped by category
        print(f"\n{'='*80}")
        print("FOCUSED ABLATION RESULTS SUMMARY")
        print(f"{'='*80}")
        
        for category in result_df['category'].unique():
            cat_df = result_df[result_df['category'] == category]
            print(f"\n{category.upper()}:")
            print(cat_df[['name'] + available_metrics].to_string(index=False))
    
    else:
        print("⚠️ No recognized metrics found in results")
        df.to_csv(agg_path, index=False)
        print(f"✅ Raw results saved to {agg_path}")
        print(f"Available columns were: {df.columns.tolist()}")
else:
    print("⚠️ No successful runs to process.")

print(f"\n{'='*80}")
print("FOCUSED ABLATION STUDY COMPLETED")
print(f"{'='*80}")
print(f"✅ Successful experiments: {len(records)}/{len(all_tasks)}")
print(f"📊 Results: {agg_path}")
print(f"📁 Logs: {logs_dir}")



"""
nohup python scripts/ablation/ablation_studies.py --multi_gpu --epochs 130 --batch_size 8 > logs/ablation_output.log 2>&1 &

"""