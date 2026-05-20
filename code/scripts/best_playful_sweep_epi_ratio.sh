#!/bin/bash
# Best hyperparameters from playful-sweep for epitope_ratio split
# Generated individual experiment script with embedded parameters for EpiFormer model

# cd into code dir and run:
# ./scripts/best_playful_sweep_epi_ratio.sh --gpu_id 0 --batch_size 8 --epochs 130 --server local

usage() {
    echo "Usage: $0 --gpu_id <gpu_id> --batch_size <batch_size> --epochs <epochs> --server <server_name> [--pretrain_epochs <pretrain_epochs>] [--wandb]"
    echo ""
    echo "Arguments:"
    echo "  --gpu_id         GPU ID to use (required)"
    echo "  --batch_size     Batch size for training (required)"
    echo "  --epochs         Number of training epochs (required)"
    echo "  --server         Server name (amai, dice, etc.) (required)"
    echo "  --pretrain_epochs Number of pretraining epochs (optional, default: 5)"
    echo "  --wandb          Enable Weights & Biases logging (optional, disabled by default)"
    echo ""
    echo "Example:"
    echo "  $0 --gpu_id 0 --batch_size 8 --epochs 130 --server amai"
    echo "  $0 --gpu_id 0 --batch_size 8 --epochs 130 --server amai --wandb"
    echo ""
    echo "Note: EpiFormer model parameters are embedded in this script."
    exit 1
}

# Parse command line arguments
use_wandb=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu_id)
            gpu_id="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        --server)
            server="$2"
            shift 2
            ;;
        --pretrain_epochs)
            pretrain_epochs="$2"
            shift 2
            ;;
        --wandb)
            use_wandb=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            ;;
    esac
done

# Check if all required parameters are provided
if [ -z "$gpu_id" ] || [ -z "$batch_size" ] || [ -z "$epochs" ] || [ -z "$server" ]; then
    echo "Error: All parameters (gpu_id, batch_size, epochs, server) are required."
    usage
fi

# Set default value for pretrain_epochs if not provided
pretrain_epochs=${pretrain_epochs:-5}

echo "- Server: $server | GPU: $gpu_id | Batch: $batch_size | Epochs: $epochs | Pretrain Epochs: $pretrain_epochs"
echo "- Model: EpiFormer (playful-sweep best config for epitope_ratio split)"

# Set logging method
if [ "$use_wandb" = true ]; then
    logging_method="wandb"
    echo "- Logging: Weights & Biases enabled"
else
    logging_method="none"
    echo "- Logging: Console only (use --wandb to enable W&B)"
fi

# Create run_id with timestamp
run_id="playful_sweep_epi_ratio_$(date +%Y%m%d-%H%M%S)"

echo "- Run ID: $run_id"

# Create logs directory if it doesn't exist
mkdir -p logs

# Execute the experiment with EpiFormer model
echo "- Starting EpiFormer training..."

nohup python trainer.py \
    mode=val \
    seed=42 \
    model.name=epiformer \
    gpu_id="$gpu_id" \
    logging_method="$logging_method" \
    dataset.split.method="epitope_ratio" \
    dataset.graph_type="raad-plm" \
    dataset.plm_type="esm2_650m" \
    dataset.graph_num_relations=4 \
    hparams.train.num_epochs="$epochs" \
    hparams.train.batch_size="$batch_size" \
    hparams.pretrain.num_epochs="$pretrain_epochs" \
    hparams.pretrain.lr=0.00005 \
    hparams.train.learning_rate=0.00009095 \
    hparams.train.weight_decay=0.00001179 \
    hparams.train.kfolds=2 \
    hparams.train.regularization.use_l2_reg=false \
    hparams.train.scheduler="reduce_lr_on_plateau" \
    run_id="$run_id" \
    num_threads=3 \
    resume=false \
    model.enable_pretraining=false \
    model.epiformer.ag_resmp_type="egnn" \
    model.epiformer.ab_resmp_type="egnn" \
    model.epiformer.residue_layers=4 \
    model.epiformer.residue_dim=128 \
    model.epiformer.residue_hidden_dim=128 \
    model.epiformer.plm_dim=128 \
    model.epiformer.n_heads=8 \
    model.epiformer.use_layer_norm=true \
    model.epiformer.use_pair_repr=false \
    model.epiformer.use_gradient_checkpointing=false \
    model.epiformer.ag_feature_fusion_type="concat" \
    model.epiformer.ab_feature_fusion_type="gated" \
    model.epiformer.activation="silu" \
    model.epiformer.dropout=0.1324459 \
    model.dropout_rates.decoder=0.072024237 \
    model.dropout_rates.projections=0.1 \
    model.decoder.type="cross_attention" \
    model.decoder.num_rbf=16 \
    model.decoder.d_k=64 \
    model.decoder.d_ff=128 \
    model.decoder.d_model=128 \
    model.decoder.n_heads=8 \
    model.decoder.decoder_layers=2 \
    model.decoder.sampling_strat="top_k_mean_2" \
    model.decoder.predict_distances=true \
    model.epi_threshold=0.3 \
    model.para_threshold=0.3 \
    model.use_layer_norm=true \
    callbacks.early_stopping.patience=10 \
    callbacks.checkpoint_interval=2 \
    loss.multi_task=true \
    loss.label_smoothing=0.1 \
    loss.class_balance.beta=0.9999 \
    loss.node_prediction.enabled=true \
    loss.node_prediction.weight=0.481570678 \
    loss.node_prediction.name="bce" \
    loss.node_prediction.task="epi_only" \
    loss.node_prediction.bce_weight=9.324872 \
    loss.node_prediction.dice_weight=2.2965577 \
    loss.node_prediction.smoothness_weight=0.01 \
    loss.node_prediction.consistency_weight=0.1 \
    loss.node_prediction.dice_enabled=true \
    loss.node_prediction.count_regularizer_enabled=true \
    loss.node_prediction.smoothness_enabled=false \
    loss.node_prediction.edge_node_consistency_enabled=false \
    loss.node_prediction.epi_pos_weight=15.28555 \
    loss.node_prediction.para_pos_weight=3 \
    loss.count_regularizer.enabled=true \
    loss.count_regularizer.per_graph_matching=true \
    loss.count_regularizer.epitope_weight=0.306774 \
    loss.count_regularizer.paratope_weight=0.1 \
    loss.count_regularizer.dataset_prior=false \
    loss.count_regularizer.epitope_prior_mean=14.6 \
    loss.count_regularizer.prior_weight=0.05 \
    loss.count_regularizer.anneal_epochs=10 \
    loss.edge_prediction.enabled=true \
    loss.edge_prediction.weight=1.0 \
    loss.edge_prediction.pos_weight=58.7076536 \
    loss.edge_count_regularizer.enabled=false \
    loss.edge_count_regularizer.weight=0.1 \
    loss.auxiliary_distance.enabled=true \
    loss.auxiliary_distance.weight=0.0513977 \
    loss.auxiliary_distance.distance_weighting=true \
    loss.auxiliary_distance.class_balancing=true \
    loss.auxiliary_distance.max_distance=32.0 \
    loss.contrastive.enabled=false \
    loss.walle.enabled=false \
    > "logs/${run_id}_output.log" 2>&1 &

pid=$!
echo "- EpiFormer experiment started successfully (PID: $pid)"
echo "- Monitor with: tail -f logs/${run_id}_output.log"
echo "- Kill with: kill $pid"
