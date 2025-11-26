#!/bin/bash
#SBATCH --nodelist=csardas
#SBATCH --job-name="extract_video_mae_features"
#SBATCH --output=/home/bickicdz/projects/Causal_VAE/slurm/logs_causal/output_%A_%a.txt
#SBATCH --error=/home/bickicdz/projects/Causal_VAE/slurm/logs_causal/error_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32        # match to your --num_workers
#SBATCH --gres=gpu:1
#SBATCH --mem=320G
#SBATCH --time=32:00:00
#SBATCH --chdir=/scratch/users/bickici/projects/causal2/causal   # start here (no permission errors)


git pull

source /scratch/users/bickici/environments/TIM/bin/activate

mkdir /scratch/users/bickici/projects/causal2/causal/checkpoint/
mkdir /scratch/users/bickici/projects/causal2/causal/checkpoint/whl
mkdir /scratch/users/bickici/projects/causal2/causal/save_max
mkdir /scratch/users/bickici/projects/causal2/causal/save_max/ek100




rm -r /scratch/users/bickici/projects/causal2/causal/checkpoint/whl
mkdir /scratch/users/bickici/projects/causal2/causal/checkpoint/whl



# Execute your training script
data="ek100"
vae_latent_dim=4
alpha=1
beta=0.1
gamma=0.1
delta=1

action_dim=3806 #106
verb_dim=1280 #19
noun_dim=1280 #53
epoch=3000
lr=3e-4
optimizer="adamw"  # Options: "adam", "adamw", or "sgd"



python3 main_lavila_causal.py \
--num_thread_reader=4 \
--pin_memory \
--cudnn_benchmark=1 \
--checkpoint_dir=whl \
--vae_latent_dim ${vae_latent_dim} \
--batch_size=32 \
--batch_size_val=32 \
--seed=0 \
--evaluate \
--dataset ${data} \
--action_dim ${action_dim} \
--verb_dim ${verb_dim} \
--noun_dim ${noun_dim} \
--epochs ${epoch} \
--lr ${lr} \
--optimizer ${optimizer} \
--beta ${beta} \
--gamma ${gamma} \
--delta ${delta} \
--gpu 0 \
--resume \
--verb_feat_train /scratch/users/bickici/data/EK100/vjepa_features/vjepa2_huge_ek100_cls_verb/ek100_cls_train_feat.pt \
--verb_feat_val /scratch/users/bickici/data/EK100/vjepa_features/vjepa2_huge_ek100_cls_verb/ek100_cls_test_feat.pt \
--noun_feat_train /scratch/users/bickici/data/EK100/vjepa_features/vjepa2_huge_ek100_cls_noun/ek100_cls_train_feat.pt \
--noun_feat_val /scratch/users/bickici/data/EK100/vjepa_features/vjepa2_huge_ek100_cls_noun/ek100_cls_test_feat.pt \
--action_feat_train /scratch/users/bickici/data/TIM/action_tokens_train/features/epic_train_feat.pt \
--action_feat_val /scratch/users/bickici/data/TIM/action_tokens_val/features/epic_val_feat.pt \
--temporal_pooling adaptive_max \
--feature_norm l2 \
--fusion_gate_clamp 0.5  # stride | adaptive_avg | adaptive_max, default stride

echo "Job finished at: $(date)"
