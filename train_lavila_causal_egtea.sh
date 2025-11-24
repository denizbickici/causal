data="egtea"
vae_latent_dim=4
alpha=1
beta=0.1
gamma=0.1
delta=1

action_dim=106
verb_dim=19
noun_dim=53
epoch=3000
lr=1e-4
optimizer="adamw"  # Options: "adam", "adamw", or "sgd"



python3 main_lavila_causal.py \
--num_thread_reader=8 \
--pin_memory \
--cudnn_benchmark=1 \
--checkpoint_dir=whl \
--vae_latent_dim ${vae_latent_dim} \
--batch_size=16 \
--batch_size_val=16 \
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

