import argparse

def get_args(description='whl'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--alpha',
                        type=float,
                        default=0.0075,
                        help='weight for recon loss in causal vae')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.0075,
                        help='weight for sparse loss in causal vae')
    parser.add_argument('--beta',
                        type=float,
                        default=0.0025,
                        help='weight for kld in causal vae')
    parser.add_argument('--delta',
                        type=float,
                        default=0.0075,
                        help='weight for structure loss in causal vae')
    parser.add_argument('--resume_epoch',
                        type=int,
                        default=0,
                        help='resume epoch')
    parser.add_argument('--eps',
                        type=float,
                        default=0.0001,
                        help='eps for rhvae')                   
    parser.add_argument('--temp',
                        type=float,
                        default=1.5,
                        help='temperature for rhvae')
    parser.add_argument('--vae_type',
                        type=str,
                        default='rhvae',
                        help='type of vae: rhvae, rhvae_dynamics, vanilla')
    parser.add_argument('--inference_type',
                        type=str,
                        default='gt',
                        help='inference type, use gt or normal')
    parser.add_argument('--setting',
                        type=str,
                        default='',
                        help='use pdpp or kepp data curation setting')
    parser.add_argument('--vae_latent_dim',
                        type=int,
                        default=128,
                        help='type of vae: rhvae, rhvae_dynamics, vanilla')
    parser.add_argument('--vae_use_act',
                        type=str,   
                        default='act+img',                     
                        help='True: action+observation, False: observation')
    parser.add_argument('--a0at_inference',
                        type=bool,
                        default=False,
                        help='if use a0 at for inference')
    parser.add_argument('--vae_path',
                        type=str,
                        default='',
                        help='path to vae')
    parser.add_argument('--act_emb_path',
                        type=str,
                        default='dataset/coin/steps_info.pickle',
                        help='action embedding path')
    parser.add_argument('--geo_type',
                        type=str,
                        default='z',
                        help='z: use embedding points, g: use riemanian metric')
    parser.add_argument('--num_rbf_center',
                        type=int,
                        default=10,
                        help='num of rbf center')
    parser.add_argument('--checkpoint_mlp',
                        type=str,
                        default='',
                        help='checkpoint path for task prediction model')
    parser.add_argument('--checkpoint_diff',
                        type=str,
                        default='',
                        help='checkpoint path for diffusion model')                                                                                    
    parser.add_argument('--mask_type',
                        type=str,
                        default='multi_add', # single_add, multi_add
                        help='action embedding mask type')        
    parser.add_argument('--attn',
                        type=str,
                        default='attention', # single_add, multi_add
                        help='WithAttention: unet with attn. NoAttention: unet without attention.')
    parser.add_argument('--infer_avg_mask',
                        type=bool,
                        default=False,
                        help='if use average mask for inference')                    
    parser.add_argument('--use_cls_mask',
                        type=bool,
                        default=False,
                        help='if use class label in diffusion mask')                                                              
    parser.add_argument('--checkpoint_root',
                        type=str,
                        default='checkpoint',
                        help='checkpoint dir root')
    parser.add_argument('--log_root',
                        type=str,
                        default='log',
                        help='log dir root')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='',
                        help='checkpoint model folder')
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help='opt algorithm')
    parser.add_argument('--num_thread_reader',
                        type=int,
                        default=40,
                        help='')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256, # 256
                        help='batch size')
    parser.add_argument('--batch_size_val',
                        type=int,
                        default=1024, # 1024
                        help='batch size eval')
    parser.add_argument('--pretrain_cnn_path',
                        type=str,
                        default='',
                        help='')
    parser.add_argument('--momemtum',
                        type=float,
                        default=0.9,
                        help='SGD momemtum')
    parser.add_argument('--log_freq',
                        type=int,
                        default=500,
                        help='how many steps do we log once')
    parser.add_argument('--save_freq',
                        type=int,
                        default=1,
                        help='how many epochs do we save once')
    parser.add_argument('--gradient_accumulate_every',
                        type=int,
                        default=1,
                        help='accumulation_steps')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.995,
                        help='')
    parser.add_argument('--step_start_ema',
                        type=int,
                        default=400,
                        help='')
    parser.add_argument('--update_ema_every',
                        type=int,
                        default=10,
                        help='')
    parser.add_argument('--crop_only',
                        type=int,
                        default=1,
                        help='random seed')
    parser.add_argument('--centercrop',
                        type=int,
                        default=0,
                        help='random seed')
    parser.add_argument('--random_flip',
                        type=int,
                        default=1,
                        help='random seed')
    parser.add_argument('--verbose',
                        type=int,
                        default=1,
                        help='')
    parser.add_argument('--fps',
                        type=int,
                        default=1,
                        help='')
    parser.add_argument('--cudnn_benchmark',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument('--horizon',
                        type=int,
                        default=3,
                        help='')
    parser.add_argument('--dataset',
                        type=str,
                        default='coin',
                        help='dataset')
    parser.add_argument('--action_dim',
                        type=int,
                        default=778,
                        help='')
    parser.add_argument('--verb_dim',
                        type=int,
                        default=1536,
                        help='')
    parser.add_argument('--noun_dim',
                        type=int,
                        default=180,
                        help='')
    parser.add_argument('--feature_norm',
                        type=str,
                        default='l2',
                        choices=['none', 'l2', 'layernorm', 'l2_layernorm'],
                        help='normalization applied to TIM/V-JEPA feature streams before fusion')
    parser.add_argument('--feature_norm_eps',
                        type=float,
                        default=1e-6,
                        help='epsilon used for feature normalization')
    parser.add_argument('--fusion_alpha',
                        type=float,
                        default=10.0,
                        help='scaling for VAE branch logits when fusing with TIM logits')
    parser.add_argument('--fusion_gate_clamp',
                        type=float,
                        default=2.0,
                        help='log-scale clamp for learnable fusion gates (exp(-clamp)..exp(clamp))')
    parser.add_argument('--disable_fusion_std_scale',
                        action='store_true',
                        help='disable std-based rescaling between TIM logits and VAE logits')
    parser.add_argument('--temporal_target_len',
                        type=int,
                        default=16,
                        help='temporal length after pooling/striding for action features')
    parser.add_argument('--temporal_pooling',
                        type=str,
                        default='stride',
                        choices=['stride', 'adaptive_avg', 'adaptive_max'],
                        help='temporal reduction strategy for action features')
    parser.add_argument('--temporal_stride_step',
                        type=int,
                        default=0,
                        help='stride step for temporal reduction when using stride mode (0 auto-selects)')
    # Feature file paths (override dataloader defaults when provided)
    parser.add_argument('--verb_feat_train',
                        type=str,
                        default='',
                        help='path to verb features (.pt) for training')
    parser.add_argument('--verb_feat_val',
                        type=str,
                        default='',
                        help='path to verb features (.pt) for validation')
    parser.add_argument('--noun_feat_train',
                        type=str,
                        default='',
                        help='path to noun features (.pt) for training')
    parser.add_argument('--noun_feat_val',
                        type=str,
                        default='',
                        help='path to noun features (.pt) for validation')
    parser.add_argument('--action_feat_train',
                        type=str,
                        default='',
                        help='path to action/TIM features (.pt) for training')
    parser.add_argument('--action_feat_val',
                        type=str,
                        default='',
                        help='path to action/TIM features (.pt) for validation')
    parser.add_argument('--n_diffusion_steps',
                        type=int,
                        default=200,
                        help='')
    parser.add_argument('--n_train_steps',
                        type=int,
                        default=200,
                        help='training_steps_per_epoch')
    parser.add_argument('--root',
                        type=str,
                        default='',
                        help='root path of dataset crosstask')
    parser.add_argument('--json_path_train',
                        type=str,
                        default='dataset/coin/train_split_T4.json', 
                        help='path of the generated json file for train')
    parser.add_argument('--json_path_val',
                        type=str,
                        default='dataset/coin/coin_mlp_T4.json', 
                        help='path of the generated json file for val')
    parser.add_argument('--epochs', default=800, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('--pretrain_vae', dest='pretrain_vae', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                        help='use pin_memory')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-file', default='dist-file', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:4000', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=217, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()
    return args

