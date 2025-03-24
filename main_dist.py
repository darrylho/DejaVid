from commons import *
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
from datasets import * # SSV2Dataset
from dba import calculate_mean_series
from dtw_model_dist import DTWModelDist
from engine_dist import train_one_epoch, validation

def get_args():
    parser = argparse.ArgumentParser(description='DTW Model')
    parser.add_argument('--num_kernel_workers', type=int, default=None, help='Number of workers for DTW kernel worker pool')
    parser.add_argument('--feat_batch_size', type=int, default=128, help='Batch size for DTW kernel computation')
    parser.add_argument('--sample_size_per_class', type=int, default=50, help='Number of samples per class for mean series calculation')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations for mean series calculation')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    parser.add_argument('--num_timesteps', type=int, default=64, help='Number of timesteps for DTW mean series')
    parser.add_argument('--num_feats_per_timestep', type=int, help='Number of features per timestep')
    parser.add_argument('--num_epochs', type=int, default=512, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=3e-2, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--shared_dir', type=str, default="", help='Shared directory for synchronization')
    parser.add_argument('--prefetch_factor', type=int, default=1400, help='Prefetch factor for data loader')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    
    parser.add_argument('--id2splits_pkl_path', type=str, help='Path to id2splits.pkl')
    parser.add_argument('--id2labels_pkl_path', type=str, help='Path to id2labels.pkl')

    parser.add_argument('--id2fname_pkl_path', type=str, help='Path to id2fname.pkl')
    parser.add_argument('--id2feats_pkl_path_root', type=str, help='Root of id2feats pkls')
    parser.add_argument('--mean_series_cached_filename_prefix', type=str, help='Filename prefix to mean series cache')

    parser.add_argument('--use_mm', type=int, help='Use MM or not')
    parser.add_argument('--first_order_penalty', type=float, default=0, help='First order penalty')
    parser.add_argument('--second_order_penalty', type=float, default=0, help='Second order penalty')

    parser.add_argument('--load_weights_path', type=str, default=None, help='Load model (feat_log_) weights')

    parser.add_argument('--validation', type=int, default=0, help='validation only')

    parser.add_argument('--use_fixed_path', type=int, default=0, help='use fixed path')

    parser.add_argument('--ms_lr_div', type=float, default=3, help='Learning rate adjustment for mean_series')
    
    parser.add_argument('--w_one_init', type=int, default=0, help='whether feat_weights are init to 1')


    parser.add_argument("--local-rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    # parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])

    
    parser.add_argument('--save_model', type=int, default=0, help='save model or not')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='num of warmup epochs')

    parser.add_argument('--b1_flw', type=float, default=0.9, help='beta1 for feat_log_weights')
    parser.add_argument('--b2_flw', type=float, default=0.999, help='beta2 for feat_log_weights')
    parser.add_argument('--b1_ms', type=float, default=0.9, help='beta1 for feat_log_weights')
    parser.add_argument('--b2_ms', type=float, default=0.999, help='beta2 for feat_log_weights')


    # XXX 
    parser.add_argument('--lmr', type=int, default=1, help='lmr')
    
    parser.add_argument('--output_pred_label_prefix', type=str, default=None, help='output model predictions and labels to path')

    return parser.parse_args()

def main(args):
    rank = int(os.environ['RANK'])  # Global rank of the process
    world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes
    if rank == 0:
        print(args, flush=True)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    id2splits = pickle.load(open(args.id2splits_pkl_path, 'rb'))
    id2labels = pickle.load(open(args.id2labels_pkl_path, 'rb'))
    id2fname = pickle.load(open(args.id2fname_pkl_path, 'rb'))

    train_idx = [idx for idx in id2fname.keys() if id2splits[idx] == 'train']
    val_idx = [idx for idx in id2fname.keys() if id2splits[idx] == 'val']
    test_idx = [idx for idx in id2fname.keys() if id2splits[idx] == 'test']

    for li in [train_idx, val_idx, test_idx]:
        li.sort(key=lambda i: id2fname[i])

    slice_idx = lambda li: li # [len(li)*args.rank//args.world_size:len(li)*(args.rank+1)//args.world_size]
    train_idx, val_idx, test_idx = slice_idx(train_idx), slice_idx(val_idx), slice_idx(test_idx)

    # assert args.batch_size % (args.world_size * torch.cuda.device_count()) == 0


    # Initialize the process group
    dist.init_process_group(
        backend='nccl',  # or 'gloo' for CPU-only training
        # init_method='env://',  # Or specify a URL if not using env variables
        rank=rank,
        world_size=world_size
    )

    local_rank = rank % torch.cuda.device_count()  # Determine local rank
    torch.cuda.set_device(local_rank)  # Set the current device to the local GPU

    DatasetCls = eval(args.dataset)

    dataset_train = DatasetCls(train_idx, id2labels, id2fname, args.id2feats_pkl_path_root, lmr=args.lmr, num_feats=args.num_feats_per_timestep, gpu_id=local_rank)
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True) # DONE support shuffle

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.prefetch_factor,
        pin_memory=False,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor)
    
    dataset_val = DatasetCls(val_idx, id2labels, id2fname, args.id2feats_pkl_path_root, lmr=args.lmr, num_feats=args.num_feats_per_timestep, gpu_id=local_rank)
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size, # //args.world_size,
        num_workers=args.prefetch_factor,
        pin_memory=False,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor)
    
    mean_series_cached_pkl_path = args.mean_series_cached_filename_prefix + f'_cached_median_djv3_{args.num_timesteps}_{args.seed}_{args.sample_size_per_class}_{args.max_iter}_{args.num_classes}_{args.num_feats_per_timestep}_mm{args.use_mm}.pkl'
    if os.path.exists(mean_series_cached_pkl_path):
        mean_series = pickle.load(open(mean_series_cached_pkl_path, 'rb'))
    else:
        if rank == 0:
            print('Calculating mean series', flush=True)
            mean_series = calculate_mean_series(data_loader_train, args.use_mm, sample_size_per_class=args.sample_size_per_class, max_iter=args.max_iter, feat_batch_size=args.feat_batch_size, target_length=args.num_timesteps)
            pickle.dump(mean_series, open(mean_series_cached_pkl_path, 'wb'))     
        exit(0)

    model = DTWModelDist(mean_series, args.use_mm, local_rank, feat_batch_size=args.feat_batch_size, weights_path=args.load_weights_path, use_fixed_path=args.use_fixed_path, w_one_init=args.w_one_init)
    model = DDP(model, device_ids=[local_rank])
    loss_fn = F.cross_entropy
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(
        [
            {'params': model.module.feat_log_weights, 'lr': args.lr, 'betas': (args.b1_flw, args.b2_flw)},
            {'params': model.module.mean_series, 'lr': args.lr/args.ms_lr_div, 'betas': (args.b1_ms, args.b2_ms)},
        ]) #, betas=(0.95, 0.999))
    # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=args.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])


    if args.validation:
        validation( data_loader_val, 
                model, 
                loss_fn, 
                val_idx, 
                args.shared_dir,
                args.output_pred_label_prefix)

        exit(0)
    
    for epoch in ((pbar := tqdm(range(args.num_epochs))) if local_rank == 0 else range(args.num_epochs)):
        sampler_train.set_epoch(epoch)
        train_one_epoch(pbar if local_rank == 0 else None, 
                        epoch, 
                        args.num_epochs, 
                        optimizer, 
                        scheduler, 
                        data_loader_train, 
                        data_loader_val, 
                        model, 
                        loss_fn, 
                        train_idx, 
                        val_idx, 
                        args.shared_dir,
                        args.first_order_penalty,
                        args.second_order_penalty,
                        args.use_fixed_path,
                        args.save_model)
        if args.use_fixed_path:
            model.module.fixed_reset()
            # fixed_feat_weights *= 0
            # model.module.fixed_feat_weights += model.module.feat_log_weights.exp().detach() # .clone().contiguous()
    
    dist.destroy_process_group()

if __name__ == '__main__':
    args = get_args()
    main(args)