from commons import *

def train_one_epoch(
        pbar, 
        epoch, 
        num_epochs, 
        optimizer, 
        scheduler, 
        data_loader_train, 
        data_loader_val, 
        model, 
        loss_fn, 
        train_idx, 
        val_idx, 
        shared_dir, 
        first_order_penalty, 
        second_order_penalty,
        use_fixed_path,
        save_model):

    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    torch.cuda.empty_cache()

    

    if global_rank == 0:
        num_correct = num_total = 0
        
    for t_id, (padded_batch, batch_lens, labels) in ((pbar2 := tqdm(enumerate(data_loader_train), total=-(len(train_idx)//-(data_loader_train.batch_size*world_size)))) if global_rank == 0 else enumerate(data_loader_train)):
        logits_per_ele_per_samp_prediv = model([x[:l].clone() for x, l in zip(padded_batch, batch_lens)])
        # logits_per_ele_per_samp.shape = num_samples x num_kernels
        logits = (logits_per_ele_per_samp_prediv - logits_per_ele_per_samp_prediv.mean(dim=1, keepdim=True)) / (logits_per_ele_per_samp_prediv.std(dim=1, keepdim=True) + 1e-18) 
        # logits should be              batch_size x num_class (x for_k_dim_loss...)
        # labels should be              batch_size (x for_k_dim_loss...)
        loss = loss_fn(logits, labels)
        
        assert loss.shape == torch.Size([])

        predictions = torch.argmax(logits, dim=1)
        # predictions.shape = num_samples
        correct_predictions = (predictions == labels)
        # correct_predictions.shape = num_samples

        it_correct = correct_predictions.sum()
        it_total = torch.tensor(len(correct_predictions), device=it_correct.device)

        dist.reduce(it_correct, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(it_total, dst=0, op=dist.ReduceOp.SUM)

        if global_rank == 0:
            num_correct += it_correct.item()
            num_total += it_total.item()
            pbar2.set_description(f'DistTrainAcc: {num_correct / num_total:.4f} ({num_correct}/{num_total})')
        

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    if global_rank == 0:
        cur_epoch_train_acc = num_correct / num_total

    scheduler.step()


    if global_rank == 0:
        num_correct = num_total = 0
        num_top5 = 0
        
    padded = (world_size - (len(val_idx) % world_size)) % world_size

    with torch.no_grad():
        for v_id, (v_padded_batch, v_batch_lens, v_labels) in ((pbar2 := tqdm(enumerate(data_loader_val), total=-(len(val_idx)//-(data_loader_val.batch_size*world_size)))) if global_rank == 0 else enumerate(data_loader_val)):
            logits_per_ele_per_samp_prediv = model([x[:l].clone() for x, l in zip(v_padded_batch, v_batch_lens)])
            # logits_per_ele_per_samp.shape = num_samples x num_kernels
            logits = (logits_per_ele_per_samp_prediv - logits_per_ele_per_samp_prediv.mean(dim=1, keepdim=True)) / (logits_per_ele_per_samp_prediv.std(dim=1, keepdim=True) + 1e-18) 
            # logits should be              batch_size x num_class (x for_k_dim_loss...)
            # labels should be              batch_size x (for_k_dim_loss...)
            loss = loss_fn(logits, v_labels)
            assert loss.shape == torch.Size([])

            predictions = torch.argmax(logits, dim=1)
            # predictions.shape = num_samples  
            correct_predictions = (predictions == v_labels)
            top5_predictions = torch.tensor(tuple((int(one_label.item()) in np.argsort(-one_logits.cpu().numpy())[:5]) \
                                           for one_label, one_logits in zip(v_labels, logits)), device=correct_predictions.device)
            
            it_correct = correct_predictions.sum()
            it_total = torch.tensor(len(correct_predictions), device=it_correct.device)
            it_top5 = top5_predictions.sum()

            if v_id == 0 and global_rank < padded: # fixes last-batch inaccuracy
                it_correct -= int(correct_predictions[0].item())
                it_total -= 1                
                it_top5 -= int(top5_predictions[0].item())

    
            dist.reduce(it_correct, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(it_total, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(it_top5, dst=0, op=dist.ReduceOp.SUM)

    
            if global_rank == 0:
                num_correct += it_correct.item()
                num_total += it_total.item()
                num_top5 += it_top5.item()
                pbar2.set_description(f'DistValAcc: {num_correct / num_total:.4f} ({num_correct}/{num_total}), DistValTop5: {num_top5 / num_total:.4f} ({num_top5}/{num_total})')


    if global_rank == 0:
        cur_epoch_val_acc = num_correct / num_total
    
    if global_rank == 0:
        pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], '+\
                            f'TrainAcc: {cur_epoch_train_acc:.4f}, '+\
                            f'ValAcc: {cur_epoch_val_acc:.4f}')
        
    if global_rank == 0 and save_model:    
        tim = time.time()
        weight_saved_filename = f'saved_models/feat_log_weights_rank_{global_rank}_{tim}.pt'
        feats_saved_filename = f'saved_models/k_feats_rank_{global_rank}_{tim}.pkl'
        with torch.no_grad():
            torch.save(model.module.feat_log_weights.data, weight_saved_filename)
            pickle.dump(model.module.mean_series.data, open(feats_saved_filename, 'wb'))
        print(f'Saved weight {weight_saved_filename}', flush=True)
        print(f'Saved k_feats {feats_saved_filename}', flush=True)

def validation(
        data_loader_val, 
        model, 
        loss_fn, 
        val_idx, 
        shared_dir,
        output_pred_label_prefix = None):

    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    if global_rank == 0:
        num_correct = num_total = 0
        num_top5 = 0

    padded = (world_size - (len(val_idx) % world_size)) % world_size
        
    with torch.no_grad():
        for v_id, (v_padded_batch, v_batch_lens, v_labels) in ((pbar2 := tqdm(enumerate(data_loader_val), total=-(len(val_idx)//-(data_loader_val.batch_size*world_size)))) if global_rank == 0 else enumerate(data_loader_val)):          
            logits_per_ele_per_samp_prediv = model([x[:l].clone() for x, l in zip(v_padded_batch, v_batch_lens)])
            # logits_per_ele_per_samp.shape = num_samples x num_kernels
            logits = (logits_per_ele_per_samp_prediv - logits_per_ele_per_samp_prediv.mean(dim=1, keepdim=True)) / (logits_per_ele_per_samp_prediv.std(dim=1, keepdim=True) + 1e-18) 
            # logits should be              batch_size x num_class (x for_k_dim_loss...)
            # labels should be              batch_size x (for_k_dim_loss...)
            loss = loss_fn(logits, v_labels)
            assert loss.shape == torch.Size([])

            predictions = torch.argmax(logits, dim=1)
            # predictions.shape = num_samples  
            correct_predictions = (predictions == v_labels)
            top5_predictions = torch.tensor(tuple((int(one_label.item()) in np.argsort(-one_logits.cpu().numpy())[:5]) \
                                           for one_label, one_logits in zip(v_labels, logits)), device=correct_predictions.device)
            
            it_correct = correct_predictions.sum()
            it_total = torch.tensor(len(correct_predictions), device=it_correct.device)
            it_top5 = top5_predictions.sum()


            if v_id == 0 and global_rank < padded: # fixes last-batch inaccuracy
                it_correct -= int(correct_predictions[0].item())
                it_total -= 1
                it_top5 -= int(top5_predictions[0].item())

    
            dist.reduce(it_correct, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(it_total, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(it_top5, dst=0, op=dist.ReduceOp.SUM)
                
            if global_rank == 0:
                num_correct += it_correct.item()
                num_total += it_total.item()
                num_top5 += it_top5.item()
                pbar2.set_description(f'DistValAcc: {num_correct / num_total:.4f} ({num_correct}/{num_total}), DistValTop5: {num_top5 / num_total:.4f} ({num_top5}/{num_total})')

    if global_rank == 0:
        cur_epoch_val_acc = num_correct / num_total
        print(f'FinalValAcc: {cur_epoch_val_acc:.4f} ({num_correct}/{num_total})', flush=True)
                
    
    
