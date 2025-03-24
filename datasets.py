from commons import *

class SSV2DatasetDist(torch.utils.data.Dataset):
    def __init__(self, sorted_idx, id2labels, id2fname, id2feats_pkl_path_root, \
                 pad=True, \
                 lmr=1, max_num_frames=150, \
                 num_feats = 1756*4, gpu_id=0):
        self.idx = sorted_idx
        self.pad = pad
        self.id2labels = id2labels
        self.id2fname = id2fname
        self.label_remap = {v:k for k, v in enumerate(sorted(set(id2labels[i] for i in id2fname.keys())))} # to make labels contiguous from 0
        self.odr2id = dict(enumerate(sorted_idx))
        self.id2odr = {v:k for k, v in self.odr2id.items()}
        self.lmr = lmr
        self.id2feats_pkl_path_root = id2feats_pkl_path_root
        self.max_num_frames = max_num_frames
        self.num_feats = num_feats
        self.gpu_id = gpu_id
                    
    def __getitem__(self, i):
        i = self.idx[i]
        
        fname = self.id2fname[i]
        id2feats = torch.load(os.path.join(self.id2feats_pkl_path_root, fname), map_location=f'cuda:{self.gpu_id}')
        ret = id2feats[i] # .to('cuda')

        
        lret = torch.tensor(len(ret), device=ret.device)
        if self.pad:
            ret = torch.cat((ret, torch.zeros(self.max_num_frames-ret.shape[0], ret.shape[1], device=ret.device)), dim=0)

        assert ret.shape[1] == self.num_feats
        return ret, lret, torch.tensor(self.label_remap[self.id2labels[i]], device=ret.device)        
        
    def __len__(self):
        return len(self.idx)


class K400DatasetDist(torch.utils.data.Dataset):
    def __init__(self, sorted_idx, id2labels, id2fname, id2feats_pkl_path_root, \
                 pad=True, \
                 lmr=1, max_num_frames=33, \
                 num_feats = 1408+400, gpu_id=0):
        self.idx = sorted_idx
        self.pad = pad
        self.id2labels = id2labels
        self.id2fname = id2fname
        self.label_remap = {v:k for k, v in enumerate(sorted(set(id2labels[i] for i in id2fname.keys())))} # to make labels contiguous from 0
        self.odr2id = dict(enumerate(sorted_idx))
        self.id2odr = {v:k for k, v in self.odr2id.items()}
        self.lmr = lmr
        self.id2feats_pkl_path_root = id2feats_pkl_path_root
        self.max_num_frames = max_num_frames
        self.num_feats = num_feats
        self.gpu_id = gpu_id
                    
    def __getitem__(self, i):
        i = self.idx[i]
        
        fname = self.id2fname[i]
        ret = torch.load(os.path.join(self.id2feats_pkl_path_root, fname), map_location=f'cuda:{self.gpu_id}')
        
        lret = torch.tensor(len(ret), device=ret.device)
        if self.pad:
            ret = torch.cat((ret, torch.zeros(self.max_num_frames-ret.shape[0], ret.shape[1], device=ret.device)), dim=0)

        assert ret.shape[1] == self.num_feats
        return ret, lret, torch.tensor(self.label_remap[self.id2labels[i]], device=ret.device)        
        
    def __len__(self):
        return len(self.idx)


class HMDB51DatasetDist(torch.utils.data.Dataset):
    def __init__(self, sorted_idx, id2labels, id2fname, id2feats_pkl_path_root, \
                 pad=True, \
                 lmr=1, max_num_frames=33, \
                 num_feats = 1408+51, gpu_id=0):
        self.idx = sorted_idx
        self.pad = pad
        self.id2labels = id2labels
        self.id2fname = id2fname
        self.label_remap = {v:k for k, v in enumerate(sorted(set(id2labels[i] for i in id2fname.keys())))} # to make labels contiguous from 0
        self.odr2id = dict(enumerate(sorted_idx))
        self.id2odr = {v:k for k, v in self.odr2id.items()}
        self.lmr = lmr
        self.id2feats_pkl_path_root = id2feats_pkl_path_root
        self.max_num_frames = max_num_frames
        self.num_feats = num_feats
        assert num_feats == 1408+51
        self.gpu_id = gpu_id
                    
    def __getitem__(self, i):
        i = self.idx[i]
        
        fname = self.id2fname[i]
        ret = torch.load(os.path.join(self.id2feats_pkl_path_root, fname), map_location=f'cuda:{self.gpu_id}')
        
        lret = torch.tensor(len(ret), device=ret.device)
        if self.pad:
            ret = torch.cat((ret, torch.zeros(self.max_num_frames-ret.shape[0], ret.shape[1], device=ret.device)), dim=0)

        assert ret.shape[1] == self.num_feats
        return ret, lret, torch.tensor(self.label_remap[self.id2labels[i]], device=ret.device)        
        
    def __len__(self):
        return len(self.idx)
