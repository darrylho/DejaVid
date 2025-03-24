# Install
1. `conda env create -f environment.yml -n dejavid && conda activate dejavid`
2. For each `.py` file in `kernel_setup`, run the last commented line to install

# Running
1. Given a video dataset, first use any video encoder(s) to convert each video into a 2D tensor (first dim time, second dim embeddings). Use torch.save to save each 2D tensor to a unique file. The `id2feats` argument in script takes the root folder of these tensor files.
2. Prepare `id2splits` (maps `MTS_id` to its split; either `'train'` or `'val'`), `id2labels` (maps `MTS_id` to its class ID (integer)), and `id2fname` (maps `MTS_id` to its 2D tensor filename) and fill their paths into the script. Each of these should be a python `dict` object stored in a pickle (`.pkl`) file. 
3. Specify in the script a cached filename for the centroid series and run the script once. The code should detect that no such cached file exists and calculate one for you.
4. Run the script again to start training and evaluation.