# [CVPR 2025] DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification

Official implementation of **"DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification"**, accepted to CVPR 2025.  
📄 [Paper (CVF)](https://openaccess.thecvf.com/content/CVPR2025/html/Ho_DejaVid_Encoder-Agnostic_Learned_Temporal_Matching_for_Video_Classification_CVPR_2025_paper.html) • 📊 [Papers with Code](https://paperswithcode.com/paper/dejavid-encoder-agnostic-learned-temporal)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dejavid-encoder-agnostic-learned-temporal/action-recognition-in-videos-on-something)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something?p=dejavid-encoder-agnostic-learned-temporal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dejavid-encoder-agnostic-learned-temporal/action-recognition-in-videos-on-hmdb-51)](https://paperswithcode.com/sota/action-recognition-in-videos-on-hmdb-51?p=dejavid-encoder-agnostic-learned-temporal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dejavid-encoder-agnostic-learned-temporal/action-classification-on-kinetics-400)](https://paperswithcode.com/sota/action-classification-on-kinetics-400?p=dejavid-encoder-agnostic-learned-temporal)


---

## 🔍 Abstract

In recent years, large transformer-based video encoder models have greatly advanced state-of-the-art performance on video classification tasks. However, these large models typically process videos by averaging embedding outputs from multiple clips over time to produce fixed-length representations. This approach fails to account for a variety of time-related features, such as variable video durations, chronological order of events, and temporal variance in feature significance. While methods for temporal modeling do exist, they often require significant architectural changes and expensive retraining, making them impractical for off-the-shelf, fine-tuned large encoders. To overcome these limitations, we propose DejaVid, an encoder-agnostic method that enhances model performance without the need for retraining or altering the architecture. Our framework converts a video into a variable-length temporal sequence of embeddings (TSE). A TSE naturally preserves temporal order and accommodates variable video durations. We then learn per-timestep, per-feature weights over the encoded TSE frames, allowing us to account for variations in feature importance over time. We introduce a new neural network architecture inspired by traditional time series alignment algorithms for this learning task. Our evaluation demonstrates that DejaVid substantially improves the performance of a state-of-the-art large encoder, achieving leading Top-1 accuracy of 77.2% on Something-Something V2, 89.1% on Kinetics-400, and 88.6% on HMDB51, while adding fewer than 1.8% additional learnable parameters and requiring less than 3 hours of training time. 

---

## 🚀 Getting Started

### Install
1. `git clone https://github.com/darrylho/DejaVid.git && cd DejaVid`
2. `conda env create -f environment.yml -n dejavid && conda activate dejavid`
3. For each `.py` file in `kernel_setup`, run the last commented line to install

### Running
1. Given a video dataset, first use any video encoder(s) to convert each video into a 2D tensor (first dim time, second dim embeddings). Use torch.save to save each 2D tensor to a unique file. The `id2feats` argument in script takes the root folder of these tensor files.
2. Prepare `id2splits` (maps `MTS_id` to its split; either `'train'` or `'val'`), `id2labels` (maps `MTS_id` to its class ID (integer)), and `id2fname` (maps `MTS_id` to its 2D tensor filename) and fill their paths into the script. Each of these should be a python `dict` object stored in a pickle (`.pkl`) file. 
3. Specify in the script a cached filename for the centroid series and run the script once. The code should detect that no such cached file exists and calculate one for you.
4. Run the script again to start training and evaluation.

---
## 📈 Results

| Task                  | Dataset               | Metric                      | Result |
|-----------------------|------------------------|------------------------------|---------|
| Action Recognition    | HMDB-51                | Top-1 Accuracy       | 88.6    |
| Action Classification | Kinetics-400           | Top-1 Accuracy               | 89.1    |
|                       |                        | Top-5 Accuracy               | 98.2    |
| Action Recognition    | Something-Something V2 | Top-1 Accuracy               | 77.2    |
|                       |                        | Top-5 Accuracy               | 96.3    |

---
## 📜 Citation

If you find this work useful, please consider citing:

```bibtex 
@inproceedings{ho2025dejavid,
  title={DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification},
  author={Ho, Darryl and Madden, Samuel},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={24023--24032},
  year={2025}
}
```

---

## 📬 Contact

For questions, issues, or collaboration requests, feel free to reach out:

Darryl Ho

📧 darrylho@csail.mit.edu
