# 3rd Place Solution - Kaggle-OTTO-Comp - Chris' Part
In Kaggle's OTTO competition, we need to build a model that predicts what a user will click, cart, and order in the future at online ecommerce website https://otto.de We are given 4.5 weeks of data and must predict the next 0.5 week. More details are explained at Kaggle [here][1] and final leaderboard is [here][2]. Our team of Benny, Chris, Giba, and Theo won 3rd place cash gold! Our detailed writeup is published [here][3](Chris), [here][4](Theo), and [here][5](Benny) with final submission Kaggle submit notebook [here][8]

# Code for 15th Place Gold Single Model
The code in this GitHub repo, will create Chris' 15th place solo gold single model which achieves CV 0.601 and LB 0.601. When we ensemble the single models of Benny, Chris, Giba, Theo, we achieve 3rd place cash gold with Private LB 0.6038 and Public LB 0.6044. Final submission Kaggle submit notebook is published [here][8]

This model is a "candidate rerank" model explained [here][6], [here][7], and [here][3]. Our challenge is to predict 20 items for each target per user (the 3 targets are click, cart, order) that we suspect user will engage in the future. First we generate 100 item candidates (per target per user) using co-visitiation matrices (and heuristic rules). Next we merge features onto the user item pairs. Lastly we train a GBT reranker model to select 20 from 100.

The following image will help understand the organization of code in this repo. First we train our model on the first 3.5 weeks of data. Then we infer our model on 4.5 weeks of data. Therefore we will basically run the same pipeline twice:

![](data/images/timeline.png)

# How To Run Code
This code ran successfully on 20xCPU 256GB and 1xGPU 32GB. Using less memory may cause memory errors. To run this code, first install libraries RAPIDS (cuDF cuML), XGBoost, and Pandarallel in addition to basic Python libraries Pandas, NumPy, Pickle, Scikit-Learn, Matplotlib, and Tqdm. The script to compute item embeddings requires PyTorch and Merlin-Dataloader. Next follow these 3 main steps with substeps:
* **(1) Download Data from Kaggle**
* => Run [/data/make_train_valid.ipynb](data/make_train_valid.ipynb)
* **(2) Train Models**
* => (2-1) compute co-visit matrices by running [/train/covisit_matrices/script.ipynb](train/covisit_matrices/script.ipynb)
  * [ ] gpu-115.ipynb
  * [ ] gpu-116.ipynb
  * [ ] gpu-155.ipynb
  * [ ] gpu-157.ipynb
  * [ ] gpu-165.ipynb
  * [ ] gpu-166.ipynb
  * [ ] gpu-167.ipynb
  * [ ] gpu-168.ipynb
  * [ ] gpu-217.ipynb
  * [ ] gpu-220.ipynb
  * [ ] gpu-226.ipynb
  * [ ] gpu-232.ipynb
  * [ ] gpu-235.ipynb
  * [ ] gpu-239.ipynb
  * [ ] gpu-700.ipynb
  * [ ] gpu-701.ipynb
  * [ ] gpu-93.ipynb
  * [x] cpu-90.ipynb
  * [ ] cpu-95.ipynb
  * [ ] cpu-99.ipynb
* => (2-2) generate candidates and scores with [/train/candidates/script.ipynb](train/candidates/script.ipynb)
  * [x] make-valid-with-d.ipynb
  * [ ] model-564-1.ipynb
  * [ ] model-564-1-wgt.ipynb
  * [ ] model-564-20.ipynb
  * [ ] model-564-20-wgt.ipynb
  * [ ] model-564-21.ipynb
  * [ ] model-564-21-wgt.ipynb
  * [ ] model-564-22.ipynb
  * [ ] model-564-22-wgt.ipynb
  * [ ] model-564-23.ipynb
  * [ ] model-564-23-wgt.ipynb
  * [ ] model-564-24.ipynb
  * [ ] model-564-24-wgt.ipynb
  * [ ] model-564-25.ipynb
  * [ ] model-564-25-wgt.ipynb
  * [ ] model-564-26.ipynb
  * [ ] model-564-26-wgt.ipynb
  * [ ] model-564-27.ipynb
  * [ ] model-564-27-wgt.ipynb
  * [ ] model-564-3.ipynb
  * [ ] model-564-3-wgt.ipynb
  * [ ] model-564-31.ipynb
  * [ ] model-564-31-wgt.ipynb
  * [ ] model-564-32.ipynb
  * [ ] model-564-32-wgt.ipynb
  * [ ] model-564-33.ipynb
  * [ ] model-564-33-wgt.ipynb
  * [ ] model-564-4.ipynb
  * [ ] model-564-4-wgt.ipynb
  * [ ] model-564-5.ipynb
  * [ ] model-564-5-wgt.ipynb
  * [ ] model-564-6.ipynb
  * [ ] model-564-6-wgt.ipynb
  * [ ] model-564-7.ipynb
  * [ ] model-564-7-wgt.ipynb
  * [ ] model-564-F.ipynb
  * [ ] model-564-F-wgt.ipynb
  * [ ] model-612.ipynb
  * [ ] model-612-wgt.ipynb
  * [ ] model-614.ipynb
  * [ ] model-614-wgt.ipynb
  * [ ] make-valid-lists.ipynb
  * [ ] model-620-wgt.ipynb
  * [ ] model-709.ipynb
  * [ ] model-709-wgt.ipynb
* => (2-3) engineer features with [/train/item_user_features/script.ipynb](train/item_user_features/script.ipynb)
  * [NG] embeddings-for-train.ipynb
  * [x] item-features-10.ipynb
  * [ ] item-features-12.ipynb
  * [ ] item-features-13.ipynb
  * [ ] item-features-20.ipynb
  * [ ] item-features-21.ipynb
  * [ ] item-features-22.ipynb
  * [ ] item-features-4.ipynb
  * [ ] item-features-40.ipynb
  * [ ] item-features-41.ipynb
  * [x] user-features-10.ipynb
  * [ ] user-features-20.ipynb
  * [ ] user-features-21.ipynb
  * [ ] user-features-4.ipynb
  * [ ] user-features-7.ipynb
* => (2-4) merge candidates and features for click model with [/train/make_parquets/script-1.ipynb](train/make_parquets/script-1.ipynb)
  * [ ] Make-152-split-users.ipynb
  * [ ] Make-152-A.ipynb
  * [ ] Make-152-B.ipynb
  * [ ] Make-152-C.ipynb
  * [ ] Make-152-D.ipynb
  * [ ] Make-152-E.ipynb
  * [ ] Make-152-A2.ipynb
  * [ ] Make-152-B2.ipynb
  * [ ] Make-152-C2.ipynb
  * [ ] Make-152-D2.ipynb
  * [ ] Make-152-E2.ipynb
* => (2-5) train click model with [/train/ranker_models/XGB-186-CLICKS.ipynb](train/ranker_models/XGB-186-CLICKS.ipynb)
* => (2-6) merge candidates and features for cart and order model with [/train/make_parquets/script-2.ipynb](train/make_parquets/script-2.ipynb)
* => (2-7) train cart model with [/train/ranker_models/XGB-406-CARTS.ipynb](train/ranker_models/XGB-406-CARTS.ipynb)
* => (2-8) train order model with [/train/ranker_models/XGB-412-ORDERS.ipynb](train/ranker_models/XGB-412-ORDERS.ipynb)
* **(3) Infer Models**
* => compute LB co-visit matrices by running [/infer/covisit_matrices_LB/script.ipynb](infer/covisit_matrices_LB/script.ipynb)
* => generate LB candidates and scores with [/infer/candidates_LB/script.ipynb](infer/candidates_LB/script.ipynb)
* => engineer LB features with [/infer/item_user_features_LB/script.ipynb](infer/item_user_features_LB/script.ipynb)
* => merge LB candidates and features for click model with [/infer/make_parquets_LB/script.ipynb](infer/make_parquets_LB/script.ipynb)
* => infer models with [/infer/inference_LB/script.ipynb](infer/inference_LB/script.ipynb)

After running the steps above, the file `/data/submission_final/submission_chris_v186v406v412.csv` is generated. 
This file will score Private LB 0.6012 and Public LB 0.6010. To achieve a better CV and LB, we can train CatBoost with the
code `/train/ranker_model/CAT-200-orders.ipynb` and `/train/ranker_model/CAT-203-carts.ipynb` and change inference to infer
CatBoost. The result is Private LB 0.6018 and Public LB 0.6016. We discovered that CatBoost was better after the competition
ended.


```
├── train
│   ├── covisit_matrices         # Compute matrices with RAPIDS cuDF
│   ├── candidates               # Generate candidates from matrices
│   ├── item_user_features       # Feature engineering with RAPIDS cuDF
│   ├── make_parquets            # Combine candidates, features, targets
│   └── ranker_models            # Train XGB model
├── infer        
│   ├── covisit_matrices_LB      # Compute matrices with RAPIDS cuDF
│   ├── candidates_LB            # Generate candidates from matrices
│   ├── item_user_features_LB    # Feature engineering with RAPIDS cuDF
│   ├── make_parquets_LB         # Combine candidates, features, targets
│   └── inference_LB             # Infer XGB model with RAPIDS FIL
├── data    
│   ├── make_train_valid.ipynb   # Run to download data
│   ├── train_data               # Train data downloaded to here
│   ├── infer_data               # Infer data downloaded to here
│   ├── covisit_matrices         # Matrices stored here
│   ├── candidate_scores         # Candidate lists and scores here
│   ├── item_user_features       # Item and user features here
│   ├── train_with_features      # Train data with features merged
│   ├── infer_with_features      # Infer data with features merged
│   ├── models                   # Trained models here
│   ├── submission_parts         # Partial submission.csv here
│   └── submission_final         # Final submission.csv here
└── README.md
```

[1]: https://www.kaggle.com/competitions/otto-recommender-system/overview
[2]: https://www.kaggle.com/competitions/otto-recommender-system/leaderboard
[3]: https://www.kaggle.com/competitions/otto-recommender-system/discussion/383013
[4]: https://www.kaggle.com/competitions/otto-recommender-system/discussion/382975
[5]: https://www.kaggle.com/competitions/otto-recommender-system/discussion/386497
[6]: https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575
[7]: https://www.kaggle.com/competitions/otto-recommender-system/discussion/370210
[8]: https://www.kaggle.com/code/cdeotte/3rd-place-team-g-b-d-t-0-604
