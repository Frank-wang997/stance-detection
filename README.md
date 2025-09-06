# stance-detection



## Dataset

### [NLPCC-2016](http://tcci.ccf.org.cn/conference/2016/pages/page05_CFPTasks.html)

|         Target         | Total  | Unlabeled | Labeled for training | Labeled for test |
| :--------------------: | :----: | :-------: | :------------------: | :--------------: |
|        iPhoneSE        | 3,800  |   3,000   |         600          |       200        |
|    Ban of fireworks    | 3,800  |   3,000   |         600          |       200        |
| Russian anti-terrorist | 3,800  |   3,000   |         600          |       200        |
|    Two-child Policy    | 3,800  |   3,000   |         600          |       200        |
|    Ban of Tricycles    | 3,800  |   3,000   |         600          |       200        |
|         Total          | 19,000 |  15,000   |        3,000         |      1,000       |



## hyperparameter

| Hyperparameter |      Setting       |
| :------------: | :----------------: |
|   Optimizer    |        Adam        |
|     Epoch      |         20         |
|   Batch size   |         16         |
| Learning rate  |        2e-5        |
| Loss function  | Cross Entropy Loss |
|       λ        |        0.8         |
