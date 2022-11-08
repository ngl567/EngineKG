# EngineKG
COLING 2022: Perform Like an Engine: A Closed-Loop Neural-Symbolic Learning Framework for Knowledge Graph Inference

## Introduction
This is the C++ implementation of our approach [EngineKG](https://aclanthology.org/2022.coling-1.119.pdf). We propose a novel and effective closed-loop neural-symbolic learning framework EngineKG via incorporating our developed KGE and rule learning modules. KGE module exploits symbolic rules and paths to enhance the semantic association between entities and relations for improving KG embeddings and interpretability. A novel rule pruning mechanism is proposed in the rule learning module by leveraging paths as initial candidate rules and employing KG embeddings together with concepts for extracting more high-quality rules.

## A Brief Overview of the EngineKG Architecture which Performs Like an Engine
![image](https://github.com/ngl567/EngineKG/blob/master/architecture.png)<br>

## The Whole Framework of the EngineKG that Conducts KG Embedding and Rule Learning Iteratively
![image](https://github.com/ngl567/EngineKG/blob/master/framework.png)

## Datasets
You can download all the datasets employed in our experiments from [Drive](https://drive.google.com/drive/folders/1SH12cApzClCPlJG6Hdh_BPtmub9qiPvy?usp=share_link).

## Train
In order to reproduce the results of EngineKG model, taking fb15k dataset for an instance, you can kindly run the following commands:  
```
g++ Train_EngineKG.cpp -o Train_EngineKG
./Train_EngineKG -lr 0.0005 -epoch 100 -margin 1.5 -margin_p 1.5 -margin_r 2.5 -res_path dc_loop -data_dir fb15k
```

## Test
In order to evaluate the EngineKG model, you can kindly run the following commands:  
```
g++ Test_EngineKG.cpp -o Test_EngineKG
./Test_EngineKG -res_path dc_loop -data_dir fb15k -hit_n 10
```

## Citation
If you use the codes, please cite the following paper:
```
@inproceedings{niu2022enginekg,
  author    = {Guanglin Niu and
               Bo Li and
               Yongfei Zhang and
               Shiliang Pu},
  title     = {Perform like an Engine: A Closed-Loop Neural-Symbolic Learning Framework for Knowledge Graph Inference},
  booktitle = {COLING},
  year      = {2022}
}
```
