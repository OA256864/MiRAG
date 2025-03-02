# Multi-Level Information Retrieval Augmented Generation for Knowledge-based Visual Question Answering (MiRAG)

Official Implementation of [Multi-Level Information Retrieval Augmented Generation for Knowledge-based Visual Question Answering (Adjali et al., EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.922/)  



# Environment
Create a virtual environment with all requirements using the provided yaml file

```
conda env create -f a100.yml
```

or create env and install dependecies as follows

```
conda create -n a100 python=3.7
conda activate a100

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install setuptools==59.5.0 
python -m spacy download en_core_web_sm
pip install -r requirements.txt
```


# Dataset

The ViQuAE dataset [(Lerner et al., SIGIR’22)](https://hal.science/hal-03650618>) and all the necessary resources (Konlwedge base and images) can be downloaded from:  https://huggingface.co/PaulLerner. For compatibility with our code, use the hugginface ```datasets``` format.

# Data Preparation

In this work, we relied heavily on Paul Lerner repository https://github.com/PaulLerner/ to deal with the ViQuAE data, in particular, to perform data preprocessing, compute embeddings (DPR and CLIP) and evaluate Information Retrieval (IR) performance at the passage and entity level. Thus, please refer to https://github.com/PaulLerner/ViQuAE/blob/main/EXPERIMENTS.rst for instructions on how to perform the aforementioned steps.

1- **Preprocessing passages**

This step splits the 1.5 million articles of the ViQuAE KB into passages. Please follows instructions at: https://github.com/PaulLerner/ViQuAE/blob/main/EXPERIMENTS.rst#preprocessing-passages, or use the preprocessed passages from https://huggingface.co/datasets/PaulLerner/viquae_passages

2- **Compute Embeddings**

To perform retrieval augmented generation, compute dense vector representations of:

- Questions and passages using DPR encoders.
- Question images and KB article titles using respectively CLIP image and text encoders.

Follows instructions at https://github.com/PaulLerner/ViQuAE/blob/main/EXPERIMENTS.rst#embedding-questions-and-passages

3- **IR evaluation**

Indexing, Search and IR evaluation are performed using the [FAISS](https://github.com/facebookresearch/faiss) and [ranx](https://amenra.github.io/ranx/) libraries. Please follows instructions at https://github.com/PaulLerner/ViQuAE/blob/main/EXPERIMENTS.rst#searching-1

# RAG and MiRAG Training

We used the [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning) as a backbone for the training and testing procedures. Edit the config file (```--config```) to adapt the different parameters of experimentation. 

Example of fine-tuning and testing  blip2 on ViQuAE using RAG:

```
python main.py  --train_batch_size 4 --eval_batch_size 4  --num_train_epochs 15 --learning_rate 2e-5  --config experiments/blip2/config_rag.json --experiment_name blip2_rag --non_strict_load --nbr_gpus 1 --grad_check
```

Example of fine-tuning and testing blip2 on ViQuAE using MiRAG:

```
python main.py  --train_batch_size 4 --eval_batch_size 4  --num_train_epochs 15 --learning_rate 2e-5  --config experiments/blip2/config_mirag.json --experiment_name blip2_mirag --non_strict_load --nbr_gpus 1 --grad_check
```





# Citation

```
@inproceedings{omar-etal-2024-multi,
    title = "Multi-Level Information Retrieval Augmented Generation for Knowledge-based Visual Question Answering",
    author = "Adjali, Omar  and
      Ferret, Olivier  and
      Ghannay, Sahar  and
      Le Borgne, Herv{\'e}",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.922/",
    doi = "10.18653/v1/2024.emnlp-main.922",
    pages = "16499--16513",
    abstract = "The Knowledge-Aware Visual Question Answering about Entity task aims to disambiguate entities using textual and visual information, as well as knowledge. It usually relies on two independent steps, information retrieval then reading comprehension, that do not benefit each other. Retrieval Augmented Generation (RAG) offers a solution by using generated answers as feedback for retrieval training. RAG usually relies solely on pseudo-relevant passages retrieved from external knowledge bases which can lead to ineffective answer generation. In this work, we propose a multi-level information RAG approach that enhances answer generation through entity retrieval and query expansion. We formulate a joint-training RAG loss such that answer generation is conditioned on both entity and passage retrievals. We show through experiments new state-of-the-art performance on the VIQuAE KB-VQA benchmark and demonstrate that our approach can help retrieve more actual relevant knowledge to generate accurate answers."
}
```

