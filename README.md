# Byte Latent Transformer (BLT) Implementation  

## Overview  

This project aims to implement Meta’s **Byte Latent Transformer (BLT)**, a novel approach that eliminates tokenization by encoding byte sequences as latent vectors for transformer input. This method enhances scalability and removes the dependence on a fixed vocabulary. The study will evaluate BLT’s performance on small-scale architectures and explore potential applications beyond text processing.  

## Running the Code

To run the code, follow these steps:
1. Clone the repository:
```bash
git clone https://github.com/danielpwarren/open-blt
cd open-blt
```

2. Activate the virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the main script:
```bash
python -m blt.train
```

## Run logging

This project logs training runs using WandB. To view current and past runs, visit the [WandB project page](https://wandb.ai/danielpwarren/open-blt).


## Research Plan and Timeline  

### Phase 1: Preparation (Weeks 5-6)  
- Conduct a thorough review of Meta’s BLT paper and related research.  
- Identify open-source implementations for reference, such as the Minimind project.  
- Select and preprocess **The Pile** dataset to ensure compatibility with a byte-based encoding approach.  

### Phase 2: Model Training and Evaluation (Weeks 7-8)  
- Implement BLT’s architecture and verify its core functionality on a subset of the dataset.  
- Train the model on language modeling tasks, using **validation loss** and **validation accuracy** as evaluation metrics.  
- Compare results with existing token-based transformer models to assess scalability improvements.

### Phase 3: Extensions (Weeks 8-9)  
- Investigate the potential of BLT for multimodal learning by experimenting with byte representations of non-textual data.  
- Fine-tune the model for specific downstream tasks, such as classification, summarization, or conversation.  

### Phase 4: Final Analysis (Weeks 9-10)  
- Conduct a comprehensive analysis of experimental results, comparing BLT’s efficiency against token-based models.  
- Document findings, challenges encountered, and potential directions for future research.  
- Prepare a final report detailing the methodology, results, and conclusions.  

## Dataset: The Pile  

**The Pile** is a **large-scale diverse text corpus** designed for language modeling. It consists of **825.18 GiB of high-quality data** drawn from a variety of sources, including:  

- **Books3** – A large collection of books.  
- **PubMed Abstracts** – Biomedical abstracts from PubMed.  
- **ArXiv** – Scientific papers across various disciplines.  
- **Github** – Code repositories.  
- **OpenWebText2** – A refined version of OpenWebText.  
- **Wikipedia (en)** – English Wikipedia.  
- **DM Mathematics** – Mathematical problem datasets.  
- **HackerNews** – Tech-related discussions.  
- **EuroParl** – European Parliament proceedings.  
- **Pile-CC** – A cleaned version of Common Crawl.  
- **USPTO Backgrounds** – U.S. patent text.  

This dataset is well-suited for evaluating **byte-level models** due to its diversity and lack of heavy pre-processing, making it a strong candidate for training the BLT model.  

## References  
- Meta’s Byte Latent Transformer [Paper](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/8) and [Code](https://github.com/facebookresearch/blt)  
- Minimind Project [Repository](https://github.com/jingyaogong/minimind)
- The Pile [Paper](https://arxiv.org/pdf/2101.00027)

## Team  
- Daniel Warren  
- Aniruddha Dasu  
- Ryan Wong  
- Ansh Mujral  