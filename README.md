# Introduction
The Financial Sentiment Classification Benchmark (FSC-Bench) provides a systematic evaluation framework for financial sentiment analysis, addressing key real-world challenges: performance on imbalanced data, temporal robustness, and domain adaptation. We benchmark three model families—traditional ML, domain-specific BERT variants, and general-purpose LLMs—across three core experiments. 

Experiment 1 (Baseline Performance) evaluates models on two financial news datasets: one small/imbalanced, one larger/balanced. Experiment 2 (Temporal Robustness) tests models trained on historical data against a recent news corpus to measure performance drift as market language evolves. Experiment 3 (Domain Robustness) assesses the same models on a high-quality, human-annotated Reddit dataset to evaluate adaptability from formal news to informal social media discourse.

Our results reveal critical insights: specialized models like FinBERT excel in-distribution but degrade under temporal and domain shifts, while general-purpose LLMs show surprising robustness to domain changes. This work raises a pivotal, underexplored research question: In a specialized domain, can a powerful, general-purpose LLM without domain-specific fine-tuning match or surpass a specialized smaller model? FSC-Bench offers reproducible code, data, and analysis to guide model selection and inspire robust financial NLP systems.

# Baselines
Loading pre-trained models​ involves downloading weights and configurations from repositories like Hugging Face, where models are pre-trained on massive text corpora, endowing them with fundamental language understanding. Fine-tuning with our data​ adapts these models to specific tasks (e.g., financial sentiment analysis) by continuing training on domain-specific datasets, requiring less time and data than training from scratch while typically yielding superior performance. Saving the model​ preserves the fine-tuned weights, configuration, and tokenizer locally, enabling direct inference, sharing, or deployment to production environments.<br>
<small>

| Model Name    | Model Version                    | Source                                                                                   |
|----------------|---------------------------------|------------------------------------------------------------------------------------------|
| MLP_TF_IDF     | Custom Implementation           | Traditional ML Baseline                                                                  |
| TextCNN        | Custom Implementation           | CNN-based Baseline                                                                       |
| BERT           | `bert-base-uncased`             | https://www.kaggle.com/xhlulu/huggingface-bert?select=bert-base-uncased                  |
| RoBERTa        | `roberta-base`                  | https://www.kaggle.com/datasets/dariussingh/huggingface-roberta                          |
| FinBERT        | `FinBERT-BaseVocab-Cased`       | https://www.kaggle.com/models/addarm/finbert                                             |
| Qwen2          | `Qwen/Qwen2-1.5B-Instruct`      | https://www.kaggle.com/models/qwen-lm/qwen2                                              |
</small>

# Datasets

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; justify-items: center;">
  <img src="./figures/sentiment_distribution.png" width="49%">
  <img src="./figures/FNME2025_sentiment_distribution.png" width="23.3%">
  <img src="./figures/Reddit_sentiment_distribution.png" width="23.1%">
</div>

<br>

**SAFN:** https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news<br>
This dataset contains the sentiments for financial news headlines from the perspective of a retail investor. Further details about the dataset can be found in: Malo, P., Sinha, A., Takala, P., Korhonen, P. and Wallenius, J. (2014): “Good debt or bad debt: Detecting semantic orientations in economic texts.” Journal of the American Society for Information Science and Technology.

**SEntFiN:** https://www.kaggle.com/datasets/ankurzing/aspect-based-sentiment-analysis-for-financial-news<br>
This file contains 10,700+ news headlines for which we have sentiment annotations for all the financial entities that appear in the headlines. Further details about the dataset can be found in: Sinha, A., Kedas, S., Kumar, R., & Malo, P. (2022). SEntFiN 1.0: Entity‐aware sentiment analysis for financial news. Journal of the Association for Information Science and Technology.

**FNME2025:** https://www.kaggle.com/datasets/pratyushpuri/financial-news-market-events-dataset-2025<br>
This synthetic dataset contains 3,024 records of financial news headlines centered around major market events from February 2025 to August 2025. The dataset captures real-time market dynamics, sentiment analysis, and trading patterns across global financial markets, making it ideal for financial analysis, sentiment modeling, and market prediction tasks.

**Reddit:** https://www.kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts/data<br>
Reddit posts from subreddit WallStreetBets, downloaded from https://www.reddit.com/r/wallstreetbets/ using praw (The Python Reddit API Wrapper). WallStreetBets (r/wallstreetbets, also known as WSB), is a subreddit where participants discuss stock and option trading. It has become notable for its profane nature and allegations of users manipulating securities.<br>
***⭐ We randomly selected 300 records from this dataset link which maintained a balanced sample ratio and manually labeled them to obtain a high-quality Reddit dataset.***


# Experiment 1 (Baseline Performance)


<small>
---------------------------------------- Table 1：Baseline Performance in SAFN -----------------------------------------
  
| Baselines | Accuracy | Macro_Precision | Macro_Recall | Macro_F1 | Macro_Specificity | Macro_AUC | Macro_AUPRC |
|------|----------|-----------------|--------------|----------|-------------------|-----------|-------------|
| mlp_tfidf | 0.6795 | 0.6124 | 0.6125 | 0.6122 | 0.8068 | 0.8160 | 0.6928 |
| textcnn | 0.7414 | 0.7064 | 0.6612 | 0.6757 | 0.8264 | 0.8387 | 0.7374 |
| bert | 0.8322 | 0.7930 | 0.8402 | 0.8127 | 0.9058 | 0.9431 | 0.8863 |
| roberta | 0.8446 | 0.8105 | 0.8639 | 0.8328 | 0.9135 | 0.9572 | 0.9254 |
| finbert | 0.8391 | 0.8057 | 0.8466 | 0.8233 | 0.9070 | 0.9257 | 0.8633 |
| qwen2 | 0.5722 | 0.4503 | 0.3728 | 0.3562 | 0.6989 | 0.6204 | 0.4212 |
</small>

<br>
<small>
---------------------------------------- Table 2：Baseline Performance in SEntFiN ----------------------------------------
  
| Baselines | Accuracy | Macro_Precision | Macro_Recall | Macro_F1 | Macro_Specificity | Macro_AUC | Macro_AUPRC |
|-----------|----------|-----------------|--------------|----------|-------------------|-----------|-------------|
| mlp_tfidf | 0.7223 | 0.7163 | 0.7114 | 0.7135 | 0.8575 | 0.8728 | 0.7949 |
| textcnn   | 0.7099 | 0.7225 | 0.6967 | 0.7057 | 0.8458 | 0.8507 | 0.7646 |
| bert      | 0.7991 | 0.7957 | 0.8082 | 0.8002 | 0.8987 | 0.9257 | 0.8737 |
| **roberta** | **0.8543** | **0.8539** | **0.8554** | **0.8546** | **0.9248** | **0.9581** | **0.9286** |
| finbert   | 0.8165 | 0.8142 | 0.8241 | 0.8174 | 0.9074 | 0.9436 | 0.9059 |
| qwen2     | 0.4867 | 0.4682 | 0.4454 | 0.4402 | 0.7267 | 0.6531 | 0.4844 |
</small>

<br>
<div style="text-align: center;">
  <img src="./figures/SAFN_roc_curves.png" width="910" height="500"><br>
  <strong style="font-size: 1.1em;">SAFN ROC</strong>
</div>

<br>
<div style="text-align: center;">
  <img src="./figures/SEntFiN_roc_curves.png" width="910" height="500"><br>
  <strong style="font-size: 1.1em;">SEntFiN ROC</strong>
</div>

<br>
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; justify-items: center;">
  <img src="./figures/mlp_TFIDF_confusion_matrices.png" width="32.7%">
  <img src="./figures/textCNN_confusion_matrices.png" width="32.7%">
  <img src="./figures/bert_confusion_matrices.png" width="32.7%">
</div>

<br>
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; justify-items: center;">
  <img src="./figures/RoBERTa_confusion_matrices.png" width="32.7%">
  <img src="./figures/FinBERT_confusion_matrices.png" width="32.7%">
  <img src="./figures/qwen2_confusion_matrices.png" width="32.7%">
</div>

**Confusion Matrice**
​

# Experiment 2 (Temporal Robustness)

<small>
---------------------------------------------- Table 3：Time Drift in SAFN -----------------------------------------------
  
| Baselines | Accuracy | Macro_Precision | Macro_Recall | Macro_F1 | Macro_Specificity | Macro_AUC | Macro_AUPRC |
|-----------|----------|-----------------|--------------|----------|-------------------|-----------|-------------|
| mlp_tfidf | 0.6795 | 0.6124 | 0.6125 | 0.6122 | 0.8068 | 0.8160 | 0.6928 |
|  **->**   | **0.3312** | **0.3351** | **0.3326** | **0.3100** | **0.6664** | **0.5045** | **0.3361** |
| textcnn   | 0.7414 | 0.7064 | 0.6612 | 0.6757 | 0.8264 | 0.8387 | 0.7374 |
|  **->**   | **0.3298** | **0.3255** | **0.3295** | **0.2842** | **0.6646** | **0.4911** | **0.3278** |
| bert      | 0.8322 | 0.7930 | 0.8402 | 0.8127 | 0.9058 | 0.9431 | 0.8863 |
|  **->**   | **0.3320** | **0.3309** | **0.3312** | **0.3219** | **0.6656** | **0.4964** | **0.3297** |
| roberta   | 0.8446 | 0.8105 | 0.8639 | 0.8328 | 0.9135 | 0.9572 | 0.9254 |
|  **->**   | **0.3320** | **0.3343** | **0.3325** | **0.3138** | **0.6663** | **0.4967** | **0.3293** |
| finbert   | 0.8391 | 0.8057 | 0.8466 | 0.8233 | 0.9070 | 0.9257 | 0.8633 |
|  **->**   | **0.3301** | **0.3303** | **0.3298** | **0.3191** | **0.6649** | **0.4919** | **0.3260** |
| qwen2     | 0.5722 | 0.4503 | 0.3728 | 0.3562 | 0.6989 | 0.6204 | 0.4212 |
|  **->**   | **0.3357** | **0.3479** | **0.3377** | **0.2812** | **0.6689** | **0.5092** | **0.3418** |
</small>

<br>
<small>
---------------------------------------------- Table 4：Time Drift in SEntFiN -----------------------------------------------
  
| Baselines | Accuracy | Macro_Precision | Macro_Recall | Macro_F1 | Macro_Specificity | Macro_AUC | Macro_AUPRC |
|-----------|----------|-----------------|--------------|----------|-------------------|-----------|-------------|
| mlp_tfidf | 0.7223 | 0.7163 | 0.7114 | 0.7135 | 0.8575 | 0.8728 | 0.7949 |
|    **->** | **0.3346** | **0.3428** | **0.3367** | **0.3170** | **0.6685** | **0.4974** | **0.3335** |
| textcnn   | 0.7099 | 0.7225 | 0.6967 | 0.7057 | 0.8458 | 0.8507 | 0.7646 |
|    **->** | **0.3235** | **0.3263** | **0.3245** | **0.3219** | **0.6624** | **0.4933** | **0.3294** |
| bert      | 0.7991 | 0.7957 | 0.8082 | 0.8002 | 0.8987 | 0.9257 | 0.8737 |
|    **->** | **0.3312** | **0.3353** | **0.3313** | **0.3211** | **0.6657** | **0.5026** | **0.3353** |
| roberta   | 0.8543 | 0.8539 | 0.8554 | 0.8546 | 0.9248 | 0.9581 | 0.9286 |
|    **->** | **0.3331** | **0.3377** | **0.3336** | **0.3151** | **0.6669** | **0.4996** | **0.3320** |
| finbert   | 0.8165 | 0.8142 | 0.8241 | 0.8174 | 0.9074 | 0.9436 | 0.9059 |
|    **->** | **0.3309** | **0.3336** | **0.3309** | **0.3203** | **0.6656** | **0.4997** | **0.3339** |
| qwen2     | 0.4867 | 0.4682 | 0.4454 | 0.4402 | 0.7267 | 0.6531 | 0.4844 |
|    **->** | **0.3279** | **0.3265** | **0.3286** | **0.3066** | **0.6643** | **0.5019** | **0.3328** |
</small>

<br>
<div style="text-align: center;">
  <img src="./figures/model_performance_comparison_SAFN_TIME.png" width="910" height="500"><br>
  <strong style="font-size: 1.1em;">SAFN</strong>
</div>

<br>
<div style="text-align: center;">
  <img src="./figures/model_performance_comparison_SEntFiN_TIME.png" width="910" height="500"><br>
  <strong style="font-size: 1.1em;">SEntFiN</strong>
</div>

<br>
<div style="text-align: center;">
  <img src="./figures/performance_decline_barcharts.png" width="910" height="500"><br>
  <strong style="font-size: 1.1em;">Decline</strong>
</div>

# Experiment 3 (Domain Robustness)​​

