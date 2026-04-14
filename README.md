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
  <img src="./figures/sentiment_distribution.png" width="50%">
  <img src="./figures/FNME2025_sentiment_distribution.png" width="23.7%">
  <img src="./figures/Reddit_sentiment_distribution.png" width="23.6%">
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

This experiment evaluated the sentiment classification performance of six types of models using two datasets with different data characteristics (SAFN and SentFiN). The SAFN dataset contains 4,846 samples with a highly imbalanced sentiment distribution (neutral 59.4%, positive 28.1%, negative 12.5%), while the SentFiN dataset contains 10,753 samples with a relatively balanced distribution (neutral 42.2%, positive 32.6%, negative 25.2%). This design allows systematic investigation of the impact of data scale and class balance on model performance.

<small>
  
Table 1：Baseline Performance Comparison
  
| Model | Dataset | Accuracy | Macro_Precision | Macro_Recall | Macro_F1 | Macro_Specificity | Macro_AUC | Macro_AUPRC |
|-------|---------|----------|-----------------|--------------|----------|-------------------|-----------|-------------|
| mlp_tfidf | SAFN | 0.6795 | 0.6124 | 0.6125 | 0.6122 | 0.8068 | 0.8160 | 0.6928 |
|           | SentFiN | 0.7223 | 0.7163 | 0.7114 | 0.7135 | 0.8575 | 0.8728 | 0.7949 |
| textcnn | SAFN | 0.7414 | 0.7064 | 0.6612 | 0.6757 | 0.8264 | 0.8387 | 0.7374 |
|         | SentFiN | 0.7099 | 0.7225 | 0.6967 | 0.7057 | 0.8458 | 0.8507 | 0.7646 |
| bert | SAFN | 0.8322 | 0.7930 | 0.8402 | 0.8127 | 0.9058 | 0.9431 | 0.8863 |
|      | SentFiN | 0.7991 | 0.7957 | 0.8082 | 0.8002 | 0.8987 | 0.9257 | 0.8737 |
| roberta | SAFN | 0.8446 | 0.8105 | 0.8639 | 0.8328 | 0.9135 | 0.9572 | 0.9254 |
|         | SentFiN | 0.8543 | 0.8539 | 0.8554 | 0.8546 | 0.9248 | 0.9581 | 0.9286 |
| finbert | SAFN | 0.8391 | 0.8057 | 0.8466 | 0.8233 | 0.9070 | 0.9257 | 0.8633 |
|         | SentFiN | 0.8165 | 0.8142 | 0.8241 | 0.8174 | 0.9074 | 0.9436 | 0.9059 |
| qwen2 | SAFN | 0.5722 | 0.4503 | 0.3728 | 0.3562 | 0.6989 | 0.6204 | 0.4212 |
|       | SentFiN | 0.4867 | 0.4682 | 0.4454 | 0.4402 | 0.7267 | 0.6531 | 0.4844 |
</small>

##  Impact of Data Characteristics on Model Performance
Data scale and class balance significantly affect overall model performance. On the SAFN dataset, the average accuracy across models is 0.678±0.059, which improves to 0.704±0.099 on SentFiN, representing a 3.8% relative improvement. This improvement is particularly pronounced in traditional models (MLP+TF-IDF, TextCNN) with an average increase of 0.042, while pretrained models (BERT series) show a smaller improvement (average 0.022), indicating that traditional models are more dependent on data quality, whereas pretrained models can leverage their knowledge base to mitigate class imbalance effects.<br>

Class imbalance substantially impairs the recognition of minority categories. On SAFN, all models show significantly reduced capability in identifying negative samples. For instance, MLP+TF-IDF achieves a recall of only 0.54 for the negative class, compared to 0.79 for neutral and 0.73 for positive. In contrast, on the balanced SentFiN dataset, RoBERTa demonstrates consistent recall across all three classes (neutral 0.84, positive 0.87, negative 0.86). ROC analysis further confirms the impact of class balance: the average AUC for the negative class in SAFN is only 0.742, while the mean AUC across all three classes in SentFiN reaches 0.876, indicating that balanced data enhances discrimination consistency across categories. This finding aligns with He and Garcia's [3] conclusions on imbalanced learning.<br>

## Model Architecture and Generalization Capability
Different model architectures exhibit distinct performance patterns across both datasets. Pretrained language models (BERT, RoBERTa, FinBERT) significantly outperform traditional models on both SAFN and SentFiN (p<0.01, paired t-test), with RoBERTa achieving the best performance in both tasks (SAFN: 0.8155, SentFiN: 0.8568). Notably, FinBERT—specifically designed for finance—slightly underperforms the general-purpose RoBERTa, possibly due to domain mismatch between its pretraining corpus (financial news) and the application context (financial social media), and reflecting the influence of model capacity (RoBERTa 125M vs FinBERT 110M parameters). This supports Devlin et al.'s [1] conclusions regarding pretrained models acquiring general language understanding through self-supervised learning.<br>

Evaluation of generalization capability shows a similar trend. The AUPRC metric indicates that traditional models (MLP+TF-IDF: 0.6455) are substantially worse than pretrained models (RoBERTa: 0.8411) on the imbalanced SAFN dataset, demonstrating superior robustness of pretrained models to data imbalance. Qwen2 performs the worst across all experiments (SentFiN accuracy only 0.4867), likely due to mismatches between its generative architecture and classification tasks, limitations of training only the classification head, and computational resource constraints. This observation aligns with Raffel et al.'s [2] findings on task-specific adaptation for text-to-text models.

<br>
<div style="text-align: center;">
  <img src="./figures/SAFN_roc_curves.png" width="907" height="500"><br>
  <strong style="font-size: 1.1em;">SAFN ROC</strong>
</div>

<br>
<div style="text-align: center;">
  <img src="./figures/SEntFiN_roc_curves.png" width="907" height="500"><br>
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

[1] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019, June). Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers) (pp. 4171-4186).

[2] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140), 1-67.

[3] He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on knowledge and data engineering, 21(9), 1263-1284.


# Experiment 2 (Temporal Robustness)

<small>
Table 2：Time Drift in SAFN
  
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
Table 3：Time Drift in SEntFiN
  
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
  <img src="./figures/model_performance_comparison_SAFN_TIME.png" width="907" height="500"><br>
  <strong style="font-size: 1.1em;">SAFN</strong>
</div>

<br>
<div style="text-align: center;">
  <img src="./figures/model_performance_comparison_SEntFiN_TIME.png" width="907" height="500"><br>
  <strong style="font-size: 1.1em;">SEntFiN</strong>
</div>

<br>
<div style="text-align: center;">
  <img src="./figures/performance_decline_barcharts.png" width="910" height="500"><br>
  <strong style="font-size: 1.1em;">Decline</strong>
</div>

# Experiment 3 (Domain Robustness)​​

