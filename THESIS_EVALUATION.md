# Evaluation Framework

## 1. Evaluation Philosophy

The evaluation framework is designed to answer three questions:

1. **Predictive performance:** How well does the model predict price movements?
2. **Statistical significance:** Are differences between models meaningful or due to chance?
3. **Practical utility:** Can the model improve trading outcomes?

## 2. Classification Metrics

### 2.1 Primary Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **Accuracy** | Correct / Total | [0, 1] | Overall correctness |
| **Macro F1** | Mean(F1_class) | [0, 1] | Class-balanced performance |
| **Weighted F1** | Σ(w_c × F1_c) | [0, 1] | Sample-weighted performance |
| **MCC** | See below | [-1, 1] | Correlation coefficient |
| **Cohen's Kappa** | (p_o - p_e)/(1 - p_e) | [-1, 1] | Agreement beyond chance |

**Matthews Correlation Coefficient (MCC):**
```
MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

For multi-class: computed as correlation between predicted and actual labels.

### 2.2 Why Macro F1 as Primary?

**Class imbalance consideration:**
- Accuracy can be misleading with imbalanced classes
- Macro F1 treats all classes equally
- Prevents model from ignoring minority classes

**Example:**
- Class distribution: Up=30%, Stat=40%, Down=30%
- Model predicting all "Stationary" achieves 40% accuracy
- But Macro F1 ≈ 0.19 (reveals poor performance)

### 2.3 Per-Class Metrics

For each class c ∈ {Down, Stationary, Up}:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision_c** | TP_c / (TP_c + FP_c) | "When predicting c, how often correct?" |
| **Recall_c** | TP_c / (TP_c + FN_c) | "Of actual c, how many detected?" |
| **F1_c** | 2 × (P_c × R_c) / (P_c + R_c) | Harmonic mean of P and R |

**Implementation:**
```python
def compute_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=[0,1,2]),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    # Per-class metrics
    for i, cls in enumerate(["down", "stat", "up"]):
        metrics[f"precision_{cls}"] = precision_score(..., labels=[i])
        metrics[f"recall_{cls}"] = recall_score(..., labels=[i])
        metrics[f"f1_{cls}"] = f1_score(..., labels=[i])
    return metrics
```

### 2.4 Probabilistic Metrics

When probability predictions are available:

| Metric | Description | Optimal |
|--------|-------------|---------|
| **ROC-AUC (macro)** | Area under ROC curve, averaged | 1.0 |
| **ROC-AUC (weighted)** | Weighted by class frequency | 1.0 |
| **ECE** | Expected Calibration Error | 0.0 |
| **MCE** | Maximum Calibration Error | 0.0 |
| **Brier Score** | Mean squared probability error | 0.0 |

## 3. Confusion Matrix Analysis

### 3.1 Structure

```
                    Predicted
                Down  Stat   Up
Actual  Down    [TN_d] [...]  [...]
        Stat    [...]  [TN_s] [...]
        Up      [...]  [...]  [TN_u]
```

### 3.2 Normalization

**Row-normalized (recall focus):**
```
cm_norm[i,j] = cm[i,j] / Σ_j cm[i,j]
```
Shows: "Of actual class i, what fraction predicted as j?"

**Column-normalized (precision focus):**
```
cm_norm[i,j] = cm[i,j] / Σ_i cm[i,j]
```
Shows: "Of predicted class j, what fraction actually i?"

### 3.3 Visualization

```python
def plot_confusion_matrix(cm, model_name, split, save_path):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)  # Row-normalize
    plt.imshow(cm_norm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{cm_norm[i,j]:.1%}")
```

## 4. Statistical Significance Testing

### 4.1 Why Statistical Tests?

**Problem:** Model A achieves 52% accuracy, Model B achieves 51%. Is A better?

**Answer:** Depends on:
- Number of test samples
- Variance in predictions
- Correlation between model errors

Statistical tests quantify the probability that observed differences are due to chance.

### 4.2 McNemar's Test

Tests whether two classifiers make different errors:

```
H0: P(A correct, B wrong) = P(A wrong, B correct)
```

**Test statistic:**
```
χ² = (|b - c| - 1)² / (b + c)
```

Where:
- b = samples where A correct, B wrong
- c = samples where A wrong, B correct

**Implementation:**
```python
def mcnemar_test(y_true, y_pred_a, y_pred_b):
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    chi2 = (abs(b - c) - 1)**2 / (b + c)
    p_value = 1 - chi2.cdf(chi2, df=1)
    return chi2, p_value
```

**Interpretation:**
- p < 0.05: Significant difference
- p ≥ 0.05: No significant difference

### 4.3 Paired Permutation Test

Non-parametric test for accuracy differences:

```python
def paired_permutation_test(y_true, y_pred_a, y_pred_b, n_permutations=10000):
    observed_diff = accuracy(y_pred_a) - accuracy(y_pred_b)
    
    perm_diffs = []
    for _ in range(n_permutations):
        # Randomly swap predictions between models
        swap = np.random.random(len(y_true)) < 0.5
        perm_pred_a = np.where(swap, y_pred_b, y_pred_a)
        perm_pred_b = np.where(swap, y_pred_a, y_pred_b)
        perm_diffs.append(accuracy(perm_pred_a) - accuracy(perm_pred_b))
    
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    return observed_diff, p_value
```

**Why permutation test?**
- No distributional assumptions
- Exact for finite samples
- Accounts for correlated errors

### 4.4 Bootstrap Confidence Intervals

Estimate uncertainty in metrics:

```python
def bootstrap_confidence_interval(y_true, y_pred, metric_fn, n_bootstrap=1000):
    point_estimate = metric_fn(y_true, y_pred)
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        bootstrap_metrics.append(metric_fn(y_true[idx], y_pred[idx]))
    
    lower = np.percentile(bootstrap_metrics, 2.5)
    upper = np.percentile(bootstrap_metrics, 97.5)
    return point_estimate, lower, upper
```

### 4.5 Multi-Model Comparison (Friedman + Nemenyi)

When comparing >2 models across multiple datasets:

**Friedman test:**
- H0: All models perform equally
- Ranks models on each dataset
- Tests if average ranks differ significantly

**Nemenyi post-hoc:**
- If Friedman rejects H0, identifies which pairs differ
- Critical difference (CD) determines significance

```python
def multi_model_comparison(scores_matrix, model_names):
    # Friedman test
    stat, p_value = friedmanchisquare(*[scores_matrix[:, i] for i in range(n_models)])
    
    # Nemenyi post-hoc if significant
    if p_value < 0.05:
        cd = nemenyi_critical_difference(n_models, n_datasets)
        avg_ranks = compute_average_ranks(scores_matrix)
        significant_pairs = [(i, j) for i, j in combinations 
                            if abs(avg_ranks[i] - avg_ranks[j]) > cd]
```

## 5. Output Formats

### 5.1 Console Output

```
======================================================================
  TEST Metrics:
  Metric                      Value
  ----------------------------------
  Accuracy                    0.5234
  Macro F1                    0.4567
  Weighted F1                 0.5012
  Mcc                         0.2845
  Cohen Kappa                 0.2756
  Balanced Accuracy           0.4892
  ----------------------------------
  ROC-AUC (macro)             0.6823
  ----------------------------------
  ECE                         0.0834
  MCE                         0.1523
  BRIER_SCORE                 0.4123
  ----------------------------------
  F1-Down                     0.4234
  F1-Stat                     0.5123
  F1-Up                       0.4345
```

### 5.2 CSV Export

```csv
Model,Ticker,Test Accuracy,Test Macro F1,Test MCC,Test ECE
TLOB,CSCO,0.5234,0.4567,0.2845,0.0834
TLOB-Decay,CSCO,0.5456,0.4789,0.3012,0.0712
DeepLOB,CSCO,0.5012,0.4234,0.2534,0.0956
```

### 5.3 JSON Export

```json
{
  "model": "TLOB-Decay",
  "ticker": "CSCO",
  "test_accuracy": 0.5456,
  "test_macro_f1": 0.4789,
  "test_mcc": 0.3012,
  "per_class": {
    "down": {"precision": 0.48, "recall": 0.42, "f1": 0.45},
    "stat": {"precision": 0.55, "recall": 0.62, "f1": 0.58},
    "up": {"precision": 0.46, "recall": 0.40, "f1": 0.43}
  },
  "calibration": {
    "ece": 0.0712,
    "mce": 0.1234,
    "brier_score": 0.3856
  }
}
```

### 5.4 LaTeX Tables

```python
def metrics_to_latex(metrics, model_name):
    return f"""
\\begin{{table}}[h]
\\centering
\\caption{{Performance metrics for {model_name}}}
\\begin{{tabular}}{{lr}}
\\toprule
Metric & Value \\\\
\\midrule
Accuracy & {metrics['accuracy']:.4f} \\\\
Macro F1 & {metrics['macro_f1']:.4f} \\\\
MCC & {metrics['mcc']:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
```

## 6. Visualization Outputs

### 6.1 Generated Plots

| Plot | Purpose | Format |
|------|---------|--------|
| `confusion_matrix_{split}.png` | Class-wise errors | PNG, 150 DPI |
| `training_history.png` | Loss/F1 over epochs | PNG, 150 DPI |
| `reliability_diagram.png` | Calibration | PNG, 150 DPI |
| `decay_curves.png` | Learned decay rates | PNG, 150 DPI |

### 6.2 Plot Style

Consistent visual style across all plots:

```python
COLORS = {
    "green": "#A8D5BA",   # Success, positive
    "red": "#F5B7B1",     # Error, negative
    "blue": "#AED6F1",    # Primary accent
    "orange": "#FAD7A0",  # Secondary accent
    "purple": "#D7BDE2",  # Tertiary
    "teal": "#A3E4D7",    # Quaternary
    "gray": "#D5D8DC",    # Neutral
    "yellow": "#F9E79F"   # Highlight
}
```

## 7. Evaluation Pipeline

### 7.1 Full Evaluation Run

```python
def run_evaluation(model, train_loader, val_loader, test_loader, output_dir):
    results = {}
    
    # 1. Evaluate on all splits
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        metrics, labels, preds, probs = evaluate_model(model, loader)
        results[split] = metrics
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        plot_confusion_matrix(cm, model_name, split, output_dir / f"cm_{split}.png")
    
    # 2. Print test metrics
    print_metrics(results["test"], "test")
    
    # 3. Save results
    save_results_json(results, output_dir / "results.json")
    save_results_csv(results, output_dir / "metrics.csv")
    
    return results
```

### 7.2 Model Comparison

```python
def compare_models(models, test_loader, y_true):
    predictions = {name: get_predictions(model, test_loader) 
                   for name, model in models.items()}
    
    # Pairwise comparisons
    comparisons = compare_all_models(y_true, predictions)
    print_comparison_table(comparisons)
    
    # Multi-model comparison
    scores = np.array([[metrics[name] for name in models] 
                       for metrics in [f1_scores]])  # Per dataset
    multi_model_result = multi_model_comparison(scores, list(models.keys()))
    
    return comparisons, multi_model_result
```

## 8. Best Practices

### 8.1 Avoid Evaluation Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Data leakage | Future info in training | Temporal split |
| Cherry-picking | Report best run only | Report mean ± std |
| Ignoring variance | Single test result | Bootstrap CIs |
| Wrong baseline | Compare to trivial | Include random/majority baseline |
| Metric mismatch | Optimize for accuracy, report F1 | Consistent metrics |

### 8.2 Reporting Checklist

- [ ] Accuracy, Macro F1, and MCC reported
- [ ] Per-class metrics included
- [ ] Confidence intervals provided
- [ ] Statistical significance tests performed
- [ ] Confusion matrices visualized
- [ ] Calibration metrics (if probabilistic)
- [ ] Comparison to baseline(s)
