import numpy as np
from scipy import stats

poisson_emb_mrr = [0.788, 0.788, 0.787, 0.787, 0.787, 0.785, 0.787, 0.784, 0.787, 0.788]
poisson_emb_recall = [0.988, 0.989, 0.989, 0.988, 0.988, 0.989, 0.989, 0.989, 0.988, 0.989]

sentence_poisson_emb_mrr = [0.783, 0.783, 0.785, 0.783, 0.784, 0.781, 0.784, 0.781, 0.784, 0.787]
sentence_poisson_emb_recall = [0.988, 0.989, 0.989, 0.989, 0.988, 0.989, 0.989, 0.989, 0.989, 0.989]

def execute_significance_test(metric_name, first_classifier_metrics, second_classifier_metrics, alpha=0.01):
    print(f'{metric_name} results (alpha={alpha}):')
    t_statistic, p_value = stats.ttest_rel(first_classifier_metrics, second_classifier_metrics)

    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")

    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference between the classifiers.")
    else:
        print("Fail to reject the null hypothesis. There is no significant difference between the classifiers.")


execute_significance_test('Mean reciprocal rank', poisson_emb_mrr, sentence_poisson_emb_mrr)
execute_significance_test('Recall', poisson_emb_recall, sentence_poisson_emb_recall)