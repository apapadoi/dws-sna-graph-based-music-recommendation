import numpy as np
from scipy import stats

poisson_emb_mrr = [0.409, 0.408, 0.408, 0.409, 0.406, 0.407, 0.409, 0.410, 0.408, 0.408]
poisson_emb_recall = [0.601, 0.601, 0.600, 0.601, 0.603, 0.599, 0.597, 0.598, 0.599, 0.600]

sentence_poisson_emb_mrr = [0.418, 0.418, 0.417, 0.418, 0.416, 0.415, 0.417, 0.420, 0.417, 0.418]
sentence_poisson_emb_recall = [0.645, 0.645, 0.645, 0.649, 0.644, 0.644, 0.643, 0.640, 0.644, 0.644]

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