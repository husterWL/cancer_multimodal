from sklearn.metrics import f1_score, classification_report, accuracy_score

def api_metric(true_labels, pred_labels):
    # return f1_score(true_labels, pred_labels)
    print(classification_report(true_labels, pred_labels, target_names = ['benign', 'malignant'], labels = [0, 1], digits = 4))
    report_dict = classification_report(true_labels, pred_labels, target_names = ['benign', 'malignant'], labels = [0, 1], digits = 4, output_dict = True)
    return accuracy_score(true_labels, pred_labels), report_dict