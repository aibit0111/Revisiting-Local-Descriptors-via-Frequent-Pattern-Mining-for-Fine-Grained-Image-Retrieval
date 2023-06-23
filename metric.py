
def average_precision(retrieved_items, relevant_items):
    rel_count = 0
    precisions = []

    for i, item in enumerate(retrieved_items, start=1):
        if item in relevant_items:
            rel_count += 1
            precision_at_i = rel_count / i
            precisions.append(precision_at_i)

    if precisions:
        avg_precision = sum(precisions) / len(precisions)
    else:
        avg_precision = 0.0


    return avg_precision

 
def average_recall(retrieved_items, relevant_items):
    rel_count = 0
    recalls = []

    for i, item in enumerate(retrieved_items, start=1):
        if item in relevant_items:
            rel_count += 1
            recall_at_i = rel_count / len(relevant_items)
            recalls.append(recall_at_i)

    if recalls:
        avg_recall = sum(recalls) / len(recalls)
    else:
        avg_recall = 0.0

    return avg_recall

 import matplotlib.pyplot as plt



avg_precisions = []
avg_recalls = []

for i in range(1, len(retrieved_items) + 1):
    avg_precisions.append(average_precision(retrieved_items[:i], relevant_items))
    avg_recalls.append(average_recall(retrieved_items[:i], relevant_items))

plt.figure(figsize=(10,6))
plt.plot(avg_recalls, avg_precisions, marker='o')
plt.xlabel('Average Recall')
plt.ylabel('Average Precision')
plt.title('Precision-Recall Curve')
plt.grid()
plt.show()



import matplotlib.pyplot as plt
from sklearn import metrics

# assuming retrieved_items and relevant_items from the previous example
retrieved_items = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
relevant_items = ['a', 'c', 'e', 'g', 'i']

# creating binary labels for our items
y_true = [1 if item in relevant_items else 0 for item in retrieved_items]

# for an ROC curve, we need scores rather than binary predictions, 
# let's assume the score is simply the reverse rank of the item in the retrieved list
y_scores = [len(retrieved_items) - i for i in range(len(retrieved_items))]

fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()



import matplotlib.pyplot as plt



