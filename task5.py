model = LogisticRegressionModel(input_dim)
criterion = nn.BCELoss()
learning_rate = 0.01
optimizer_with_L2 = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)
for epoch in range(1000):
    model.train()
    optimizer_with_L2.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer_with_L2.step()

model.eval()
torch.no_grad()
test_predictions = model(X_test)

def confusion_matrix(predictions, threshold):
    classifications = (predictions > threshold).float()
    TP = ((classifications == 1.0) & (y_test == 1.0)).sum().item()
    TN = ((classifications == 0.0) & (y_test == 0.0)).sum().item()
    FP = ((classifications == 1.0) & (y_test == 0.0)).sum().item()
    FN = ((classifications == 0.0) & (y_test == 1.0)).sum().item()
    return TP, TN, FP, FN

TP, TN, FP, FN = confusion_matrix(test_predictions, threshold=0.5)
print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN)
actual_positive_cnt = (y_test == 1.0).sum().item()
actual_negative_cnt = (y_test == 0.0).sum().item()

recall = float(TP) / actual_positive_cnt if actual_positive_cnt > 0 else 0.0
fpr = float(FP) / actual_negative_cnt if actual_negative_cnt > 0 else 0.0
precision = float(TP) / (TP + FP) if (TP + FP) > 0 else 0.0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
print(f'TPR (Recall): {recall:.4f}')
print(f'FPR: {fpr:.4f}')
print(f'Precision: {precision:.4f}')
print(f'F1-score: {f1_score:.4f}')

import numpy as np
import matplotlib.pyplot as plt
tpr_list, fpr_list = [],[]
for thresh in torch.arange(0, 1.05, 0.05):
    TP, TN, FP, FN = confusion_matrix(test_predictions, threshold=thresh.item())
    tpr = TP / actual_positive_cnt if actual_positive_cnt > 0 else 0.0
    fpr = FP / actual_negative_cnt if actual_negative_cnt > 0 else 0.0
    tpr_list.append(tpr)
    fpr_list.append(fpr)

# Sort points by FPR (important for correct AUC calculation)
fpr_array = np.array(fpr_list)
tpr_array = np.array(tpr_list)
sorted_indices = np.argsort(fpr_array)
fpr_array = fpr_array[sorted_indices]
tpr_array = tpr_array[sorted_indices]

# Compute AUC using trapezoidal rule
auc = np.trapezoid(tpr_array, fpr_array)
print(f"AUC: {auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr_array, tpr_array, marker='o', label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()