# Save the model
torch.save(model.state_dict(), 'L2_trained_AH_model.pth')

# Load the model
loaded_model = LogisticRegressionModel(input_dim)
loaded_model.load_state_dict(torch.load('L2_trained_AH_model.pth'))

# Ensure the loaded model is in evaluation mode
loaded_model.eval()
torch.no_grad()

# Evaluate the loaded model
test_predictions = loaded_model(X_test)
TP, TN, FP, FN = confusion_matrix(test_predictions, threshold=0.5)
accuracyRate = float(TP + TN) / (TP + TN + FP + FN)
print('accuracy:', accuracyRate)
print('True Positives:', TP, 'True Negatives:', TN, 'False Positives:', FP, 'False Negatives:', FN)

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