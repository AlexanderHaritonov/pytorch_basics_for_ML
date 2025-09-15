criterion = nn.BCELoss()
def train_and_compute_accuracy(learning_rate, epochs):
    model = LogisticRegressionModel(input_dim)
    optimizer_with_L2 = torch.optim.SGD(model.parameters(), lr, weight_decay=0.01)
    for epoch in range(epochs):
        model.train()
        optimizer_with_L2.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer_with_L2.step()
    model.eval()
    torch.no_grad()
    test_predictions = model(X_test)
    test_classifications = (test_predictions > threshold).float()
    accuracy = (test_classifications == y_test).float().mean()
    return accuracy, model

learning_rates_to_test = [0.01, 0.05, 0.1]
accuracies, models = [], []
for lr in learning_rates_to_test:
    accuracy, model = train_and_compute_accuracy(learning_rate, 100)
    accuracies.append(accuracy)
    models.append(model)

best_index = max(range(len(learning_rates_to_test)), key=lambda i: accuracies[i])
print('best learning rate is', learning_rates_to_test[best_index], ', accuracy: ', accuracies[best_index].item())