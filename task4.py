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
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs_number}], Loss: {loss.item():.4f}")
        
model.eval()
torch.no_grad()
test_predictions = model(X_test)
train_predictions = model(X_train)

threshold = 0.5
test_classifications = (test_predictions > threshold).float()
train_classifications = (train_predictions > threshold).float()

test_accuracy = (test_classifications == y_test).float().mean()
train_accuracy = (train_classifications == y_train).float().mean()
print(f"Test Accuracy: {test_accuracy.item():.4f}")
print(f"Train Accuracy: {train_accuracy.item():.4f}")