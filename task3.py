epochs_number = 1000
for epoch in range(epochs_number):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
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