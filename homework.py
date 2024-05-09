"""
This program trains an MLP to recognize symmetric binary numbers.

"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def create_dataset(size, n_digits=6):
    X = np.random.randint(0, 2, (size, n_digits), dtype=int)
    y = np.ones(size, dtype=int)
    for i in range(n_digits):
        y *= X[:, i] == X[:, n_digits-i-1]
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


def create_model(n_digits=6, hidden_units=2, output_size=2, weight_range=0.3):

    # Create a simple MLP
    model = nn.Sequential(
        nn.Linear(n_digits, hidden_units),
        nn.Sigmoid(),
        nn.Linear(hidden_units, output_size),
        nn.Softmax(dim=1)
    )

    # Initialize weights within the given range
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            torch.nn.init.uniform_(layer.weight, -weight_range, weight_range)
            torch.nn.init.uniform_(layer.bias, -weight_range, weight_range)

    return model


def main(n_epoch=1000, size=500):
    X, y = create_dataset(size)
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    accuracy = []

    for epoch in range(n_epoch):
        optimizer.zero_grad()
        output = model(X.float())
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        acc = (output.argmax(dim=1) == y).float().mean()
        accuracy.append(acc)

    # print("\nTesting model")
    X_test, y_test = create_dataset(size)
    output = model(X_test.float())

    accuracy = (output.argmax(dim=1) == y_test).float().mean()
    print(f"\nAccuracy: {accuracy.item()}")

    return accuracy


if __name__ == '__main__':
    for _ in range(3):
        main()
