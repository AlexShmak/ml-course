from copy import deepcopy
from itertools import product

import numpy as np
from matplotlib import pyplot as plt

from howework.task6.utils.nn import NeuralNetwork
from howework.task6.utils.optimizer import SGD, Momentum, Adam


def grid_search_hyperparameters(X_train, y_train, X_test, y_test, classes_num, nn_model_factory):
    """
    Perform grid search to find optimal hyperparameters for neural network.

    Parameters:
    - X_train, y_train: Training data and labels
    - X_test, y_test: Test data and labels
    - classes_num: Number of output classes
    - nn_model_factory: Function that creates and configures a new neural network
                       with the specified hyperparameters

    Returns:
    - best_params: Dictionary of best hyperparameters
    - best_model: Trained model with best hyperparameters
    - results: List of dictionaries with results for all parameter combinations
    """
    # Define hyperparameter grid including optimizer type
    param_grid = {
        'hidden_neurons_num': [32, 64, 128],
        'learning_rate': [0.001, 0.01, 0.1],
        'reg_strength': [1e-3, 1e-4, 1e-5],
        'batch_size': [16, 32, 64],
        'epochs': [500],  # Fixed for grid search to save time
        'optimizer': ['SGD', 'Momentum', 'Adam']
    }

    # Generate all combinations of hyperparameters
    param_combinations = list(product(
        param_grid['hidden_neurons_num'],
        param_grid['learning_rate'],
        param_grid['reg_strength'],
        param_grid['batch_size'],
        param_grid['epochs'],
        param_grid['optimizer']
    ))

    # Store results
    results = []
    best_val_accuracy = 0
    best_model = None
    best_params = None

    print(f"Running grid search with {len(param_combinations)} hyperparameter combinations...")

    # Run grid search
    for i, (hidden_neurons_num, learning_rate, reg_strength, batch_size, epochs, optimizer_type) in enumerate(
            param_combinations):
        print(f"\nCombination {i + 1}/{len(param_combinations)}")
        print(f"Parameters: hidden_neurons_num={hidden_neurons_num}, learning_rate={learning_rate}, " +
              f"reg_strength={reg_strength}, batch_size={batch_size}, optimizer={optimizer_type}")

        # Initialize model with current hyperparameters using the factory function
        model = nn_model_factory(
            input_size=X_train.shape[1],
            hidden_size=hidden_neurons_num,
            output_size=classes_num,
            reg_strength=reg_strength
        )

        # Choose the optimizer based on the type
        if optimizer_type == 'SGD':
            optimizer = SGD(learning_rate=learning_rate)
        elif optimizer_type == 'Momentum':
            optimizer = Momentum(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_type == 'Adam':
            optimizer = Adam(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Train model with the chosen optimizer
        model.fit(
            X_train, y_train,
            optimizer=optimizer,
            batch_size=batch_size,
            epochs=epochs,
            reg_strength=reg_strength
        )

        # Evaluate on test set
        test_preds = model.predict(X_test)
        test_accuracy = np.mean(test_preds == y_test)

        # Calculate train accuracy at the end of training
        train_preds = model.predict(X_train)
        train_accuracy = np.mean(train_preds == y_train)

        # Store results
        result = {
            'hidden_neurons_num': hidden_neurons_num,
            'learning_rate': learning_rate,
            'reg_strength': reg_strength,
            'batch_size': batch_size,
            'epochs': epochs,
            'optimizer': optimizer_type,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'final_loss': model.loss_history[-1]
        }
        results.append(result)

        print(f"Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")

        # Update best model if needed
        if test_accuracy > best_val_accuracy:
            best_val_accuracy = test_accuracy
            best_model = deepcopy(model)
            best_params = {
                'hidden_neurons_num': hidden_neurons_num,
                'learning_rate': learning_rate,
                'reg_strength': reg_strength,
                'batch_size': batch_size,
                'epochs': epochs,
                'optimizer': optimizer_type
            }
            print("New best model found!")

    # Sort results by test accuracy
    results.sort(key=lambda x: x['test_accuracy'], reverse=True)

    print("\nGrid Search Results:")
    print(f"Best test accuracy: {best_val_accuracy:.4f}")
    print(f"Best hyperparameters: {best_params}")

    return best_params, best_model, results


def visualize_hyperparameter_search(results, param_name):
    """
    Visualize the effect of a specific hyperparameter on model performance.

    Parameters:
    - results: List of dictionaries with grid search results
    - param_name: Name of the hyperparameter to visualize
    """
    # Group results by the parameter of interest
    param_values = sorted(set(result[param_name] for result in results))

    # Calculate average test accuracy for each parameter value
    avg_accuracies = []
    for value in param_values:
        matching_results = [r for r in results if r[param_name] == value]
        avg_accuracy = sum(r['test_accuracy'] for r in matching_results) / len(matching_results)
        avg_accuracies.append(avg_accuracy)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, avg_accuracies, 'o-', linewidth=2)
    plt.xlabel(param_name)
    plt.ylabel('Average Test Accuracy')
    plt.title(f'Effect of {param_name} on Model Performance')
    plt.grid(True)

    # For learning rate, use log scale on x-axis
    if param_name == 'learning_rate':
        plt.xscale('log')

    plt.tight_layout()
    plt.show()


def visualize_best_model_training(model):
    """
    Visualize the training history of the best model.

    Parameters:
    - model: Trained neural network model
    """
    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(model.accuracy_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Example usage:
def create_neural_network(input_size, hidden_size, output_size, reg_strength):
    """
    Factory function to create and configure a neural network.

    Parameters:
    - input_size: Number of input features
    - hidden_size: Number of neurons in hidden layer
    - output_size: Number of output classes
    - reg_strength: L2 regularization strength

    Returns:
    - Configured NeuralNetwork instance
    """
    from howework.task6.utils.layer import FullyConnected, ReLU, BatchNorm

    model = NeuralNetwork()

    # Example structure: Input -> FC -> BatchNorm -> ReLU -> FC
    model.build(
        FullyConnected(input_size, hidden_size),
        ReLU(),
        BatchNorm(hidden_size),
        FullyConnected(hidden_size, output_size)
    )

    return model
