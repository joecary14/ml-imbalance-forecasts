import matplotlib.pyplot as plt

def plot_predictions_vs_actuals(predicted, actual):
    """
    Plots predicted data vs actual data as a scatter plot, alongside the line y = x.

    Parameters:
    predicted (array-like): Predicted data
    actual (array-like): Actual data
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.5, label='Predicted vs Actual')
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color='red', linestyle='--', label='y = x')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual Scatter Plot')
    plt.legend()
    plt.grid(True)
    plt.show()