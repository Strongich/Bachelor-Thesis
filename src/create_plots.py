from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt


def analyze_series(series):
    plt.figure(figsize=(12, 6))
    # 1. Output describe()
    describe_output = series.describe()
    print("Describe:")
    print(describe_output)
    print()

    # 2. Calculate skewness and kurtosis
    series_skewness = skew(series)
    series_kurtosis = kurtosis(series)

    print("Skewness:", series_skewness)
    print("Kurtosis:", series_kurtosis)
    plt.hist(series)
    plt.xlabel("Price")
    plt.grid(True)
    plt.show()
    

def plot_dictionaries(dict1, dict2, initial_train, initial_val, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with two subplots

    # Plot for dictionary 1
    for key, values in dict1.items():
        axes[0].plot(range(1, len(values) + 1), values, label=key, marker="*")
        axes[0].legend()
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("MSE Loss")
        axes[0].set_title(f"Train losses, initial: {initial_train.round(2)}")
        axes[0].grid(True)
    # Plot for dictionary 2
    for key, values in dict2.items():
        axes[1].plot(range(1, len(values) + 1), values, label=key, marker="*")
        axes[1].legend()
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("MSE Loss")
        axes[1].set_title(f"Validation losses, initial: {initial_val.round(2)}")
        axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
