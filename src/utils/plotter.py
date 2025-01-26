import matplotlib.pyplot as plt

epochs = [1, 2, 3]
training_loss = [0.6426, 0.5873, 0.5756]
validation_loss = [0.6124, 0.5721, 0.5648]

checkpoints = ["Epoch 1", "Epoch 2", "Epoch 3"]
wer_metrics = [1.2, 0.95, 0.84]  # WER in percentage
cer_metrics = [0.5, 0.3, 0.24]   # CER in percentage

# Plot 1: Loss Curve
def plot_loss_curve(epochs, training_loss, validation_loss):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, training_loss, marker='o', label='Training Loss')
    plt.plot(epochs, validation_loss, marker='s', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")  # Save the figure as a file
    plt.show()

# Plot 2: WER/CER Analysis
def plot_wer_cer_analysis(checkpoints, wer_metrics, cer_metrics):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(checkpoints))
    width = 0.4  # Width of the bars
    ax.bar([p - width/2 for p in x], wer_metrics, width=width, label='WER (%)')
    ax.bar([p + width/2 for p in x], cer_metrics, width=width, label='CER (%)')

    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints)
    ax.set_xlabel('Checkpoints')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('WER and CER Metrics Across Checkpoints')
    ax.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("wer_cer_analysis.png")  # Save the figure as a file
    plt.show()

# Call the plotting functions
plot_loss_curve(epochs, training_loss, validation_loss)
plot_wer_cer_analysis(checkpoints, wer_metrics, cer_metrics)
