import matplotlib.pyplot as plt
import os

def plot_epsilon_delta():
    # Data from Table 1 in the paper
    epsilon = [0.1, 0.2, 0.3, 0.5]
    mean_delta = [0.0, -0.018, -0.028, -0.050]

    plt.figure(figsize=(8, 6))
    plt.plot(epsilon, mean_delta, marker='o', linestyle='-', color='b', linewidth=2)
    plt.axhline(0, linestyle='--', color='r', alpha=0.5, label="Clean Baseline")
    
    plt.xlabel("Perturbation Strength ($\epsilon$)", fontsize=12)
    plt.ylabel("Mean Entropy Shift ($\Delta$)", fontsize=12)
    plt.title("Entropy Response Under Visual Perturbation", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Save figure
    output_dir = "results/figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "figure_1_epsilon_delta.png")
    plt.savefig(output_path, dpi=300)
    print(f"Figure 1 saved to {output_path}")

if __name__ == "__main__":
    plot_epsilon_delta()
