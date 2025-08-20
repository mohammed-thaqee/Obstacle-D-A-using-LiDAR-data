import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
train_df = pd.read_csv("data3d/train.csv")
test_df = pd.read_csv("data3d/test.csv")

# Plot train distribution
plt.figure(figsize=(6,4))
train_df["recommended_maneuver"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Train Label Distribution")
plt.ylabel("Count")
plt.xlabel("Maneuver")
plt.tight_layout()
plt.savefig("train_distribution.png")
plt.show()

# Plot test distribution
plt.figure(figsize=(6,4))
test_df["recommended_maneuver"].value_counts().plot(kind="bar", color="lightgreen")
plt.title("Test Label Distribution")
plt.ylabel("Count")
plt.xlabel("Maneuver")
plt.tight_layout()
plt.savefig("test_distribution.png")
plt.show()


