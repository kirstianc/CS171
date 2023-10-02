# to plot the log loss values vs. iteration for report

import matplotlib.pyplot as plt

# read log loss values from the text file
with open('log_loss.txt', 'r') as file:
    log_losses = [float(line.strip()) for line in file]

# create a plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(log_losses) + 1), log_losses, marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('Log Loss vs. Iteration')
plt.grid(True)
plt.show()
