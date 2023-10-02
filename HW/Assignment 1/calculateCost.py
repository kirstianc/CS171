# calculate the total cost of the model, log loss already calculated in log_loss.txt

with open("log_loss.txt", "r") as file:
    log_loss_values = [float(line.strip()) for line in file]

total_cost = sum(log_loss_values) / len(log_loss_values)

print("Total Cost:", total_cost)
