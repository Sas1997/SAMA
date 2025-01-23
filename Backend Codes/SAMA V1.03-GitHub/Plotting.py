import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Read data from .csv file, only the first and second columns
path = 'Outputforplotting - Sacramento.csv'
Plotting = pd.read_csv(path, header=None).values

# Get values from the first and second columns
software_1_values = np.array(Plotting[:, 6])
software_2_values = np.array(Plotting[:, 7])

# Compute the absolute differences
absolute_differences = np.abs(software_1_values - software_2_values)

# Compute the relative errors (in percentage)
relative_errors = (absolute_differences / software_2_values) * 100

MAE=(np.mean(absolute_differences)/8760)*100

# Create a range representing hours in a year
hours = range(8760)
plt.rcParams["font.family"] = "Times New Roman"

# Increase font size
plt.rcParams.update({'font.size': 28})

# Create a new figure and set its size and dpi for clarity
plt.figure(figsize=[20,10], dpi=300)

plt.plot(hours, software_1_values, label='SAMA', linestyle='solid', color='r', linewidth=1)
plt.plot(hours, software_2_values, label='Homer Pro', linestyle='dashed', color='g', linewidth=1)

# Set limits for the x-axis and y-axis
plt.xlim(-100, 8860)  # Adjust the limits as needed for your specific data
plt.ylim(-0.125, 1.175 * max(software_2_values))  # Adjust the limits as needed

# Create a new subplot to set custom x-axis tick labels
ax = plt.gca()

# Set custom tick labels for the x-axis using your method
hours_per_month = [0, 31 * 24, 59 * 24, 90 * 24, 120 * 24, 151 * 24, 181 * 24, 212 * 24, 243 * 24, 273 * 24, 304 * 24, 334 * 24]
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(hours_per_month)
ax.set_xticklabels(month_labels)

# Plot the relative errors
#plt.plot(hours, relative_errors, label='Relative Error', linestyle=':', color='r', linewidth=1)
# Plot the absolute mean error as a horizontal line
#plt.axhline(y=MAE, color='r', linestyle='--', label='Absolute Mean Error')
# Set labels and title
plt.xlabel('Month', labelpad=20)
plt.ylabel('Battery Power Out [kW]', labelpad=20)
#plt.title('Comparison between SAMA and Homer Pro - Battery Charge Out')


# Show the legend
plt.legend(loc='upper left', ncol = 2)
# Add minor ticks to the y-axis
#minor_locator = MultipleLocator(10)  # Customize the minor tick spacing as needed (here, every 10 units)
#ax.yaxis.set_minor_locator(minor_locator)

plt.tight_layout()
# Show the plot
plt.show()

plt.figure()
#plt.scatter(software_1_values, software_2_values)
plt.show()