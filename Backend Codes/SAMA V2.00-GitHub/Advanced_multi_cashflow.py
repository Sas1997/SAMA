import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
from matplotlib import pyplot
n = 25
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.ticker as mticker
Cash_Flow_adv = 1
# Advanced multi-Cash Flow Chart

if Cash_Flow_adv == 1:
        # Function to load data from CSV file
        def load_data_from_csv(filename):
            data = []
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    data.append(list(map(eval, row)))
            return data

        # Load all data from CSV
        csv_filename = 'cash_flow_data.csv'
        all_data = load_data_from_csv(csv_filename)

        # Define colors and hatch patterns
        colors_curve = ['black', 'blue', 'purple', 'red', 'yellow', 'green']

        plt.figure(figsize=(10, 6))
        bar_legend_handles = []  # List to store handles for bar legends
        curve_labels = []  # List to store labels for curve legends
        curve_handles = []  # List to store handles for curve legends

        for idx, data in enumerate(all_data):
            R_Cost, MO_Cost, C_Fu, Salvage, avoided_costs, Grid_Cost_pos, Grid_Cost_neg, I_Cost = data
            years = list(range(n + 1))
            yearly_total_cost = [sum(x) + g + a for x, g, a in zip(zip(R_Cost, MO_Cost, C_Fu, Salvage, Grid_Cost_pos), Grid_Cost_neg, avoided_costs)]
            yearly_total_cost = [-I_Cost] + yearly_total_cost
            cumulative_total_cost = [sum(yearly_total_cost[:i + 1]) for i in range(n + 1)]

            # Adjust x-axis positions for bars
            bar_width = 0.075  # to adjust the width of bars
            offset = bar_width * idx  # to shift bars horizontally
            bar_positions = [x + offset for x in years[1:]]

            # Plot costs
            if any(R_Cost):
                plt.bar(bar_positions, R_Cost, label='Replacement Cost', color='blue', edgecolor='black', width=0.5)
            if any(MO_Cost):
                plt.bar(bar_positions, MO_Cost, bottom=R_Cost, label='Maintenance & Operating Cost', color='brown', edgecolor='black', width=0.5)
            if any(C_Fu):
                plt.bar(bar_positions, C_Fu, bottom=[i + j for i, j in zip(R_Cost, MO_Cost)], label='Fuel Cost', color='orange', edgecolor='black', width=0.5)
            if any(Grid_Cost_pos):
                plt.bar(bar_positions, Grid_Cost_pos, bottom=[i + j + k for i, j, k in zip(R_Cost, MO_Cost, C_Fu)], label='Grid Cost', color='purple', edgecolor='black', width=0.5)

            # Plot grid revenues
            if any(Grid_Cost_neg):
                plt.bar(bar_positions, Grid_Cost_neg, label='Grid Revenue', color='pink', alpha=1, edgecolor='black', width=0.5)
            if any(avoided_costs):
                plt.bar(bar_positions, avoided_costs, bottom=Grid_Cost_neg, label='Avoided Costs', color='cyan', alpha=1, edgecolor='black', width=0.5)
            # Plot salvage revenues (Start from x-axis)
            if any(Salvage):
                plt.bar(bar_positions, Salvage, bottom=[i + j for i, j in zip(Grid_Cost_neg, avoided_costs)], label='Salvage', color='green', edgecolor='black', width=0.5)

            # Plot initial investment as a red bar
            plt.bar(0, -I_Cost, label='Initial Investment' if idx == 0 else None, color='red', edgecolor='black', width = 0.5)

            # Store handles for legends of the bars in the first loop iteration
            if idx == 0:
                bar_legend_handles, _ = plt.gca().get_legend_handles_labels()
            LG_lines = [7.5, 10, 12.5, 15, 17.5, 20]
            # Plot curves with different colors and names
            color_idx = idx % len(colors_curve)  # Ensure idx stays within the bounds of colors_curve
            curve_labels.append(f'Total System Cost (inflation rate =  {idx*2}%)')
            curve_handles.extend(plt.plot(years, cumulative_total_cost, linestyle='-', marker='o', linewidth=0.8, color=colors_curve[color_idx]))

        # Plot legends for bars outside of the loop
        legend1 = plt.legend(handles=bar_legend_handles, loc='best',fontsize=13)
        # Plot legends for curves
        plt.legend(curve_handles, curve_labels, loc='center left', fontsize=13)
        pyplot.gca().add_artist(legend1)
        # Add details and labels
        plt.xlabel('Year', fontsize=20)
        plt.ylabel('Cash Flow [$]', fontsize=20)
        plt.xticks([i for i in years], years, fontsize=14)
        plt.axhline(0, color='black', linewidth=0.8)
        # Customize y-axis ticks
        y_ticks = range(int(min(cumulative_total_cost) // 10000) * 5000,
                        int(max(cumulative_total_cost) // 5000) * 5000 + 5000, 5000)
        plt.yticks(y_ticks, [f'{y_tick:,}' for y_tick in y_ticks], fontsize=14)
        # Set y-axis limit
        y_min, y_max = plt.ylim()
        y_margin = (y_max - y_min) * 0.025
        plt.ylim(y_min - y_margin, y_max + y_margin)
        plt.tight_layout()
        plt.savefig('output/figs_inflationpaper/Multiple_Cash_Flow_San_Diego.png', dpi=300)