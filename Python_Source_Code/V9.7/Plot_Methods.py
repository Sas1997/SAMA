import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_convergence(x, y):
    # plt.ion
    plt.figure()
    plt.plot(x, y, color='r')
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("Convergence characteristic")
    # plt.xticks(x, x)
    # plt.savefig('Convergence.jpg')
    plt.ioff()
    plt.draw()


def plot_cashflow(chash_flow):
    # create data

    y, n = np.shape(chash_flow)
    x = np.arange(n)
    chash_flow.insert(0, x)
    chash_flow = np.transpose(chash_flow)

    df = pd.DataFrame(chash_flow,
                      columns=['Year', 'Capital', 'Operating', 'Salvage', 'Fuel', 'Replacement'])

    # print(df)

    ax = df.plot(x='Year', kind='bar', stacked=True, title='Cash Flow')
    ax.set_xticklabels(x + 1, rotation=90)
    plt.ioff()
    plt.draw()


def plot_my_plots(vec, legends, title, xlabel, ylabel):
    plt.figure()
    for i in range(len(vec)):
        plt.plot(vec[i], label=legends[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.draw()


def plot_one_day(title,n_row,n_col,titles,indices,x_labels,y_labels,x_limits, values,is_bar):
    plt.figure()
    plt.title(title)
    plt.xlim()
    for i in range(len(titles)):
        axi = plt.subplot(n_row, n_col, indices[i])
        axi.set_xlabel(x_labels[i])
        axi.set_ylabel(y_labels[i])
        axi.set_xlim(x_limits[i])
        if is_bar[i]:
            t1=x_limits[i][0]
            t2=x_limits[i][1]
            axi.bar(list(range(t1,t2+1)),values[i][t1-1:t2])
        else:
            axi.plot(values[i])

    plt.tight_layout()
    plt.draw()

def plot_heatmap(title,matrix,YData):

    import seaborn as sns
    import matplotlib.pyplot as plt
    # Define the plot
    fig, ax = plt.subplots()

    # Set the font size and the distance of the title from the plot
    plt.title(title)
    ttl = ax.title
    ttl.set_position([0.5, 1.05])

    # Hide ticks for X & Y axis
    #ax.set_xticks([])
    ax.set_yticks([])

    # Remove the axes
    matrix=np.transpose(matrix)
    # Use the heatmap function from the seaborn package
    sns.heatmap(matrix, xticklabels=YData, annot=True, fmt='g', cmap='jet')
    plt.tight_layout()
    plt.draw()

def plot_imshow(title, matrix):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6, 10))

    ax1.imshow(matrix, extent=[0, 100, 0, 1])
    ax1.set_title('Default')

    ax2.imshow(matrix, extent=[0, 100, 0, 1], aspect='auto')
    ax2.set_title('Auto-scaled Aspect')

    ax3.imshow(matrix, extent=[0, 100, 0, 1], aspect=100)
    ax3.set_title('Manually Set Aspect')
    plt.title(title)
    plt.tight_layout()
    plt.draw()


