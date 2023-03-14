from Plot_Methods import *
from InputData import InData


def Utility_results(Psell):
    if InData.Grid == 1:
        # Utility
        np.savetxt('Cbuy.csv', InData.Cbuy, delimiter=',')
        # -----
        p_c = np.zeros((12, 31))
        index = 0
        for m in range(12):
            index1 = index
            for d in range(InData.daysInMonth[m]):
                cost = np.mean(InData.Cbuy[index1:index1 + 24])
                p_c[m, d] = cost
                index1 = index1 + 24

            index = (24 * InData.daysInMonth[m]) + index

        #
        L_c = np.zeros((12, 31))
        index = 0
        for m in range(12):
            index1 = index
            for d in range(InData.daysInMonth[m]):
                Total_daily_load = np.sum(InData.Eload[index1:index1 + 24])
                L_c[m, d] = Total_daily_load
                index1 = index1 + 24

            index = (24 * InData.daysInMonth[m]) + index

        E_c = np.round(p_c * L_c * (1 + InData.Grid_Tax), 2)
        #
        # figure(7)
        p_c[p_c == 0] = None
        title = "p_c"
        YData = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"]
        plot_heatmap(title, np.round(p_c, 2), YData)

        # figure(8)
        E_c[E_c == 0] = None
        title = "E_c"
        YData = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"]
        plot_heatmap(title, E_c, YData)

        # figure(9)
        title ='southoutside'
        plot_imshow(title, InData.Cbuy)

        if np.sum(Psell) > 0.1:
            S_c = np.zeros((12, 31))
            index = 1
            for m in range(12):
                index1 = index
                for d in range(InData.daysInMonth[m]):
                    Total_daily_sell = np.sum(Psell[index1:index1 + 24])
                    S_c[m, d] = Total_daily_sell
                    index1 = index1 + 24
                index = (24 * InData.daysInMonth[m]) + index

            Ss_c = np.round(InData.Csell * S_c, 2)

            # figure(10)
            Ss_c[InData.Ss_c == 0] = None
            title = "Ss_c"
            YData = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                     "November", "December"]
            plot_heatmap(title, Ss_c, YData)
