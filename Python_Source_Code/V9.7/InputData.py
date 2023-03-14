import numpy as np

from InputDataMethods import *
import Data


class InputData:
    def __init__(self, path):
        self.Eload, self.G, self.T, self.Vw = Data.read_data(path)  # the path of data file.
        self.NT = len(self.Eload)

        # ----------Other needed parameters---------
        self.Ppv_r, self.Pwt_r, self.Cbt_r, self.Cdg_r, self.Tc_noct, self.fpv, self.Gref, self.Tcof, self.Tref, \
        self.h_hub, self.h0, self.alfa_wind_turbine, self.v_cut_in, self.v_cut_out, self.v_rated, \
        self.R_B, self.Q_lifetime, self.ef_bat, self.b, self.C_fuel, self.R_DG, self.TL_DG, self.MO_DG, self.SOC_max, \
        self.SOC_min, self.SOC_initial, self.n_I, self.Grid, self.Cbuy, self.a, self.LR_DG, self.Pbuy_max, self.Psell_max, \
        self.self_discharge_rate, self.alfa_battery, self.c, self.k, self.Imax, self.Vnom, self.C_PV, self.C_WT, self.C_DG, \
        self.C_B, self.C_I, self.C_CH, self.n, self.R_PV, self.ir, self.L_PV, self.R_WT, self.L_WT, self.L_B, self.R_I, \
        self.R_CH, self.MO_PV, self.MO_WT, self.MO_B, self.MO_I, self.MO_CH, self.RT_PV, self.RT_WT, self.RT_B, self.RT_I, \
        self.L_I, self.RT_CH, self.L_CH, self.CO2, self.NOx, self.SO2, self.E_CO2, self.E_SO2, self.E_NOx, self.Csell, \
        self.EM, self.LPSP_max, self.RE_min, self.Budget, self.Ta_noct, self.G_noct, self.n_PV, self.gama, self.PV, self.WT, \
        self.Bat, self.DG,self.RE_incentives,self.Engineering_Costs,self.System_Tax,self.Grid_Tax,self.daysInMonth\
            = other_parameters(self.Eload, self.Eload)
        # those values that can be calculated before the main loop is started.
        # Module Temprature
        self.Tc = np.divide((self.T + (self.Tc_noct - self.Ta_noct) * (self.G / self.G_noct) * (
                1 - (self.n_PV * (1 - self.Tcof * self.Tref) / self.gama))), (
                               1 + (self.Tc_noct - self.Ta_noct) * (self.G / self.G_noct) * (
                               (self.Tcof * self.n_PV) / self.gama)))
        # Variable Number and the respective boundaries
        # Variable: PV number, WT number, Battery number, number of DG, Rated Power Inverter
        VarMin = np.array([0, 0, 0, 0, 0])  # Lower Bound of Variables
        VarMax = np.array([100, 100, 60, 10, 20])  # Upper Bound of Variables

        self.minimum_var_range = np.multiply(VarMin, np.array([self.PV, self.WT, self.Bat, self.DG, 1]))
        self.maximum_var_range = np.multiply(VarMax, np.array([self.PV, self.WT, self.Bat, self.DG, 1]))
        self.number_of_variables, = VarMin.shape

InData=InputData('Data.csv')