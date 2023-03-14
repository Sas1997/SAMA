import math

from Battery_Model import battery_model

"""
Energy management 
"""


def energy_management(Ppv, Pwt, Cn_B, Nbat, Pn_DG, Pinv_max, cc_gen, Cbw, InData):
    # ^^^^^^^^^^^^^^READ INPUTS^^^^^^^^^^^^^^^^^^

    Eb = []
    Pch = []
    Pdch = []
    Ech = []
    Edch = []
    Pdg = []
    Edump = []
    Ens = []
    Psell = []
    Pbuy = []
    Pinv = []
    Ebmax = InData.SOC_max * Cn_B
    Ebmin = InData.SOC_min * Cn_B
    Eb.append(InData.SOC_initial * Cn_B)
    LrPn = InData.LR_DG * Pn_DG
    dt = 1
    sqrt_ef = math.sqrt(InData.ef_bat)
    if InData.Grid == 0:
        Pbuy_max = 0
        Psell_max = 0

    P_RE = Ppv + Pwt  # (8760,)
    Pdg_min = 0.05 * Pn_DG  # LR_DG # float

    for t in range(InData.NT):
        Pch.append(0)
        Pdch.append(0)
        Ech.append(0)
        Edch.append(0)
        Pdg.append(0)
        Edump.append(0)
        Ens.append(0)
        Psell.append(0)
        Pbuy.append(0)
        Pinv.append(0)
        Pch_max, Pdch_max = battery_model(Cn_B, Nbat, Eb[t], InData)  # kW

        Elpn = InData.Eload[t] / InData.n_I
        if P_RE[t] >= Elpn and InData.Eload[t] <= Pinv_max:  # if PV + Pwt greater than load  (battery should charge)
            # Battery charge power calculated based on surEloadus energy and battery empty capacity
            Eb_e = (Ebmax - Eb[t]) / sqrt_ef
            Pch[t] = min([Eb_e, P_RE[t] - Elpn, Pch_max])

            Psur_AC = InData.n_I * (P_RE[t] - Pch[t]) - InData.Eload[t]  # surplus Energy

            Psell[t] = min([Psur_AC, Psell_max, max([0, Pinv_max - InData.Eload[t]])])
            Edump[t] = P_RE[t] - Pch[t] - (InData.Eload[t] + Psell[t]) / InData.n_I

            # if load greater than PV+Pwt
        else:

            Edef_AC = InData.Eload[t] - min([Pinv_max, InData.n_I * P_RE[t]])

            price_dg = cc_gen + InData.a * InData.C_fuel  # DG cost($ / kWh)

            if InData.Cbuy[t] <= price_dg <= Cbw:  # Grid, DG, Bat: 1

                Pbuy[t] = min([Edef_AC, Pbuy_max])

                Pdg[t] = min([Edef_AC - Pbuy[t], Pn_DG])
                Pdg[t] = Pdg[t] * (Pdg[t] >= LrPn) + LrPn * (Pdg[t] < LrPn) * (Pdg[t] > Pdg_min)

                Edef_AC = InData.Eload[t] - Pdg[t] - Pbuy[t] - min([Pinv_max, InData.n_I * P_RE[t]])
                Edef_DC = Edef_AC / InData.n_I * (Edef_AC > 0)
                Eb_e = (Eb[t] - Ebmin) * sqrt_ef
                Pdch[t] = min([Eb_e, Edef_DC, Pdch_max])

                Esur_AC = -Edef_AC * (Edef_AC < 0)
                Pbuy[t] = Pbuy[t] - Esur_AC * (InData.Grid == 1)

            elif InData.Cbuy[t] <= Cbw < price_dg:  # Grid, Bat, DG: 2

                Pbuy[t] = min([Edef_AC, Pbuy_max])

                Edef_DC = (InData.Eload[t] - Pbuy[t]) / InData.n_I - P_RE[t]
                Eb_e = (Eb[t] - Ebmin) * sqrt_ef
                Pdch[t] = min([Eb_e, Edef_DC, Pdch_max])

                Edef_AC = InData.Eload[t] - Pbuy[t] - min([Pinv_max, InData.n_I * (P_RE[t] + Pdch[t])])
                Pdg[t] = min([Edef_AC, Pn_DG])
                Pdg[t] = Pdg[t] * (Pdg[t] >= LrPn) + LrPn * (Pdg[t] < LrPn) * (Pdg[t] > Pdg_min)


            elif price_dg < InData.Cbuy[t] <= Cbw: # % DG, Grid, Bat: 3
                Pdg[t] = min([Edef_AC, Pn_DG])
                Pdg[t] = Pdg[t] * (Pdg[t] >= LrPn) + LrPn * (Pdg[t] < LrPn) * (Pdg[t] > Pdg_min)

                Pbuy[t] = max([0, min([Edef_AC - Pdg[t], Pbuy_max])])
                Psell[t] = max([0, min([Pdg[t] - Edef_AC, Psell_max])])

                Edef_DC = (InData.Eload[t] - Pbuy[t] - Pdg[t]) / InData.n_I - P_RE[t]
                Edef_DC = Edef_DC * (Edef_DC > 0)
                Eb_e = (Eb[t] - Ebmin) * sqrt_ef
                Pdch[t] = min([Eb_e, Edef_DC,Pdch_max])

            elif price_dg < Cbw < InData.Cbuy[t]:  # DG, Bat, Grid: 4
                Pdg[t] = min([Edef_AC, Pn_DG])
                Pdg[t] = Pdg[t] * (Pdg[t] >= LrPn) + LrPn * (Pdg[t] < LrPn) * (Pdg[t] > Pdg_min)

                Edef_DC = (InData.Eload[t] - Pdg[t]) / InData.n_I - P_RE[t]
                Edef_DC = Edef_DC * (Edef_DC > 0)
                Eb_e = (Eb[t] - Ebmin) * sqrt_ef
                Pdch[t] = min([Eb_e, Edef_DC,Pdch_max])

                Edef_AC = InData.Eload[t] - Pdg[t] - min(Pinv_max, InData.n_I * (P_RE[t] + Pdch[t]))
                Pbuy[t] = max([0, min([Edef_AC, Pbuy_max])])
                Psell[t] = max([0, min([-Edef_AC, Psell_max])])

            elif Cbw < price_dg < InData.Cbuy[t]:  # Bat, DG, Grid: 5
                Edef_DC = Elpn - P_RE[t]
                Eb_e = (Eb[t] - Ebmin) * sqrt_ef
                Pdch[t] = min([Eb_e, Edef_DC,Pdch_max])

                Edef_AC = InData.Eload[t] - min([Pinv_max, InData.n_I * (P_RE[t] + Pdch[t])])
                Pdg[t] = min([Edef_AC, Pn_DG])
                Pdg[t] = Pdg[t] * (Pdg[t] >= LrPn) + LrPn * (Pdg[t] < LrPn) * (Pdg[t] > Pdg_min)

                Pbuy[t] = max([0, min([Edef_AC - Pdg[t], Pbuy_max])])
                Psell[t] = max([0, min([Pdg[t] - Edef_AC, Psell_max])])

            else:  # Bat, Grid, DG: 6

                Edef_DC = min([Pinv_max, Elpn]) - P_RE[t]
                Eb_e = (Eb[t] - Ebmin) * sqrt_ef
                Pdch[t] = min([Eb_e, Edef_DC]) * (Edef_DC > 0)
                Pdch[t] = min([Pdch[t], Pdch_max])

                Edef_AC = InData.Eload[t] - min(Pinv_max, InData.n_I * (P_RE[t] + Pdch[t]))
                Pbuy[t] = min([Edef_AC, Pbuy_max])

                Pdg[t] = min([Edef_AC - Pbuy[t], Pn_DG])
                Pdg[t] = Pdg[t] * (Pdg[t] >= LrPn) + LrPn * (Pdg[t] < LrPn) * (Pdg[t] > Pdg_min)

            Edef_DC = (InData.Eload[t] + Psell[t] - Pdg[t] - Pbuy[t]) / InData.n_I - (P_RE[t] + Pdch[t] - Pch[t])

            if Edef_DC < 0:
                Eb_e = (Ebmax - Eb[t]) / sqrt_ef
                Pch[t] = min([Eb_e, Pch[t] - Edef_DC,Pch_max])

            Esur = InData.Eload[t] + Psell[t] - Pbuy[t] - Pdg[t] - min([Pinv_max,
                                                                        (P_RE[t] + Pdch[t] - Pch[t]) * InData.n_I])
            Ens[t] = Esur * (Esur > 0)
            Edump[t] = -Esur * (Esur < 0)

        # Battery charging and dischargin energy is determined based on charging and discharging power and the battery charge
        # level is updated.
        Ech[t] = Pch[t] * dt
        Edch[t] = Pdch[t] * dt
        Eb.append((1 - InData.self_discharge_rate) * Eb[t] + sqrt_ef * Ech[t] - Edch[t] / math.sqrt(
            InData.ef_bat))

    return Eb[:-1], Pdg, Edump, Ens, Pch, Pdch, Pbuy, Psell, Pinv
