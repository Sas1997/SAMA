import numpy as np

"""
ENERGY BALANCE CLARIFICATION:

Your EMS (Energy Management Strategy) already decides:
- P_RE: How much renewable to use
- Pdg: How much diesel to generate  
- Pbuy: How much to buy from grid
- Psell: How much to sell to grid (removes energy)
- Edump: How much excess to dump (removes energy)
- Pdch/Pch: Battery discharge/charge
- Pev_dch/Pev_ch: EV discharge/charge
- Ens: Unmet demand (shortage)

The fundamental energy balance your EMS must satisfy:
    SOURCES = SINKS + LOSSES
    P_RE + Pdg + Pbuy + Pdch*n_I + Pev_dch*n_I = 
        Eload_eh + HP_heating + HP_cooling + Pch/n_I + Pev_ch/n_I + Psell + Edump + Ens

Key insights:
1. Psell and Edump REMOVE energy (they're sinks, not sources)
2. Ens represents UNMET demand (shortage) - it's in the equation to balance when generation is insufficient
3. The functions below don't change your EMS decisions, they just TRACK where HP energy came from
4. They use marginal costs to attribute energy sources to HP for cost accounting
"""


def calculate_hp_grid_allocation(P_RE, Pdg, Pbuy, Psell, Edump, Pch, Pdch, n_I,
                                 Ens, Eload_eh, power_hp_heating, power_hp_cooling,
                                 Pev_ch, Pev_dch, HP):
    """
    Calculate grid power allocation for heat pump heating and cooling

    Parameters:
    -----------
    P_RE : array - Renewable energy generation [kW]
    Pdg : array - Diesel generator output [kW]
    Pbuy : array - Grid purchase [kW]
    Psell : array - Grid sales [kW]
    Edump : array - Dumped energy [kW]
    Pch : array - Battery charging [kW]
    Pdch : array - Battery discharging [kW]
    n_I : float - Inverter efficiency
    Ens : array - Energy not served [kW]
    Eload_eh : array - Electrical household load [kW]
    power_hp_heating : array - Heat pump heating power [kW]
    power_hp_cooling : array - Heat pump cooling power [kW]
    Pev_ch : array - EV charging power [kW]
    Pev_dch : array - EV discharging power (V2H) [kW]
    HP : int - Heat pump flag (1 if installed, 0 otherwise)

    Returns:
    --------
    Pbuy_heating : array - Grid power for heating [kW]
    Pbuy_cooling : array - Grid power for cooling [kW]
    Pbuy_other : array - Grid power for other loads [kW]
    energy_balance_check : array - Should be ~0 if balanced [kW]
    """

    # Total energy available (sources)
    E_available = P_RE + Pdg + Pbuy + Pdch * n_I + Pev_dch * n_I

    # Total energy consumed (sinks)
    # Note: Eload_eh is satisfied load only (Ens is unmet demand, separate)
    # Psell and Edump remove energy from the system
    E_consumed = Eload_eh + power_hp_heating + power_hp_cooling + Pch / n_I + Psell + Edump + Pev_ch / n_I

    # Energy balance check (should be approximately zero if EMS is correct)
    # Note: Ens (energy not served) is not included because it represents UNMET demand
    # If Ens > 0, it means there was insufficient generation, so balance will show deficit
    energy_balance_check = E_available - E_consumed

    # Calculate net available renewable + local generation (before grid)
    E_local = P_RE + Pdg + Pdch * n_I + Pev_dch * n_I

    # Total demand (all loads including losses)
    E_total_demand = Eload_eh + power_hp_heating + power_hp_cooling + Pch / n_I + Edump + Ens + Pev_ch / n_I

    # Only proceed if heat pump is installed
    if HP == 0:
        return np.zeros_like(power_hp_heating), np.zeros_like(power_hp_cooling), Pbuy.copy(), energy_balance_check

    # Calculate the deficit that requires grid power
    deficit = E_total_demand - E_local

    # Initialize allocation arrays
    Pbuy_heating = np.zeros_like(power_hp_heating)
    Pbuy_cooling = np.zeros_like(power_hp_cooling)
    Pbuy_other = np.zeros_like(power_hp_heating)

    for i in range(len(power_hp_heating)):
        if deficit[i] <= 0:
            # Sufficient local generation - no grid purchase for HP
            Pbuy_heating[i] = 0
            Pbuy_cooling[i] = 0
            Pbuy_other[i] = 0
        else:
            # Grid purchase needed - allocate proportionally to loads
            total_load = Eload_eh[i] + power_hp_heating[i] + power_hp_cooling[i] + Pch[i] / n_I + Pev_ch[i] / n_I

            if total_load > 0:
                # Proportional allocation based on load magnitude
                ratio_heating = power_hp_heating[i] / total_load
                ratio_cooling = power_hp_cooling[i] / total_load
                ratio_other = (Eload_eh[i] + Pch[i] / n_I + Pev_ch[i] / n_I) / total_load

                # Allocate grid purchase proportionally
                Pbuy_heating[i] = Pbuy[i] * ratio_heating
                Pbuy_cooling[i] = Pbuy[i] * ratio_cooling
                Pbuy_other[i] = Pbuy[i] * ratio_other
            else:
                Pbuy_heating[i] = 0
                Pbuy_cooling[i] = 0
                Pbuy_other[i] = 0

    return Pbuy_heating, Pbuy_cooling, Pbuy_other, energy_balance_check


def calculate_hp_energy_sources_economic(P_RE, Pdg, Pbuy, Pdch, n_I, Pev_dch,
                                         Eload_eh, power_hp_heating, power_hp_cooling,
                                         Pch, Psell, Edump, Ens, Pev_ch, HP,
                                         price_dg, Cbw, Cevw, Cbuy):
    """
    Economic dispatch method: Track heat pump energy sources based on marginal cost priority
    Priority order: RE (free) -> Lowest cost -> Highest cost

    Parameters:
    -----------
    price_dg : array - Diesel generator marginal cost [$/kWh] (hourly)
    Cbw : float - Battery discharge marginal cost [$/kWh]
    Cevw : float - EV discharge marginal cost [$/kWh]
    Cbuy : array or float - Grid purchase price [$/kWh] (can be hourly or fixed)
    Psell : array - Power sold to grid [kW] (energy leaving system)
    Edump : array - Dumped energy [kW] (excess generation wasted)
    Ens : array - Energy not served [kW] (unmet demand)

    Returns:
    --------
    HP_from_RE : array - HP power from renewables [kW]
    HP_from_DG : array - HP power from diesel [kW]
    HP_from_Battery : array - HP power from battery [kW]
    HP_from_EV : array - HP power from EV [kW]
    HP_from_Grid : array - HP power from grid [kW]
    HP_heating_from_Grid : array - Heating specifically from grid [kW]
    HP_cooling_from_Grid : array - Cooling specifically from grid [kW]
    HP_cost_breakdown : dict - Cost breakdown by source
    """

    n_hours = len(power_hp_heating)

    # Initialize tracking arrays
    HP_from_RE = np.zeros(n_hours)
    HP_from_DG = np.zeros(n_hours)
    HP_from_Battery = np.zeros(n_hours)
    HP_from_EV = np.zeros(n_hours)
    HP_from_Grid = np.zeros(n_hours)
    HP_heating_from_Grid = np.zeros(n_hours)
    HP_cooling_from_Grid = np.zeros(n_hours)

    # Cost tracking
    HP_cost_from_RE = np.zeros(n_hours)
    HP_cost_from_DG = np.zeros(n_hours)
    HP_cost_from_Battery = np.zeros(n_hours)
    HP_cost_from_EV = np.zeros(n_hours)
    HP_cost_from_Grid = np.zeros(n_hours)

    if HP == 0:
        return (HP_from_RE, HP_from_DG, HP_from_Battery, HP_from_EV, HP_from_Grid,
                HP_heating_from_Grid, HP_cooling_from_Grid, {
                    'RE': 0, 'DG': 0, 'Battery': 0, 'EV': 0, 'Grid': 0, 'Total': 0
                }, np.zeros(n_hours), np.zeros(n_hours))

    # Ensure Cbuy is an array
    if not isinstance(Cbuy, np.ndarray):
        Cbuy_array = np.full(n_hours, Cbuy)
    else:
        Cbuy_array = Cbuy

    # Verify energy balance for Method 2
    energy_balance_method2 = np.zeros(n_hours)
    HP_curtailed = np.zeros(n_hours)  # Track HP curtailment due to Ens

    for i in range(n_hours):
        # Check energy balance at this hour
        sources = P_RE[i] + Pdg[i] + Pbuy[i] + Pdch[i] * n_I + Pev_dch[i] * n_I
        sinks = (Eload_eh[i] + power_hp_heating[i] + power_hp_cooling[i] +
                 Pch[i] / n_I + Pev_ch[i] / n_I + Psell[i] + Edump[i])

        # When Ens > 0, there's a shortage - sources < (desired sinks + Ens)
        # The balance equation becomes: sources = sinks (actual served)
        # And: desired_demand = sinks + Ens
        energy_balance_method2[i] = sources - sinks

        # Total HP demand (what HP WANTED)
        HP_total_desired = power_hp_heating[i] + power_hp_cooling[i]

        if HP_total_desired == 0:
            continue

        # Handle energy shortage (Ens > 0)
        if Ens[i] > 0:
            # Total desired demand (before curtailment)
            total_desired_demand = Eload_eh[i] + HP_total_desired + Pch[i] / n_I + Pev_ch[i] / n_I + Ens[i]

            # Proportional curtailment: assume all loads reduced proportionally
            curtailment_factor = Ens[i] / total_desired_demand

            # HP actually served (after curtailment)
            HP_total = HP_total_desired * (1 - curtailment_factor)
            HP_curtailed[i] = HP_total_desired - HP_total

            # Adjust HP components proportionally
            HP_heating_served = power_hp_heating[i] * (1 - curtailment_factor)
            HP_cooling_served = power_hp_cooling[i] * (1 - curtailment_factor)
        else:
            # No shortage - full demand satisfied
            HP_total = HP_total_desired
            HP_heating_served = power_hp_heating[i]
            HP_cooling_served = power_hp_cooling[i]

        if HP_total == 0:
            continue

        # Other loads that were actually satisfied
        # Note: Eload_eh, Pch, Pev_ch already reflect satisfied amounts from EMS
        other_loads = Eload_eh[i] + Pch[i] / n_I + Pev_ch[i] / n_I

        # Total energy generated/available at this hour
        # These are the TOTAL amounts - the EMS has already decided to split them
        # between loads, storage, grid sales, and dumps
        available_RE = P_RE[i]
        available_DG = Pdg[i]
        available_Battery = Pdch[i] * n_I
        available_EV = Pev_dch[i] * n_I
        available_Grid = Pbuy[i]

        # IMPORTANT: The energy balance is:
        # Sources: P_RE + Pdg + Pbuy + Pdch*n_I + Pev_dch*n_I
        # Uses: Eload_eh + HP_heat + HP_cool + Pch/n_I + Pev_ch/n_I + Psell + Edump
        #
        # Psell and Edump are where EXCESS energy goes (sinks)
        # We're now doing economic attribution: which sources served which loads?

        # Marginal costs at this hour
        cost_RE = 0  # Renewable is free (already paid capital cost)
        cost_DG = price_dg[i]
        cost_Battery = Cbw
        cost_EV = Cevw
        cost_Grid = Cbuy_array[i]

        # Create priority list based on marginal cost
        # Format: (source_name, available_power, marginal_cost)
        sources = [
            ('RE', available_RE, cost_RE),
            ('DG', available_DG, cost_DG),
            ('Battery', available_Battery, cost_Battery),
            ('EV', available_EV, cost_EV),
            ('Grid', available_Grid, cost_Grid)
        ]

        # Sort by marginal cost (lowest to highest)
        sources_sorted = sorted(sources, key=lambda x: x[2])

        # Total demand to serve
        total_demand = other_loads + HP_total

        # Track what's used for other loads vs HP
        remaining_demand = total_demand
        remaining_HP = HP_total

        # Dispatch sources in order of cost
        for source_name, available, cost in sources_sorted:
            if remaining_demand <= 0:
                break

            # How much from this source?
            used_from_source = min(available, remaining_demand)

            if used_from_source > 0:
                # First serve other loads, then HP
                if remaining_demand > remaining_HP:
                    # Still serving other loads
                    other_load_from_source = min(used_from_source, remaining_demand - remaining_HP)
                    hp_from_source = used_from_source - other_load_from_source
                else:
                    # Only HP remaining
                    hp_from_source = used_from_source

                # Track HP allocation
                if source_name == 'RE':
                    HP_from_RE[i] = hp_from_source
                    HP_cost_from_RE[i] = hp_from_source * cost
                elif source_name == 'DG':
                    HP_from_DG[i] = hp_from_source
                    HP_cost_from_DG[i] = hp_from_source * cost
                elif source_name == 'Battery':
                    HP_from_Battery[i] = hp_from_source
                    HP_cost_from_Battery[i] = hp_from_source * cost
                elif source_name == 'EV':
                    HP_from_EV[i] = hp_from_source
                    HP_cost_from_EV[i] = hp_from_source * cost
                elif source_name == 'Grid':
                    HP_from_Grid[i] = hp_from_source
                    HP_cost_from_Grid[i] = hp_from_source * cost

                remaining_demand -= used_from_source
                remaining_HP -= hp_from_source

        # Split grid contribution between heating and cooling proportionally
        if HP_total > 0:
            HP_heating_from_Grid[i] = HP_from_Grid[i] * (HP_heating_served / HP_total)
            HP_cooling_from_Grid[i] = HP_from_Grid[i] * (HP_cooling_served / HP_total)

    # Calculate total costs
    HP_cost_breakdown = {
        'RE': HP_cost_from_RE.sum(),
        'DG': HP_cost_from_DG.sum(),
        'Battery': HP_cost_from_Battery.sum(),
        'EV': HP_cost_from_EV.sum(),
        'Grid': HP_cost_from_Grid.sum(),
        'Total': (HP_cost_from_RE.sum() + HP_cost_from_DG.sum() +
                  HP_cost_from_Battery.sum() + HP_cost_from_EV.sum() + HP_cost_from_Grid.sum())
    }

    return (HP_from_RE, HP_from_DG, HP_from_Battery, HP_from_EV, HP_from_Grid,
            HP_heating_from_Grid, HP_cooling_from_Grid, HP_cost_breakdown,
            energy_balance_method2, HP_curtailed)


    # Method 1: Proportional allocation
    Pbuy_h, Pbuy_c, Pbuy_o, balance = calculate_hp_grid_allocation(
        P_RE, Pdg, Pbuy, Psell, Edump, Pch, Pdch, n_I,
        Ens, Eload_eh, power_hp_heating, power_hp_cooling,
        Pev_ch, Pev_dch, HP
    )

    # Method 2: Economic dispatch with marginal cost priority
    (HP_RE, HP_DG, HP_Batt, HP_EV, HP_Grid, HP_h_Grid, HP_c_Grid,
     cost_breakdown, balance_method2, HP_curtailed) = calculate_hp_energy_sources_economic(
        P_RE, Pdg, Pbuy, Pdch, n_I, Pev_dch,
        Eload_eh, power_hp_heating, power_hp_cooling,
        Pch, Psell, Edump, Ens, Pev_ch, HP,
        price_dg, Cbw, Cevw, Cbuy
    )
