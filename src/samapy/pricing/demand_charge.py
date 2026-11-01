"""
demand_charge.py
================
Helper module for computing commercial / industrial electricity demand charges.

Demand charges are billed per kW of metered peak demand within each billing
period (typically monthly), in addition to the standard per-kWh energy charges.

Four demand-charge structures are supported (Input_Data.py demand_charge_type):

  Type 1 – Flat monthly       : single $/kW/month rate applied all year.
  Type 2 – Seasonal           : separate summer / winter $/kW/month rates.
  Type 3 – Monthly varying    : 12 individual $/kW/month rates.
  Type 4 – TOU (on/off-peak)  : separate on-peak and off-peak $/kW/month rates,
                                 with user-defined on-peak hours per season.

Additional options:
  Option B – NEM demand netting     : peak measured on net import (Pbuy−Psell)
                                      instead of gross import (Pbuy).
  Option C – NEM demand rate        : a separate (usually lower) $/kW/month rate
                                      for customers on a NEM tariff.
  Minimum charge rule               : monthly bill cannot fall below
                                      max(minimum_charge, base_charge + demand_charge),
                                      regardless of NEM exports.
  Ratchet clause                    : bill based on at least X% of the 12-month
                                      annual peak demand.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Monthly rate arrays from the user inputs
# ---------------------------------------------------------------------------

def build_demand_rates(d):
    """
    Set ``d.Demand_charge_rate_monthly`` and ``d.Demand_charge_offpeak_rate_monthly``
    ($/kW/month, 12 values each) from the demand charge inputs on ``d``
    (the InData object). Called by Input_Data and by the config loader.

    Type 1 uses ``demand_flat_rate``; type 2 ``demand_seasonal_rates``
    [summer, winter] with ``demand_season``; type 3 ``demand_monthly_rates``;
    type 4 ``demand_tou_onpeak_rate`` and ``demand_tou_offpeak_rate``
    [summer, winter] with ``demand_tou_season``. With Option C
    (``nem_demand_rate_override`` = 1) the NEM-specific rates replace them.
    When demand charges are disabled both arrays are zero.
    """
    zeros = np.zeros(12)
    if not d.demand_charge_enabled:
        d.Demand_charge_rate_monthly = zeros.copy()
        d.Demand_charge_offpeak_rate_monthly = zeros.copy()
        return

    t = int(d.demand_charge_type)
    if t == 1:
        on = np.full(12, float(d.demand_flat_rate))
        off = zeros.copy()
    elif t == 2:
        rates = np.asarray(d.demand_seasonal_rates, dtype=float)
        season = np.asarray(d.demand_season, dtype=int)
        on = np.where(season == 1, rates[0], rates[1]).astype(float)
        off = zeros.copy()
    elif t == 3:
        on = np.asarray(d.demand_monthly_rates, dtype=float).copy()
        off = zeros.copy()
    elif t == 4:
        season = np.asarray(d.demand_tou_season, dtype=int)
        on_r = np.asarray(d.demand_tou_onpeak_rate, dtype=float)
        off_r = np.asarray(d.demand_tou_offpeak_rate, dtype=float)
        on = np.where(season == 1, on_r[0], on_r[1]).astype(float)
        off = np.where(season == 1, off_r[0], off_r[1]).astype(float)
    else:
        raise ValueError(f"demand_charge_type must be 1, 2, 3 or 4, got {d.demand_charge_type}")

    if getattr(d, 'nem_demand_rate_override', 0):
        on = np.asarray(d.nem_demand_rate_monthly, dtype=float).copy()
        off = np.asarray(d.nem_demand_offpeak_rate_monthly, dtype=float).copy()

    for name, arr in (('on-peak', on), ('off-peak', off)):
        if arr.shape != (12,):
            raise ValueError(f"demand charge {name} rates must have 12 monthly values, got {arr.shape}")
    d.Demand_charge_rate_monthly = on
    d.Demand_charge_offpeak_rate_monthly = off


# ---------------------------------------------------------------------------
# Core demand charge computation
# ---------------------------------------------------------------------------

def compute_monthly_peak_demand(
        Pbuy,
        daysInMonth,
        demand_charge_enabled,
        Demand_charge_rate_monthly,
        demand_charge_type=1,
        Demand_charge_offpeak_rate_monthly=None,
        demand_tou_onpeak_hours=None,
        demand_tou_season=None,
        demand_ratchet_enabled=0,
        demand_ratchet_rate=0.85,
        nem_demand_netting=0,
        Psell=None,
        tou_weekdays_only=0,
        year=None,
        holidays=None):
    """
    Compute monthly demand charges from an hourly grid-purchase profile.

    Parameters
    ----------
    Pbuy : array-like, shape (8760,)
        Hourly power purchased from the grid [kW].
    daysInMonth : array-like, shape (12,)
        Number of calendar days in each month.
    demand_charge_enabled : int
        1 = demand charges active; 0 = disabled (residential tariff).
    Demand_charge_rate_monthly : array-like, shape (12,)
        $/kW/month – non-coincident (or on-peak for Type 4) demand rate.
        When Option C (NEM rate override) is active this already contains
        the NEM-specific rate (set in Input_Data.py before being passed here).
    demand_charge_type : int, optional
        1=flat, 2=seasonal, 3=monthly, 4=TOU on/off-peak.  Default 1.
    Demand_charge_offpeak_rate_monthly : array-like, shape (12,) or None
        $/kW/month – off-peak demand rate.  Only used for Type 4.
    demand_tou_onpeak_hours : array of lists, shape (2,) or None
        [summer_hours, winter_hours] – lists of hour-of-day indices (0-23)
        that constitute the on-peak window.  Required for Type 4.
    demand_tou_season : array-like, shape (12,) or None
        1 = summer month, 0 = winter month.  Required for Type 4.
    demand_ratchet_enabled : int, optional
        1 = apply ratchet clause; 0 = disabled.  Default 0.
    demand_ratchet_rate : float, optional
        Ratchet fraction (e.g. 0.85 = bill ≥ 85 % of annual peak).
        Default 0.85.
    nem_demand_netting : int, optional
        **Option B** – 0 = peak measured on gross import Pbuy (standard,
        most utilities); 1 = peak measured on net import max(Pbuy−Psell, 0)
        per hour, for utilities that allow NEM demand netting.  Default 0.
    Psell : array-like, shape (8760,) or None, optional
        Hourly power sold to the grid [kW].  Required when
        nem_demand_netting=1; ignored otherwise.
    tou_weekdays_only : int, optional
        Type 4 only. 0 = the on-peak hours apply every day (default);
        1 = they apply on weekdays only, and weekend days and ``holidays``
        are entirely off-peak, as in the TOU energy rate calculator.
    year : int, optional
        Calendar year that sets the day of the week; required when
        tou_weekdays_only=1.
    holidays : array-like, optional
        Holidays as day-of-year numbers (1 = 1 January).

    Returns
    -------
    annual_demand_cost : float
        Total demand charge for the simulated year [$].
    monthly_peak_demand : ndarray, shape (12,)
        Metered non-coincident (or on-peak for Type 4) peak demand per
        month [kW].  With Option B this is the net-import peak.
    monthly_demand_costs : ndarray, shape (12,)
        Demand charge billed each month [$].  For Type 4 this is the sum
        of the on-peak and off-peak demand charges for each month.

    Notes
    -----
    * With hourly data the peak demand equals the maximum hourly kW value
      in the billing period (equivalent to a 1-hour demand interval).
    * The ratchet clause compares each month's metered demand to the
      annual maximum and substitutes the ratchet floor when it is higher.
    * For Type 4 the ratchet is applied to the on-peak window only;
      off-peak demand is billed on the actual measured peak.
    * Option C (NEM rate override) is handled entirely in Input_Data.py by
      substituting the NEM rate into Demand_charge_rate_monthly before
      calling this function; no extra parameter is needed here.
    """
    if not demand_charge_enabled:
        return 0.0, np.zeros(12), np.zeros(12)

    Pbuy = np.asarray(Pbuy, dtype=float)

    # Option B: prepare net-import profile if netting is requested
    if nem_demand_netting and Psell is not None:
        Psell_arr = np.asarray(Psell, dtype=float)
        Pbuy_eff  = np.maximum(Pbuy - Psell_arr, 0.0)
    else:
        Pbuy_eff  = Pbuy

    monthly_peak_demand         = np.zeros(12)
    monthly_peak_demand_offpeak = np.zeros(12)
    monthly_demand_costs        = np.zeros(12)

    if Demand_charge_offpeak_rate_monthly is None:
        Demand_charge_offpeak_rate_monthly = np.zeros(12)
    else:
        Demand_charge_offpeak_rate_monthly = np.asarray(
            Demand_charge_offpeak_rate_monthly, dtype=float)

    Demand_charge_rate_monthly = np.asarray(Demand_charge_rate_monthly, dtype=float)

    hours_in_month = (np.asarray(daysInMonth, dtype=int) * 24)
    hour_idx = 0

    special_day = None
    if demand_charge_type == 4 and tou_weekdays_only:
        if year is None:
            raise ValueError("tou_weekdays_only needs the calendar year")
        special_day = special_days(year, holidays, int(np.sum(daysInMonth)))

    for m in range(12):
        n_hours       = int(hours_in_month[m])
        month_Pbuy    = Pbuy_eff[hour_idx: hour_idx + n_hours]

        if (demand_charge_type == 4
                and demand_tou_onpeak_hours is not None
                and demand_tou_season is not None):
            # ---- TOU demand charge (Type 4) --------------------------------
            season_idx    = 1 if demand_tou_season[m] == 1 else 0
            on_peak_h_set = {int(h) for h in demand_tou_onpeak_hours[season_idx]}

            # Build hour-of-day index for every hour in the month
            n_full_days = n_hours // 24
            hod = np.tile(np.arange(24), n_full_days)
            remainder = n_hours - len(hod)
            if remainder > 0:
                hod = np.concatenate([hod, np.arange(remainder)])

            onpeak_mask  = np.isin(hod, np.array(sorted(on_peak_h_set), dtype=int))
            if special_day is not None:
                # weekend days and holidays are off-peak all day
                day_of_year = (hour_idx + np.arange(n_hours)) // 24
                onpeak_mask &= ~special_day[np.minimum(day_of_year, len(special_day) - 1)]
            offpeak_mask = ~onpeak_mask

            on_vals  = month_Pbuy[onpeak_mask]
            off_vals = month_Pbuy[offpeak_mask]

            peak_on  = float(np.max(on_vals))  if len(on_vals)  > 0 else 0.0
            peak_off = float(np.max(off_vals)) if len(off_vals) > 0 else 0.0

            monthly_peak_demand[m]         = peak_on
            monthly_peak_demand_offpeak[m] = peak_off

            monthly_demand_costs[m] = (
                peak_on  * Demand_charge_rate_monthly[m] +
                peak_off * Demand_charge_offpeak_rate_monthly[m]
            )

        else:
            # ---- Non-coincident peak (Types 1 / 2 / 3) --------------------
            peak = float(np.max(month_Pbuy)) if n_hours > 0 else 0.0
            monthly_peak_demand[m]  = peak
            monthly_demand_costs[m] = peak * Demand_charge_rate_monthly[m]

        hour_idx += n_hours

    # ---- Optional ratchet clause ------------------------------------------
    if demand_ratchet_enabled:
        annual_peak      = float(np.max(monthly_peak_demand))
        ratchet_floor_kw = annual_peak * demand_ratchet_rate

        for m in range(12):
            if monthly_peak_demand[m] < ratchet_floor_kw:
                if (demand_charge_type == 4
                        and demand_tou_onpeak_hours is not None):
                    monthly_demand_costs[m] = (
                        ratchet_floor_kw * Demand_charge_rate_monthly[m] +
                        monthly_peak_demand_offpeak[m] *
                        Demand_charge_offpeak_rate_monthly[m]
                    )
                else:
                    monthly_demand_costs[m] = (
                        ratchet_floor_kw * Demand_charge_rate_monthly[m]
                    )

    annual_demand_cost = float(np.sum(monthly_demand_costs))
    return annual_demand_cost, monthly_peak_demand, monthly_demand_costs


# ---------------------------------------------------------------------------
# Minimum charge rule
# ---------------------------------------------------------------------------

def compute_minimum_charge_adjustment(
        Pbuy, Psell, Cbuy, Csell,
        daysInMonth,
        Service_charge,
        Grid_Tax, Grid_Tax_amount,
        Annual_expenses, Grid_credit,
        minimum_charge_enabled,
        minimum_charge_monthly,
        monthly_demand_costs,
        Grid=1,
        credit_carryover=0):
    """
    Compute the annual upward adjustment to the electricity bill due to the
    minimum charge rule.

    The rule (verbatim from a typical commercial utility tariff):
        "Regardless of whether excess energy is delivered to the Company's
        electric grid, the customer will be required to pay the greater of
        (1) the minimum charge as stated in their applicable rate schedule,
        or (2) the applicable base charge plus the applicable demand charge
        for the maximum measured demand during the billing period in
        accordance with the provisions of their applicable rate schedule."

    This rule prevents NEM exports from reducing the monthly bill below a
    floor equal to max(minimum_charge, base_charge + demand_charge).

    Parameters
    ----------
    Pbuy : array (8760,)
        Hourly power purchased from the grid [kW].
    Psell : array (8760,) or None
        Hourly power sold to the grid [kW].
    Cbuy : array (8760,)
        Hourly electricity buy rate [$/kWh].
    Csell : array (8760,) or None
        Hourly electricity sell rate [$/kWh].
    daysInMonth : array (12,)
        Calendar days per month.
    Service_charge : array (12,)
        Monthly fixed customer (base) charge [$].
    Grid_Tax : float
        Sales tax rate applied to the utility bill (decimal).
    Grid_Tax_amount : float
        Per-kWh tax charged on grid purchases [$/kWh].
    Annual_expenses : float
        Annual fixed grid-related expenses [$] (spread equally over 12 months).
    Grid_credit : float
        Annual grid credit [$] (spread equally over 12 months).
    minimum_charge_enabled : int
        1 = active; 0 = disabled.
    minimum_charge_monthly : array (12,)
        Monthly minimum charge amounts [$] — (1) in the rule above.
    monthly_demand_costs : array (12,)
        Demand charge per month [$] — output of compute_monthly_peak_demand.
        Pass np.zeros(12) if demand charges are not enabled.
    Grid : int
        1 = grid-connected; 0 = off-grid (rule does not apply off-grid).
    credit_carryover : int, optional
        0 = a month whose bill is below the floor is billed at the floor and
        the export credit it cannot use is lost (default). 1 = that credit is
        carried forward and offsets later monthly charges down to their
        floor; credit left at the end of the year is lost
        (see apply_minimum_charge).

    Returns
    -------
    annual_adjustment : float
        Additional post-tax payment [$] required to meet the monthly floor,
        summed over the year.  Add this directly to the year-1 grid cost
        before escalation / discounting (it is already in the same post-tax
        units as Grid_Cost_net).
    monthly_adjustments : ndarray (12,)
        Per-month adjustment [$].

    Notes
    -----
    The floor is calculated AFTER NEM credits are applied, matching the
    rule's phrase "regardless of whether excess energy is delivered."
    Both the minimum_charge and the base_charge + demand_charge amounts
    are treated as pre-tax; the sales tax is applied to the floor before
    comparison with the normal (post-tax, post-credit) bill.
    """
    if not minimum_charge_enabled or not Grid:
        return 0.0, np.zeros(12)

    Pbuy  = np.asarray(Pbuy,  dtype=float)
    Cbuy  = np.asarray(Cbuy,  dtype=float)
    Psell = np.zeros_like(Pbuy) if Psell is None else np.asarray(Psell, dtype=float)
    Csell = np.zeros_like(Pbuy) if Csell is None else np.asarray(Csell, dtype=float)
    minimum_charge_monthly = np.asarray(minimum_charge_monthly, dtype=float)
    monthly_demand_costs   = np.asarray(monthly_demand_costs,   dtype=float)

    hours_in_month      = (np.asarray(daysInMonth, dtype=int) * 24)
    monthly_adjustments = np.zeros(12)
    normal_bills        = np.zeros(12)
    floors              = np.zeros(12)
    credits             = np.zeros(12)
    hour_idx = 0

    for m in range(12):
        n_hours  = int(hours_in_month[m])
        m_Pbuy   = Pbuy [hour_idx: hour_idx + n_hours]
        m_Psell  = Psell[hour_idx: hour_idx + n_hours]
        m_Cbuy   = Cbuy [hour_idx: hour_idx + n_hours]
        m_Csell  = Csell[hour_idx: hour_idx + n_hours]

        # --- Normal monthly bill (same structure as calculate_electricity_bills)
        energy_cost   = float(np.sum(m_Pbuy  * m_Cbuy))
        sell_revenue  = float(np.sum(m_Psell * m_Csell))
        grid_tax_cost = Grid_Tax_amount * float(np.sum(m_Pbuy))
        demand_cost_m = float(monthly_demand_costs[m])
        service_m     = float(Service_charge[m])

        pretax      = (energy_cost + grid_tax_cost + service_m +
                       Annual_expenses / 12.0 + demand_cost_m)
        normal_bill = pretax * (1.0 + Grid_Tax) - sell_revenue - Grid_credit / 12.0

        # --- Minimum charge floor -------------------------------------------
        # (1) minimum_charge  vs  (2) base_charge + demand_charge
        # base_charge = Service_charge[m]  (the fixed monthly customer charge)
        floor_pretax  = max(float(minimum_charge_monthly[m]),
                            service_m + demand_cost_m)
        floor_posttax = floor_pretax * (1.0 + Grid_Tax)

        normal_bills[m] = normal_bill
        floors[m]       = floor_posttax
        credits[m]      = sell_revenue + Grid_credit / 12.0

        hour_idx += n_hours

    # --- Adjustment = billed amount minus the normal bill -------------------
    monthly_adjustments = apply_minimum_charge(normal_bills, floors, credit_carryover, credits) - normal_bills
    return float(np.sum(monthly_adjustments)), monthly_adjustments


def apply_minimum_charge(normal_bills, floors, credit_carryover=0, monthly_credits=None):
    """
    Monthly amounts billed under the minimum charge rule.

    normal_bills are the post-tax monthly bills after this month's credits
    (export revenue and grid credit; negative when exports exceed purchases),
    floors the post-tax monthly minimums and monthly_credits those credits.
    Without carry-over each month is billed max(normal, floor) and credit the
    floor prevents a month from using is lost. With carry-over, that unused
    credit is carried to the next months and offsets their charges down to
    their floors. Only export and grid credit is carried, never the minimum
    charge itself, and credit left after December is lost (the conservative
    assumption for an annual true-up).
    """
    normal_bills = np.asarray(normal_bills, dtype=float)
    floors = np.asarray(floors, dtype=float)
    if not credit_carryover:
        return np.maximum(normal_bills, floors)
    if monthly_credits is None:
        raise ValueError("credit carry-over needs the monthly credits")
    credits = np.asarray(monthly_credits, dtype=float)
    billed = np.zeros(len(normal_bills))
    carried = 0.0
    for m in range(len(normal_bills)):
        charges = normal_bills[m] + credits[m]          # bill before any credit
        available = carried + credits[m]
        used = min(available, max(0.0, charges - floors[m]))
        billed[m] = max(floors[m], charges - available)
        carried = available - used
    return billed


def special_days(year, holidays, n_days=365):
    """Boolean array per day of the year: True for Saturdays, Sundays and holidays."""
    from datetime import date
    first = date(int(year), 1, 1).weekday()  # Monday = 0
    weekday = (first + np.arange(n_days)) % 7
    out = weekday >= 5
    if holidays is not None:
        h = np.asarray(holidays, dtype=int) - 1  # day-of-year numbers start at 1
        h = h[(h >= 0) & (h < n_days)]
        out[h] = True
    return out


# ---------------------------------------------------------------------------
# Convenience wrapper for NPC calculation
# ---------------------------------------------------------------------------

def demand_charge_npc(
        demand_charge_year1,
        cumulative_escalation,
        ir,
        n,
        Grid_Tax=0.0,
        Grid=1):
    """
    Present-value lifecycle cost of annual demand charges over the project
    horizon, using the same escalation / discount pattern as all other
    grid costs in the software.

    Parameters
    ----------
    demand_charge_year1 : float
        Annual demand charge for Year 1 [$].
    cumulative_escalation : ndarray, shape (n,)
        ``np.cumprod(1 + Grid_escalation)`` – cumulative price escalation
        factors already computed in Fitness.py / Results.py.
    ir : float
        Real discount rate (decimal).
    n : int
        Project lifetime in years.
    Grid_Tax : float, optional
        Sales-tax rate applied to utility bills (decimal).  Default 0.
    Grid : int, optional
        1 if grid-connected, 0 if off-grid.  Default 1.

    Returns
    -------
    ndarray, shape (n,)
        Year-by-year present-value demand charge [$], ready to be summed
        and added to the NPC.
    """
    if Grid == 0 or demand_charge_year1 == 0.0:
        return np.zeros(n)

    discount_factors = cumulative_escalation / (
        (1.0 + ir) ** np.arange(1, n + 1))

    return demand_charge_year1 * (1.0 + Grid_Tax) * discount_factors
