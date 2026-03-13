import numpy as np


def calculate_electricity_bills(daysInMonth, Pbuy, Cbuy, Service_charge,
                                Grid_Tax, Grid_Tax_amount, Annual_expenses,
                                Grid_escalation, Grid_credit, NEM_fee,
                                Grid, Psell=None, Csell=None):
    """
    Calculate monthly and annual electricity bills with option for selling electricity back to grid.

    Parameters:
    -----------
    daysInMonth : array (12,)
        Days in each month
    Pbuy : array (8760,)
        Hourly electricity BOUGHT from grid in kWh
    Cbuy : array (8760,)
        Hourly electricity rate in $/kWh
    Service_charge : array (12,)
        Monthly service charges in $
    Grid_Tax : float
        Sales tax rate (as decimal, e.g., 0.00986 for 0.986%)
    Grid_Tax_amount : float
        Grid tax amount in $/kWh
    Annual_expenses : float
        Annual fixed expenses in $
    Grid_escalation : array (25,)
        Yearly escalation rate (as decimal)
    Grid_credit : float
        Credits offered by grid in $
    NEM_fee : float
        Net metering one-time setup fee in $
    Grid : int
        Grid connection flag (1 = connected, 0 = not connected)
    Psell : array (8760,), optional
        Hourly power sold to grid in kWh
    Csell : array (8760,), optional
        Hourly sell rate in $/kWh
    """
    hours_in_month = daysInMonth * 24
    total_consumption = Pbuy  # This is what we actually buy from grid

    # Initialize arrays for monthly calculations
    monthly_energy_costs = np.zeros(12)
    monthly_sell_revenue = np.zeros(12)
    monthly_net_energy_costs = np.zeros(12)
    monthly_tax_costs = np.zeros(12)
    monthly_grid_tax = np.zeros(12)
    monthly_kwh_bought = np.zeros(12)
    monthly_kwh_sold = np.zeros(12)
    monthly_bills = np.zeros(12)

    # Check if selling is enabled
    has_sell = (Psell is not None and Csell is not None and Grid > 0)

    if not has_sell:
        Psell = np.zeros(8760)
        Csell = np.zeros(8760)

    # Calculate hour indices for each month
    hour_idx = 0

    for month in range(12):
        hours_this_month = hours_in_month[month]
        month_hours = slice(hour_idx, hour_idx + hours_this_month)

        # Extract data for this month
        month_consumption = total_consumption[month_hours]
        month_rates = Cbuy[month_hours]
        month_sell = Psell[month_hours]
        month_sell_rates = Csell[month_hours]

        # Calculate total kWh bought for the month
        total_kwh_bought = np.sum(month_consumption)
        monthly_kwh_bought[month] = total_kwh_bought

        # Calculate energy cost (hourly rate * hourly consumption)
        energy_cost = np.sum(month_consumption * month_rates)
        monthly_energy_costs[month] = energy_cost

        # Calculate sell revenue
        if has_sell:
            sell_revenue = np.sum(month_sell * month_sell_rates)
            monthly_sell_revenue[month] = sell_revenue
            total_kwh_sold = np.sum(month_sell)
            monthly_kwh_sold[month] = total_kwh_sold
        else:
            sell_revenue = 0
            monthly_sell_revenue[month] = 0
            monthly_kwh_sold[month] = 0

        # Calculate net energy cost (for reporting only)
        net_energy_cost = energy_cost - sell_revenue
        monthly_net_energy_costs[month] = net_energy_cost

        # Calculate grid tax (per kWh tax on purchased energy)
        grid_tax_cost = total_kwh_bought * Grid_Tax_amount
        monthly_grid_tax[month] = grid_tax_cost

        # Monthly bill formula matching Grid_Cost_net:
        #   (energy_cost + grid_tax + service + annual_exp/12) * (1 + sales_tax)
        #   - sell_revenue                                    ← subtracted AFTER tax
        #   - grid_credit/12
        #
        # Sell revenue is subtracted AFTER applying sales tax to the bought portion.
        # This is consistent with the user's Grid_Cost_net formula structure and
        # ensures that sum(monthly_bills) * df_year1 == year_1 projection value.
        monthly_before_tax = energy_cost + grid_tax_cost + Service_charge[month] + (Annual_expenses / 12)
        monthly_with_tax = monthly_before_tax * (1 + Grid_Tax)
        monthly_after_credit = monthly_with_tax - sell_revenue - (Grid_credit / 12)

        monthly_bills[month] = monthly_after_credit
        monthly_tax_costs[month] = monthly_before_tax * Grid_Tax

        hour_idx += hours_this_month

    # Calculate annual totals
    total_energy_cost = np.sum(monthly_energy_costs)
    total_sell_revenue = np.sum(monthly_sell_revenue)
    total_net_energy_cost = np.sum(monthly_net_energy_costs)
    total_service_charges = np.sum(Service_charge)
    total_grid_tax = np.sum(monthly_grid_tax)
    total_sales_tax = np.sum(monthly_tax_costs)

    # Annual cost: sum of monthly bills
    annual_cost = np.sum(monthly_bills)

    # Create detailed breakdown
    breakdown = {
        'total_energy_cost': total_energy_cost,
        'total_sell_revenue': total_sell_revenue,
        'total_net_energy_cost': total_net_energy_cost,
        'total_service_charges': total_service_charges,
        'total_grid_tax': total_grid_tax,
        'total_sales_tax': total_sales_tax,
        'annual_expenses': Annual_expenses,
        'nem_fee': NEM_fee,
        'grid_credit': Grid_credit,
        'total_kwh_bought': np.sum(monthly_kwh_bought),
        'total_kwh_sold': np.sum(monthly_kwh_sold),
        'net_kwh': np.sum(monthly_kwh_bought) - np.sum(monthly_kwh_sold)
    }

    return {
        'monthly_bills': monthly_bills,
        'annual_cost': annual_cost,
        'monthly_energy_costs': monthly_energy_costs,
        'monthly_sell_revenue': monthly_sell_revenue,
        'monthly_net_energy_costs': monthly_net_energy_costs,
        'monthly_tax_costs': monthly_tax_costs,
        'monthly_grid_tax': monthly_grid_tax,
        'monthly_kwh_bought': monthly_kwh_bought,
        'monthly_kwh_sold': monthly_kwh_sold,
        'monthly_service_charges': Service_charge,
        'breakdown': breakdown
    }


def calculate_nyear_projection(annual_cost_year1, Grid_escalation, Grid_credit,
                               Grid, n, ir,
                               Pbuy=None, Cbuy=None, Service_charge=None,
                               Grid_Tax=None, Grid_Tax_amount=None, Annual_expenses=None,
                               Psell=None, Csell=None):
    """
    Calculate n-year cost projection.

    Formula (matching calculate_electricity_bills):
        net_base = Annual_expenses + sum(Service_charge) + sum(Pbuy*Cbuy)
                   - sum(Psell*Csell) + Grid_Tax_amount * sum(Pbuy)

        Net_cost_year_k = (net_base * discount_factor_k) * (1 + Grid_Tax)
                          - Grid_credit * discount_factor_k

    IMPORTANT: Sell revenue reduces the taxable base (same as monthly bill calculation).
    This ensures the Year 1 projection value exactly matches the sum of Year 1 monthly bills.
    """

    # Calculate cumulative escalation factors
    cumulative_escalation = np.cumprod(1 + Grid_escalation[:n])

    # Calculate present value discount factors
    discount_factors = cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1))

    # Check if we have raw data to do proper calculation
    has_sell = (Psell is not None and Csell is not None)
    has_raw_data = (Pbuy is not None and Cbuy is not None and
                    Service_charge is not None and Grid_Tax is not None and
                    Grid_Tax_amount is not None and Annual_expenses is not None)

    if has_raw_data:
        # Grid_Cost_net formula (matching user's codebase):
        #   Grid_Cost_net = bought_base * (1+Tax) * df
        #                   - sold_base * df           ← sell subtracted AFTER tax
        #                   - credit * df
        #
        # This is consistent with the monthly bill formula in calculate_electricity_bills
        # which does: (buy + grid_tax + service + annual/12)*(1+Tax) - sell - credit/12
        # Summing 12 months and multiplying by df_year1 gives exactly this formula.
        #
        # bought_base = Annual_expenses + sum(Service_charge) + sum(Pbuy*Cbuy)
        #               + Grid_Tax_amount * sum(Pbuy)
        # sold_base   = sum(Psell * Csell)   [raw sell revenue, no tax multiplier]

        bought_base = (Annual_expenses + np.sum(Service_charge) +
                       np.sum(Pbuy * Cbuy) +
                       Grid_Tax_amount * np.sum(Pbuy))
        sold_base = np.sum(Psell * Csell) if has_sell else 0.0

        # Apply escalation, discount, and sales tax to bought; subtract raw sell and credit
        yearly_net_costs_pv = (bought_base * (1 + Grid_Tax) * discount_factors
                               - sold_base * discount_factors
                               - Grid_credit * discount_factors)
        yearly_net_costs_nominal = (bought_base * (1 + Grid_Tax) * cumulative_escalation
                                    - sold_base * cumulative_escalation
                                    - Grid_credit * cumulative_escalation)

        # Store bought/sold breakdowns for reporting
        bought_electricity_pv = bought_base * (1 + Grid_Tax) * discount_factors
        bought_electricity_nominal = bought_base * (1 + Grid_Tax) * cumulative_escalation

        # sold_electricity_pv stores raw sell revenue escalated/discounted (no tax multiplier)
        # This is the true economic value of energy sold back to grid.
        # Note: bought_pv - sold_pv - credit_pv == net_cost_pv (identity holds exactly)
        if has_sell:
            sold_electricity_pv = sold_base * discount_factors
            sold_electricity_nominal = sold_base * cumulative_escalation
        else:
            sold_electricity_pv = np.zeros(n)
            sold_electricity_nominal = np.zeros(n)

        yearly_costs_pv = bought_electricity_pv
        yearly_costs_nominal = bought_electricity_nominal
        yearly_sell_revenue_pv = sold_electricity_pv
        yearly_sell_revenue_nominal = sold_electricity_nominal

    else:
        # Fallback: raw data not provided, use year1 annual cost directly
        if has_sell:
            annual_sell_revenue_year1 = np.sum(Psell * Csell)
        else:
            annual_sell_revenue_year1 = 0

        base_cost_year1 = annual_cost_year1 + Grid_credit + annual_sell_revenue_year1

        yearly_costs_nominal = base_cost_year1 * cumulative_escalation - Grid_credit * cumulative_escalation
        yearly_sell_revenue_nominal = annual_sell_revenue_year1 * cumulative_escalation if has_sell else np.zeros(n)
        yearly_net_costs_nominal = yearly_costs_nominal - yearly_sell_revenue_nominal

        yearly_costs_pv = base_cost_year1 * discount_factors - Grid_credit * discount_factors
        yearly_sell_revenue_pv = annual_sell_revenue_year1 * discount_factors if has_sell else np.zeros(n)
        yearly_net_costs_pv = yearly_costs_pv - yearly_sell_revenue_pv

    # Calculate totals
    total_nominal_cost = np.sum(yearly_net_costs_nominal)
    total_pv_cost = np.sum(yearly_net_costs_pv)
    total_sell_revenue_nominal = np.sum(yearly_sell_revenue_nominal)
    total_sell_revenue_pv = np.sum(yearly_sell_revenue_pv)

    return {
        'yearly_costs_nominal': yearly_costs_nominal,
        'yearly_costs_pv': yearly_costs_pv,
        'yearly_sell_revenue_nominal': yearly_sell_revenue_nominal,
        'yearly_sell_revenue_pv': yearly_sell_revenue_pv,
        'yearly_net_costs_nominal': yearly_net_costs_nominal,
        'yearly_net_costs_pv': yearly_net_costs_pv,
        'total_nominal_cost': total_nominal_cost,
        'total_pv_cost': total_pv_cost,
        'total_sell_revenue_nominal': total_sell_revenue_nominal,
        'total_sell_revenue_pv': total_sell_revenue_pv,
        'npv': total_pv_cost,
        'average_annual_nominal': total_nominal_cost / n,
        'average_annual_pv': total_pv_cost / n,
        'discount_rate': ir,
        'n_years': n,
        'has_sell': has_sell
    }


def calculate_ng_lifecycle_cost(Hload, Cbuy_NG, Service_charge_NG, Grid_Tax_NG,
                                Grid_Tax_amount_NG, Annual_expenses_NG, Grid_escalation_NG,
                                Grid_credit_NG, HP, n, ir, daysInMonth=None):
    """
    Calculate Natural Gas lifecycle cost for scenarios with only gas furnace (no heat pump).

    Uses formula:
    NG_Grid_Cost_onlyG = (((Annual_expenses_NG + sum(Service_charge_NG) + sum(Hload * Cbuy_NG) +
                           Grid_Tax_amount_NG * sum(Hload)) * discount_factors) * (1 + Grid_Tax_NG) -
                          (Grid_credit_NG * discount_factors)) * (HP > 0)

    Parameters:
    -----------
    Hload : array (8760,)
        Hourly heating load in kWh
    Cbuy_NG : array (8760,)
        Hourly natural gas rate in $/kWh
    Service_charge_NG : array (12,)
        Monthly NG service charges in $
    Grid_Tax_NG : float
        NG sales tax rate (as decimal)
    Grid_Tax_amount_NG : float
        NG grid tax amount in $/kWh
    Annual_expenses_NG : float
        Annual fixed NG expenses in $
    Grid_escalation_NG : array (n,)
        Yearly NG escalation rates
    Grid_credit_NG : float
        NG credits offered in $
    HP : int
        Heat pump flag (1 = has heat pump, 0 = no heat pump)
    n : int
        Number of years
    ir : float
        Discount/interest rate as decimal
    daysInMonth : array (12,), optional
        Days in each month. Defaults to standard non-leap year [31,28,...,31].
        Should be passed from compare_scenarios to ensure consistent hour indexing.

    Returns:
    --------
    dict : Contains NG costs in PV and nominal terms, plus monthly Year-1 breakdown
    """

    # Only calculate if HP > 0 (heat pump is present, meaning we saved on NG)
    if HP <= 0 or Hload is None or Cbuy_NG is None:
        return {
            'yearly_ng_costs_pv': np.zeros(n),
            'yearly_ng_costs_nominal': np.zeros(n),
            'monthly_ng_costs_year1': np.zeros(12),
            'total_ng_cost_pv': 0.0,
            'total_ng_cost_nominal': 0.0,
            'has_ng_savings': False
        }

    # Use provided daysInMonth or fall back to standard non-leap year
    if daysInMonth is None:
        days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    else:
        days_per_month = np.asarray(daysInMonth)

    # Calculate cumulative escalation factors
    cumulative_escalation_NG = np.cumprod(1 + Grid_escalation_NG[:n])

    # Calculate present value discount factors
    discount_factors = cumulative_escalation_NG / ((1 + ir) ** np.arange(1, n + 1))
    df_year1 = discount_factors[0]

    # --- Monthly NG costs (base year, for accurate monthly display) ---
    # Compute per-month NG bills from actual hourly Hload profile.
    monthly_ng_costs_base = np.zeros(12)
    hour_idx = 0
    for m in range(12):
        hours = days_per_month[m] * 24
        month_slice = slice(hour_idx, hour_idx + hours)
        hload_m = Hload[month_slice]
        cbuy_ng_m = Cbuy_NG[month_slice]
        energy_cost_m = np.sum(hload_m * cbuy_ng_m)
        grid_tax_m = np.sum(hload_m) * Grid_Tax_amount_NG
        # Per-month share of annual fixed costs (service + annual expenses)
        fixed_m = Service_charge_NG[m] + Annual_expenses_NG / 12
        before_tax_m = energy_cost_m + grid_tax_m + fixed_m
        monthly_ng_costs_base[m] = before_tax_m * (1 + Grid_Tax_NG) - Grid_credit_NG / 12
        hour_idx += hours

    # Scale to Year 1 PV
    monthly_ng_costs_year1 = monthly_ng_costs_base * df_year1

    # --- Annual NG lifecycle cost ---
    ng_base = (Annual_expenses_NG + np.sum(Service_charge_NG) +
               np.sum(Hload * Cbuy_NG) + Grid_Tax_amount_NG * np.sum(Hload))

    # Apply escalation, discount, and tax
    ng_bought_pv = (ng_base * discount_factors) * (1 + Grid_Tax_NG)
    ng_bought_nominal = (ng_base * cumulative_escalation_NG) * (1 + Grid_Tax_NG)

    # Apply NG credits
    ng_credits_pv = Grid_credit_NG * discount_factors
    ng_credits_nominal = Grid_credit_NG * cumulative_escalation_NG

    # Net NG costs
    yearly_ng_costs_pv = ng_bought_pv - ng_credits_pv
    yearly_ng_costs_nominal = ng_bought_nominal - ng_credits_nominal

    total_ng_cost_pv = np.sum(yearly_ng_costs_pv)
    total_ng_cost_nominal = np.sum(yearly_ng_costs_nominal)

    return {
        'yearly_ng_costs_pv': yearly_ng_costs_pv,
        'yearly_ng_costs_nominal': yearly_ng_costs_nominal,
        'monthly_ng_costs_year1': monthly_ng_costs_year1,
        'total_ng_cost_pv': total_ng_cost_pv,
        'total_ng_cost_nominal': total_ng_cost_nominal,
        'has_ng_savings': True
    }


def compare_scenarios(daysInMonth, Pbuy_no_sell, Pbuy_with_sell, Cbuy, Service_charge,
                      Grid_Tax, Grid_Tax_amount, Annual_expenses,
                      Grid_escalation, Grid_credit, NEM_fee,
                      Grid, n, ir, Psell=None, Csell=None,
                      HP=0, Hload=None, Cbuy_NG=None, Service_charge_NG=None,
                      Grid_Tax_NG=None, Grid_Tax_amount_NG=None, Annual_expenses_NG=None,
                      Grid_escalation_NG=None, Grid_credit_NG=None):
    """
    Compare electricity costs with and without selling electricity to grid.
    Includes Natural Gas savings when heat pumps are present.

    Parameters:
    -----------
    daysInMonth : array (12,)
        Days in each month
    Pbuy_no_sell : array (8760,)
        Hourly electricity BOUGHT from grid in "no sell" scenario (kWh)
    Pbuy_with_sell : array (8760,)
        Hourly electricity BOUGHT from grid in "with sell" scenario (kWh)
    Cbuy : array (8760,)
        Hourly electricity rate in $/kWh
    Service_charge : array (12,)
        Monthly service charges in $
    Grid_Tax : float
        Sales tax rate (as decimal)
    Grid_Tax_amount : float
        Grid tax amount in $/kWh
    Annual_expenses : float
        Annual fixed expenses in $
    Grid_escalation : array (n,)
        Yearly escalation rates
    Grid_credit : float
        Credits offered by grid in $
    NEM_fee : float
        Net metering one-time setup fee in $
    Grid : int
        Grid connection flag (1 = connected, 0 = not connected)
    n : int
        Number of years to project
    ir : float
        Discount/interest rate as decimal
    Psell : array (8760,), optional
        Hourly power sold to grid in kWh
    Csell : array (8760,), optional
        Hourly sell rate in $/kWh
    HP : int, optional
        Heat pump flag (1 = has heat pump, 0 = no heat pump)
    Hload : array (8760,), optional
        Hourly heating load in kWh (for NG calculation)
    Cbuy_NG : array (8760,), optional
        Hourly natural gas rate in $/kWh
    Service_charge_NG : array (12,), optional
        Monthly NG service charges in $
    Grid_Tax_NG : float, optional
        NG sales tax rate (as decimal)
    Grid_Tax_amount_NG : float, optional
        NG grid tax amount in $/kWh
    Annual_expenses_NG : float, optional
        Annual fixed NG expenses in $
    Grid_escalation_NG : array (n,), optional
        Yearly NG escalation rates
    Grid_credit_NG : float, optional
        NG credits offered in $
    """

    # Scenario 1: WITHOUT selling
    results_no_sell = calculate_electricity_bills(daysInMonth,
                                                  Pbuy_no_sell, Cbuy, Service_charge,
                                                  Grid_Tax, Grid_Tax_amount, Annual_expenses,
                                                  Grid_escalation, Grid_credit, NEM_fee,
                                                  Grid, Psell=None, Csell=None
                                                  )

    projection_no_sell = calculate_nyear_projection(
        results_no_sell['annual_cost'],
        Grid_escalation,
        Grid_credit,
        Grid, n=n, ir=ir,
        Pbuy=Pbuy_no_sell, Cbuy=Cbuy, Service_charge=Service_charge,
        Grid_Tax=Grid_Tax, Grid_Tax_amount=Grid_Tax_amount, Annual_expenses=Annual_expenses,
        Psell=None, Csell=None
    )

    # Scenario 2: WITH selling
    results_with_sell = calculate_electricity_bills(daysInMonth,
                                                    Pbuy_with_sell, Cbuy, Service_charge,
                                                    Grid_Tax, Grid_Tax_amount, Annual_expenses,
                                                    Grid_escalation, Grid_credit, NEM_fee,
                                                    Grid=Grid, Psell=Psell, Csell=Csell
                                                    )

    projection_with_sell = calculate_nyear_projection(
        results_with_sell['annual_cost'],
        Grid_escalation,
        Grid_credit,
        Grid=Grid, n=n, ir=ir,
        Pbuy=Pbuy_with_sell, Cbuy=Cbuy, Service_charge=Service_charge,
        Grid_Tax=Grid_Tax, Grid_Tax_amount=Grid_Tax_amount, Annual_expenses=Annual_expenses,
        Psell=Psell, Csell=Csell
    )

    # Calculate Natural Gas costs if heat pump is present
    ng_costs = calculate_ng_lifecycle_cost(
        Hload, Cbuy_NG, Service_charge_NG, Grid_Tax_NG,
        Grid_Tax_amount_NG, Annual_expenses_NG, Grid_escalation_NG,
        Grid_credit_NG, HP, n, ir, daysInMonth=daysInMonth
    )

    # Apply NG costs to the baseline "no HES" scenario
    # - Without HES: Grid electricity + Natural Gas furnace
    # - With HES: Solar + Grid electricity + Heat pump (no NG furnace)
    if ng_costs['has_ng_savings']:
        projection_no_sell_adjusted = projection_no_sell.copy()
        projection_no_sell_adjusted['yearly_net_costs_pv'] = (
                projection_no_sell['yearly_net_costs_pv'] + ng_costs['yearly_ng_costs_pv']
        )
        projection_no_sell_adjusted['yearly_net_costs_nominal'] = (
                projection_no_sell['yearly_net_costs_nominal'] + ng_costs['yearly_ng_costs_nominal']
        )
        projection_no_sell_adjusted['total_pv_cost'] = np.sum(
            projection_no_sell_adjusted['yearly_net_costs_pv'])
        projection_no_sell_adjusted['total_nominal_cost'] = np.sum(
            projection_no_sell_adjusted['yearly_net_costs_nominal'])
        projection_no_sell_adjusted['average_annual_pv'] = (
            projection_no_sell_adjusted['total_pv_cost'] / n)
        projection_no_sell_adjusted['average_annual_nominal'] = (
            projection_no_sell_adjusted['total_nominal_cost'] / n)

        projection_with_sell_adjusted = projection_with_sell
    else:
        projection_no_sell_adjusted = projection_no_sell
        projection_with_sell_adjusted = projection_with_sell

    # Calculate savings
    monthly_savings = results_no_sell['monthly_bills'] - results_with_sell['monthly_bills']
    annual_savings_year0 = results_no_sell['annual_cost'] - results_with_sell['annual_cost']
    annual_savings_year1 = (projection_no_sell_adjusted['yearly_net_costs_nominal'][0] -
                            projection_with_sell_adjusted['yearly_net_costs_nominal'][0])
    annual_savings_year1_pv = (projection_no_sell_adjusted['yearly_net_costs_pv'][0] -
                               projection_with_sell_adjusted['yearly_net_costs_pv'][0])

    total_savings_nominal = (projection_no_sell_adjusted['total_nominal_cost'] -
                             projection_with_sell_adjusted['total_nominal_cost'])
    total_savings_pv = (projection_no_sell_adjusted['total_pv_cost'] -
                        projection_with_sell_adjusted['total_pv_cost'])

    yearly_savings_nominal = (projection_no_sell_adjusted['yearly_net_costs_nominal'] -
                              projection_with_sell_adjusted['yearly_net_costs_nominal'])
    yearly_savings_pv = (projection_no_sell_adjusted['yearly_net_costs_pv'] -
                         projection_with_sell_adjusted['yearly_net_costs_pv'])

    return {
        'no_sell': {
            'results': results_no_sell,
            'projection': projection_no_sell_adjusted
        },
        'with_sell': {
            'results': results_with_sell,
            'projection': projection_with_sell_adjusted
        },
        'ng_costs': ng_costs,
        'params': {
            'Grid_escalation': Grid_escalation,
            'ir': ir,
            'Grid_Tax': Grid_Tax,
            'Service_charge': Service_charge,
            'Annual_expenses': Annual_expenses,
            'Grid_credit': Grid_credit
        },
        'savings': {
            'monthly_savings': monthly_savings,
            'annual_savings_year0': annual_savings_year0,
            'annual_savings_year1': annual_savings_year1,
            'annual_savings_year1_pv': annual_savings_year1_pv,
            'total_savings_nominal': total_savings_nominal,
            'total_savings_pv': total_savings_pv,
            'yearly_savings_nominal': yearly_savings_nominal,
            'yearly_savings_pv': yearly_savings_pv,
            'annual_sell_revenue': results_with_sell['breakdown']['total_sell_revenue'],
            'ng_savings_pv': ng_costs['total_ng_cost_pv'] if ng_costs['has_ng_savings'] else 0,
            'ng_savings_nominal': ng_costs['total_ng_cost_nominal'] if ng_costs['has_ng_savings'] else 0,
            'percent_savings_year0': (
                (annual_savings_year0 / results_no_sell['annual_cost']) * 100
                if results_no_sell['annual_cost'] > 0 else 0),
            'percent_savings_year1': (
                (annual_savings_year1 / projection_no_sell_adjusted['yearly_net_costs_nominal'][0]) * 100
                if projection_no_sell_adjusted['yearly_net_costs_nominal'][0] > 0 else 0),
            'percent_savings_year1_pv': (
                (annual_savings_year1_pv / projection_no_sell_adjusted['yearly_net_costs_pv'][0]) * 100
                if projection_no_sell_adjusted['yearly_net_costs_pv'][0] > 0 else 0),
            'percent_savings_25year': (
                (total_savings_nominal / projection_no_sell_adjusted['total_nominal_cost']) * 100
                if projection_no_sell_adjusted['total_nominal_cost'] > 0 else 0)
        }
    }


def print_comparison(comparison, show_plot, save_path=None):
    """Print detailed comparison between scenarios with and without selling.

    Parameters:
    -----------
    comparison : dict
        Dictionary returned by compare_scenarios()
    show_plot : bool
        If True, display matplotlib comparison plots
    """
    import matplotlib.pyplot as plt

    results_no_sell = comparison['no_sell']['results']
    results_with_sell = comparison['with_sell']['results']
    proj_no_sell = comparison['no_sell']['projection']
    proj_with_sell = comparison['with_sell']['projection']
    savings = comparison['savings']
    ng_costs = comparison['ng_costs']
    params = comparison['params']

    # Extract parameters
    Grid_escalation = params['Grid_escalation']
    ir = params['ir']
    Grid_Tax = params['Grid_Tax']
    Service_charge = params['Service_charge']
    Annual_expenses = params['Annual_expenses']
    Grid_credit = params['Grid_credit']

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print("\n" + "=" * 100)
    print("SCENARIO COMPARISON: WITH vs WITHOUT HES")
    print("=" * 100)

    # Executive Summary Table
    print("\n" + "╔" + "=" * 98 + "╗")
    print(f"║ {'EXECUTIVE SUMMARY':^96} ║")
    print("╠" + "=" * 98 + "╣")
    print(f"║ {'Metric':<50} {'Without HES':>20} {'With HES':>20} ║")
    print("╠" + "-" * 98 + "╣")

    # Year 1 costs (PRESENT VALUE, with escalation and discounting applied)
    year1_cost_no_sell_pv = proj_no_sell['yearly_net_costs_pv'][0]
    year1_cost_with_sell_pv = proj_with_sell['yearly_net_costs_pv'][0]

    if ng_costs['has_ng_savings']:
        year1_elec_no_sell = year1_cost_no_sell_pv - ng_costs['yearly_ng_costs_pv'][0]
        year1_ng_cost = ng_costs['yearly_ng_costs_pv'][0]

        print(f"║ {'Year 1 Electricity Cost (Present Value)':<50} "
              f"${year1_elec_no_sell:>19,.2f} ${year1_cost_with_sell_pv:>19,.2f} ║")
        print(f"║ {'Year 1 Natural Gas Cost (Present Value)':<50} "
              f"${year1_ng_cost:>19,.2f} ${0:>19,.2f} ║")
        print("╠" + "-" * 98 + "╣")

    print(f"║ {'Year 1 Total Annual Cost (Present Value)':<50} "
          f"${year1_cost_no_sell_pv:>19,.2f} ${year1_cost_with_sell_pv:>19,.2f} ║")
    print(f"║ {'Year 1 Monthly Average (Present Value)':<50} "
          f"${year1_cost_no_sell_pv / 12:>19,.2f} ${year1_cost_with_sell_pv / 12:>19,.2f} ║")
    print(f"║ {'Year 1 Savings (Present Value)':<50} "
          f"{'-':>20} ${savings['annual_savings_year1_pv']:>19,.2f} ║")

    if ng_costs['has_ng_savings']:
        year1_elec_savings = year1_elec_no_sell - year1_cost_with_sell_pv
        year1_ng_savings = year1_ng_cost
        print(f"║ {'  - Electricity Savings':<50} {'-':>20} ${year1_elec_savings:>19,.2f} ║")
        print(f"║ {'  - Natural Gas Savings':<50} {'-':>20} ${year1_ng_savings:>19,.2f} ║")

    print(f"║ {'Year 1 Savings % (Present Value)':<50} "
          f"{'-':>20} {savings['percent_savings_year1_pv']:>19.1f}% ║")

    print("╠" + "-" * 98 + "╣")

    # 25-year projections
    if ng_costs['has_ng_savings']:
        total_elec_no_sell = proj_no_sell['total_nominal_cost'] - ng_costs['total_ng_cost_nominal']
        total_elec_no_sell_pv = proj_no_sell['total_pv_cost'] - ng_costs['total_ng_cost_pv']

        print(f"║ {'25-Year Electricity Cost (Nominal)':<50} "
              f"${total_elec_no_sell:>19,.2f} ${proj_with_sell['total_nominal_cost']:>19,.2f} ║")
        print(f"║ {'25-Year Natural Gas Cost (Nominal)':<50} "
              f"${ng_costs['total_ng_cost_nominal']:>19,.2f} ${0:>19,.2f} ║")
        print("╠" + "-" * 98 + "╣")

    print(f"║ {'25-Year Total Cost (Nominal)':<50} "
          f"${proj_no_sell['total_nominal_cost']:>19,.2f} ${proj_with_sell['total_nominal_cost']:>19,.2f} ║")

    if ng_costs['has_ng_savings']:
        print(f"║ {'25-Year Electricity Cost (Present Value)':<50} "
              f"${total_elec_no_sell_pv:>19,.2f} ${proj_with_sell['total_pv_cost']:>19,.2f} ║")
        print(f"║ {'25-Year Natural Gas Cost (Present Value)':<50} "
              f"${ng_costs['total_ng_cost_pv']:>19,.2f} ${0:>19,.2f} ║")
        print("╠" + "-" * 98 + "╣")

    print(f"║ {'25-Year Total Cost (Present Value)':<50} "
          f"${proj_no_sell['total_pv_cost']:>19,.2f} ${proj_with_sell['total_pv_cost']:>19,.2f} ║")
    print(f"║ {'25-Year Total Savings (Nominal)':<50} "
          f"{'-':>20} ${savings['total_savings_nominal']:>19,.2f} ║")

    if ng_costs['has_ng_savings']:
        total_elec_savings_nom = total_elec_no_sell - proj_with_sell['total_nominal_cost']
        print(f"║ {'  - Electricity Savings (Nominal)':<50} "
              f"{'-':>20} ${total_elec_savings_nom:>19,.2f} ║")
        print(f"║ {'  - Natural Gas Savings (Nominal)':<50} "
              f"{'-':>20} ${ng_costs['total_ng_cost_nominal']:>19,.2f} ║")
        print("╠" + "-" * 98 + "╣")

    print(f"║ {'25-Year Total Savings (PV)':<50} "
          f"{'-':>20} ${savings['total_savings_pv']:>19,.2f} ║")

    if ng_costs['has_ng_savings']:
        total_elec_savings_pv = total_elec_no_sell_pv - proj_with_sell['total_pv_cost']
        print(f"║ {'  - Electricity Savings (PV)':<50} "
              f"{'-':>20} ${total_elec_savings_pv:>19,.2f} ║")
        print(f"║ {'  - Natural Gas Savings (PV)':<50} "
              f"{'-':>20} ${ng_costs['total_ng_cost_pv']:>19,.2f} ║")

    print(f"║ {'25-Year Savings %':<50} "
          f"{'-':>20} {savings['percent_savings_25year']:>19.1f}% ║")

    print("╠" + "-" * 98 + "╣")
    print("╠" + "-" * 98 + "╣")

    # Energy metrics
    print(f"║ {'Total kWh Bought (Base Year)':<50} "
          f"{results_no_sell['breakdown']['total_kwh_bought']:>19,.1f} "
          f"{results_with_sell['breakdown']['total_kwh_bought']:>19,.1f} ║")
    print(f"║ {'Total kWh Sold (Base Year)':<50} "
          f"{0:>19,.1f} "
          f"{results_with_sell['breakdown']['total_kwh_sold']:>19,.1f} ║")
    print(f"║ {'Net kWh (Buy - Sell, Base Year)':<50} "
          f"{results_no_sell['breakdown']['net_kwh']:>19,.1f} "
          f"{results_with_sell['breakdown']['net_kwh']:>19,.1f} ║")

    print("╠" + "-" * 98 + "╣")

    # Averages
    print(f"║ {'Average Annual Cost - 25 Years (Nominal)':<50} "
          f"${proj_no_sell['average_annual_nominal']:>19,.2f} "
          f"${proj_with_sell['average_annual_nominal']:>19,.2f} ║")
    print(f"║ {'Average Annual Cost - 25 Years (PV)':<50} "
          f"${proj_no_sell['average_annual_pv']:>19,.2f} "
          f"${proj_with_sell['average_annual_pv']:>19,.2f} ║")

    print("╚" + "=" * 98 + "╝\n")

    # -------------------------------------------------------------------------
    # MONTHLY BILL COMPARISON - YEAR 1 (Present Value)
    # -------------------------------------------------------------------------
    # Apply Year 1 discount factor to base monthly electricity bills
    escalation_year1 = 1 + Grid_escalation[0]
    discount_year1 = escalation_year1 / (1 + ir)

    monthly_bills_no_sell_year1 = results_no_sell['monthly_bills'] * discount_year1
    monthly_bills_with_sell_year1 = results_with_sell['monthly_bills'] * discount_year1

    # Year 1 PV totals from projection (electricity only, NG handled separately)
    year1_elec_pv_no_sell = proj_no_sell['yearly_net_costs_pv'][0]
    year1_elec_pv_with_sell = proj_with_sell['yearly_net_costs_pv'][0]

    if ng_costs['has_ng_savings']:
        # Subtract the NG component to get electricity-only Year 1 PV
        year1_elec_pv_no_sell -= ng_costs['yearly_ng_costs_pv'][0]

    monthly_sum_no_sell = np.sum(monthly_bills_no_sell_year1)
    monthly_sum_with_sell = np.sum(monthly_bills_with_sell_year1)

    print("\nMONTHLY BILL COMPARISON - YEAR 1 (Present Value)")
    print(f"Verification: Monthly elec sum Without HES = ${monthly_sum_no_sell:,.2f}, "
          f"Year 1 Elec (PV) = ${year1_elec_pv_no_sell:,.2f}")
    print(f"              Monthly elec sum With HES    = ${monthly_sum_with_sell:,.2f}, "
          f"Year 1 Elec (PV) = ${year1_elec_pv_with_sell:,.2f}")
    print("-" * 100)

    if ng_costs['has_ng_savings']:
        # --- FIX: Use accurate monthly NG from calculate_ng_lifecycle_cost ---
        # monthly_ng_costs_year1 is already computed per-month from actual heating load,
        # NOT proportionally distributed from annual NG cost.
        monthly_ng_year1 = ng_costs['monthly_ng_costs_year1']

        print(f"{'Month':<6} {'Elec':>10} {'NG':>10} {'Total':>12} {'With HES':>12} "
              f"{'Elec Sav':>12} {'NG Sav':>10} {'Total Sav':>12} {'% Saved':>10}")
        print(f"{'':>6} {'No HES':>10} {'No HES':>10} {'No HES':>12} "
              f"{'(Elec)':>12} {'':>12} {'':>10} {'':>12} {'':>10}")
    else:
        print(f"{'Month':<6} {'No HES':>12} {'With HES':>12} {'Savings':>12} {'% Saved':>10}")

    print("-" * 100)

    for i in range(12):
        if ng_costs['has_ng_savings']:
            # Electricity no HES = electricity-only monthly bill (no NG)
            elec_no_hes = monthly_bills_no_sell_year1[i]
            ng_no_hes = monthly_ng_year1[i]
            total_no_hes = elec_no_hes + ng_no_hes

            # With HES: only electricity (net, can be negative when exporting)
            with_hes = monthly_bills_with_sell_year1[i]

            # Savings breakdown
            elec_savings = elec_no_hes - with_hes
            ng_savings = ng_no_hes          # all NG cost is saved (heat pump replaces furnace)
            total_savings = total_no_hes - with_hes

            percent = (total_savings / total_no_hes) * 100 if total_no_hes != 0 else 0

            print(f"{months[i]:<6} ${elec_no_hes:>9,.2f} ${ng_no_hes:>9,.2f} "
                  f"${total_no_hes:>11,.2f} "
                  f"${with_hes:>11,.2f} "
                  f"${elec_savings:>11,.2f} ${ng_savings:>9,.2f} "
                  f"${total_savings:>11,.2f} "
                  f"{percent:>9.1f}%")
        else:
            percent = ((monthly_bills_no_sell_year1[i] - monthly_bills_with_sell_year1[i]) /
                       monthly_bills_no_sell_year1[i]) * 100 \
                if monthly_bills_no_sell_year1[i] > 0 else 0
            print(f"{months[i]:<6} ${monthly_bills_no_sell_year1[i]:>11.2f} "
                  f"${monthly_bills_with_sell_year1[i]:>11.2f} "
                  f"${monthly_bills_no_sell_year1[i] - monthly_bills_with_sell_year1[i]:>11.2f} "
                  f"{percent:>9.1f}%")

    print("-" * 100)

    if ng_costs['has_ng_savings']:
        total_elec_no_hes = np.sum(monthly_bills_no_sell_year1)
        total_ng_no_hes = np.sum(monthly_ng_year1)
        total_total_no_hes = total_elec_no_hes + total_ng_no_hes
        total_with_hes = np.sum(monthly_bills_with_sell_year1)
        total_elec_savings = total_elec_no_hes - total_with_hes
        total_ng_savings = total_ng_no_hes
        total_total_savings = total_total_no_hes - total_with_hes
        pct = (total_total_savings / total_total_no_hes) * 100 if total_total_no_hes != 0 else 0

        print(f"{'TOTAL':<6} ${total_elec_no_hes:>9,.2f} ${total_ng_no_hes:>9,.2f} "
              f"${total_total_no_hes:>11,.2f} "
              f"${total_with_hes:>11,.2f} "
              f"${total_elec_savings:>11,.2f} ${total_ng_savings:>9,.2f} "
              f"${total_total_savings:>11,.2f} "
              f"{pct:>9.1f}%")
    else:
        total_sav = np.sum(monthly_bills_no_sell_year1) - np.sum(monthly_bills_with_sell_year1)
        total_no = np.sum(monthly_bills_no_sell_year1)
        pct = (total_sav / total_no) * 100 if total_no > 0 else 0
        print(f"{'TOTAL':<6} ${total_no:>11.2f} "
              f"${np.sum(monthly_bills_with_sell_year1):>11.2f} "
              f"${total_sav:>11.2f} "
              f"{pct:>9.1f}%")

    print(f"\nNote: Year 1 values with escalation and discounting applied (Present Value)")
    if ng_costs['has_ng_savings']:
        print(f"      Natural Gas costs are included in 'Without HES' scenario (grid electricity + gas furnace)")
        print(f"      'With HES' uses heat pump instead of gas furnace")
        print(f"      Monthly NG costs derived from hourly heating load profile (accurate seasonal distribution)")
    print(f"      Monthly elec sums should match Year 1 Elec (PV) from the year-by-year table")

    # 25-year projection comparison summary
    print("\n" + "=" * 100)
    print("25-YEAR PROJECTION SUMMARY")
    print("=" * 100)
    print(f"{'Metric':<40} {'Without HES':>20} {'With HES':>20} {'Savings':>20}")
    print("-" * 100)

    if ng_costs['has_ng_savings']:
        total_elec_no_sell_nom = proj_no_sell['total_nominal_cost'] - ng_costs['total_ng_cost_nominal']
        total_elec_no_sell_pv = proj_no_sell['total_pv_cost'] - ng_costs['total_ng_cost_pv']

        print(f"{'Electricity Cost (Nominal)':<40} ${total_elec_no_sell_nom:>19,.2f} "
              f"${proj_with_sell['total_nominal_cost']:>19,.2f} "
              f"${total_elec_no_sell_nom - proj_with_sell['total_nominal_cost']:>19,.2f}")
        print(f"{'Natural Gas Cost (Nominal)':<40} ${ng_costs['total_ng_cost_nominal']:>19,.2f} "
              f"${0:>19,.2f} "
              f"${ng_costs['total_ng_cost_nominal']:>19,.2f}")
        print(f"{'-' * 40} {'-' * 20} {'-' * 20} {'-' * 20}")

    print(f"{'Total Cost (Nominal)':<40} ${proj_no_sell['total_nominal_cost']:>19,.2f} "
          f"${proj_with_sell['total_nominal_cost']:>19,.2f} "
          f"${savings['total_savings_nominal']:>19,.2f}")

    if ng_costs['has_ng_savings']:
        print(f"{'Electricity Cost (Present Value)':<40} ${total_elec_no_sell_pv:>19,.2f} "
              f"${proj_with_sell['total_pv_cost']:>19,.2f} "
              f"${total_elec_no_sell_pv - proj_with_sell['total_pv_cost']:>19,.2f}")
        print(f"{'Natural Gas Cost (Present Value)':<40} ${ng_costs['total_ng_cost_pv']:>19,.2f} "
              f"${0:>19,.2f} "
              f"${ng_costs['total_ng_cost_pv']:>19,.2f}")
        print(f"{'-' * 40} {'-' * 20} {'-' * 20} {'-' * 20}")

    print(f"{'Total Cost (Present Value)':<40} ${proj_no_sell['total_pv_cost']:>19,.2f} "
          f"${proj_with_sell['total_pv_cost']:>19,.2f} "
          f"${savings['total_savings_pv']:>19,.2f}")
    print(f"{'Savings Percentage (25-year)':<40} {'-':>20} {'-':>20} "
          f"{savings['percent_savings_25year']:>19.1f}%")

    # Yearly breakdown table
    n_years = proj_no_sell['n_years']

    if ng_costs['has_ng_savings']:
        print("\n" + "=" * 200)
        print("YEAR-BY-YEAR BREAKDOWN (25-YEAR PROJECTION) - WITH ELECTRICITY AND NATURAL GAS BREAKDOWN")
        print("Note: 'Without HES' costs separated into Electricity and Natural Gas components")
        print("      'With HES' uses heat pump (electricity only, no gas)")
        print("=" * 200)
        print(f"{'':>4} {'WITHOUT HES':>60} {'WITH HES':>28} {'SAVINGS':>84}")
        print(f"{'Year':>4} {'Elec':>11} {'Elec':>11} {'NG':>11} {'NG':>11} "
              f"{'Total':>11} {'Total':>11} "
              f"{'Elec':>11} {'Elec':>11} {'Elec':>11} {'Elec':>11} "
              f"{'NG':>11} {'NG':>11} "
              f"{'Total':>11} {'Total':>11}")
        print(f"{'':>4} {'(Nom)':>11} {'(PV)':>11} {'(Nom)':>11} {'(PV)':>11} "
              f"{'(Nom)':>11} {'(PV)':>11} "
              f"{'(Nom)':>11} {'(PV)':>11} {'Sav(Nom)':>11} {'Sav(PV)':>11} "
              f"{'Sav(Nom)':>11} {'Sav(PV)':>11} "
              f"{'Sav(Nom)':>11} {'Sav(PV)':>11}")
        print("-" * 200)

        cumulative_savings_nom = 0
        cumulative_savings_pv = 0

        for year in range(n_years):
            cumulative_savings_nom += savings['yearly_savings_nominal'][year]
            cumulative_savings_pv += savings['yearly_savings_pv'][year]

            year_elec_no_sell_nom = (proj_no_sell['yearly_net_costs_nominal'][year] -
                                     ng_costs['yearly_ng_costs_nominal'][year])
            year_elec_no_sell_pv = (proj_no_sell['yearly_net_costs_pv'][year] -
                                    ng_costs['yearly_ng_costs_pv'][year])
            year_ng_nom = ng_costs['yearly_ng_costs_nominal'][year]
            year_ng_pv = ng_costs['yearly_ng_costs_pv'][year]
            year_elec_savings_nom = year_elec_no_sell_nom - proj_with_sell['yearly_net_costs_nominal'][year]
            year_elec_savings_pv = year_elec_no_sell_pv - proj_with_sell['yearly_net_costs_pv'][year]

            print(f"{year + 1:>4} ${year_elec_no_sell_nom:>10,.2f} ${year_elec_no_sell_pv:>10,.2f} "
                  f"${year_ng_nom:>10,.2f} ${year_ng_pv:>10,.2f} "
                  f"${proj_no_sell['yearly_net_costs_nominal'][year]:>10,.2f} "
                  f"${proj_no_sell['yearly_net_costs_pv'][year]:>10,.2f} "
                  f"${proj_with_sell['yearly_net_costs_nominal'][year]:>10,.2f} "
                  f"${proj_with_sell['yearly_net_costs_pv'][year]:>10,.2f} "
                  f"${year_elec_savings_nom:>10,.2f} ${year_elec_savings_pv:>10,.2f} "
                  f"${year_ng_nom:>10,.2f} ${year_ng_pv:>10,.2f} "
                  f"${savings['yearly_savings_nominal'][year]:>10,.2f} "
                  f"${savings['yearly_savings_pv'][year]:>10,.2f}")

        print("-" * 200)
        total_elec_no_sell_nom = proj_no_sell['total_nominal_cost'] - ng_costs['total_ng_cost_nominal']
        total_elec_no_sell_pv = proj_no_sell['total_pv_cost'] - ng_costs['total_ng_cost_pv']
        total_elec_savings_nom = total_elec_no_sell_nom - proj_with_sell['total_nominal_cost']
        total_elec_savings_pv = total_elec_no_sell_pv - proj_with_sell['total_pv_cost']

        print(f"{'TOT':>4} ${total_elec_no_sell_nom:>10,.2f} ${total_elec_no_sell_pv:>10,.2f} "
              f"${ng_costs['total_ng_cost_nominal']:>10,.2f} ${ng_costs['total_ng_cost_pv']:>10,.2f} "
              f"${proj_no_sell['total_nominal_cost']:>10,.2f} "
              f"${proj_no_sell['total_pv_cost']:>10,.2f} "
              f"${proj_with_sell['total_nominal_cost']:>10,.2f} "
              f"${proj_with_sell['total_pv_cost']:>10,.2f} "
              f"${total_elec_savings_nom:>10,.2f} ${total_elec_savings_pv:>10,.2f} "
              f"${ng_costs['total_ng_cost_nominal']:>10,.2f} ${ng_costs['total_ng_cost_pv']:>10,.2f} "
              f"${cumulative_savings_nom:>10,.2f} "
              f"${cumulative_savings_pv:>10,.2f}")
        print("=" * 200 + "\n")
    else:
        print("\n" + "=" * 130)
        print("YEAR-BY-YEAR BREAKDOWN (25-YEAR PROJECTION)")
        print("=" * 130)
        print(f"{'Year':>4} {'No HES (Nom)':>15} {'No HES (PV)':>15} {'With HES (Nom)':>17} "
              f"{'With HES (PV)':>15} {'Savings (Nom)':>15} {'Savings (PV)':>15}")
        print("-" * 130)

        cumulative_savings_nom = 0
        cumulative_savings_pv = 0

        for year in range(n_years):
            cumulative_savings_nom += savings['yearly_savings_nominal'][year]
            cumulative_savings_pv += savings['yearly_savings_pv'][year]

            print(f"{year + 1:>4} ${proj_no_sell['yearly_net_costs_nominal'][year]:>14,.2f} "
                  f"${proj_no_sell['yearly_net_costs_pv'][year]:>14,.2f} "
                  f"${proj_with_sell['yearly_net_costs_nominal'][year]:>16,.2f} "
                  f"${proj_with_sell['yearly_net_costs_pv'][year]:>14,.2f} "
                  f"${savings['yearly_savings_nominal'][year]:>14,.2f} "
                  f"${savings['yearly_savings_pv'][year]:>14,.2f}")

        print("-" * 130)
        print(f"{'TOTAL':>4} ${proj_no_sell['total_nominal_cost']:>14,.2f} "
              f"${proj_no_sell['total_pv_cost']:>14,.2f} "
              f"${proj_with_sell['total_nominal_cost']:>16,.2f} "
              f"${proj_with_sell['total_pv_cost']:>14,.2f} "
              f"${cumulative_savings_nom:>14,.2f} "
              f"${cumulative_savings_pv:>14,.2f}")
        print("=" * 130 + "\n")

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    fig = None
    fig_monthly = None

    if show_plot:
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        years = np.arange(1, n_years + 1)

        if ng_costs['has_ng_savings']:
            elec_costs_no_sell = proj_no_sell['yearly_net_costs_pv'] - ng_costs['yearly_ng_costs_pv']
            ng_costs_array = ng_costs['yearly_ng_costs_pv']
            elec_costs_with_sell = proj_with_sell['yearly_net_costs_pv']
        else:
            elec_costs_no_sell = proj_no_sell['yearly_net_costs_pv']
            ng_costs_array = np.zeros(n_years)
            elec_costs_with_sell = proj_with_sell['yearly_net_costs_pv']

        # Plot 1: Yearly costs (Present Value)
        ax1 = axes[0, 0]
        width = 0.35
        ax1.set_axisbelow(True)

        if ng_costs['has_ng_savings']:
            ax1.bar(years - width / 2, elec_costs_no_sell, width,
                    label='Without HES - Electricity', color='#e74c3c', alpha=0.8,
                    edgecolor='#c0392b', linewidth=1.5)
            ax1.bar(years - width / 2, ng_costs_array, width,
                    bottom=elec_costs_no_sell,
                    label='Without HES - Natural Gas', color='#f39c12', alpha=0.8,
                    edgecolor='#d68910', linewidth=1.5, hatch='///')
            ax1.bar(years + width / 2, elec_costs_with_sell, width,
                    label='With HES - Electricity', color='#27ae60', alpha=0.8,
                    edgecolor='#229954', linewidth=1.5)
        else:
            ax1.bar(years - width / 2, proj_no_sell['yearly_net_costs_pv'], width,
                    label='Without HES', color='#e74c3c', alpha=0.8, edgecolor='#c0392b', linewidth=1.5)
            ax1.bar(years + width / 2, proj_with_sell['yearly_net_costs_pv'], width,
                    label='With HES', color='#27ae60', alpha=0.8, edgecolor='#229954', linewidth=1.5)

        ax1.set_xlabel('Year', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Annual Cost ($)', fontsize=16, fontweight='bold')
        ax1.set_title('a) Yearly Utility Costs (Present Value)', fontsize=18, fontweight='bold', loc='left')
        ax1.legend(fontsize=12, loc='upper left')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xlim(0, n_years + 1)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Plot 2: Cumulative costs (Present Value)
        ax2 = axes[0, 1]
        ax2.set_axisbelow(True)
        cumulative_no_sell_pv = np.cumsum(proj_no_sell['yearly_net_costs_pv'])
        cumulative_with_sell_pv = np.cumsum(proj_with_sell['yearly_net_costs_pv'])

        if ng_costs['has_ng_savings']:
            cumulative_elec_no_sell = np.cumsum(elec_costs_no_sell)
            cumulative_ng = np.cumsum(ng_costs_array)
            ax2.fill_between(years, 0, cumulative_elec_no_sell,
                             alpha=0.7, color='#e74c3c', label='Without HES - Electricity',
                             edgecolor='#c0392b', linewidth=1.5)
            ax2.fill_between(years, cumulative_elec_no_sell, cumulative_no_sell_pv,
                             alpha=0.7, color='#f39c12', label='Without HES - Natural Gas',
                             edgecolor='#d68910', linewidth=1.5, hatch='///')
            ax2.plot(years, cumulative_no_sell_pv, 'o-', linewidth=2.5, markersize=6,
                     color='#922b21', label='Without HES - Total')
            ax2.plot(years, cumulative_with_sell_pv, 's-', linewidth=2.5, markersize=6,
                     color='#27ae60', label='With HES - Electricity')
        else:
            ax2.plot(years, cumulative_no_sell_pv, 'o-', label='Without HES',
                     linewidth=2.5, markersize=7, color='#e74c3c')
            ax2.plot(years, cumulative_with_sell_pv, 's-', label='With HES',
                     linewidth=2.5, markersize=7, color='#27ae60')
            ax2.fill_between(years, cumulative_no_sell_pv, cumulative_with_sell_pv,
                             alpha=0.3, color='#3498db', label='Cumulative Savings')

        ax2.set_xlabel('Year', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Cumulative Cost ($)', fontsize=16, fontweight='bold')
        ax2.set_title('b) Cumulative Utility Costs (Present Value)', fontsize=18, fontweight='bold', loc='left')
        ax2.legend(fontsize=12, loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, n_years + 1)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Plot 3: Annual savings (Present Value)
        ax3 = axes[1, 0]
        ax3.set_axisbelow(True)

        if ng_costs['has_ng_savings']:
            elec_savings = elec_costs_no_sell - elec_costs_with_sell
            ng_savings = ng_costs_array
            ax3.bar(years, elec_savings, color='#3498db', alpha=0.8,
                    edgecolor='#2874a6', linewidth=1.5, label='Electricity Savings')
            ax3.bar(years, ng_savings, bottom=elec_savings,
                    color='#f39c12', alpha=0.8, edgecolor='#d68910',
                    linewidth=1.5, hatch='///', label='Natural Gas Savings')
        else:
            ax3.bar(years, savings['yearly_savings_pv'], color='#27ae60', alpha=0.8,
                    edgecolor='#229954', linewidth=1.5)

        ax3.set_xlabel('Year', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Annual Savings ($)', fontsize=16, fontweight='bold')
        ax3.set_title('c) Year-by-Year Savings (Present Value)', fontsize=18, fontweight='bold', loc='left')
        if ng_costs['has_ng_savings']:
            ax3.legend(fontsize=12, loc='upper left')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xlim(0, n_years + 1)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Plot 4: Cumulative savings (Present Value)
        ax4 = axes[1, 1]
        ax4.set_axisbelow(True)
        cumulative_savings_pv = np.cumsum(savings['yearly_savings_pv'])

        if ng_costs['has_ng_savings']:
            cumulative_elec_savings = np.cumsum(elec_savings)
            cumulative_ng_savings = np.cumsum(ng_savings)
            ax4.fill_between(years, 0, cumulative_elec_savings, alpha=0.7,
                             color='#3498db', label='Electricity Savings',
                             edgecolor='#2874a6', linewidth=1.5)
            ax4.fill_between(years, cumulative_elec_savings, cumulative_savings_pv,
                             alpha=0.7, color='#f39c12', label='Natural Gas Savings',
                             edgecolor='#d68910', linewidth=1.5, hatch='///')
            ax4.plot(years, cumulative_savings_pv, 'o-', linewidth=2.5, markersize=6,
                     color='#196f3d', label='Total Savings')
        else:
            ax4.fill_between(years, 0, cumulative_savings_pv, alpha=0.6, color='#27ae60',
                             label='Cumulative Savings')
            ax4.plot(years, cumulative_savings_pv, 'o-', linewidth=2.5, markersize=7,
                     color='#229954')

        ax4.set_xlabel('Year', fontsize=16, fontweight='bold')
        ax4.set_ylabel('Cumulative Savings ($)', fontsize=16, fontweight='bold')
        ax4.set_title('d) Cumulative Savings Over Time (Present Value)', fontsize=18, fontweight='bold', loc='left')
        ax4.legend(fontsize=12, loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, n_years + 1)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.tick_params(axis='both', which='major', labelsize=14)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        total_savings_plot = cumulative_savings_pv[-1]
        ax4.annotate(f'Total: ${total_savings_plot:,.0f}',
                     xy=(n_years, total_savings_plot),
                     xytext=(n_years - 5, total_savings_plot * 1.1),
                     fontsize=15, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

        plt.tight_layout()
        if save_path:
            base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
            annual_path = f"{base_path}_annual.png"
            fig.savefig(annual_path, dpi=300, bbox_inches='tight')

        # FIGURE 2: Monthly comparison
        fig_monthly, axes_monthly = plt.subplots(2, 1, figsize=(12, 10))
        x_pos = np.arange(12)
        width = 0.35

        if ng_costs['has_ng_savings']:
            monthly_ng_plot = ng_costs['monthly_ng_costs_year1']
            monthly_elec_no_sell = monthly_bills_no_sell_year1
            monthly_elec_with_sell = monthly_bills_with_sell_year1
        else:
            monthly_elec_no_sell = monthly_bills_no_sell_year1
            monthly_ng_plot = np.zeros(12)
            monthly_elec_with_sell = monthly_bills_with_sell_year1

        ax5 = axes_monthly[0]
        ax5.set_axisbelow(True)

        if ng_costs['has_ng_savings']:
            ax5.bar(x_pos - width / 2, monthly_elec_no_sell, width,
                    label='Without HES - Electricity', color='#e74c3c', alpha=0.8,
                    edgecolor='#c0392b', linewidth=1.5)
            ax5.bar(x_pos - width / 2, monthly_ng_plot, width,
                    bottom=monthly_elec_no_sell,
                    label='Without HES - Natural Gas', color='#f39c12', alpha=0.8,
                    edgecolor='#d68910', linewidth=1.5, hatch='///')
            ax5.bar(x_pos + width / 2, monthly_elec_with_sell, width,
                    label='With HES - Electricity', color='#27ae60', alpha=0.8,
                    edgecolor='#229954', linewidth=1.5)
        else:
            ax5.bar(x_pos - width / 2, monthly_bills_no_sell_year1, width,
                    label='Without HES', color='#e74c3c', alpha=0.8,
                    edgecolor='#c0392b', linewidth=1.5)
            ax5.bar(x_pos + width / 2, monthly_bills_with_sell_year1, width,
                    label='With HES', color='#27ae60', alpha=0.8,
                    edgecolor='#229954', linewidth=1.5)

        ax5.set_xlabel('Month', fontsize=16, fontweight='bold')
        ax5.set_ylabel('Monthly Cost ($)', fontsize=16, fontweight='bold')
        ax5.set_title('a) Monthly Utility Costs Comparison', fontsize=18, fontweight='bold', loc='left')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(months, fontsize=14)
        ax5.legend(fontsize=13, loc='best')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.tick_params(axis='both', which='major', labelsize=14)
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        ax6 = axes_monthly[1]
        ax6.set_axisbelow(True)

        if ng_costs['has_ng_savings']:
            monthly_elec_savings = monthly_elec_no_sell - monthly_elec_with_sell
            monthly_ng_savings = monthly_ng_plot

            ax6.bar(x_pos, monthly_elec_savings, color='#3498db', alpha=0.8,
                    edgecolor='#2874a6', linewidth=1.5, label='Electricity Savings')
            ax6.bar(x_pos, monthly_ng_savings, bottom=monthly_elec_savings,
                    color='#f39c12', alpha=0.8, edgecolor='#d68910',
                    linewidth=1.5, hatch='///', label='Natural Gas Savings')

            for i in range(12):
                total_sav = (monthly_elec_no_sell[i] + monthly_ng_plot[i]) - monthly_elec_with_sell[i]
                height = monthly_elec_savings[i] + monthly_ng_savings[i]
                ax6.text(x_pos[i], height, f'${total_sav:,.0f}',
                         ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            monthly_sav_plot = monthly_bills_no_sell_year1 - monthly_bills_with_sell_year1
            colors_savings = ['#27ae60' if s >= 0 else '#e74c3c' for s in monthly_sav_plot]
            bars = ax6.bar(x_pos, monthly_sav_plot, color=colors_savings, alpha=0.8,
                           edgecolor='#2c3e50', linewidth=1.5)
            for i, (bar, val) in enumerate(zip(bars, monthly_sav_plot)):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width() / 2., height,
                         f'${val:,.0f}',
                         ha='center', va='bottom' if val >= 0 else 'top',
                         fontsize=12, fontweight='bold')

        ax6.set_xlabel('Month', fontsize=16, fontweight='bold')
        ax6.set_ylabel('Monthly Savings ($)', fontsize=16, fontweight='bold')
        ax6.set_title('b) Monthly Savings', fontsize=18, fontweight='bold', loc='left')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(months, fontsize=14)
        if ng_costs['has_ng_savings']:
            ax6.legend(fontsize=13, loc='upper left')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax6.tick_params(axis='both', which='major', labelsize=14)
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()
        if save_path:
            monthly_path = f"{base_path}_monthly.png"
            fig_monthly.savefig(monthly_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
        plt.close(fig_monthly)
        plt.rcParams.update(plt.rcParamsDefault)

    return (fig, fig_monthly) if show_plot else (None, None)