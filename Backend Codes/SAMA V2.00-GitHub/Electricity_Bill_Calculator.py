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

        # Calculate net energy cost
        net_energy_cost = energy_cost - sell_revenue
        monthly_net_energy_costs[month] = net_energy_cost

        # Calculate grid tax (per kWh tax on purchased energy)
        grid_tax_cost = total_kwh_bought * Grid_Tax_amount
        monthly_grid_tax[month] = grid_tax_cost

        # Calculate sales tax to match original formula
        taxable_amount = net_energy_cost + grid_tax_cost + Service_charge[month] + (Annual_expenses / 12)
        sales_tax_cost = max(0, taxable_amount) * Grid_Tax
        monthly_tax_costs[month] = sales_tax_cost

        # Calculate total monthly bill
        monthly_bill = (net_energy_cost + grid_tax_cost + sales_tax_cost + Service_charge[month])
        monthly_bills[month] = monthly_bill

        hour_idx += hours_this_month

    # Calculate annual totals
    total_energy_cost = np.sum(monthly_energy_costs)
    total_sell_revenue = np.sum(monthly_sell_revenue)
    total_net_energy_cost = np.sum(monthly_net_energy_costs)
    total_service_charges = np.sum(Service_charge)
    total_grid_tax = np.sum(monthly_grid_tax)
    total_sales_tax = np.sum(monthly_tax_costs)

    annual_cost_before_tax = (total_net_energy_cost + total_service_charges +
                              total_grid_tax + Annual_expenses + NEM_fee)
    annual_cost_before_credits = annual_cost_before_tax + total_sales_tax
    annual_cost = annual_cost_before_credits - Grid_credit

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
        'subtotal_before_credits': annual_cost_before_credits,
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
    Calculate n-year cost projection matching Grid_Cost_net formula structure.

    When raw data (Pbuy, Cbuy, etc.) is provided, uses the formula:
    Bought_electricity = ((Annual_expenses + Service_charge + Pbuy*Cbuy + Grid_Tax_amount*Pbuy) * discount_factors) * (1 + Grid_Tax)
    Sold_electricity = (Psell*Csell * discount_factors)
    Grid_Cost_net = Bought_electricity - Sold_electricity - Grid_credits
    """

    # Calculate cumulative escalation factors
    cumulative_escalation = np.cumprod(1 + Grid_escalation[:n])

    # Calculate present value discount factors
    discount_factors = cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1))

    # Check if we have raw data to do proper calculation
    has_sell = (Psell is not None and Csell is not None and Grid > 0)
    has_raw_data = (Pbuy is not None and Cbuy is not None and
                    Service_charge is not None and Grid_Tax is not None and
                    Grid_Tax_amount is not None and Annual_expenses is not None)

    if has_raw_data:
        # Calculate year 1 base costs using raw data (matching Grid_Cost_net formula)
        # Pbuy is the actual electricity bought from grid

        # Bought electricity (year 1 base, before escalation)
        bought_base = (Annual_expenses + np.sum(Service_charge) +
                       np.sum(Pbuy * Cbuy) + Grid_Tax_amount * np.sum(Pbuy))

        # Apply escalation, discount, and tax to bought electricity
        bought_electricity_pv = (bought_base * discount_factors) * (1 + Grid_Tax) * (Grid > 0)
        bought_electricity_nominal = (bought_base * cumulative_escalation) * (1 + Grid_Tax) * (Grid > 0)

        if has_sell:
            # Sold electricity (year 1 base)
            sold_base = np.sum(Psell * Csell)
            sold_electricity_pv = sold_base * discount_factors * (Grid > 0)
            sold_electricity_nominal = sold_base * cumulative_escalation * (Grid > 0)
        else:
            sold_electricity_pv = np.zeros(n)
            sold_electricity_nominal = np.zeros(n)

        # Grid credits
        total_grid_credits_pv = Grid_credit * discount_factors * (Grid > 0)
        total_grid_credits_nominal = Grid_credit * cumulative_escalation * (Grid > 0)

        # Net costs = Bought - Sold - Credits (matching Grid_Cost_net formula)
        yearly_net_costs_pv = bought_electricity_pv - sold_electricity_pv - total_grid_credits_pv
        yearly_net_costs_nominal = bought_electricity_nominal - sold_electricity_nominal - total_grid_credits_nominal

        # Store components
        yearly_costs_pv = bought_electricity_pv
        yearly_costs_nominal = bought_electricity_nominal
        yearly_sell_revenue_pv = sold_electricity_pv
        yearly_sell_revenue_nominal = sold_electricity_nominal

    else:
        # Fallback to old method when raw data not provided
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


def compare_scenarios(daysInMonth, Pbuy_no_sell, Pbuy_with_sell, Cbuy, Service_charge,
                      Grid_Tax, Grid_Tax_amount, Annual_expenses,
                      Grid_escalation, Grid_credit, NEM_fee,
                      Grid, n, ir, Psell=None, Csell=None):
    """
    Compare electricity costs with and without selling electricity to grid.

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

    # Calculate savings
    monthly_savings = results_no_sell['monthly_bills'] - results_with_sell['monthly_bills']
    annual_savings = results_no_sell['annual_cost'] - results_with_sell['annual_cost']
    total_savings_nominal = projection_no_sell['total_nominal_cost'] - projection_with_sell['total_nominal_cost']
    total_savings_pv = projection_no_sell['total_pv_cost'] - projection_with_sell['total_pv_cost']

    yearly_savings_nominal = projection_no_sell['yearly_net_costs_nominal'] - projection_with_sell[
        'yearly_net_costs_nominal']
    yearly_savings_pv = projection_no_sell['yearly_net_costs_pv'] - projection_with_sell['yearly_net_costs_pv']

    return {
        'no_sell': {
            'results': results_no_sell,
            'projection': projection_no_sell
        },
        'with_sell': {
            'results': results_with_sell,
            'projection': projection_with_sell
        },
        'savings': {
            'monthly_savings': monthly_savings,
            'annual_savings': annual_savings,
            'total_savings_nominal': total_savings_nominal,
            'total_savings_pv': total_savings_pv,
            'yearly_savings_nominal': yearly_savings_nominal,
            'yearly_savings_pv': yearly_savings_pv,
            'annual_sell_revenue': results_with_sell['breakdown']['total_sell_revenue'],
            'percent_savings_annual': (annual_savings / results_no_sell['annual_cost']) * 100 if results_no_sell[
                                                                                                     'annual_cost'] > 0 else 0,
            'percent_savings_25year': (total_savings_nominal / projection_no_sell['total_nominal_cost']) * 100 if
            projection_no_sell['total_nominal_cost'] > 0 else 0
        }
    }


def print_comparison(comparison, show_plot):
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

    # Year 1 costs
    print(f"║ {'First Year Annual Cost':<50} ${results_no_sell['annual_cost']:>19,.2f} ${results_with_sell['annual_cost']:>19,.2f} ║")
    print(f"║ {'First Year Monthly Average':<50} ${results_no_sell['annual_cost']/12:>19,.2f} ${results_with_sell['annual_cost']/12:>19,.2f} ║")
    #print(f"║ {'First Year Sell Revenue':<50} ${0:>19,.2f} ${results_with_sell['breakdown']['total_sell_revenue']:>19,.2f} ║")
    print(f"║ {'First Year Savings':<50} {'-':>20} ${savings['annual_savings']:>19,.2f} ║")
    print(f"║ {'First Year Savings %':<50} {'-':>20} {savings['percent_savings_annual']:>19.1f}% ║")

    print("╠" + "-" * 98 + "╣")

    # 25-year projections
    print(f"║ {'25-Year Total Cost (Nominal)':<50} ${proj_no_sell['total_nominal_cost']:>19,.2f} ${proj_with_sell['total_nominal_cost']:>19,.2f} ║")
    print(f"║ {'25-Year Total Cost (Present Value)':<50} ${proj_no_sell['total_pv_cost']:>19,.2f} ${proj_with_sell['total_pv_cost']:>19,.2f} ║")
    print(f"║ {'25-Year Total Savings (Nominal)':<50} {'-':>20} ${savings['total_savings_nominal']:>19,.2f} ║")
    print(f"║ {'25-Year Total Savings (PV)':<50} {'-':>20} ${savings['total_savings_pv']:>19,.2f} ║")
    print(f"║ {'25-Year Savings %':<50} {'-':>20} {savings['percent_savings_25year']:>19.1f}% ║")

    # if proj_with_sell['has_sell']:
    #     print("╠" + "-" * 98 + "╣")
    #     print(f"║ {'25-Year Total Sell Revenue (Nominal)':<50} ${0:>19,.2f} ${proj_with_sell['total_sell_revenue_nominal']:>19,.2f} ║")
    #     print(f"║ {'25-Year Total Sell Revenue (PV)':<50} ${0:>19,.2f} ${proj_with_sell['total_sell_revenue_pv']:>19,.2f} ║")
    #
    # print("╠" + "-" * 98 + "╣")

    # Energy metrics
    print(f"║ {'Total kWh Bought (Year 1)':<50} {results_no_sell['breakdown']['total_kwh_bought']:>19,.1f} {results_with_sell['breakdown']['total_kwh_bought']:>19,.1f} ║")
    print(f"║ {'Total kWh Sold (Year 1)':<50} {0:>19,.1f} {results_with_sell['breakdown']['total_kwh_sold']:>19,.1f} ║")
    print(f"║ {'Net kWh (Buy - Sell, Year 1)':<50} {results_no_sell['breakdown']['net_kwh']:>19,.1f} {results_with_sell['breakdown']['net_kwh']:>19,.1f} ║")

    print("╠" + "-" * 98 + "╣")

    # Averages
    print(f"║ {'Average Annual Cost - 25 Years (Nominal)':<50} ${proj_no_sell['average_annual_nominal']:>19,.2f} ${proj_with_sell['average_annual_nominal']:>19,.2f} ║")
    print(f"║ {'Average Annual Cost - 25 Years (PV)':<50} ${proj_no_sell['average_annual_pv']:>19,.2f} ${proj_with_sell['average_annual_pv']:>19,.2f} ║")

    print("╚" + "=" * 98 + "╝\n")

    # Monthly comparison
    print("\nMONTHLY BILL COMPARISON")
    print("-" * 100)
    print(f"{'Month':<6} {'No HES':>12} {'With HES':>12} {'Savings':>12} {'% Saved':>10}") # Sell Revenue Excluded here
    print("-" * 100)

    for i in range(12):
        percent = (savings['monthly_savings'][i] / results_no_sell['monthly_bills'][i]) * 100 if \
        results_no_sell['monthly_bills'][i] > 0 else 0
        print(f"{months[i]:<6} ${results_no_sell['monthly_bills'][i]:>11.2f} "
              f"${results_with_sell['monthly_bills'][i]:>11.2f} "
              #f"${results_with_sell['monthly_sell_revenue'][i]:>12.2f} "
              f"${savings['monthly_savings'][i]:>11.2f} "
              f"{percent:>9.1f}%")

    print("-" * 100)
    print(f"{'TOTAL':<6} ${np.sum(results_no_sell['monthly_bills']):>11.2f} "
          f"${np.sum(results_with_sell['monthly_bills']):>11.2f} "
          f"${results_with_sell['breakdown']['total_sell_revenue']:>12.2f} "
          f"${savings['annual_savings']:>11.2f} "
          f"{savings['percent_savings_annual']:>9.1f}%")

    # 25-year projection comparison summary
    print("\n" + "=" * 100)
    print("25-YEAR PROJECTION SUMMARY")
    print("=" * 100)
    print(f"{'Metric':<40} {'Without HES':>20} {'With HES':>20} {'Savings':>20}")
    print("-" * 100)
    print(f"{'Total Cost (Nominal)':<40} ${proj_no_sell['total_nominal_cost']:>19,.2f} "
          f"${proj_with_sell['total_nominal_cost']:>19,.2f} "
          f"${savings['total_savings_nominal']:>19,.2f}")
    print(f"{'Total Cost (Present Value)':<40} ${proj_no_sell['total_pv_cost']:>19,.2f} "
          f"${proj_with_sell['total_pv_cost']:>19,.2f} "
          f"${savings['total_savings_pv']:>19,.2f}")

    # if proj_with_sell['has_sell']:
    #     print(f"{'Total Sell Revenue (Nominal)':<40} ${0:>19,.2f} "
    #           f"${proj_with_sell['total_sell_revenue_nominal']:>19,.2f} "
    #           f"${proj_with_sell['total_sell_revenue_nominal']:>19,.2f}")
    #     print(f"{'Total Sell Revenue (PV)':<40} ${0:>19,.2f} "
    #           f"${proj_with_sell['total_sell_revenue_pv']:>19,.2f} "
    #           f"${proj_with_sell['total_sell_revenue_pv']:>19,.2f}")

    print(f"{'Savings Percentage (25-year)':<40} {'-':>20} {'-':>20} {savings['percent_savings_25year']:>19.1f}%")

    # Yearly breakdown table
    n_years = proj_no_sell['n_years']
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

        print(f"{year+1:>4} ${proj_no_sell['yearly_net_costs_nominal'][year]:>14,.2f} "
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

    # Create comparison plots (Present Value only)
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Electricity Cost Comparison: With vs Without HES (Present Value)',
                    fontsize=16, fontweight='bold')

        years = np.arange(1, n_years + 1)

        # Plot 1: Yearly costs (Present Value)
        ax1 = axes[0, 0]
        ax1.plot(years, proj_no_sell['yearly_net_costs_pv'], 'o-', label='Without HES',
                linewidth=2, markersize=6, color='#e74c3c')
        ax1.plot(years, proj_with_sell['yearly_net_costs_pv'], 's-', label='With HES',
                linewidth=2, markersize=6, color='#27ae60')
        ax1.fill_between(years, proj_no_sell['yearly_net_costs_pv'],
                         proj_with_sell['yearly_net_costs_pv'],
                         alpha=0.3, color='#3498db', label='Savings')
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Annual Cost ($)', fontsize=12)
        ax1.set_title('Yearly Costs (Present Value)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, n_years + 1)

        # Plot 2: Cumulative costs (Present Value)
        ax2 = axes[0, 1]
        cumulative_no_sell_pv = np.cumsum(proj_no_sell['yearly_net_costs_pv'])
        cumulative_with_sell_pv = np.cumsum(proj_with_sell['yearly_net_costs_pv'])
        ax2.plot(years, cumulative_no_sell_pv, 'o-', label='Without HES',
                linewidth=2, markersize=6, color='#e74c3c')
        ax2.plot(years, cumulative_with_sell_pv, 's-', label='With HES',
                linewidth=2, markersize=6, color='#27ae60')
        ax2.fill_between(years, cumulative_no_sell_pv, cumulative_with_sell_pv,
                         alpha=0.3, color='#3498db', label='Cumulative Savings')
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Cumulative Cost ($)', fontsize=12)
        ax2.set_title('Cumulative Costs (Present Value)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, n_years + 1)

        # Plot 3: Annual savings (Present Value)
        ax3 = axes[1, 0]
        ax3.bar(years, savings['yearly_savings_pv'], color='#27ae60', alpha=0.8,
               edgecolor='#229954', linewidth=1.5)
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('Annual Savings ($)', fontsize=12)
        ax3.set_title('Year-by-Year Savings (Present Value)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xlim(0, n_years + 1)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Plot 4: Cumulative savings (Present Value)
        ax4 = axes[1, 1]
        cumulative_savings_pv = np.cumsum(savings['yearly_savings_pv'])
        ax4.fill_between(years, 0, cumulative_savings_pv, alpha=0.6, color='#27ae60',
                        label='Cumulative Savings')
        ax4.plot(years, cumulative_savings_pv, 'o-', linewidth=2.5, markersize=6,
                color='#229954')
        ax4.set_xlabel('Year', fontsize=12)
        ax4.set_ylabel('Cumulative Savings ($)', fontsize=12)
        ax4.set_title('Cumulative Savings Over Time (Present Value)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, n_years + 1)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add total savings annotation to cumulative savings plot
        total_savings = cumulative_savings_pv[-1]
        ax4.annotate(f'Total: ${total_savings:,.0f}',
                    xy=(n_years, total_savings),
                    xytext=(n_years-5, total_savings*1.1),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

        plt.tight_layout()
        plt.show()

    return fig if show_plot else None