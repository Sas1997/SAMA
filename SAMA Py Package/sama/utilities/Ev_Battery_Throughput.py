import pandas as pd

def calculate_ev_battery_throughput(initial_capacity_kwh, degradation_percent, total_distance_km, range_per_full_charge_km, step_km):
    """
    Calculate energy throughput and capacity degradation of an EV battery over distance in steps.

    Parameters:
    - initial_capacity_kwh: Initial usable battery capacity in kWh
    - degradation_per_1000km_percent: Linear degradation per 1000 km (as percent)
    - total_distance_km: Total distance to simulate (in km)
    - range_per_full_charge_km: Range per full battery (in km)
    - step_km: Step size in km (default is 1000)

    Returns:
    - DataFrame with throughput and capacity by step
    - Total energy throughput (kWh)
    """

    num_steps = total_distance_km // step_km
    distance_bins = []
    capacity_kwh = []
    energy_throughput_kwh = []

    for i in range(num_steps):
        degradation_fraction = (i * degradation_percent) / 100
        capacity_now = initial_capacity_kwh * (1 - degradation_fraction)
        throughput = capacity_now / range_per_full_charge_km * step_km

        distance_bins.append(f"{i * step_km}-{(i + 1) * step_km} km")
        capacity_kwh.append(round(capacity_now, 2))
        energy_throughput_kwh.append(round(throughput, 2))

    df = pd.DataFrame({
        "Distance Interval": distance_bins,
        "Capacity at Start (kWh)": capacity_kwh,
        "Energy Throughput (kWh)": energy_throughput_kwh
    })

    total_throughput = sum(energy_throughput_kwh)

    return df, total_throughput