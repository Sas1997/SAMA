"""Demand charges and the minimum charge rule (samapy.pricing.demand_charge).

The monthly charges are checked against hand calculations on small synthetic
profiles, then the integration into the bill calculator and the fitness
function is checked against the exact formulas they implement.
"""
import types

import numpy as np
import pytest

from samapy.core.Input_Data import InData
from samapy.pricing.demand_charge import (build_demand_rates, compute_minimum_charge_adjustment,
                                          compute_monthly_peak_demand)

DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
STARTS = np.concatenate([[0], np.cumsum(DAYS * 24)[:-1]])  # first hour of each month


def profile_with_peaks(base=1.0, peaks=None, hour_of_day=12):
    """Flat hourly profile with one peak per month on day 2 at hour_of_day."""
    p = np.full(8760, base)
    for m, kw in enumerate(peaks if peaks is not None else np.arange(1, 13) * 10.0):
        p[STARTS[m] + 24 + hour_of_day] = kw
    return p


def charge(Pbuy, rates, **kw):
    return compute_monthly_peak_demand(Pbuy, DAYS, 1, rates, **kw)


def test_disabled_returns_zero():
    total, peaks, costs = compute_monthly_peak_demand(profile_with_peaks(), DAYS, 0, np.full(12, 15.0))
    assert total == 0.0 and not peaks.any() and not costs.any()


def test_flat_rate_bills_each_monthly_peak():
    peaks = np.arange(1, 13) * 10.0
    total, monthly_peak, costs = charge(profile_with_peaks(peaks=peaks), np.full(12, 15.0))
    np.testing.assert_allclose(monthly_peak, peaks)
    np.testing.assert_allclose(costs, 15.0 * peaks)
    assert total == pytest.approx(15.0 * peaks.sum())


def test_monthly_rates():
    peaks = np.full(12, 40.0)
    rates = np.arange(1, 13, dtype=float)
    total, _, costs = charge(profile_with_peaks(peaks=peaks), rates)
    np.testing.assert_allclose(costs, 40.0 * rates)


def test_tou_on_and_off_peak():
    # on-peak 16-20 h; the monthly peak (50 kW) is at noon, i.e. off-peak,
    # and the highest on-peak value is a 20 kW hour at 17 h
    p = profile_with_peaks(peaks=np.full(12, 50.0), hour_of_day=12)
    for m in range(12):
        p[STARTS[m] + 48 + 17] = 20.0
    on_hours = np.empty(2, dtype=object)
    on_hours[0] = on_hours[1] = [16, 17, 18, 19, 20]
    season = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    on_rate = np.where(season == 1, 20.0, 15.0)
    off_rate = np.where(season == 1, 2.0, 1.0)
    total, monthly_on_peak, costs = charge(
        p, on_rate, demand_charge_type=4, Demand_charge_offpeak_rate_monthly=off_rate,
        demand_tou_onpeak_hours=on_hours, demand_tou_season=season)
    np.testing.assert_allclose(monthly_on_peak, 20.0)
    np.testing.assert_allclose(costs, 20.0 * on_rate + 50.0 * off_rate)


def test_ratchet_floor():
    peaks = np.array([100.0] + [10.0] * 11)
    total, _, costs = charge(profile_with_peaks(peaks=peaks), np.full(12, 10.0),
                             demand_ratchet_enabled=1, demand_ratchet_rate=0.85)
    # January bills its own 100 kW; every other month bills 85 % of it
    np.testing.assert_allclose(costs, [1000.0] + [850.0] * 11)


def test_nem_demand_netting_uses_net_import():
    Pbuy = profile_with_peaks(peaks=np.full(12, 30.0))
    Psell = np.zeros(8760)
    Psell[STARTS + 24 + 12] = 25.0  # export in the same hour as each peak
    gross = charge(Pbuy, np.full(12, 10.0))[1]
    net = charge(Pbuy, np.full(12, 10.0), nem_demand_netting=1, Psell=Psell)[1]
    np.testing.assert_allclose(gross, 30.0)
    np.testing.assert_allclose(net, 5.0)  # max(30 - 25, base 1 - 0)


def test_minimum_charge_lifts_bill_to_floor():
    # 1 kW all year at 0.10 $/kWh, large exports make the bill negative
    Pbuy = np.ones(8760)
    Psell = np.full(8760, 2.0)
    Cbuy = np.full(8760, 0.10)
    Csell = np.full(8760, 0.10)
    service = np.full(12, 20.0)
    dc = np.full(12, 30.0)
    tax = 0.13
    total, monthly = compute_minimum_charge_adjustment(
        Pbuy, Psell, Cbuy, Csell, DAYS, service, tax, 0.0, 0.0, 0.0,
        1, np.full(12, 40.0), dc, 1)
    hours = DAYS * 24
    normal = (0.10 * hours + 20.0 + 30.0) * (1 + tax) - 0.2 * hours
    floor = max(40.0, 20.0 + 30.0) * (1 + tax)
    np.testing.assert_allclose(monthly, np.maximum(0.0, floor - normal))
    assert total == pytest.approx(monthly.sum())
    # off-grid systems have no bill
    assert compute_minimum_charge_adjustment(Pbuy, Psell, Cbuy, Csell, DAYS, service, tax,
                                             0, 0, 0, 1, np.full(12, 40.0), dc, 0)[0] == 0.0


def rate_inputs(**kw):
    d = types.SimpleNamespace(
        demand_charge_enabled=1, demand_charge_type=1, demand_flat_rate=15.0,
        demand_seasonal_rates=np.array([18.0, 12.0]),
        demand_season=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]),
        demand_monthly_rates=np.arange(1.0, 13.0),
        demand_tou_onpeak_rate=np.array([20.0, 15.0]),
        demand_tou_offpeak_rate=np.array([2.0, 1.0]),
        demand_tou_season=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]),
        nem_demand_rate_override=0, nem_demand_rate_monthly=np.full(12, 10.0),
        nem_demand_offpeak_rate_monthly=np.zeros(12))
    for k, v in kw.items():
        setattr(d, k, v)
    build_demand_rates(d)
    return d.Demand_charge_rate_monthly, d.Demand_charge_offpeak_rate_monthly


def test_rate_arrays_for_each_type():
    summer = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]) == 1
    np.testing.assert_allclose(rate_inputs(demand_charge_type=1)[0], 15.0)
    np.testing.assert_allclose(rate_inputs(demand_charge_type=2)[0], np.where(summer, 18.0, 12.0))
    np.testing.assert_allclose(rate_inputs(demand_charge_type=3)[0], np.arange(1.0, 13.0))
    on, off = rate_inputs(demand_charge_type=4)
    np.testing.assert_allclose(on, np.where(summer, 20.0, 15.0))
    np.testing.assert_allclose(off, np.where(summer, 2.0, 1.0))
    np.testing.assert_allclose(rate_inputs(nem_demand_rate_override=1)[0], 10.0)
    assert not rate_inputs(demand_charge_enabled=0)[0].any()
    with pytest.raises(ValueError):
        rate_inputs(demand_charge_type=5)


def test_default_inputs_have_demand_charges_off():
    assert InData.demand_charge_enabled == 0 and InData.minimum_charge_enabled == 0
    assert not np.asarray(InData.Demand_charge_rate_monthly).any()


def test_bill_calculator_adds_taxed_demand_charge():
    from samapy.pricing.Electricity_Bill_Calculator import calculate_electricity_bills

    Pbuy = profile_with_peaks()
    args = (DAYS, Pbuy, np.full(8760, 0.1), np.full(12, 20.0), 0.13, 0.01, 0.0,
            np.full(25, 0.02), 0.0, 0.0, 1)
    base = calculate_electricity_bills(*args)
    rates = np.full(12, 15.0)
    with_dc = calculate_electricity_bills(*args, demand_charge_enabled=1, Demand_charge_rate_monthly=rates)
    expected_dc = charge(Pbuy, rates)[2]
    np.testing.assert_allclose(with_dc['monthly_demand_charges'], expected_dc)
    np.testing.assert_allclose(with_dc['monthly_bills'] - base['monthly_bills'], expected_dc * 1.13)


@pytest.fixture
def demand_inputs(monkeypatch):
    """Enable demand charges on InData for one test; monkeypatch restores it."""
    def enable(**kw):
        values = dict(demand_charge_enabled=1, demand_charge_type=1, demand_flat_rate=15.0,
                      demand_ratchet_enabled=0)
        values.update(kw)
        for k, v in values.items():
            monkeypatch.setattr(InData, k, v)
        d = types.SimpleNamespace(**{k: getattr(InData, k) for k in (
            'demand_charge_enabled', 'demand_charge_type', 'demand_flat_rate', 'demand_seasonal_rates',
            'demand_season', 'demand_monthly_rates', 'demand_tou_onpeak_rate', 'demand_tou_offpeak_rate',
            'demand_tou_season', 'nem_demand_rate_override', 'nem_demand_rate_monthly',
            'nem_demand_offpeak_rate_monthly')})
        build_demand_rates(d)
        monkeypatch.setattr(InData, 'Demand_charge_rate_monthly', d.Demand_charge_rate_monthly)
        monkeypatch.setattr(InData, 'Demand_charge_offpeak_rate_monthly', d.Demand_charge_offpeak_rate_monthly)
    return enable


def lifetime_factor():
    import samapy.core.Fitness as F
    return np.sum(np.cumprod(1 + F.Grid_escalation) / ((1 + F.ir) ** np.arange(1, F.n + 1)))


def test_fitness_npc_includes_demand_charge(demand_inputs):
    import samapy.core.Fitness as F

    X = np.array([12.0, 0.0, 2.0, 0.0, 8.0])
    base = F.evaluate(X)
    assert base["demand_charge_year1"] == 0.0
    demand_inputs()
    m = F.evaluate(X)
    assert m["demand_charge_year1"] > 0
    # dispatch does not change, so NPC rises by the taxed, escalated, discounted charge
    expected = m["demand_charge_year1"] * (1 + F.Grid_Tax) * lifetime_factor()
    assert m["NPC"] - base["NPC"] == pytest.approx(expected, rel=1e-9)


def test_fitness_npc_includes_minimum_charge(monkeypatch):
    import samapy.core.Fitness as F

    X = np.array([12.0, 0.0, 2.0, 0.0, 8.0])
    base = F.evaluate(X)
    monkeypatch.setattr(InData, 'minimum_charge_enabled', 1)
    monkeypatch.setattr(InData, 'minimum_charge_monthly', np.full(12, 500.0))
    m = F.evaluate(X)
    assert m["min_charge_adj_year1"] > 0
    expected = m["min_charge_adj_year1"] * lifetime_factor()
    assert m["NPC"] - base["NPC"] == pytest.approx(expected, rel=1e-9)


def test_tou_weekdays_only_moves_weekend_peak_off_peak():
    from datetime import date
    from samapy.pricing.demand_charge import special_days

    year = 2023  # 1 January 2023 is a Sunday
    special = special_days(year, np.array([1, 51]))
    assert special[0] and special[6] and not special[1]  # Sun 1 Jan, Sat 7 Jan, Mon 2 Jan
    assert special[50] and date(2023, 2, 20).weekday() == 0  # holiday 51 is a Monday
    assert special.sum() == 105 + 1  # 105 weekend days in 2023 plus the Monday holiday

    # 40 kW at 17 h on Saturday 7 January, 25 kW at 17 h on Monday 9 January
    p = np.ones(8760)
    p[6 * 24 + 17] = 40.0
    p[8 * 24 + 17] = 25.0
    on_hours = np.empty(2, dtype=object)
    on_hours[0] = on_hours[1] = [16, 17, 18, 19, 20]
    season = np.zeros(12, dtype=int)
    kw = dict(demand_charge_type=4, Demand_charge_offpeak_rate_monthly=np.full(12, 1.0),
              demand_tou_onpeak_hours=on_hours, demand_tou_season=season)
    every_day = charge(p, np.full(12, 10.0), **kw)
    weekdays = charge(p, np.full(12, 10.0), tou_weekdays_only=1, year=year, holidays=None, **kw)
    # every day: both peaks are on-peak, the off-peak peak is the 1 kW base
    assert every_day[1][0] == 40.0 and every_day[2][0] == 40.0 * 10 + 1.0 * 1
    # weekday-only: the Saturday peak is off-peak, the Monday peak is on-peak
    assert weekdays[1][0] == 25.0 and weekdays[2][0] == 25.0 * 10 + 40.0 * 1
    with pytest.raises(ValueError):
        charge(p, np.full(12, 10.0), tou_weekdays_only=1, **kw)


def test_minimum_charge_credit_carryover():
    from samapy.pricing.demand_charge import apply_minimum_charge

    floors = np.full(12, 40.0)
    # monthly charges before credits and export credits
    charges = np.array([40.0, 100.0, 70.0, 30.0] + [60.0] * 8)
    credits = np.array([90.0, 0.0, 0.0, 0.0] + [0.0] * 8)
    normal = charges - credits
    np.testing.assert_allclose(apply_minimum_charge(normal, floors, 0, credits), np.maximum(normal, floors))
    # Jan: pays the 40 floor, the 90 credit is unused and carried
    # Feb: 100 - 90 = 10 < 40, pays 40 and uses 60, carries 30
    # Mar: 70 - 30 = 40, pays 40, credit used up
    # Apr: charges 30 < floor and no credit: pays the 40 floor and carries nothing
    np.testing.assert_allclose(apply_minimum_charge(normal, floors, 1, credits), [40, 40, 40, 40] + [60.0] * 8)
    # a minimum charge without any export credit is never carried as credit
    no_credit = np.array([30.0] + [60.0] * 11)
    np.testing.assert_allclose(apply_minimum_charge(no_credit, floors, 1, np.zeros(12)),
                               np.maximum(no_credit, floors))
    with pytest.raises(ValueError):
        apply_minimum_charge(normal, floors, 1)


def test_carryover_never_bills_more_than_without_and_matches_bill_calculator():
    from samapy.pricing.Electricity_Bill_Calculator import calculate_electricity_bills

    rng = np.random.default_rng(3)
    Pbuy = rng.random(8760) * 2
    Psell = rng.random(8760) * 3 * (np.arange(8760) % 24 > 10)  # midday exports
    Cbuy = np.full(8760, 0.12)
    args = (DAYS, Pbuy, Cbuy, np.full(12, 20.0), 0.13, 0.01, 0.0, np.full(25, 0.02), 0.0, 0.0, 1)
    rule = dict(minimum_charge_enabled=1, minimum_charge_monthly=np.full(12, 60.0))
    lost = calculate_electricity_bills(*args, Psell=Psell, Csell=Cbuy, **rule)
    carried = calculate_electricity_bills(*args, Psell=Psell, Csell=Cbuy, minimum_charge_credit_carryover=1, **rule)
    assert carried['annual_cost'] <= lost['annual_cost'] + 1e-9
    for flag, res in ((0, lost), (1, carried)):
        adj, _ = compute_minimum_charge_adjustment(Pbuy, Psell, Cbuy, Cbuy, DAYS, np.full(12, 20.0), 0.13, 0.01,
                                                   0.0, 0.0, 1, np.full(12, 60.0), np.zeros(12), 1,
                                                   credit_carryover=flag)
        plain = calculate_electricity_bills(*args, Psell=Psell, Csell=Cbuy)
        assert res['annual_cost'] == pytest.approx(plain['annual_cost'] + adj, rel=1e-12)


@pytest.mark.parametrize("options", [
    dict(demand_charge_type=1),
    dict(demand_charge_type=4, demand_ratchet_enabled=1, demand_tou_weekdays_only=1,
         minimum_charge_enabled=1, minimum_charge_monthly=np.full(12, 150.0)),
    dict(demand_charge_type=2, minimum_charge_enabled=1, minimum_charge_monthly=np.full(12, 150.0),
         minimum_charge_credit_carryover=1),
])
def test_fitness_grid_cost_equals_monthly_bills(demand_inputs, monkeypatch, options):
    """Year-1 grid cost in the optimization equals the sum of the 12 monthly bills."""
    import sys
    import samapy.core.Fitness as F
    from samapy.pricing.Electricity_Bill_Calculator import calculate_electricity_bills

    rules = {k: options.pop(k) for k in list(options) if not k.startswith('demand_charge_type')
             and k != 'demand_ratchet_enabled'}
    demand_inputs(**options)
    for k, v in rules.items():
        monkeypatch.setattr(InData, k, v)
    captured = {}

    def keep_locals(frame, event, arg):
        if event == 'return' and frame.f_code.co_name == 'evaluate':
            captured.update(frame.f_locals)

    sys.setprofile(keep_locals)
    try:
        F.evaluate(np.array([12.0, 0.0, 2.0, 0.0, 8.0]))
    finally:
        sys.setprofile(None)
    df1 = (1 + F.Grid_escalation[0]) / (1 + F.ir)
    bills = calculate_electricity_bills(
        InData.daysInMonth, captured['Pbuy'], F.Cbuy, F.Service_charge, F.Grid_Tax, F.Grid_Tax_amount,
        F.Annual_expenses, F.Grid_escalation, F.Grid_credit, F.NEM_fee, F.Grid,
        Psell=captured['Psell'], Csell=F.Csell,
        demand_charge_enabled=1, Demand_charge_rate_monthly=InData.Demand_charge_rate_monthly,
        Demand_charge_offpeak_rate_monthly=InData.Demand_charge_offpeak_rate_monthly,
        demand_charge_type=InData.demand_charge_type, demand_tou_onpeak_hours=InData.demand_tou_onpeak_hours,
        demand_tou_season=InData.demand_tou_season, demand_ratchet_enabled=InData.demand_ratchet_enabled,
        demand_ratchet_rate=InData.demand_ratchet_rate, nem_demand_netting=InData.nem_demand_netting,
        minimum_charge_enabled=InData.minimum_charge_enabled, minimum_charge_monthly=InData.minimum_charge_monthly,
        demand_tou_weekdays_only=InData.demand_tou_weekdays_only, year=InData.year, holidays=InData.holidays,
        minimum_charge_credit_carryover=InData.minimum_charge_credit_carryover)
    assert captured['Grid_Cost_net'][0] / df1 == pytest.approx(bills['annual_cost'], rel=1e-12)
