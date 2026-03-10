"""
SAMA CLI ─ runner.py   (v4.0)
==============================
Entry point for the `sama-run` console script.

Usage
-----
    sama-run                          # auto-finds sama_config_COMPLETE.yaml in cwd,
                                      # uses algorithm stored in the config
    sama-run -c my_config.yaml        # explicit config path
    sama-run -c config.yaml -a pso    # override algorithm
    sama-run --verbose                # show full load/patch log + config summary
    sama-run --dry-run --verbose      # validate config without running

OUTPUT DIRECTORY ROUTING
--------------------------
Results.py exposes:  Gen_Results(X, output_dir='sama_outputs')
The optimizers call Gen_Results() without passing output_dir, so it always
defaults to 'sama_outputs' relative to cwd.

We redirect this WITHOUT touching Results.py or the optimizers by
monkey-patching sama.results.Results.Gen_Results with a wrapper that
injects the user's chosen output_directory from the YAML config.
"""

import sys
import os
import argparse
import time
import functools
from pathlib import Path


def _redirect_gen_results(output_dir: str):
    """
    Tell Results.py to save everything to `output_dir`.

    Results.py now has a module-level OUTPUT_DIR variable.  Gen_Results()
    reads it when called with no output_dir argument (i.e. from optimizers).
    Setting it here means all optimizer calls automatically use the right folder.

    Also redirect the optimizer convergence plot (Optimization.png) which is
    saved via plt.savefig with a hardcoded 'sama_outputs/...' path.
    """
    import sama.results.Results as _R
    # Tell Results.py where to save — works even when optimizers call
    # Gen_Results() with no args (their locally-captured function reference
    # reads OUTPUT_DIR at call time, not at import time).
    _R.OUTPUT_DIR = output_dir

    # Redirect optimizer convergence plot: plt.savefig('sama_outputs/Optimization.png')
    import matplotlib.pyplot as _plt
    _orig = _plt.savefig

    def _patched(fname, *args, **kwargs):
        if isinstance(fname, str):
            n = fname.replace('\\', '/')
            if n.startswith('sama_outputs/') or n.startswith('output/figs/'):
                fname = os.path.join(output_dir, os.path.basename(n))
                os.makedirs(os.path.dirname(os.path.abspath(fname)), exist_ok=True)
        return _orig(fname, *args, **kwargs)

    _plt.savefig = _patched


def _patch_fitness(indata, verbose=False):
    import sama.core.Fitness as _F
    d = indata

    _F.Eload          = d.Eload
    _F.Eload_eh       = d.Eload_eh
    _F.Eload_hp       = d.Eload_hp
    _F.Ppv_r          = d.Ppv_r
    _F.Pwt_r          = d.Pwt_r
    _F.Cbt_r          = d.Cbt_r
    _F.Cdg_r          = d.Cdg_r
    _F.T              = d.T
    _F.Tc_noct        = d.Tc_noct
    _F.G              = d.G
    _F.fpv            = d.fpv
    _F.Gref           = d.Gref
    _F.Tcof           = d.Tcof
    _F.Tref           = d.Tref
    _F.Ta_noct        = d.Ta_noct
    _F.G_noct         = d.G_noct
    _F.n_PV           = d.n_PV
    _F.gama           = d.gama
    _F.Vw             = d.Vw
    _F.h_hub          = d.h_hub
    _F.h0             = getattr(d, 'h0', 43.6)
    _F.alfa_wind_turbine = d.alfa_wind_turbine
    _F.v_cut_in       = d.v_cut_in
    _F.v_cut_out      = d.v_cut_out
    _F.v_rated        = d.v_rated
    _F.P              = d.P

    _F.R_B                   = d.R_B
    _F.Q_lifetime_leadacid   = d.Q_lifetime_leadacid
    _F.ef_bat_leadacid        = d.ef_bat_leadacid
    _F.alfa_battery_leadacid = d.alfa_battery_leadacid
    _F.c                     = d.c
    _F.k_lead_acid           = d.k_lead_acid
    _F.Ich_max_leadacid      = d.Ich_max_leadacid
    _F.Vnom_leadacid         = d.Vnom_leadacid
    _F.Lead_acid             = d.Lead_acid
    _F.Li_ion                = d.Li_ion
    _F.Ich_max_Li_ion        = d.Ich_max_Li_ion
    _F.Idch_max_Li_ion       = d.Idch_max_Li_ion
    _F.Vnom_Li_ion           = d.Vnom_Li_ion
    _F.Cnom_Li               = d.Cnom_Li
    _F.ef_bat_Li             = d.ef_bat_Li
    _F.Q_lifetime_Li         = d.Q_lifetime_Li
    _F.alfa_battery_Li_ion   = d.alfa_battery_Li_ion
    _F.SOC_max               = d.SOC_max
    _F.SOC_min               = d.SOC_min
    _F.SOC_initial           = d.SOC_initial
    _F.self_discharge_rate   = d.self_discharge_rate

    _F.b          = d.b
    _F.a          = d.a
    _F.C_fuel     = d.C_fuel
    _F.R_DG       = d.R_DG
    _F.TL_DG      = d.TL_DG
    _F.MO_DG      = d.MO_DG
    _F.LR_DG      = d.LR_DG
    _F.C_fuel_adj = d.C_fuel_adj

    _F.n_I         = d.n_I
    _F.DC_AC_ratio = d.DC_AC_ratio

    _F.Grid             = d.Grid
    _F.Cbuy             = d.Cbuy
    _F.Csell            = d.Csell
    _F.Pbuy_max         = d.Pbuy_max
    _F.Psell_max        = d.Psell_max
    _F.Grid_Tax         = d.Grid_Tax
    _F.Grid_Tax_amount  = d.Grid_Tax_amount
    _F.Grid_credit      = d.Grid_credit
    _F.Grid_escalation  = d.Grid_escalation
    _F.Annual_expenses  = d.Annual_expenses
    _F.Service_charge   = d.Service_charge
    _F.NEM              = d.NEM
    _F.NEM_fee          = d.NEM_fee
    _F.Eload_Previous   = getattr(d, 'Eload_Previous', d.Eload)
    _F.EloadPrevious    = getattr(d, 'EloadPrevious',  d.Eload)

    _F.CO2   = d.CO2
    _F.NOx   = d.NOx
    _F.SO2   = d.SO2
    _F.E_CO2 = d.E_CO2
    _F.E_SO2 = d.E_SO2
    _F.E_NOx = d.E_NOx

    _F.RE_incentives     = d.RE_incentives
    _F.C_PV              = d.C_PV
    _F.C_WT              = d.C_WT
    _F.C_DG              = d.C_DG
    _F.C_B               = d.C_B
    _F.C_I               = d.C_I
    _F.C_CH              = d.C_CH
    _F.Engineering_Costs = d.Engineering_Costs
    _F.n                 = d.n
    _F.L_PV              = d.L_PV
    _F.R_PV              = d.R_PV
    _F.ir                = d.ir
    _F.L_WT              = d.L_WT
    _F.R_WT              = d.R_WT
    _F.L_B               = d.L_B
    _F.L_I               = d.L_I
    _F.R_I               = d.R_I
    _F.L_CH              = d.L_CH
    _F.R_CH              = d.R_CH
    _F.MO_PV             = d.MO_PV
    _F.MO_WT             = d.MO_WT
    _F.MO_B              = d.MO_B
    _F.MO_I              = getattr(d, 'MO_I', 0.0)
    _F.MO_CH             = d.MO_CH
    _F.RT_PV             = d.RT_PV
    _F.RT_WT             = d.RT_WT
    _F.RT_B              = d.RT_B
    _F.RT_I              = d.RT_I
    _F.RT_CH             = d.RT_CH
    _F.System_Tax        = d.System_Tax
    _F.EM                = d.EM
    _F.LPSP_max          = d.LPSP_max
    _F.RE_min            = d.RE_min
    _F.Budget            = d.Budget

    _F.HP       = d.HP
    _F.HP_brand = d.HP_brand
    _F.Hload    = d.Hload
    _F.Cload    = d.Cload
    _F.L_HP     = d.L_HP
    _F.R_HP     = d.R_HP
    _F.RT_HP    = d.RT_HP
    _F.MO_HP    = d.MO_HP
    _F.C_HP     = d.C_HP
    _F.Php_r    = d.Php_r

    _F.NG_Grid             = d.NG_Grid
    _F.Cbuy_NG             = d.Cbuy_NG
    _F.Grid_Tax_NG         = d.Grid_Tax_NG
    _F.Grid_Tax_amount_NG  = d.Grid_Tax_amount_NG
    _F.Grid_credit_NG      = d.Grid_credit_NG
    _F.Grid_escalation_NG  = d.Grid_escalation_NG
    _F.Annual_expenses_NG  = d.Annual_expenses_NG
    _F.Service_charge_NG   = d.Service_charge_NG

    _F.EV                     = d.EV
    _F.self_discharge_rate_ev = d.self_discharge_rate_ev
    _F.Tin                    = d.Tin
    _F.Tout                   = d.Tout
    _F.C_ev                   = d.C_ev
    _F.Pev_max                = d.Pev_max
    _F.SOCe_initial           = d.SOCe_initial
    _F.SOC_dep                = d.SOC_dep
    _F.SOC_arr                = d.SOC_arr
    _F.SOCe_min               = d.SOCe_min
    _F.SOCe_max               = d.SOCe_max
    _F.n_e                    = d.n_e
    _F.Q_lifetime_ev          = d.Q_lifetime_ev
    _F.EV_p                   = d.EV_p
    _F.R_EVB                  = d.R_EVB
    _F.L_EV                   = d.L_EV
    _F.RT_EV                  = d.RT_EV
    _F.Cost_EV                = d.Cost_EV
    _F.MO_EV                  = d.MO_EV

    _F.cap_option                         = d.cap_option
    _F.cap_size                           = d.cap_size
    _F.available_roof_surface             = d.available_roof_surface
    _F.PVPanel_surface_per_rated_capacity = d.PVPanel_surface_per_rated_capacity
    _F.generation_cap                     = d.generation_cap

    if hasattr(d, 'c2'):
        _F.c2 = d.c2

    if verbose:
        print("  ✅ Fitness module patched (90+ variables)")


def _patch_results(indata, verbose=False):
    """
    Patch all module-level InData captures in sama.results.Results.

    Results.py captures ~60 variables at import time — before config_loader
    updates InData.  The most critical ones that cause wrong results:
      - HP_brand, hp_model, HP_size  → wrong HP model shown in results
      - Cbuy, Csell                  → wrong rate structure in cost calcs
      - Eload, G, T, Vw              → wrong load/weather arrays
      - All economic parameters      → wrong cost outputs
    """
    import sama.results.Results as _R
    d = indata

    _R.daysInMonth        = d.daysInMonth
    _R.Eload              = d.Eload
    _R.Ppv_r              = d.Ppv_r
    _R.Pwt_r              = d.Pwt_r
    _R.Cbt_r              = d.Cbt_r
    _R.Cdg_r              = d.Cdg_r
    _R.T                  = d.T
    _R.Tc_noct            = d.Tc_noct
    _R.G                  = d.G
    _R.c2                 = d.c2
    _R.fpv                = d.fpv
    _R.Gref               = d.Gref
    _R.Tcof               = d.Tcof
    _R.Tref               = d.Tref
    _R.Ta_noct            = d.Ta_noct
    _R.G_noct             = d.G_noct
    _R.Vw                 = d.Vw
    _R.h_hub              = d.h_hub
    _R.h0                 = getattr(d, 'h0', 43.6)
    _R.alfa_wind_turbine  = d.alfa_wind_turbine
    _R.v_cut_in           = d.v_cut_in
    _R.v_cut_out          = d.v_cut_out
    _R.v_rated            = d.v_rated
    _R.R_B                = d.R_B
    _R.Q_lifetime_leadacid= d.Q_lifetime_leadacid
    _R.ef_bat_leadacid    = d.ef_bat_leadacid
    _R.b                  = d.b
    _R.C_fuel             = d.C_fuel
    _R.R_DG               = d.R_DG
    _R.TL_DG              = d.TL_DG
    _R.MO_DG              = d.MO_DG
    _R.SOC_max            = d.SOC_max
    _R.SOC_min            = d.SOC_min
    _R.SOC_initial        = d.SOC_initial
    _R.n_I                = d.n_I
    _R.Grid               = d.Grid
    _R.Cbuy               = d.Cbuy
    _R.Csell              = d.Csell
    _R.a                  = d.a
    _R.LR_DG              = d.LR_DG
    _R.Pbuy_max           = d.Pbuy_max
    _R.Psell_max          = d.Psell_max
    _R.self_discharge_rate= d.self_discharge_rate
    _R.alfa_battery_leadacid = d.alfa_battery_leadacid
    _R.c                  = d.c
    _R.k_lead_acid        = d.k_lead_acid
    _R.Ich_max_leadacid   = d.Ich_max_leadacid
    _R.Vnom_leadacid      = d.Vnom_leadacid
    _R.RE_incentives      = d.RE_incentives
    _R.C_PV               = d.C_PV
    _R.C_WT               = d.C_WT
    _R.C_DG               = d.C_DG
    _R.C_B                = d.C_B
    _R.Grid_escalation    = d.Grid_escalation
    _R.C_fuel_adj         = d.C_fuel_adj
    _R.Grid_Tax_amount    = d.Grid_Tax_amount
    _R.Grid_credit        = d.Grid_credit
    _R.NEM                = d.NEM
    _R.NEM_fee            = d.NEM_fee
    _R.Lead_acid          = d.Lead_acid
    _R.Li_ion             = d.Li_ion
    # Li-ion battery
    _R.Ich_max_Li_ion     = d.Ich_max_Li_ion
    _R.Idch_max_Li_ion    = d.Idch_max_Li_ion
    _R.Vnom_Li_ion        = d.Vnom_Li_ion
    _R.Cnom_Li            = d.Cnom_Li
    _R.ef_bat_Li          = d.ef_bat_Li
    _R.Q_lifetime_Li      = d.Q_lifetime_Li
    _R.alfa_battery_Li_ion= d.alfa_battery_Li_ion
    # Financial / misc
    _R.Cash_Flow_adv      = d.Cash_Flow_adv
    _R.Eload_eh           = d.Eload_eh
    _R.h0                 = d.h0
    # HP - all sizing, cost, and performance variables
    _R.P                  = d.P
    _R.HP                 = d.HP
    _R.HP_brand           = d.HP_brand
    _R.hp_model           = d.hp_model
    _R.HP_size            = d.HP_size
    _R.Php_r              = d.Php_r
    _R.L_HP               = d.L_HP
    _R.R_HP               = d.R_HP
    _R.RT_HP              = d.RT_HP
    _R.MO_HP              = d.MO_HP
    _R.C_HP               = d.C_HP
    _R.NG_Grid            = d.NG_Grid
    _R.power_hp_heating   = d.power_hp_heating
    _R.power_hp_cooling   = d.power_hp_cooling
    _R.COP_hp_heating     = d.COP_hp_heating
    _R.COP_hp_cooling     = d.COP_hp_cooling
    _R.Eload_hp           = d.Eload_hp
    _R.Hload              = d.Hload
    _R.Cload              = d.Cload
    # Natural gas (used by electricity comparison figures)
    _R.Cbuy_NG            = d.Cbuy_NG
    _R.Service_charge_NG  = d.Service_charge_NG
    _R.Grid_Tax_NG        = d.Grid_Tax_NG
    _R.Grid_Tax_amount_NG = d.Grid_Tax_amount_NG
    _R.Annual_expenses_NG = d.Annual_expenses_NG
    _R.Grid_escalation_NG = d.Grid_escalation_NG
    _R.Grid_credit_NG     = d.Grid_credit_NG
    # EV
    _R.EV                 = d.EV
    _R.EV_p               = d.EV_p
    _R.C_ev               = d.C_ev
    _R.Pev_max            = d.Pev_max
    _R.SOCe_initial       = d.SOCe_initial
    _R.SOC_dep            = d.SOC_dep
    _R.SOC_arr            = d.SOC_arr
    _R.SOCe_min           = d.SOCe_min
    _R.SOCe_max           = d.SOCe_max
    _R.n_e                = d.n_e
    _R.Q_lifetime_ev      = d.Q_lifetime_ev
    _R.self_discharge_rate_ev = d.self_discharge_rate_ev
    _R.Tin                = d.Tin
    _R.Tout               = d.Tout
    _R.R_EVB              = d.R_EVB
    _R.L_EV               = d.L_EV
    _R.RT_EV              = d.RT_EV
    _R.Cost_EV            = d.Cost_EV
    _R.MO_EV              = d.MO_EV

    if verbose:
        print("  ✅ Results module patched")


def _patch_optimizer(algo, indata, verbose=False):
    d = indata
    try:
        if algo == 'ade':
            import sama.optimizers.AdvancedDifferentialEvolution as _mod
            _mod.PV = d.PV; _mod.WT = d.WT; _mod.Bat = d.Bat; _mod.DG = d.DG
            _mod.Run_Time = d.Run_Time; _mod.nPop = d.nPop; _mod.MaxIt = d.MaxIt
            _mod.F_min = d.F_min; _mod.F_max = d.F_max
            _mod.CR_min = d.CR_min; _mod.CR_max = d.CR_max
            _mod.VarMin = d.VarMin; _mod.VarMax = d.VarMax
        elif algo == 'pso':
            import sama.optimizers.swarm as _mod
            _mod.PV = d.PV; _mod.WT = d.WT; _mod.Bat = d.Bat; _mod.DG = d.DG
            _mod.Run_Time = d.Run_Time; _mod.nPop = d.nPop; _mod.MaxIt = d.MaxIt
            _mod.w = d.w; _mod.wdamp = d.wdamp; _mod.c1 = d.c1; _mod.c2 = d.c2
            _mod.VarMin = d.VarMin; _mod.VarMax = d.VarMax
        elif algo == 'gwo':
            import sama.optimizers.GreyWolfOptimizer as _mod
            _mod.PV = d.PV; _mod.WT = d.WT; _mod.Bat = d.Bat; _mod.DG = d.DG
            _mod.Run_Time = d.Run_Time; _mod.nPop = d.nPop; _mod.MaxIt = d.MaxIt
            _mod.VarMin = d.VarMin; _mod.VarMax = d.VarMax
        elif algo == 'abc':
            import sama.optimizers.ArtificialBeeColony as _mod
            _mod.PV = d.PV; _mod.WT = d.WT; _mod.Bat = d.Bat; _mod.DG = d.DG
            _mod.Run_Time = d.Run_Time; _mod.nPop = d.nPop; _mod.MaxIt = d.MaxIt
            # Note: ABC class reads these from self.* not module-level, so patching
            # is informational — the class __init__ sets its own values from InData
            _mod.VarMin = d.VarMin; _mod.VarMax = d.VarMax
        if verbose:
            print(f"  ✅ {algo.upper()} optimizer module patched")
    except Exception as e:
        print(f"  ⚠️  {algo.upper()} patch warning: {e}")


def _print_summary(d, algo):
    print(f"""
{'─'*60}
CONFIGURATION SUMMARY
{'─'*60}
  Algorithm   : {algo.upper()}
  MaxIt       : {d.MaxIt}
  nPop        : {d.nPop}
  Run_Time    : {d.Run_Time}
  PV          : {d.PV}
  WT          : {d.WT}
  DG          : {d.DG}
  Bat         : {d.Bat}
  Grid        : {d.Grid}
  HP          : {d.HP}
  EV          : {d.EV}
  rateStructure: {d.rateStructure}
  n (years)   : {d.n}
  Budget ($)  : {d.Budget:,.0f}
  VarMin      : {d.VarMin}
  VarMax      : {d.VarMax}
{'─'*60}
""")


def main():
    parser = argparse.ArgumentParser(
        prog='sama-run',
        description='Run SAMA optimization from a wizard-generated YAML config')
    parser.add_argument('-c', '--config', default=None,
                        help='Path to YAML config (default: sama_config_COMPLETE.yaml in cwd)')
    parser.add_argument('-a', '--algorithm', default=None,
                        choices=['pso', 'ade', 'gwo', 'abc'],
                        help='Override algorithm (default: read from config)')
    parser.add_argument('--output',  default=None,
                        help='Override output directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate config without running optimization')
    parser.add_argument('--no-gui',  action='store_true',
                        help='Disable matplotlib GUI (use Agg backend)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed load/patch log and config summary')
    args = parser.parse_args()

    # ── Headless detection ─────────────────────────────────────────────────────
    if (args.no_gui
            or not os.environ.get('DISPLAY', '')
            or os.environ.get('SAMA_HEADLESS', '0') == '1'):
        import matplotlib
        matplotlib.use('Agg')

    # ── Resolve config path ────────────────────────────────────────────────────
    config_path = Path(args.config) if args.config else Path.cwd() / 'sama_config_COMPLETE.yaml'
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        print("Run  sama-config  first to generate a configuration file.")
        sys.exit(1)

    # ── Load & apply config ────────────────────────────────────────────────────
    import io, contextlib
    from sama.cli.config_loader import load_config, apply_config

    config = load_config(config_path)

    if args.verbose:
        print(f"📂 Loading: {config_path.name}  ({len(config)} parameters)")
        indata = apply_config(config)
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            indata = apply_config(config)
        # Surface warnings/errors even in quiet mode
        for line in buf.getvalue().splitlines():
            if '⚠️' in line or 'failed' in line.lower() or 'error' in line.lower():
                print(line)

    # Regenerate Inputs.csv with updated values
    import pandas as pd
    os.makedirs('sama_inputs', exist_ok=True)
    data = {
        'Eload': indata.Eload,
        'Eload_eh': indata.Eload_eh,
        'Eload_hp': indata.Eload_hp if hasattr(indata, 'Eload_hp') and not isinstance(indata.Eload_hp,
                                                                                      int) else [0] * 8760,
        'G': indata.G,
        'T': indata.T,
        'Vw': indata.Vw,
        'Cbuy': indata.Cbuy,
        'EV_P': indata.EV_p,
    }
    pd.DataFrame(data).to_csv('sama_inputs/Inputs.csv', index=False)

    # ── Algorithm: CLI flag > config value > fallback ──────────────────────────
    algo = (args.algorithm or config.get('optimization_algorithm', 'ade')).lower()

    # ── Patch Fitness + optimizer modules ──────────────────────────────────────
    _patch_fitness(indata, verbose=args.verbose)
    _patch_results(indata, verbose=args.verbose)
    _patch_optimizer(algo, indata, verbose=args.verbose)

    # ── Output directory ───────────────────────────────────────────────────────
    # Priority: --output flag  >  config output_directory  >  'sama_outputs'
    out_dir = args.output or config.get('output_directory', 'sama_outputs')
    os.makedirs(out_dir, exist_ok=True)

    # Redirect Gen_Results to user's chosen directory (no changes to Results.py)
    _redirect_gen_results(out_dir)

    # ── Dry run ────────────────────────────────────────────────────────────────
    if args.dry_run:
        print("✅ Config valid — dry run complete.")
        if args.verbose:
            _print_summary(indata, algo)
        return

    if args.verbose:
        _print_summary(indata, algo)

    # ── Run optimization ───────────────────────────────────────────────────────
    print(f"\n🚀 Starting {algo.upper()} optimization...")
    t0 = time.time()

    try:
        if algo == 'ade':
            from sama.optimizers.AdvancedDifferentialEvolution import AdvancedDifferentialEvolution
            AdvancedDifferentialEvolution().optimize()
        elif algo == 'pso':
            from sama.optimizers.swarm import Swarm
            Swarm().optimize()
        elif algo == 'gwo':
            from sama.optimizers.GreyWolfOptimizer import GreyWolfOptimizer
            GreyWolfOptimizer().optimize()
        elif algo == 'abc':
            from sama.optimizers.ArtificialBeeColony import ImprovedArtificialBeeColony
            ImprovedArtificialBeeColony().optimize()
    except Exception as e:
        import traceback
        print(f"\n❌ Optimization failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n✅ Optimization completed in {elapsed:.1f}s")
    print(f"📁 Results saved to: {os.path.abspath(out_dir)}")


if __name__ == '__main__':
    main()
