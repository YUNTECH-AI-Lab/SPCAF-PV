"""Generate one month of data for 30 houses in Sydney and print the key results."""
from synpv import Simulation, default_config

cfg = default_config(
    LATITUDE=-33.8688, LONGITUDE=151.2093, SITE_NAME="Sydney, Australia",
    START_DATE="2023-01-01", END_DATE="2023-01-31",
    N_HOUSES=30, OUTPUT_DIR="sim_output_sydney_jan2023",
)
result = Simulation(cfg).run()

m = result.metrics
print(f"\n{m['n_pv_houses']} PV houses, {m['energy_kwh']:.0f} kWh, PR {m['performance_ratio_mean']:.1%}")
print("Files written to", result.output_dir)

# the arrays are also available directly, e.g. AC power of the first PV house:
import pandas as pd
p = pd.read_csv(f"{result.output_dir}/pv_output_5min.csv.gz", index_col=0, parse_dates=True)
print(p.loc[:, (p.max() > 0)].iloc[:, 0].describe())
