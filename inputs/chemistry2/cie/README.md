# Collisional ionization equilibrium tables for chemistry2

`tigress_coolftn_ncr.txt`: the TIGRESS NCR hot-gas table, copied from Tigris
(`inputs/tables/tigress_coolftn_ncr.txt`). Read by `src/chemistry2/thermo/hot_gas.hpp`.

- Header: `nvar nx1`, then the log10 T limits (4, 9), then 251 rows uniform in log10 T.
- Columns: log10 T; Lambda_H, Lambda_He, Lambda_metal [erg cm^3 s^-1, cooling per H =
  n_H Lambda, metals at Z = 1]; xe_He, xe_metal [electrons per H]; four columns not used.
- Used above `NCR_temp_hot0` (default 2e4 K), with the hot-gas weight; see the module's
  design note P4 for the blend.

The per-element CHIANTI v11 tables (`ioneq_<El>.txt`, `cool_<El>.txt`) are not used yet:
summed over stages, their hydrogen cooling is 0.50-0.73 of Lambda_H here at
10^5.4-10^7 K, which points to recombination radiation filed under the recombined stage.
They are to be rebuilt before the per-element (ion-ladder) form uses them.
