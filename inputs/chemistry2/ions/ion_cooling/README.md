# Per-ion line cooling Lambda_q(T, n_e)

Collisional-excitation line cooling per ion for the GOW17 ion ladders:
line cooling per unit volume = n_e n_q Lambda_q. One file per ion,
`lambda_<ion>.txt` (CHIANTI names: o_2 = O II, s_4 = S IV, ...).

- Source: CHIANTI 11.0.2 via ChiantiPy 0.16.0, `ion.populate()` with every
  CHIANTI level (O II 196, O III 177, S II 70, S III 53, S IV 52, N II 58,
  N III 204, Ne II 138, Ne III 86); electron and proton collisions,
  n_p = 0.85 n_e. ChiantiPy 0.16's `upsilonDescale(prot=1)` treats `.psplups`
  rates as collision strengths (protons ~1e8 too small); the build wraps it.
  Protons change Lambda by < 0.25% at 8000 K, < 0.8% at 1e5 K.
  No recombination into levels (CHIANTI 11 has no .rrlvl for these ions).
- Grid: log10 T 3.00-9.00 step 0.05 (rows), log10 n_e -2.0-6.0 step 0.5
  (columns); values log10 Lambda [erg cm^3 s^-1], -300 where it underflows.
  Tigris resamples onto the thermo-table T rows at start-up.
- Checked against Cloudy 25 at 8000 K, n_H = 100 cm^-3 (per-ion cooling with
  Cloudy's ion fractions and n_e), 5 zones: O II 0.974-0.978, O III 1.007-1.020,
  S II 0.994-0.997, S III 0.998-0.999, S IV 1.019-1.021, N II 1.000-1.002,
  N III 1.014-1.017.
- Build and check scripts: Research/Photchem-Method/photchem-method/reproduce/
  ions/ion_cooling/ (build_ion_cooling.py, check_ion_cooling.py,
  lambda_table.py).
