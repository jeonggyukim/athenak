# Atomic rate and cross-section tables

Read by `src/photchem/rates/`. One file per published fit, in the units the
source gives, with the conversion done on load rather than in the file, so a
table can be checked against its paper without arithmetic.

Photoionization, recombination, collisional ionization and charge transfer with
hydrogen are here. Charge transfer with helium or H2 as the collider is not,
and the list at the end says where it comes from so that adding it is a
download rather than a search.

`N` means the electron count of the species named, so it is the count before
the reaction in every table: hydrogen recombining is `Z = 1, N = 0`, and
hydrogen being photoionized or collisionally ionized is `Z = 1, N = 1`.

## Photoionization cross sections

| file | source | rows | coverage | what it gives |
|---|---|---|---|---|
| `verner96_photx.dat` | Verner, Ferland, Korista & Yakovlev (1996), ApJ 465, 487, Table 1 | 184 | Z = 1--14, 16, 18, 20, 26, all stages | the valence shell alone, from the ionization threshold up to the first inner-shell edge |
| `verner_yakovlev95_photx.dat` | Verner & Yakovlev (1995), A&AS 109, 125, Table 1, CDS catalogue `J/A+AS/109/125` | 1696 | Z = 1--30, all stages | every subshell, at every energy |

Two tables rather than one because neither is right over a wide band. The 1996
fits rest on Opacity Project data and describe the valence shell better near
its threshold, but they stop where the first inner shell opens. The 1995 fits
span every shell and every energy from a central-field calculation that is
weaker near an outer threshold. `PhotoIon::TotalXsec` takes each where it is
good, which is what Cloudy's `phfit` does, Verner having written both.

The 1996 table covers only the elements the Opacity Project treated. P, Cl, K
and everything above Z = 20 other than Fe are absent, and those run on the 1995
fits at every energy.

`verner96_photx.dat` came from
`pyathena/data/microphysics/verner96_photx.dat`; `verner_yakovlev95_photx.dat`
was fetched from CDS, whose ReadMe gives the byte-by-byte format the header
comment reproduces.

## Recombination

| file | source | rows | what it gives |
|---|---|---|---|
| `badnell_rr_2023.dat` | Badnell, `amdpp.phys.strath.ac.uk/tamoc/RR/`, compilation of 2023-05-11 | 1355 | radiative recombination, `alpha = A / [sqrt(T/T0) (1+sqrt(T/T0))^(1-B) (1+sqrt(T/T1))^(1+B)]` |
| `badnell_dr_C_2023.dat`, `badnell_dr_E_2023.dat` | Badnell, `amdpp.phys.strath.ac.uk/tamoc/DR/`, compilation of 2023-05-12 | 1322 each | dielectronic recombination, `alpha = T^(-3/2) sum_m c_m exp(-E_m/T)`; the two files carry the same rows and are read together |

Read by `PhotoIon`'s neighbour `Rates::RecombRate`. `N` in these tables is the
electron count *before* recombination, so hydrogen recombining is `Z = 1,
N = 0`.

Coverage is not complete. Iron reaches eighteen electrons, so neutral iron
recombination has no fit and the lookup says so rather than returning zero.

The recombination files carry a date rather than a paper year because Badnell
publishes the fits on the TAMOC pages and revises them as isoelectronic
sequences are recalculated; Badnell (2006), ApJS 167, 334 and A&A 447, 1165 are
the method. The first line of each file is its version.

Take the `_2023` files. Pyathena holds both vintages side by side, and the
unsuffixed `badnell_rr.dat`, `badnell_dr_C.dat` and `badnell_dr_E.dat` are the
2017-07-07 compilation, which `rec_rate.py` no longer reads. The newer set
covers 90 more ions in each file, 1322 rows against 1232 for DR and 1355
against 1265 for RR, so the difference is coverage as well as revision.

Hydrogen needs a choice these tables cannot make. The Badnell fit is case A,
counting recombination to the ground state, and an H II region wants case B,
where that photon is reabsorbed where it was emitted. `RecombRate` defaults to
the case-B expression `NCRRates` already uses, so a multi-ion network extending
NCR cannot end up with two hydrogen recombination rates.

## Collisional ionization

| file | source | rows | what it gives |
|---|---|---|---|
| `voronov97_coll_ion.dat` | Voronov (1997), ADNDT 65, 1, as distributed with Cloudy | 468 | `k(T) = A (1 + P sqrt(U)) / (X + U) U^K exp(-U)` with `U = dE/kT`, zero above `U = 80` |

Read by `Rates::CollIonRate`. The file counts both `Z` and `N` from zero and
puts the electron count first, which is Cloudy's convention; the reader
converts, so callers use the same `(Z, N)` as everywhere else and atomic
hydrogen is `Z = 1, N = 1`. A row of `-1 -1` ends the data.

## Charge transfer

| file | source | rows | what it gives |
|---|---|---|---|
| `kingdon_ferland96_ct_rec.dat` | Kingdon & Ferland (1996), ApJS 106, 205, as `data/ctrecombdata.dat` in Cloudy | 120 | `X^(q+1) + H0 -> X^q + H+`, `k = a 1e-9 (T/1e4)^b [1 + c exp(d T/1e4)]` |
| `kingdon_ferland96_ct_ion.dat` | the same, as `data/ctiondata.dat` | 120 | `X^q + H+ -> X^(q+1) + H0`, the same form times `exp(-dE_barrier 1e4 / T)` |

Read by `Rates::ChargeTransferRate`. Hydrogen is the only collider, which is
the choice Katz (2022) makes deliberately, dropping helium as a collider and
metal-metal transfer because those densities are too low to matter.

These rows carry no labels. They run over thirty elements from hydrogen with
four charge states each, and both files index a pair by its lower charge state,
so the recombination table is labelled by the *product*. `Load` converts to the
electron count of the reacting ion, the convention above, which puts the charge
transfer rate for one charge state at the same `(Z, N)` as the radiative
recombination or collisional ionization it competes with.

Coverage stops at `q = 3`, the last charge state Kingdon & Ferland fit, which
is the boundary of the planned ion sets rather than a gap inside them. Above it
the recombining direction falls back to Dalgarno's estimate, `1.92e-9 (q+1)`,
as Cloudy does, and the ionizing direction returns zero, as Cloudy also does.
Pequignot (1996) and Stancil et al. (1999) are where the fits continue.

## Still to add

Each is a table in pyathena under `data/microphysics/`, read there by the
module named in the last column.

| what | source | file | pyathena module |
|---|---|---|---|
| charge transfer, helium collider | Kingdon & Ferland (1996) | `ugacxdb/ctr_he.dat`, `ugacxdb/cti_he.dat` | `ct_rate.py` |
| charge transfer, H2 collider | Kingdon & Ferland (1996) | `ugacxdb/ct_h2.dat` | `ct_rate.py` |

The recombination and charge-transfer sets are what the sixteen-ion sweep of
`metal-ions.tex` needs; the photoionization tables here are the third leg of
that same set.
