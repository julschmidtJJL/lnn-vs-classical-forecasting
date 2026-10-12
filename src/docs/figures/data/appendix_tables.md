## Appendix Table 1 — Run 2a (raw exog), MAE mean ± std across 5 seeds

Lowest MAE per ticker in **bold**; DA = mean directional accuracy. Rows sorted by (MAE, model name) — identical to the CSV.

| Ticker | Model condition        | MAE                   | DA    |
|--------|------------------------|-----------------------|-------|
| AAPL   | **cfc_pure_ncp**       | **0.01304 ± 0.00000** | 0.513 |
| AAPL   | sarimax                | 0.01310 ± 0.00000     | 0.471 |
| AAPL   | sarimax_arma11         | 0.01310 ± 0.00000     | 0.478 |
| AAPL   | sarimax_ma1            | 0.01310 ± 0.00000     | 0.487 |
| AAPL   | sarimax_aic            | 0.01311 ± 0.00000     | 0.432 |
| AAPL   | cfc_default_ncp_mm     | 0.01485 ± 0.00019     | 0.481 |
| AAPL   | cfc_pure_fc            | 0.01485 ± 0.00012     | 0.485 |
| AAPL   | cfc_default_fc_bb128   | 0.01502 ± 0.00031     | 0.488 |
| AAPL   | cfc_default_ncp        | 0.01503 ± 0.00018     | 0.479 |
| AAPL   | ltc_ncp                | 0.01510 ± 0.00022     | 0.490 |
| AAPL   | cfc_default_fc         | 0.01527 ± 0.00021     | 0.488 |
| AAPL   | ltc_fc                 | 0.01531 ± 0.00015     | 0.487 |
| AAPL   | neural_ode             | 0.01543 ± 0.00045     | 0.481 |
| AAPL   | cfc_no_gate_fc         | 0.01550 ± 0.00050     | 0.485 |
| AAPL   | cfc_no_gate_ncp        | 0.01553 ± 0.00024     | 0.485 |
| AAPL   | lstm                   | 0.01564 ± 0.00011     | 0.484 |
| AAPL   | ctrnn                  | 0.01596 ± 0.00025     | 0.476 |
| AAPL   | ltc_ncp_mm             | 0.01674 ± 0.00037     | 0.478 |
| AAPL   | naive                  | 0.01921 ± 0.00000     | 0.452 |
| JPM    | **cfc_pure_ncp**       | **0.01217 ± 0.00000** | 0.511 |
| JPM    | sarimax_ma1            | 0.01226 ± 0.00000     | 0.472 |
| JPM    | sarimax                | 0.01229 ± 0.00000     | 0.458 |
| JPM    | sarimax_arma11         | 0.01229 ± 0.00000     | 0.478 |
| JPM    | sarimax_aic            | 0.01236 ± 0.00000     | 0.452 |
| JPM    | ltc_fc                 | 0.01383 ± 0.00017     | 0.496 |
| JPM    | cfc_pure_fc            | 0.01396 ± 0.00018     | 0.491 |
| JPM    | cfc_default_fc_bb128   | 0.01413 ± 0.00030     | 0.493 |
| JPM    | cfc_default_ncp_mm     | 0.01418 ± 0.00013     | 0.484 |
| JPM    | cfc_default_ncp        | 0.01426 ± 0.00012     | 0.487 |
| JPM    | ltc_ncp                | 0.01440 ± 0.00039     | 0.494 |
| JPM    | cfc_no_gate_fc         | 0.01464 ± 0.00030     | 0.481 |
| JPM    | neural_ode             | 0.01473 ± 0.00014     | 0.487 |
| JPM    | lstm                   | 0.01478 ± 0.00026     | 0.488 |
| JPM    | cfc_default_fc         | 0.01479 ± 0.00017     | 0.492 |
| JPM    | cfc_no_gate_ncp        | 0.01486 ± 0.00012     | 0.490 |
| JPM    | ctrnn                  | 0.01498 ± 0.00028     | 0.476 |
| JPM    | ltc_ncp_mm             | 0.01569 ± 0.00025     | 0.483 |
| JPM    | naive                  | 0.01773 ± 0.00000     | 0.470 |
| SPY    | **cfc_pure_ncp**       | **0.00794 ± 0.00000** | 0.530 |
| SPY    | sarimax                | 0.00808 ± 0.00000     | 0.467 |
| SPY    | sarimax_arma11         | 0.00808 ± 0.00000     | 0.485 |
| SPY    | sarimax_ma1            | 0.00808 ± 0.00000     | 0.475 |
| SPY    | sarimax_aic            | 0.00809 ± 0.00000     | 0.490 |
| SPY    | cfc_default_fc_bb128   | 0.00897 ± 0.00018     | 0.491 |
| SPY    | ltc_fc                 | 0.00925 ± 0.00004     | 0.483 |
| SPY    | cfc_default_ncp_mm     | 0.00941 ± 0.00007     | 0.485 |
| SPY    | cfc_pure_fc            | 0.00942 ± 0.00022     | 0.487 |
| SPY    | ltc_ncp                | 0.00949 ± 0.00026     | 0.490 |
| SPY    | cfc_default_ncp        | 0.00962 ± 0.00009     | 0.491 |
| SPY    | neural_ode             | 0.00966 ± 0.00027     | 0.486 |
| SPY    | ctrnn                  | 0.00971 ± 0.00020     | 0.491 |
| SPY    | lstm                   | 0.00984 ± 0.00010     | 0.481 |
| SPY    | cfc_no_gate_fc         | 0.00993 ± 0.00014     | 0.485 |
| SPY    | cfc_default_fc         | 0.01009 ± 0.00020     | 0.487 |
| SPY    | cfc_no_gate_ncp        | 0.01009 ± 0.00010     | 0.484 |
| SPY    | ltc_ncp_mm             | 0.01036 ± 0.00032     | 0.479 |
| SPY    | naive                  | 0.01185 ± 0.00000     | 0.458 |

## Appendix Table 2 — Run 2b (stationarized exog), MAE mean ± std across 5 seeds

Lowest MAE per ticker in **bold**; DA = mean directional accuracy. Rows sorted by (MAE, model name) — identical to the CSV.

| Ticker | Model condition        | MAE                   | DA    |
|--------|------------------------|-----------------------|-------|
| AAPL   | **cfc_pure_ncp**       | **0.01305 ± 0.00000** | 0.513 |
| AAPL   | sarimax_aic            | 0.01308 ± 0.00000     | 0.431 |
| AAPL   | sarimax                | 0.01310 ± 0.00000     | 0.472 |
| AAPL   | sarimax_arma11         | 0.01311 ± 0.00000     | 0.478 |
| AAPL   | sarimax_ma1            | 0.01311 ± 0.00000     | 0.487 |
| AAPL   | cfc_pure_fc            | 0.01467 ± 0.00024     | 0.488 |
| AAPL   | cfc_default_ncp_mm     | 0.01499 ± 0.00010     | 0.481 |
| AAPL   | cfc_default_ncp        | 0.01505 ± 0.00017     | 0.483 |
| AAPL   | cfc_default_fc_bb128   | 0.01521 ± 0.00027     | 0.486 |
| AAPL   | ltc_fc                 | 0.01526 ± 0.00012     | 0.488 |
| AAPL   | neural_ode             | 0.01541 ± 0.00042     | 0.478 |
| AAPL   | ctrnn                  | 0.01559 ± 0.00037     | 0.485 |
| AAPL   | cfc_no_gate_ncp        | 0.01560 ± 0.00007     | 0.488 |
| AAPL   | cfc_default_fc         | 0.01561 ± 0.00017     | 0.479 |
| AAPL   | cfc_no_gate_fc         | 0.01572 ± 0.00028     | 0.490 |
| AAPL   | ltc_ncp                | 0.01597 ± 0.00016     | 0.482 |
| AAPL   | lstm                   | 0.01605 ± 0.00009     | 0.485 |
| AAPL   | ltc_ncp_mm             | 0.01730 ± 0.00071     | 0.484 |
| AAPL   | naive                  | 0.01922 ± 0.00000     | 0.452 |
| JPM    | **cfc_pure_ncp**       | **0.01218 ± 0.00000** | 0.511 |
| JPM    | sarimax_ma1            | 0.01230 ± 0.00000     | 0.469 |
| JPM    | sarimax                | 0.01233 ± 0.00000     | 0.460 |
| JPM    | sarimax_arma11         | 0.01233 ± 0.00000     | 0.477 |
| JPM    | sarimax_aic            | 0.01237 ± 0.00000     | 0.452 |
| JPM    | cfc_pure_fc            | 0.01379 ± 0.00023     | 0.491 |
| JPM    | cfc_default_fc_bb128   | 0.01382 ± 0.00024     | 0.485 |
| JPM    | ltc_fc                 | 0.01396 ± 0.00006     | 0.488 |
| JPM    | ltc_ncp                | 0.01407 ± 0.00015     | 0.493 |
| JPM    | cfc_default_ncp_mm     | 0.01411 ± 0.00019     | 0.485 |
| JPM    | cfc_default_ncp        | 0.01429 ± 0.00008     | 0.493 |
| JPM    | cfc_default_fc         | 0.01452 ± 0.00022     | 0.490 |
| JPM    | cfc_no_gate_fc         | 0.01454 ± 0.00026     | 0.495 |
| JPM    | ctrnn                  | 0.01474 ± 0.00034     | 0.490 |
| JPM    | neural_ode             | 0.01476 ± 0.00042     | 0.490 |
| JPM    | cfc_no_gate_ncp        | 0.01481 ± 0.00021     | 0.487 |
| JPM    | lstm                   | 0.01495 ± 0.00011     | 0.481 |
| JPM    | ltc_ncp_mm             | 0.01548 ± 0.00038     | 0.488 |
| JPM    | naive                  | 0.01771 ± 0.00000     | 0.473 |
| SPY    | **cfc_pure_ncp**       | **0.00795 ± 0.00000** | 0.531 |
| SPY    | sarimax                | 0.00808 ± 0.00000     | 0.468 |
| SPY    | sarimax_arma11         | 0.00808 ± 0.00000     | 0.487 |
| SPY    | sarimax_ma1            | 0.00808 ± 0.00000     | 0.475 |
| SPY    | sarimax_aic            | 0.00810 ± 0.00000     | 0.491 |
| SPY    | ltc_ncp                | 0.00932 ± 0.00005     | 0.480 |
| SPY    | ltc_fc                 | 0.00955 ± 0.00007     | 0.482 |
| SPY    | cfc_default_ncp_mm     | 0.00969 ± 0.00018     | 0.488 |
| SPY    | cfc_default_fc_bb128   | 0.00972 ± 0.00030     | 0.488 |
| SPY    | cfc_pure_fc            | 0.00982 ± 0.00021     | 0.487 |
| SPY    | cfc_default_ncp        | 0.00988 ± 0.00013     | 0.487 |
| SPY    | neural_ode             | 0.00995 ± 0.00020     | 0.482 |
| SPY    | cfc_no_gate_ncp        | 0.01030 ± 0.00020     | 0.493 |
| SPY    | ctrnn                  | 0.01034 ± 0.00023     | 0.483 |
| SPY    | lstm                   | 0.01034 ± 0.00014     | 0.480 |
| SPY    | cfc_no_gate_fc         | 0.01042 ± 0.00009     | 0.481 |
| SPY    | ltc_ncp_mm             | 0.01053 ± 0.00010     | 0.482 |
| SPY    | cfc_default_fc         | 0.01061 ± 0.00015     | 0.481 |
| SPY    | naive                  | 0.01185 ± 0.00000     | 0.458 |
