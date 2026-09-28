# qwen3-4b-q8 / stance

| tier | items | condition | accuracy | log loss | ECE |
|---|---|---|---|---|---|
| test | 3145 | raw | 0.680 [0.664, 0.697] | 4.786 [4.502, 5.102] | 0.276 [0.261, 0.294] |
| test | 3145 | h2 | 0.755 [0.739, 0.770] | 0.624 [0.598, 0.651] | 0.017 [0.012, 0.033] |
| dev | 2100 | raw | 0.549 [0.527, 0.570] | 7.057 [6.665, 7.485] | 0.396 [0.375, 0.419] |
| dev | 2100 | h2 | 0.536 [0.514, 0.560] | 1.194 [1.145, 1.247] | 0.209 [0.189, 0.232] |
| final | 1343 (315 groups) | raw | 0.778 [0.755, 0.801] | 3.185 [2.812, 3.589] | 0.187 [0.167, 0.212] |
| final | 1343 (315 groups) | h2 | 0.773 [0.749, 0.795] | 0.594 [0.562, 0.628] | 0.052 [0.034, 0.077] |

## Fresh families with a recorded caveat (final-flagged), reported beside final

Held out like the final tier and read with it, but not pooled into its numbers and judging no bar:

- healthfc: The evidence sentences are the fact-checkers' own summary and often state the verdict ("we could not find any meaningful scientific evidence", verdict headings such as "Not checked"), so the no-bearing label (the verdict "insufficient evidence") can often be read off the wording, the flaw that excluded PubHealth; and 740 of the 749 claims are yes/no questions, not claims.

| tier | items | condition | accuracy | log loss | ECE |
|---|---|---|---|---|---|
| final-flagged | 749 (749 groups) | raw | 0.718 [0.684, 0.752] | 3.605 [3.081, 4.081] | 0.227 [0.196, 0.260] |
| final-flagged | 749 (749 groups) | h2 | 0.541 [0.506, 0.575] | 0.928 [0.877, 0.979] | 0.150 [0.130, 0.192] |

## Secondary evaluation: held-out families seen during development (final-seen)

An earlier held-out tier whose families were read while judgly was developed, so these numbers are
not a held-out result; they are reported as a secondary evaluation.

| tier | items | condition | accuracy | log loss | ECE |
|---|---|---|---|---|---|
| final-seen | 2100 | raw | 0.571 [0.548, 0.592] | 5.891 [5.525, 6.277] | 0.354 [0.333, 0.377] |
| final-seen | 2100 | h2 | 0.548 [0.527, 0.570] | 0.997 [0.966, 1.032] | 0.100 [0.084, 0.122] |

## Selective accuracy (h2): accuracy / share answered

| tier | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| test | 0.79 / 89% | 0.83 / 76% | 0.87 / 60% | 0.91 / 44% | 0.95 / 23% | 0.98 / 6% | n/a / 0% |
| dev | 0.56 / 87% | 0.58 / 74% | 0.61 / 62% | 0.64 / 48% | 0.71 / 27% | 0.88 / 7% | n/a / 0% |
| final | 0.81 / 87% | 0.86 / 71% | 0.88 / 58% | 0.91 / 40% | 0.94 / 17% | 0.96 / 4% | n/a / 0% |
| final-flagged | 0.55 / 86% | 0.55 / 65% | 0.57 / 44% | 0.64 / 27% | 0.84 / 4% | n/a / 0% | n/a / 0% |
| final-seen | 0.59 / 76% | 0.61 / 54% | 0.65 / 38% | 0.67 / 22% | 0.79 / 5% | 0.83 / 1% | n/a / 0% |

## By family

| tier | family | items | raw accuracy | h2 accuracy | raw ECE | h2 ECE |
|---|---|---|---|---|---|---|
| test | fit_claims | 1261 | 0.776 [0.752, 0.798] | 0.816 [0.793, 0.837] | 0.183 [0.163, 0.207] | 0.028 [0.021, 0.050] |
| test | fit_nli | 1570 | 0.604 [0.582, 0.629] | 0.727 [0.704, 0.748] | 0.352 [0.330, 0.376] | 0.035 [0.023, 0.059] |
| test | fit_scientific | 314 | 0.678 [0.627, 0.733] | 0.646 [0.596, 0.701] | 0.272 [0.221, 0.322] | 0.052 [0.034, 0.112] |
| dev | dev_claims | 700 | 0.633 [0.597, 0.667] | 0.569 [0.530, 0.606] | 0.304 [0.271, 0.341] | 0.133 [0.105, 0.173] |
| dev | dev_nli | 700 | 0.466 [0.430, 0.501] | 0.456 [0.419, 0.494] | 0.453 [0.418, 0.491] | 0.226 [0.191, 0.263] |
| dev | dev_scientific | 700 | 0.549 [0.510, 0.584] | 0.583 [0.547, 0.621] | 0.431 [0.397, 0.470] | 0.273 [0.241, 0.311] |
| final | final_covid_claims | 1343 (315 groups) | 0.778 [0.754, 0.800] | 0.773 [0.750, 0.797] | 0.187 [0.167, 0.212] | 0.052 [0.034, 0.078] |
| final-flagged | final_health_claims | 749 (749 groups) | 0.718 [0.688, 0.749] | 0.541 [0.505, 0.575] | 0.227 [0.197, 0.259] | 0.150 [0.129, 0.193] |
| final-seen | final_scientific | 2100 | 0.571 [0.550, 0.594] | 0.548 [0.527, 0.570] | 0.354 [0.331, 0.375] | 0.100 [0.083, 0.121] |

## Confusion (h2): gold (rows) by predicted (columns)

**test**

| gold | contradicts | no_bearing | supports |
|---|---|---|---|
| contradicts | 864 | 80 | 137 |
| no_bearing | 123 | 407 | 218 |
| supports | 72 | 142 | 1102 |

**dev**

| gold | contradicts | no_bearing | supports |
|---|---|---|---|
| contradicts | 403 | 31 | 395 |
| no_bearing | 217 | 100 | 134 |
| supports | 147 | 51 | 622 |

**final**

| gold | contradicts | no_bearing | supports |
|---|---|---|---|
| contradicts | 390 | 18 | 43 |
| no_bearing | 116 | 260 | 66 |
| supports | 30 | 32 | 388 |

**final-flagged**

| gold | contradicts | no_bearing | supports |
|---|---|---|---|
| contradicts | 112 | 7 | 6 |
| no_bearing | 275 | 140 | 7 |
| supports | 18 | 31 | 153 |

**final-seen**

| gold | contradicts | no_bearing | supports |
|---|---|---|---|
| contradicts | 292 | 56 | 111 |
| no_bearing | 220 | 333 | 253 |
| supports | 187 | 122 | 526 |

