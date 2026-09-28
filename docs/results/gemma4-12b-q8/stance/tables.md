# gemma4-12b-q8 / stance

| tier | items | condition | accuracy | log loss | ECE |
|---|---|---|---|---|---|
| test | 3145 | raw | 0.721 [0.703, 0.735] | 2.786 [2.619, 2.976] | 0.247 [0.232, 0.263] |
| test | 3145 | h2 | 0.786 [0.771, 0.800] | 0.550 [0.525, 0.580] | 0.021 [0.015, 0.036] |
| dev | 2100 | raw | 0.654 [0.635, 0.674] | 3.607 [3.336, 3.867] | 0.306 [0.285, 0.326] |
| dev | 2100 | h2 | 0.624 [0.605, 0.647] | 1.029 [0.974, 1.082] | 0.122 [0.106, 0.143] |
| final | 1343 (315 groups) | raw | 0.827 [0.806, 0.848] | 1.603 [1.370, 1.855] | 0.142 [0.123, 0.162] |
| final | 1343 (315 groups) | h2 | 0.812 [0.790, 0.836] | 0.509 [0.472, 0.550] | 0.048 [0.034, 0.072] |

## Fresh families with a recorded caveat (final-flagged), reported beside final

Held out like the final tier and read with it, but not pooled into its numbers and judging no bar:

- healthfc: The evidence sentences are the fact-checkers' own summary and often state the verdict ("we could not find any meaningful scientific evidence", verdict headings such as "Not checked"), so the no-bearing label (the verdict "insufficient evidence") can often be read off the wording, the flaw that excluded PubHealth; and 740 of the 749 claims are yes/no questions, not claims.

| tier | items | condition | accuracy | log loss | ECE |
|---|---|---|---|---|---|
| final-flagged | 749 (749 groups) | raw | 0.750 [0.722, 0.782] | 1.978 [1.665, 2.262] | 0.201 [0.174, 0.229] |
| final-flagged | 749 (749 groups) | h2 | 0.634 [0.597, 0.668] | 0.809 [0.762, 0.856] | 0.123 [0.093, 0.158] |

## Secondary evaluation: held-out families seen during development (final-seen)

An earlier held-out tier whose families were read while judgly was developed, so these numbers are
not a held-out result; they are reported as a secondary evaluation.

| tier | items | condition | accuracy | log loss | ECE |
|---|---|---|---|---|---|
| final-seen | 2100 | raw | 0.632 [0.610, 0.652] | 3.630 [3.404, 3.870] | 0.320 [0.302, 0.342] |
| final-seen | 2100 | h2 | 0.626 [0.605, 0.647] | 0.889 [0.851, 0.924] | 0.067 [0.051, 0.089] |

## Selective accuracy (h2): accuracy / share answered

| tier | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| test | 0.82 / 90% | 0.87 / 77% | 0.90 / 64% | 0.93 / 50% | 0.96 / 32% | 0.97 / 18% | 1.00 / 0% |
| dev | 0.66 / 84% | 0.69 / 69% | 0.71 / 58% | 0.73 / 46% | 0.76 / 31% | 0.77 / 17% | 0.67 / 0% |
| final | 0.84 / 90% | 0.87 / 78% | 0.90 / 64% | 0.92 / 50% | 0.95 / 32% | 0.96 / 16% | 0.80 / 0% |
| final-flagged | 0.64 / 77% | 0.65 / 57% | 0.71 / 41% | 0.81 / 26% | 0.91 / 11% | 0.90 / 3% | n/a / 0% |
| final-seen | 0.68 / 81% | 0.72 / 63% | 0.76 / 47% | 0.78 / 31% | 0.80 / 15% | 0.82 / 5% | 1.00 / 0% |

## By family

| tier | family | items | raw accuracy | h2 accuracy | raw ECE | h2 ECE |
|---|---|---|---|---|---|---|
| test | fit_claims | 1261 | 0.815 [0.794, 0.836] | 0.856 [0.837, 0.875] | 0.163 [0.144, 0.184] | 0.017 [0.013, 0.038] |
| test | fit_nli | 1570 | 0.645 [0.624, 0.666] | 0.743 [0.719, 0.762] | 0.319 [0.299, 0.342] | 0.031 [0.021, 0.051] |
| test | fit_scientific | 314 | 0.717 [0.666, 0.768] | 0.723 [0.675, 0.774] | 0.230 [0.190, 0.286] | 0.044 [0.041, 0.104] |
| dev | dev_claims | 700 | 0.691 [0.657, 0.726] | 0.623 [0.587, 0.657] | 0.268 [0.238, 0.305] | 0.068 [0.045, 0.105] |
| dev | dev_nli | 700 | 0.644 [0.610, 0.676] | 0.629 [0.590, 0.664] | 0.303 [0.274, 0.338] | 0.088 [0.070, 0.127] |
| dev | dev_scientific | 700 | 0.627 [0.587, 0.663] | 0.621 [0.586, 0.657] | 0.349 [0.316, 0.390] | 0.228 [0.197, 0.264] |
| final | final_covid_claims | 1343 (315 groups) | 0.827 [0.805, 0.847] | 0.812 [0.790, 0.833] | 0.142 [0.124, 0.164] | 0.048 [0.033, 0.071] |
| final-flagged | final_health_claims | 749 (749 groups) | 0.750 [0.718, 0.778] | 0.634 [0.601, 0.668] | 0.201 [0.175, 0.231] | 0.123 [0.094, 0.157] |
| final-seen | final_scientific | 2100 | 0.632 [0.610, 0.651] | 0.626 [0.606, 0.648] | 0.320 [0.302, 0.343] | 0.067 [0.051, 0.087] |

## Confusion (h2): gold (rows) by predicted (columns)

**test**

| gold | contradicts | no_bearing | supports |
|---|---|---|---|
| contradicts | 901 | 107 | 73 |
| no_bearing | 95 | 438 | 215 |
| supports | 41 | 141 | 1134 |

**dev**

| gold | contradicts | no_bearing | supports |
|---|---|---|---|
| contradicts | 463 | 59 | 307 |
| no_bearing | 164 | 185 | 102 |
| supports | 66 | 91 | 663 |

**final**

| gold | contradicts | no_bearing | supports |
|---|---|---|---|
| contradicts | 415 | 21 | 15 |
| no_bearing | 95 | 269 | 78 |
| supports | 10 | 33 | 407 |

**final-flagged**

| gold | contradicts | no_bearing | supports |
|---|---|---|---|
| contradicts | 110 | 7 | 8 |
| no_bearing | 217 | 199 | 6 |
| supports | 16 | 20 | 166 |

**final-seen**

| gold | contradicts | no_bearing | supports |
|---|---|---|---|
| contradicts | 247 | 98 | 114 |
| no_bearing | 98 | 512 | 196 |
| supports | 96 | 183 | 556 |

