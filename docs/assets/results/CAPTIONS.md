# Results figures

Made by `make figures` (scripts/make_figures.py) from the snapshot in docs/results only; every
plotted number is checked against record.json. All figures show the fresh final tier: task
families frozen before any head was scored on them, and never used to fit or choose a head.
Brackets are 95% intervals; on this tier the bootstrap resamples whole groups of items that
share a claim, a table, a template or a query.

## reliability.svg

Reliability diagrams on the fresh final tier: for ten equal-width bins of the top probability, the
mean confidence (x) against the share of correct answers (y), with 95% Wilson intervals; empty
bins are left out (the Wilson intervals treat items as independent). On the dotted diagonal, confidence equals accuracy. Gemma 4 12B general (n = 7,879 in 6,803 groups): ECE raw 0.194 [0.186, 0.204], head 0.020 [0.015, 0.031]; Gemma 4 12B stance (n = 1,343 in 315 groups): ECE raw 0.142 [0.123, 0.162], head 0.048 [0.034, 0.072]; Qwen3-4B general (n = 7,879 in 6,803 groups): ECE raw 0.268 [0.259, 0.279], head 0.030 [0.025, 0.040]; Qwen3-4B stance (n = 1,343 in 315 groups): ECE raw 0.187 [0.167, 0.212], head 0.052 [0.034, 0.077].

## selective.svg

Selective accuracy on the fresh final tier: answer only when the top probability is above a
threshold, swept from 0 to 0.999; x is the share of questions answered, y the accuracy on those.
Markers are the record's thresholds 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 and 0.99. Curves stop where
fewer than 50 questions are answered. The raw curves stop far from 0%: without a head the
models give many answers a top probability above 0.999, so no threshold can hold them back. general: n = 7,879 in 6,803 groups per model; stance: n = 1,343 in 315 groups per model.

## families-gemma4-12b-q8.svg

Gemma 4 12B, fresh final tier by task family: accuracy and ECE, raw and with the fitted
head, with the calibration record's 95% percentile bootstrap intervals (1,000 resamples of
whole groups). Each label names the family, its tasks and its number of items. The stance
format's fresh final tier is one family (the Check-COVID task). The flagged families
(politeness, HealthFC) and the families seen during development are not shown; their
numbers are in the tables of docs/results.

## families-qwen3-4b-q8.svg

Qwen3-4B, fresh final tier by task family: accuracy and ECE, raw and with the fitted
head, with the calibration record's 95% percentile bootstrap intervals (1,000 resamples of
whole groups). Each label names the family, its tasks and its number of items. The stance
format's fresh final tier is one family (the Check-COVID task). The flagged families
(politeness, HealthFC) and the families seen during development are not shown; their
numbers are in the tables of docs/results.

<!-- external comparison: written by scripts/make_compare_figures.py -->

# External comparison figures

Made by `make compare-figures` (scripts/make_compare_figures.py) from the record in
docs/results/external-comparison and judgly's committed per-item dumps; every item is rescored
and every plotted number checked against the record's result files before anything is written.
The external models were run as served by Ollama 0.35.0 (`/v1/systemone`), uncalibrated;
judgly's calibration was fitted by us on similar question types (see the caveats in the
README and in docs/methods.md).

## compare-tiers.svg

*What it shows:* accuracy (y) against ECE (x) on six test sets, for Nimble 9B, Tev1 4B, Tev1
0.8B and judgly's two packs with their default calibration (filled), and judgly raw, without
calibration (hollow, faint). Bars are 95% percentile bootstrap intervals over the tier's groups
of related items (1,000 resamples): for the confirm and final tiers they are the record's
(score_external.py); for typed-decisions and JevBench, which the record reports together as
bench with point values per source, each source was resampled on its own by this script (seed
20260930; compare-by-source.json). Each external model is scored on the items it answered and
judgly on every item: Tev1 refused 36 JevBench items longer than its context, so in the JevBench
panel the Tev1 points are on 195 of the 231 items and judgly's on all 231, different item sets
(the README and docs/methods.md also give judgly on Tev1's 195 items, where judgly Gemma 4 12B
is more accurate than on all 231). On the confirm tiers, judgly's defaults are the ones the
pre-registered confirmation selected from its read of this same tier (the temperature where it was
confirmed; H2, the 0.1.0 default, for Gemma 4 12B on general questions). *How to read it:* up and
to the left is better; points
whose intervals overlap are not clearly different, and the paired differences in
docs/methods.md are the sharper test. ECE is never negative, so its intervals lean upward near
0. *What it says:* judgly Gemma 4 12B is the most accurate or level with the most accurate on
every set. judgly's defaults have the lowest ECE on the confirm and general final tiers; on
stance final and typed-decisions Tev1 4B is as well or better calibrated. Without its
calibration judgly is among the worst calibrated on every set. The numbers: confirm, general: Nimble 9B accuracy 0.481 [0.461, 0.501], ECE 0.229 [0.210, 0.249]; Tev1 4B accuracy 0.479 [0.456, 0.501], ECE 0.142 [0.121, 0.165]; Tev1 0.8B accuracy 0.378 [0.356, 0.398], ECE 0.154 [0.136, 0.177]; judgly Gemma 4 12B, default accuracy 0.509 [0.488, 0.530], ECE 0.051 [0.041, 0.072]; judgly Qwen3-4B, default accuracy 0.484 [0.466, 0.506], ECE 0.047 [0.032, 0.068]. confirm, stance: Nimble 9B accuracy 0.595 [0.553, 0.640], ECE 0.264 [0.211, 0.296]; Tev1 4B accuracy 0.630 [0.603, 0.680], ECE 0.185 [0.152, 0.215]; Tev1 0.8B accuracy 0.451 [0.380, 0.481], ECE 0.304 [0.279, 0.364]; judgly Gemma 4 12B, default accuracy 0.655 [0.617, 0.698], ECE 0.052 [0.028, 0.079]; judgly Qwen3-4B, default accuracy 0.568 [0.526, 0.607], ECE 0.106 [0.078, 0.151]. final, general: Nimble 9B accuracy 0.674 [0.663, 0.684], ECE 0.129 [0.120, 0.139]; Tev1 4B accuracy 0.657 [0.647, 0.668], ECE 0.052 [0.044, 0.061]; Tev1 0.8B accuracy 0.504 [0.493, 0.516], ECE 0.086 [0.075, 0.096]; judgly Gemma 4 12B, default accuracy 0.737 [0.727, 0.746], ECE 0.020 [0.015, 0.030]; judgly Qwen3-4B, default accuracy 0.656 [0.645, 0.666], ECE 0.034 [0.025, 0.043]. final, stance: Nimble 9B accuracy 0.817 [0.796, 0.837], ECE 0.113 [0.096, 0.136]; Tev1 4B accuracy 0.826 [0.804, 0.845], ECE 0.041 [0.032, 0.066]; Tev1 0.8B accuracy 0.577 [0.557, 0.598], ECE 0.111 [0.090, 0.135]; judgly Gemma 4 12B, default accuracy 0.827 [0.807, 0.848], ECE 0.087 [0.068, 0.107]; judgly Qwen3-4B, default accuracy 0.778 [0.755, 0.799], ECE 0.093 [0.070, 0.115]. typed-decisions: Nimble 9B accuracy 0.702 [0.678, 0.726], ECE 0.052 [0.040, 0.077]; Tev1 4B accuracy 0.613 [0.588, 0.638], ECE 0.038 [0.026, 0.063]; Tev1 0.8B accuracy 0.434 [0.411, 0.456], ECE 0.186 [0.164, 0.209]; judgly Gemma 4 12B, default accuracy 0.700 [0.678, 0.720], ECE 0.028 [0.020, 0.049]; judgly Qwen3-4B, default accuracy 0.576 [0.549, 0.600], ECE 0.137 [0.122, 0.159]. JevBench, public items: Nimble 9B accuracy 0.779 [0.722, 0.836], ECE 0.118 [0.078, 0.163]; Tev1 4B accuracy 0.790 [0.729, 0.848], ECE 0.104 [0.068, 0.164]; Tev1 0.8B accuracy 0.672 [0.595, 0.740], ECE 0.093 [0.065, 0.169]; judgly Gemma 4 12B, default accuracy 0.844 [0.795, 0.887], ECE 0.067 [0.058, 0.117]; judgly Qwen3-4B, default accuracy 0.693 [0.617, 0.752], ECE 0.084 [0.054, 0.142].

## compare-reliability.svg

*What it shows:* reliability diagrams on the two confirm tiers for Nimble 9B, Tev1 4B and
judgly's two packs with their default calibration: for ten equal-width bins of the top
probability, the mean confidence (x) against the share of correct answers (y), with 95% Wilson
intervals; empty bins are left out. *How to read it:* on the dotted diagonal, confidence equals
accuracy; points below it are overconfident. The Wilson intervals treat items as independent,
which on stance (1,780 items in 70 linked groups) makes them too narrow. judgly's defaults here
were selected by the pre-registered confirmation from its read of this same tier (H2, the 0.1.0
default, stayed for Gemma 4 12B on general questions); nothing was fitted on it. With H2 instead,
ECE was 0.076 (Qwen3-4B) on general and 0.078 (Gemma 4 12B) and 0.183 (Qwen3-4B) on stance.
*What it says:* Nimble 9B and Tev1 4B, as served, are overconfident on these tiers. judgly's
defaults lie closer to the diagonal, but both are overconfident on stance above 0.5 (Gemma 4 12B
by 0.03 to 0.13 per bin, Qwen3-4B by 0.09 to 0.14), and Gemma 4 12B on general questions in the
0.5 to 0.6 bin (by 0.12, 796 items). ECE: general (n = 2,500): Nimble 9B 0.229, Tev1 4B 0.142, judgly Gemma 4 12B, default 0.051, judgly Qwen3-4B, default 0.047; stance (n = 1,780): Nimble 9B 0.264, Tev1 4B 0.185, judgly Gemma 4 12B, default 0.052, judgly Qwen3-4B, default 0.106.

## compare-timing.svg

*What it shows:* the median time per request (bar) and its 95th percentile (whisker) for each
system, from final/timing.json: 100 items (the first 50 of each confirm tier by the SHA-256 of
their id), one question per request, one request at a time, one uncounted warm-up request per
system, on an Apple M3 Max (64 GB). *How to read it:* shorter is faster. The bars do not measure
the same thing: judgly ran in-process through its Python API and asked each question in up to
four option orders; the Ollama models were asked over HTTP and read each question once. The
timings were taken after the comparison run and were not part of its frozen protocol. *What it
says:* medians: Tev1 0.8B 0.078 s; Tev1 4B 0.289 s; judgly Qwen3-4B 0.321 s; Nimble 9B 0.533 s; judgly Gemma 4 12B 0.960 s.
