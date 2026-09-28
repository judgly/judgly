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
