"""
This script produces the figures and tables of the article.
"""
from figures_and_tables.blendedICU_stats import Blendedicu_stats

s = Blendedicu_stats()

# basic statistics on how many ingredients and drugnames were included
#s.medication_inclusion_stats()
# Flat statistics that make the Tables 1 and 2 in the manuscript
s.flat_stats()
# Frequency of usage for included drugs: percentage of icu stays where each 
# drug was given at least once.
med_freq = s.med_frequencies()
# number of datapoints per patient day for each timeseries
ts_freq = s.ts_frequencies()
# Figure 2 of the manuscript
s.ts_kde()
