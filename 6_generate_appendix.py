from figures_and_tables import make_appendix

appendix_cat = make_appendix.get_appendix_cat()
appendix_med = make_appendix.get_appendix_med()
appendix_ts = make_appendix.get_appendix_ts()

appendix_med = appendix_med.loc[['abciximab', 'acyclovir',
                                 'alteplase', 'amiodarone',
                                 'amlodipine']]

s = make_appendix.latexify_appendix_med(appendix_med)
