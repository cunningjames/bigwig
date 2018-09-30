from sklearn import ensemble
from functools import singledispatch

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _plot_importance(importance_list, feature_names, label, color_sign,
                     num_features=10, bar=True, horizontal=True,
                     color="lightgrey", fill="lightgrey"):
    imp_df = (pd.DataFrame(columns=['feature_name', 'importance'],
                           data=list(zip(feature_names, importance_list))).
              assign(the_sign=lambda df: df['importance'] > 0,
                     importance=lambda df: df['importance'].abs()).
              sort_values('importance', ascending=False).
              head(num_features))

    if color_sign:
        if horizontal:
            imp_plot = sns.barplot('importance', 'feature_name',
                                   'the_sign', imp_df, dodge=False,
                                   edgecolor=color)
        else:
            imp_plot = sns.barplot('feature_name', "importance",
                                   'the_sign', imp_df, dodge=False,
                                   edgecolor=color, orient="v")
        imp_plot.legend_.remove()
    else:
        if horizontal:
            imp_plot = sns.barplot('importance', 'feature_name',
                                   data=imp_df, dodge=False,
                                   edgecolor=color, color=fill)
        else:
            imp_plot = sns.barplot('feature_name', 'importance',
                                   data=imp_df, dodge=False,
                                   edgecolor=color, color=fill, orient="v")

    imp_plot.set_ylabel('')
    imp_plot.set_xlabel(f'Variable Importance ({label})')

    return imp_plot

@singledispatch
def vip(model, *args, **kwargs):
    pass

@vip.register(ensemble.gradient_boosting.GradientBoostingRegressor)
def _(model, feature_names, relative=True, num_features=10, bar=True,
      horizontal=True, color="lightgrey", fill="lightgrey"):

    feature_importances = list(model.feature_importances_ /
                               model.feature_importances_.max()
                           if relative else model.feature_importances_)

    return _plot_importance(feature_importances,
                            list(feature_names), "GBM", False,
                            num_features, bar, horizontal, color, fill)

@vip.register(ensemble.forest.RandomForestRegressor)
def _(model, feature_names, relative=True, num_features=10, bar=True,
      horizontal=True, color="lightgrey", fill="lightgrey"):

    feature_importances = list(model.feature_importances_ /
                               model.feature_importances_.max()
                           if relative else model.feature_importances_)

    return _plot_importance(feature_importances,
                            list(feature_names), "Random Forest", False,
                            num_features, bar, horizontal, color, fill)
