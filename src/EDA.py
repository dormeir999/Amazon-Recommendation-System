import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List


def plot_target_bar(the_df, the_target):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns of plots

    # First subplot - Normalized distribution of the target
    target_val_counts_normalized = the_df[the_target].value_counts(normalize=True, dropna=False)
    axs[0].bar(target_val_counts_normalized.index, target_val_counts_normalized.values)
    axs[0].set_title(f'Normalized Distribution of {the_target}')
    axs[0].set_xlabel(the_target)
    axs[0].set_ylabel('Normalized Frequency')
    axs[0].set_xticks(target_val_counts_normalized.index)

    # Second subplot - Absolute count distribution of the target
    target_val_counts_absolute = the_df[the_target].value_counts(normalize=False, dropna=False)
    axs[1].bar(target_val_counts_absolute.index, target_val_counts_absolute.values)
    axs[1].set_title(f'Absolute Count Distribution of {the_target}')
    axs[1].set_xlabel(the_target)
    axs[1].set_ylabel('Count')
    axs[1].set_xticks(target_val_counts_absolute.index)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def get_mode_and_freq(the_df):

    # Define a top values dict for the top value and its frequency for each feature
    top_vals = {}
    for feature in the_df:
        # get the value_counts() for each feature
        top_val_feature = the_df[feature].value_counts(dropna=False, normalize=True)
        # the first index of the value_counts() is the top value for the feature
        top_val = top_val_feature.index[0]
        # the first value of the value_counts() is the frequency of the top value for the feature
        top_freq = top_val_feature.iloc[0]
        # assign both to a dictionary nested in each feature key of the first dictionary
        top_vals[feature] = {'top value':top_val, 'frequency':top_freq}
    # transform the dictionary into a dataframe, for an easier view of the results
    top_vals = pd.DataFrame(top_vals)
    # transpose the dataframe so that the indexes will be the feautre names, and the columns will be the top value and frequency
    top_vals = top_vals.T
    # sort the dataframe by descneding order of the frequncies
    top_vals = top_vals.sort_values('frequency', ascending=False)
    # Change the order of the columns, to first top value and than frequency
    top_vals = pd.concat([top_vals['top value'],top_vals['frequency']], axis=1)
    # print the result
    return top_vals


def get_correlation_stats(the_df, method='pearson', strong_corr_val=0.5, figsize=None, annot=True, rotation=45,
                          linewidths=.5, cmap=None, to_plot=True):
    # Set heatmap configs
    figsize = figsize if not figsize is None else (6.4,4.8)
    cmap = cmap if not cmap is None else sns.diverging_palette(220, 20, as_cmap=True)
    # Assign the table of pearson correlations of all features to a feature
    df_corr = the_df.corr(method=method, numeric_only=True)
    # Show strong correlations, i.e. higher than absolute value of strong_corr_val,
    # and don't show correlations of features to themselves with value of 1
    print(f"# {method} correlations:")
    display(df_corr)
    if to_plot:
        # plot the correlations table using a seaborn package heatmap
        plt.figure(figsize = figsize)
        sns.heatmap(df_corr, annot=annot, linewidths=linewidths, cmap=cmap)
        #sns.heatmap(df_corr, annot=True)
        # change the xticks rotation
        plt.xticks(rotation=rotation)
        # Make sure the text is shown completly
        plt.tight_layout()
        plt.show()
    print(f"# {method} correlations >= {strong_corr_val}:")
    df_corr_strong = df_corr[(df_corr.abs()>=strong_corr_val)&(df_corr.abs()!=1)]
    # Drop features with no strong correlations
    df_corr_strong = df_corr_strong.dropna(how='all', axis=1).dropna(how='all', axis=0)
    return df_corr_strong


def get_col_frequencies(the_df, col_name='userName', sort_index=True):
    if sort_index:
        the_users_frequencies = pd.concat([the_df[col_name].value_counts(normalize=False, dropna=False).sort_index(),
                                           the_df[col_name].value_counts(normalize=True, dropna=False).sort_index()], axis=1)
    else:
        the_users_frequencies = pd.concat([the_df[col_name].value_counts(normalize=False, dropna=False),
                                           the_df[col_name].value_counts(normalize=True, dropna=False)], axis=1)
    the_users_frequencies.columns = ['counts', 'pct']
    the_users_frequencies['cumsum_pct'] = the_users_frequencies['pct'].cumsum()
    return the_users_frequencies


def get_col_unique_counts_on_groupby_col(the_df, col='brand', groupby_col='itemName', dropna=False, sort_index=True):
    col_groupbied_col = the_df.groupby(groupby_col)[col].nunique(dropna=dropna)
    if sort_index:
        unique_counts_on_groupby = pd.concat([col_groupbied_col.value_counts(normalize=False, dropna=dropna).sort_index(),
                                              col_groupbied_col.value_counts(normalize=True, dropna=dropna).sort_index()], axis=1)
    else:
        unique_counts_on_groupby = pd.concat([col_groupbied_col.value_counts(normalize=False, dropna=dropna),
                                              col_groupbied_col.value_counts(normalize=True, dropna=dropna)], axis=1)
    unique_counts_on_groupby.columns = ['counts', 'pct']
    unique_counts_on_groupby['cumsum_pct'] = unique_counts_on_groupby['pct'].cumsum()
    if not isinstance(col, List):
        unique_counts_on_groupby = unique_counts_on_groupby.add_prefix(col + "_" + groupby_col + "_")
    return unique_counts_on_groupby