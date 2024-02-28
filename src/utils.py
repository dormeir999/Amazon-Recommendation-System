import os
import pandas as pd
from functools import partial
from typing import List

def print_function_name(func):
    """
    Decorator that prints the name of the function when it is called.

    Parameters:
    - func (callable): The function to be decorated.

    Returns:
    - callable: The wrapped function.
    """

    def wrapper(*args, **kwargs):
        print(f"{func.__name__}()")
        return func(*args, **kwargs)

    return wrapper

@print_function_name
def move_cols_to_first(the_df, first_cols):
    """
    Rearrange df columns so that first_cols will be first
    :param the_df: A dataframe
    :param first_cols: a list of columns to be first cols in the df
    :return: the dataframe rearranged
    """
    the_df = pd.concat([the_df[first_cols], the_df.loc[:, ~the_df.columns.isin(first_cols)]], axis=1)
    return the_df


@print_function_name
def import_data(filename: str = "winequalityN.csv", data_dir: str = "data/raw/") -> pd.DataFrame:
    """
    Imports data from a specified file located in a given directory.

    Parameters:
    - filename (str): Name of the file to be imported. Defaults to "winequalityN.csv".
    - data_dir (str): Relative path to the directory containing the data file. Defaults to "../data/raw/".

    Returns:
    - pd.DataFrame: DataFrame containing the imported data.
    """
    # Determine the path to the directory containing this script
    module_dir = os.getcwd()
    if os.path.split(os.getcwd())[-1] == 'src':
        os.chdir("..")
        module_dir = os.getcwd()
    # Construct the path to the data file
    data_dir = os.path.join(module_dir, data_dir)
    file_path = os.path.join(data_dir, filename)
    print("Attempting to load data from:", file_path)

    # Check if the file exists
    if not os.path.isfile(file_path):  # If doesn't exit, try the educative coding environment location
        file_path = "/usercode/" + filename
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

    # Read and return the data
    print("Data imported successfully!")
    return pd.read_csv(file_path)

@print_function_name
def transform_numeric_target_feature_to_binary(the_df: pd.DataFrame, target_col: str = 'quality',
                                               threshold: int = 7, keep_orig_target=False) -> pd.DataFrame:
    """
   Transform a numeric target feature in a DataFrame into a binary representation.

   Parameters:
   - the_df (pd.DataFrame): DataFrame containing the target feature.
   - target_col (str): Name of the target column. Defaults to 'quality'.
   - threshold (int): Threshold value for binarization. Defaults to 7.
   - keep_orig_target (bool): Flag to keep the original target as a copy. Defaults to False.

   Returns:
   - pd.DataFrame: Modified DataFrame with the target feature binarized.
   """
    if keep_orig_target:
        target_col_orig = target_col + "_orig"
        the_df[target_col_orig] = the_df[target_col].copy(deep=True)
    the_df[target_col] = (the_df[target_col] >= threshold) * 1

    return the_df


@print_function_name
def replace_columns_spaces_with_underscores(the_df):
    """
    Replace spaces in DataFrame column names with underscores.

    Parameters:
    - the_df (pd.DataFrame): DataFrame whose column names need modification.

    Returns:
    - pd.DataFrame: DataFrame with updated column names.
    """
    the_df.columns = the_df.columns.str.replace("_/_", "/")
    the_df.columns = the_df.columns.str.replace(" ", "_")
    return the_df


@print_function_name
def get_categorical_features_from_dtypes(the_df):
    """
    Identifies categorical features in a DataFrame based on their data types.

    This function assumes that categorical features are those with 'object' or 'category' data types.
    It prints a warning message to inform the user of this assumption.

    Parameters:
    - the_df (pd.DataFrame): The DataFrame from which to identify categorical features.

    Returns:
    - list: A list of column names that are considered categorical features.

    Note:
    This function assumes that all 'object' and 'category' dtype columns are categorical features.
    This may not always hold true, so use with caution and validate assumptions as needed.
    """
    print("** Warning: assuming categorical features as features with ['object', 'category'] dtypes **")
    categorical_features = the_df.select_dtypes(include=['object', 'category']).columns.tolist()
    return categorical_features

@print_function_name
def replace_categoricals_missing_values_with_NA_string(the_df, categorical_features, NA_string='NA'):
    """
    Replaces missing values in categorical features of a DataFrame with a specified string.

    This function fills missing values in specified categorical columns with a given string,
    such as 'NA'. It can also automatically identify categorical features based on their data types
    if not explicitly provided.

    Parameters:
    - the_df (pd.DataFrame): The DataFrame in which to replace missing values.
    - categorical_features (list, optional): List of categorical feature names. If None, the function
      will attempt to identify them based on data types.
    - NA_string (str): The string to use for replacing missing values. Defaults to 'NA'.

    Returns:
    - pd.DataFrame: The modified DataFrame with missing values in categorical columns replaced.
    """
    if categorical_features is None:
        categorical_features = get_categorical_features_from_dtypes(the_df)
    the_df[categorical_features] = the_df[categorical_features].fillna(NA_string)
    return the_df, categorical_features

@print_function_name
def imputate_missing_values(dataset_name, dataset, the_train_statistics, method='mean', n_rows_to_show=5, add_print=True):
    """
    Impute missing values in a dataset using the mean values from training statistics.

    Parameters:
    - dataset_name (str): Name of the dataset being processed.
    - dataset (pd.DataFrame): The dataset for imputation.
    - the_train_statistics (pd.DataFrame): Training statistics for reference.
    - n_rows_to_show (int): Number of rows to display for demonstration. Defaults to 5.
    - add_print (bool): Flag to print the demonstration rows. Defaults to True.

    Returns:
    - pd.DataFrame: Dataset with missing values imputed.
    """
    # get a dict of values to fill (mean) for missing values per each feature
    if method == 'mean':
        fill_values = the_train_statistics["mean"].dropna().to_dict()
    elif method == 'median':
        fill_values = the_train_statistics["50%"].dropna().to_dict()
    # get a dict of values to fill (top) for missing values per each features
    top_values = the_train_statistics["top"].dropna().to_dict()
    # update means with tops
    fill_values.update(top_values)
    # impute missing values, and save those indexes
    missing_indexes = dataset[dataset.isna().any(axis=1)].index
    if add_print:
        to_show = dataset.loc[missing_indexes][:n_rows_to_show]
        print(f"# First {n_rows_to_show} original {dataset_name} missing values:\n{to_show}\n")
    # fill the missing values in X_train with the mean values
    dataset = dataset.fillna(value=fill_values)
    current_missing_values = dataset.isna().sum()
    assert current_missing_values.sum() == 0, "There are still missing values in the dataset"
    if add_print:
        to_show = dataset.loc[missing_indexes][:n_rows_to_show]
        print(f"# First {n_rows_to_show} imputed {dataset_name} missing values:\n{to_show}\n")
        missing_values = dataset.isna().sum()
        print(f"# The number of missing values in columns in {dataset_name}:\n{missing_values}\n")
    return dataset

@print_function_name
def get_train_features_with_property(the_train_statistics, the_property):
    """
    Extracts a list of features from the train statistics DataFrame that have a specified property.

    Parameters:
    - the_train_statistics (pd.DataFrame): DataFrame containing training statistics.
    - the_property (str): The property based on which to filter features.

    Returns:
    - list: List of feature names that have the specified property.
    """
    the_features = the_train_statistics[the_train_statistics[the_property] == 1].index.tolist()
    return the_features


@print_function_name
def one_hot_encode_categoricals(the_df, the_train_statistics, drop_one=True,
                                categorical_features=None, categories_to_use_from_train=None):
    """
    Performs one-hot encoding on categorical features in a DataFrame. This function is designed to work
    for both training and non-training datasets, ensuring consistency in category encoding across different sets.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be encoded.
    - the_train_statistics (pd.DataFrame): DataFrame containing the training statistics, used for determining
      categorical features and categories used in training.
    - drop_one (bool): Flag to drop the first category in each feature to avoid multicollinearity. Defaults to True.
    - categorical_features (list, optional): List of categorical feature names. If None, features will be identified
      based on training statistics or data types.
    - categories_to_use_from_train (list, optional): List of category columns used in training for consistent encoding
      in non-training datasets.

    Returns:
    - tuple: A tuple containing two elements:
        1. Modified DataFrame with one-hot encoded categorical features.
        2. Updated training statistics DataFrame reflecting changes in the categorical features.

    Note:
    - For non-training datasets (validation, test, or new data), the function aligns the one-hot encoded categories
      with those used in training, dropping extra categories and adding missing categories as zero-filled columns.
    - For training datasets or when `categories_to_use_from_train` is None, the function updates the training
      statistics to include information about new categorical encodings.
    """
    # Set categorical_features; and categories_to_use_from_train for non-train dataset
    if categorical_features is None:
        # For non-train datasets:
        if 'is_categorical_to_drop' in the_train_statistics and 'is_category' in the_train_statistics:
            categorical_features = get_train_features_with_property(the_train_statistics, 'is_categorical_to_drop')
            categories_to_use_from_train = get_train_features_with_property(the_train_statistics, 'is_category')
        # We're in train, and categorical_features were not explicitly specified - which is inadvisable!
        else:
            categorical_features = get_categorical_features_from_dtypes(the_df)

    # Create the numerical one hot encodings - for both train and non-train - only if there are categories!
    if len(categorical_features) > 0:
        one_hot_encodings = pd.get_dummies(the_df[categorical_features], drop_first=drop_one)

        # if val or test or new data, filter categories to the ones that were used in train
        if not categories_to_use_from_train is None:
            # Extra categories in new data
            one_hot_encodings_extra = [cat for cat in one_hot_encodings.columns if cat not in categories_to_use_from_train]
            # Categories missing from new data
            missing_categories = [cat for cat in categories_to_use_from_train if cat not in one_hot_encodings.columns]

            # Drop extra categories
            one_hot_encodings = one_hot_encodings.drop(columns=one_hot_encodings_extra, errors='ignore')
            # Add missing categories in new data as zero columns
            for cat in missing_categories:
                one_hot_encodings[cat] = 0

            # Ensure the final one_hot_encodings has columns in the same order as categories_to_use_from_train
            one_hot_encodings = one_hot_encodings[categories_to_use_from_train]

        # Add the encodings to the dataset - both train and non-train datasets
        the_df = pd.concat([the_df, one_hot_encodings], axis=1)
        # Drop the original categorical_features
        the_df = the_df.drop(columns=categorical_features)
        # if train, update the_train_statistics
        if categories_to_use_from_train is None:
            train_categories = one_hot_encodings.columns.tolist()
            # Add new train cateogories statistics to train_statistics
            # Add proprty 'is_categorical_to_drop' to original cateogorical features
            the_train_statistics = add_binary_property_to_train_statistics(the_train_statistics,
                                                                           'is_categorical_to_drop',
                                                                           categorical_features)
            # Add proprty 'is_category' to newly created categories one-hot-encoding features
            the_train_statistics = add_new_features_statistics_to_train_statistics(the_df, the_train_statistics,
                                                                                   train_categories)
            the_train_statistics = add_binary_property_to_train_statistics(the_train_statistics, 'is_category',
                                                                           train_categories)
    elif categories_to_use_from_train is None:
        # In train but not categories - mark no categories are found in the the_train_statistics
        the_train_statistics['is_category'] = 0
        the_train_statistics['is_categorical_to_drop'] = 0


    return the_df, the_train_statistics

@print_function_name
def get_train_features_with_suffix(the_train_statistics, remove_suffix=True, the_suffix='is_outlier'):
    """
    Extracts a list of features from the train statistics DataFrame that have a specified suffix.

    Parameters:
    - the_train_statistics (pd.DataFrame): DataFrame containing training statistics.
    - remove_suffix (bool, optional): If True, removes the suffix from the feature names. Defaults to True.
    - the_suffix (str, optional): The suffix to filter features. Defaults to 'is_outlier'.

    Returns:
    - list: List of feature names with the specified suffix, optionally without the suffix.
    """
    the_train_statistics_features = the_train_statistics.index.to_list()
    feautres_with_suffix = [feature for feature in the_train_statistics_features if feature.endswith(the_suffix)]
    if remove_suffix:
        feautres_with_suffix = [feature.split("_" + the_suffix)[0] for feature in feautres_with_suffix]
    return feautres_with_suffix


def add_outlier_indicator(the_df: pd.DataFrame, the_feature: pd.Series, the_train_statistics: pd.DataFrame,
                          outlier_col_suffix='is_outlier', is_train=False) -> pd.DataFrame:
    """
    Add an outlier indicator column for a specific feature in the DataFrame.
    Outliers are defined as points distant more than 3 std from the mean.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be processed.
    - the_feature (str): Name of the feature to check for outliers.
    - the_train_statistics (pd.DataFrame): Training statistics for reference.
    - outlier_col_suffix (str): Suffix for the outlier column name. Defaults to 'is_outlier'.
    - is_train (bool): Flag to indicate if the dataset is training data. Defaults to False.

    Returns:
    - tuple: Modified DataFrame and the name of the new outlier column.
    """
    outlier_col = the_feature + "_" + outlier_col_suffix
    # Create is_outlier col if it doesn't exist, and fill with 0 (i.e. no outliers)
    if not outlier_col in the_df:
        the_df[outlier_col] = 0
    # The formula for calculating a z-score is: Z = (X - μ) / σ
    X = the_df[the_feature]
    mu = the_train_statistics.loc[the_feature, 'mean']
    sigma = the_train_statistics.loc[the_feature, 'std']
    obs_z_scores = (X - mu) / sigma
    # Get all rows with outliers
    outliers = obs_z_scores.abs() > 3
    # Mark outliers
    if sum(outliers) > 0:
        the_df.loc[outliers, outlier_col] = 1
    else:
        if is_train:  # if train and no outliers, drop column. if val or test, keep zeros.
            the_df = the_df.drop(columns=outlier_col)

    return the_df, outlier_col

@print_function_name
def add_outlier_indicators_on_features(the_df: pd.DataFrame, the_train_statistics: pd.DataFrame,
                                       X_train_numeric_features: List = None,
                                       outlier_col_suffix='is_outlier') -> pd.DataFrame:
    """
    Add outlier indicator columns for multiple features in the DataFrame.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be processed.
    - the_train_statistics (pd.DataFrame): Training statistics for reference.
    - X_train_numeric_features (List): List of numeric feature names to check for outliers.
    - outlier_col_suffix (str): Suffix for the outlier column names. Defaults to 'is_outlier'.

    Returns:
    - tuple: Modified DataFrame and a list of new outlier column names.
    """
    # If the_features not defined (first run - on train), filter out non-numeric features and run on all
    if not X_train_numeric_features:
        is_train = True
        categories = get_train_features_with_property(the_train_statistics, 'is_category')
        X_train_numeric_features = [col for col in the_df.columns if not col in categories]
    else:
        is_train = False  # either validation or test or new data
    new_outlier_cols = []
    for feature in X_train_numeric_features:
        the_df, new_outlier_col = add_outlier_indicator(the_df, feature, the_train_statistics,
                                                        outlier_col_suffix=outlier_col_suffix, is_train=is_train)
        new_outlier_cols = new_outlier_cols + [new_outlier_col]
    return the_df, new_outlier_cols



@print_function_name
def winsorize_outliers(the_df, the_train_statistics, percentiles=None, outlier_col_suffix='is_outlier'):
    """
    Apply winsorization to handle outliers in the DataFrame based on training statistics.

    Parameters:
    - the_df (pd.DataFrame): DataFrame containing the data.
    - the_train_statistics (pd.DataFrame): Training statistics with winsorization values.
    - percentiles (list): List of percentiles used for winsorization. Defaults to [.05, .95].

    Returns:
    - pd.DataFrame: The DataFrame after applying winsorization to outliers.
    """
    # extract original outlier call and is_outliers cols
    if percentiles is None:
        percentiles = [.05, .95]
    remove_suffix = False
    train_outlier_cols = get_train_features_with_suffix(the_train_statistics, the_suffix=outlier_col_suffix,
                                                        remove_suffix=remove_suffix)
    remove_suffix = True
    train_orig_outlier_cols = get_train_features_with_suffix(the_train_statistics, the_suffix=outlier_col_suffix,
                                                             remove_suffix=remove_suffix)
    # If outliers exist, continue to imputation
    if len(train_orig_outlier_cols) > 0:
        outlier_cols_mapper = dict(zip(train_orig_outlier_cols, train_outlier_cols))
        # extract winsorization values
        percentile_col_names = [str(col).split(".")[1].replace("0", "") + "%" for col in percentiles]
        winsorization_values = the_train_statistics.loc[train_orig_outlier_cols, percentile_col_names].T
        # replace min/max outliers with min_winzor/max_winzor
        for orig_col, is_outlier_col in outlier_cols_mapper.items():
            min_winzor = winsorization_values[orig_col].min()
            max_winzor = winsorization_values[orig_col].max()
            outlier_rows = the_df[is_outlier_col] == 1
            min_outliers = the_df[orig_col] <= min_winzor
            max_outliers = the_df[orig_col] >= max_winzor
            the_df.loc[(outlier_rows) & (min_outliers), orig_col] = min_winzor
            the_df.loc[(outlier_rows) & (max_outliers), orig_col] = max_winzor

    return the_df

@print_function_name
def add_binary_property_to_train_statistics(the_train_statistics, the_property, features_list_with_property):
    """
    Add a binary property to a group of features in the training statistics table.

    Parameters:
    - the_train_statistics (pd.DataFrame): DataFrame of training statistics.
    - the_property (str): Name of the binary property to be added.
    - features_list_with_property (list): List of features to which the property applies.

    Returns:
    - pd.DataFrame: Updated training statistics with the new property.
    """
    if not the_property in the_train_statistics:
        the_train_statistics[the_property] = 0
    if len(features_list_with_property) == 1:
        features_list_with_property = features_list_with_property[0]
    the_train_statistics.loc[features_list_with_property, the_property] = 1
    return the_train_statistics

@print_function_name
def add_new_features_statistics_to_train_statistics(the_train, the_train_statistics, new_features):
    """
    Add descriptive statistics for newly created features to the training statistics table.

    Parameters:
    - the_train (pd.DataFrame): The training dataset.
    - the_train_statistics (pd.DataFrame): DataFrame containing statistics for training data.
    - new_features (list): List of new feature names added to the dataset.

    Returns:
    - pd.DataFrame: Updated training statistics with new features included.
    """
    train_new_features_statistics = the_train[new_features].describe(include='all').T
    the_train_statistics = pd.concat([the_train_statistics, train_new_features_statistics], axis=0)
    return the_train_statistics


# Constants
METRICS_FILENAME = 'model_metrics.csv'
#target = 'quality' # Wine data target is quality
alpha = 0.01  # significance level
NA_string = 'NA'
drop_one = True # drop one category in one-hot-encoding
outlier_col_suffix = 'is_outlier'

