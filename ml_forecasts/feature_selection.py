import pandas as pd  
import misc.constants as ct
import misc.excel_handler as excel_handler

from sklearn.feature_selection import mutual_info_regression

def perform_feature_selection(independent_data_df : pd.DataFrame, dependent_series, threshold):
    excel_handler.open_excel_without_saving(independent_data_df.describe().reset_index())
    print(independent_data_df.describe())
    get_pairwise_correlations(independent_data_df, threshold)
    get_mutual_information_scores(independent_data_df, dependent_series)
    
def get_pairwise_correlations(lagged_data_df : pd.DataFrame, threshold):
    correlation_matrix = lagged_data_df.corr()
    excel_handler.open_excel_without_saving(correlation_matrix.reset_index())
    print(correlation_matrix)
    high_correlation_pairs = [(correlation_matrix.index[row], correlation_matrix.columns[col])
                              for row in range(correlation_matrix.shape[0])
                              for col in range(correlation_matrix.shape[1])
                              if abs(correlation_matrix.iat[row, col]) > threshold and row != col]
    if len(high_correlation_pairs) == 0:
        print(f'No highly correlated data series found with a threshold of {threshold}.')
    else:
        print(f'Highly correlated data series are: {high_correlation_pairs}')
    
def get_mutual_information_scores(independent_variables, dependent_series):
    mutual_info = mutual_info_regression(independent_variables, dependent_series)
    mutual_info_df = pd.DataFrame(mutual_info, index=independent_variables.columns, columns=['Mutual Information'])
    mutual_info_df = mutual_info_df.sort_values(by='Mutual Information', ascending=False)
    mutual_info_df.reset_index(inplace=True)
    excel_handler.open_excel_without_saving(mutual_info_df)
    mutual_info_series = pd.Series(mutual_info, index=independent_variables.columns)
    print(mutual_info_series.sort_values(ascending=False))