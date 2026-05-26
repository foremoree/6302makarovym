
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def read_chunks(file_path: str):
    chunks = pd.read_csv(file_path, chunksize=5000)
    for chunk in chunks:
        yield chunk


def filter_age(chunks):
    for chunk in chunks:
        needed_cols = pd.Series(['Department', 'Object Begin Date'])
        if not needed_cols.isin(chunk.columns).all():
            continue
        filtered = chunk[needed_cols].copy()
        filtered = filtered.dropna(subset=['Object Begin Date'])
        filtered['Object Begin Date'] = pd.to_numeric(filtered['Object Begin Date'], errors='coerce')
        filtered = filtered.dropna(subset=['Object Begin Date'])
        if not filtered.empty:
            yield filtered


def aggregate_chunk_stats(chunks):
    for chunk in chunks:
        chunk_stats = chunk.groupby('Department')['Object Begin Date'].agg([
            'count',
            'sum',
            ('sum_sq', lambda x: (x ** 2).sum())
        ])
        yield chunk_stats, chunk[['Department', 'Object Begin Date']]


def combine_all_stats(chunks):
    accumulator = pd.DataFrame()
    all_data = pd.DataFrame(columns=pd.Index(['Department', 'Object Begin Date']))

    for chunk_stats, chunk_data in chunks:
        accumulator = accumulator.add(chunk_stats, fill_value=0)
        all_data = pd.concat([all_data, chunk_data], ignore_index=True)

    accumulator['mean'] = accumulator['sum'] / accumulator['count']
    accumulator['variance'] = (accumulator['sum_sq'] / accumulator['count']) - (accumulator['mean'] ** 2)
    accumulator['variance'] = accumulator['variance'].clip(lower=0)
    accumulator['std'] = np.sqrt(accumulator['variance'])
    accumulator['sem'] = accumulator['std'] / np.sqrt(accumulator['count'])

    accumulator['ci_low'] = accumulator.apply(
        lambda row: stats.t.interval(0.95, df=row['count'] - 1, loc=row['mean'], scale=row['sem'])[0]
        if row['count'] > 1 else row['mean'], axis=1
    )
    accumulator['ci_high'] = accumulator.apply(
        lambda row: stats.t.interval(0.95, df=row['count'] - 1, loc=row['mean'], scale=row['sem'])[1]
        if row['count'] > 1 else row['mean'], axis=1
    )

    accumulator = accumulator.reset_index()
    accumulator.columns = pd.Index([
        'Department', 'Count', 'Sum', 'Sum_Sq', 'Mean_Age',
        'Variance', 'Std', 'SEM', 'CI_Low', 'CI_High'
    ])

    percentiles_df = pd.DataFrame(columns=pd.Index(['Department', 'Scatter_Low', 'Scatter_High', 'All_Values']))

    for dept in accumulator['Department']:
        dept_values = all_data[all_data['Department'] == dept]['Object Begin Date'].values
        if len(dept_values) > 0:
            new_row = pd.DataFrame(
                {
                    'Department': pd.Series([dept]),
                    'Scatter_Low': pd.Series([np.percentile(dept_values, 2.5)]),
                    'Scatter_High': pd.Series([np.percentile(dept_values, 97.5)]),
                    'All_Values': pd.Series([dept_values])
                }
            )
            percentiles_df = pd.concat([percentiles_df, new_row], ignore_index=True)

    return accumulator, percentiles_df


def plot_main_analysis(stats_df, percentiles_df):
    df_results = stats_df.merge(percentiles_df[['Department', 'Scatter_Low', 'Scatter_High']], on='Department')
    df_results = df_results[df_results['Count'] >= 2].sort_values('Mean_Age', ascending=False)

    max_std_dept = df_results.loc[df_results['Std'].idxmax(), 'Department']
    max_std_data = percentiles_df[percentiles_df['Department'] == max_std_dept]['All_Values'].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(df_results))
    means = df_results['Mean_Age'].values
    depts = df_results['Department'].values

    ci_low_arr = df_results['CI_Low'].values
    ci_high_arr = df_results['CI_High'].values
    scatter_low_arr = df_results['Scatter_Low'].values
    scatter_high_arr = df_results['Scatter_High'].values

    ci_errors_low = pd.Series(np.abs(means - ci_low_arr))
    ci_errors_high = pd.Series(np.abs(ci_high_arr - means))
    ci_errors = pd.concat([ci_errors_low, ci_errors_high], axis=1).T.values

    scatter_errors_low = pd.Series(np.abs(means - scatter_low_arr))
    scatter_errors_high = pd.Series(np.abs(scatter_high_arr - means))
    scatter_errors = pd.concat([scatter_errors_low, scatter_errors_high], axis=1).T.values

    ax.bar(x_pos, means, yerr=ci_errors, capsize=5,
           label='Средний возраст с 95% ДИ', alpha=0.7, color='navy')
    ax.errorbar(x_pos, means, yerr=scatter_errors, fmt='none', ecolor='red',
                capsize=3, label='95% интервал рассеяния', linewidth=2, alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(depts, rotation=90, fontsize=8)
    ax.set_ylabel('Год создания объекта', fontsize=12)
    ax.set_xlabel('Отдел музея', fontsize=12)
    ax.set_title('Возраст объектов по отделам', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    if len(max_std_data) > 0:
        ages_series = pd.Series(max_std_data).sort_values().reset_index(drop=True)
        window = max(3, int(len(ages_series) * 0.05))
        rolling_mean = ages_series.rolling(window=window, center=True).mean()

        plt.figure(figsize=(12, 5))
        plt.plot(ages_series.values, alpha=0.3, marker='.', markersize=2,
                 label='Исходные данные', color='blue')
        plt.plot(rolling_mean, color='red', linewidth=2,
                 label=f'Скользящее среднее (окно={window})')
        plt.xlabel('Порядковый номер объекта', fontsize=10)
        plt.ylabel('Год создания объекта', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main(file_path):
    stats_df, percentiles_df = combine_all_stats(
        aggregate_chunk_stats(
            filter_age(
                read_chunks(file_path)
            )
        )
    )
    plot_main_analysis(stats_df, percentiles_df)


if __name__ == "__main__":
    main("MetObjects.csv")
