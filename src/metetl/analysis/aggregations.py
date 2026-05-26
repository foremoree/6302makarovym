import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os
import logging

logger = logging.getLogger("metetl")

def analyze_csv(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    def read_chunks(file_path):
        for chunk in pd.read_csv(file_path, chunksize=5000):
            yield chunk

    def filter_age(chunks):
        for chunk in chunks:
            if 'Department' not in chunk.columns or 'Object Begin Date' not in chunk.columns:
                continue
            subset = chunk[['Department', 'Object Begin Date']].dropna(subset=['Object Begin Date'])
            subset['Object Begin Date'] = pd.to_numeric(subset['Object Begin Date'], errors='coerce')
            subset = subset.dropna()
            if not subset.empty:
                yield subset

    acc = pd.DataFrame()
    all_data_frames = []
    for chunk in filter_age(read_chunks(csv_path)):
        group = chunk.groupby('Department')['Object Begin Date'].agg(['count', 'sum', lambda x: (x**2).sum()])
        acc = acc.add(group, fill_value=0)
        all_data_frames.append(chunk)

    if acc.empty:
        logger.error("Нет данных для анализа")
        return

    all_data = pd.concat(all_data_frames, ignore_index=True)
    acc['mean'] = acc['sum'] / acc['count']
    acc['var'] = acc['<lambda_0>'] / acc['count'] - acc['mean']**2
    acc['var'] = acc['var'].clip(lower=0)
    acc['std'] = np.sqrt(acc['var'])
    acc['sem'] = acc['std'] / np.sqrt(acc['count'])
    acc['ci_low'] = acc.apply(
        lambda row: stats.t.interval(0.95, df=max(int(row['count'])-1, 1), loc=row['mean'], scale=row['sem'])[0], axis=1)
    acc['ci_high'] = acc.apply(
        lambda row: stats.t.interval(0.95, df=max(int(row['count'])-1, 1), loc=row['mean'], scale=row['sem'])[1], axis=1)
    acc = acc.reset_index()
    acc.columns = ['Department', 'Count', 'Sum', 'SumSq', 'Mean_Age', 'Var', 'Std', 'SEM', 'CI_Low', 'CI_High']

    percentile_data = []
    for dept in acc['Department']:
        dept_values = all_data[all_data['Department'] == dept]['Object Begin Date'].values
        if len(dept_values) > 0:
            percentile_data.append({
                'Department': dept,
                'Scatter_Low': np.percentile(dept_values, 2.5),
                'Scatter_High': np.percentile(dept_values, 97.5),
                'All_Values': dept_values
            })
    percentiles = pd.DataFrame(percentile_data)

    final = acc.merge(percentiles[['Department', 'Scatter_Low', 'Scatter_High']], on='Department')
    final = final[final['Count'] >= 2].sort_values('Mean_Age', ascending=False)

    x = np.arange(len(final))
    means = final['Mean_Age'].values
    plt.figure(figsize=(12, 6))
    plt.bar(x, means, yerr=[means - final['CI_Low'].values, final['CI_High'].values - means],
            capsize=5, color='navy', alpha=0.7, label='95% доверительный интервал')
    plt.errorbar(x, means, yerr=[means - final['Scatter_Low'].values, final['Scatter_High'].values - means],
                 fmt='none', ecolor='red', capsize=3, label='95% интервал рассеяния')
    plt.xticks(x, final['Department'], rotation=90, fontsize=8)
    plt.ylabel('Год создания объекта')
    plt.title('Средний возраст объектов по отделам')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'department_age_bar.png'))
    plt.close()
    logger.info("Столбцовая диаграмма сохранена в %s", output_dir)

    max_std_dept = final.loc[final['Std'].idxmax(), 'Department']
    dept_data = percentiles[percentiles['Department'] == max_std_dept]['All_Values'].iloc[0]
    if len(dept_data) > 0:
        ages_series = pd.Series(dept_data).sort_values().reset_index(drop=True)
        window = max(3, int(len(ages_series) * 0.05))
        rolling_mean = ages_series.rolling(window=window, center=True).mean()

        plt.figure(figsize=(12, 5))
        plt.plot(ages_series, alpha=0.3, marker='.', markersize=2, label='Исходные данные')
        plt.plot(rolling_mean, color='red', linewidth=2, label=f'Скользящее среднее (окно {window})')
        plt.xlabel('Порядковый номер объекта')
        plt.ylabel('Год создания')
        plt.title(f'Отдел: {max_std_dept} – временной график')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{max_std_dept}_timeline.png'))
        plt.close()
        logger.info("Временной график для отдела %s сохранён", max_std_dept)