import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

import config


def load_eval_data(data_path: str) -> pd.DataFrame:
    with open(data_path, 'r') as f:
        model_evaluation_results = [dict(tmp_tuple) for tmp_tuple in
                                    {tuple(tmp_dict.items()) for tmp_dict in json.load(f)}]

    model_evaluation_results = list(map(lambda x: {
        'coherence': x.get('coherence_adjusted', None),
        'consistency': x.get('consistency_adjusted', None),
        'fluency': x.get('fluency_adjusted', None),
        'relevance': x.get('relevance_adjusted', None),
    }, model_evaluation_results))

    return pd.DataFrame(model_evaluation_results)


base_data = load_eval_data(f'{config.LOGS_DIR}/model_evaluation_results_base_scored.json')
ft_data = load_eval_data(f'{config.LOGS_DIR}/model_evaluation_results_ft_scored.json')
dpo_data = load_eval_data(f'{config.LOGS_DIR}/model_evaluation_results_dpo_scored.json')

combined_df = pd.concat([base_data, ft_data, dpo_data], axis=0, ignore_index=True)
scaler = MinMaxScaler()
scaler.fit(combined_df)

base_data[base_data.columns] = scaler.transform(base_data)
ft_data[base_data.columns] = scaler.transform(ft_data)
dpo_data[base_data.columns] = scaler.transform(dpo_data)

color_palette = ['#4E79A7', '#F28E2C', '#E15759']


def create_grouped_bar_chart(base_data, ft_data, dpo_data, title="Grouped Bar Chart"):
    bar_width = 0.25

    r1 = np.arange(len(base_data.columns))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.figure(figsize=(12, 6))

    plt.bar(r1, base_data.mean(), color=color_palette[0], width=bar_width, label='Base')
    plt.bar(r2, ft_data.mean(), color=color_palette[1], width=bar_width, label='Fine-tuned')
    plt.bar(r3, dpo_data.mean(), color=color_palette[2], width=bar_width, label='DPO')

    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title(title)
    plt.xticks([r + bar_width for r in range(len(base_data.columns))], base_data.columns)

    plt.legend()
    plt.show()


create_grouped_bar_chart(base_data, ft_data, dpo_data)


def create_grouped_boxplot(base_data, ft_data, dpo_data, title="Grouped Box Plot"):
    base_melt = base_data.melt(var_name='Column', value_name='Value')
    base_melt['DataFrame'] = 'Base'

    ft_melt = ft_data.melt(var_name='Column', value_name='Value')
    ft_melt['DataFrame'] = 'Fine-tuned'

    dpo_melt = dpo_data.melt(var_name='Column', value_name='Value')
    dpo_melt['DataFrame'] = 'DPO'

    combined_data = pd.concat([base_melt, ft_melt, dpo_melt], ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.set_palette(color_palette)
    sns.boxplot(x='Column', y='Value', hue='DataFrame', data=combined_data)

    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Values')

    plt.legend(title='')

    plt.show()


create_grouped_boxplot(base_data, ft_data, dpo_data)
