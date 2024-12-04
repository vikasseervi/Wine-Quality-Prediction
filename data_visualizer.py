import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    @staticmethod
    def plot_quality_distribution(data):
        sns.catplot(x='quality', data=data, kind='count')
        plt.show()

    @staticmethod
    def plot_feature_vs_quality(data, feature):
        plt.figure(figsize=(5, 5))
        sns.barplot(x='quality', y=feature, data=data, ci=None)
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(data):
        correlation = data.corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
        plt.show()
