"""
Solar Anomaly Detection - Visualization Utilities
Generates publication-quality charts for EDA, anomaly detection results, and model comparisons.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def plot_missing_values(df, save_path=None):
    """
    Visualize missing value counts per column.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    missing = df.isnull().sum()
    bars = ax.bar(missing.index, missing.values, color='#4A90E2', edgecolor='black', alpha=0.85)
    ax.set_title("Missing Values Count Across Features", fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel("Count of Missing Entries", fontsize=11)
    ax.set_xlabel("Feature Name", fontsize=11)
    plt.xticks(rotation=30, ha='right')
    ax.set_ylim(0, max(1, missing.max() * 1.2))
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.05, f"{int(h)}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_feature_distributions(df, features, save_path=None):
    """
    Subplots showing histograms and KDEs for key numerical features.
    """
    n_feats = len(features)
    ncols = 3
    nrows = (n_feats + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, color='#2B5B84', edgecolor='none', alpha=0.6)
        ax.set_title(f"Distribution of {col}", fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Density / Count", fontsize=10)

    # Hide extra subplots
    for j in range(n_feats, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Sensor Feature Distributions (Solar Plant 1)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_correlation_heatmap(df, features, save_path=None):
    """
    Compute and plot correlation matrix across features.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold', pad=12)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_dc_vs_ac(df, anomaly_col=None, save_path=None):
    """
    Scatter plot of DC Power vs AC Power with optional anomaly highlighting.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if anomaly_col and anomaly_col in df.columns:
        normal_data = df[df[anomaly_col] == 1]
        anomaly_data = df[df[anomaly_col] == -1]
        ax.scatter(normal_data['DC_POWER'], normal_data['AC_POWER'],
                   c='#4A90E2', alpha=0.4, s=15, label='Normal Observation')
        ax.scatter(anomaly_data['DC_POWER'], anomaly_data['AC_POWER'],
                   c='#E74C3C', alpha=0.8, s=30, marker='x', label='Detected Anomaly')
    else:
        ax.scatter(df['DC_POWER'], df['AC_POWER'], c='#2E86AB', alpha=0.5, s=15)
        
    ax.set_title("DC Power vs AC Power Conversion Relationship", fontsize=14, fontweight='bold')
    ax.set_xlabel("DC Power (kW)", fontsize=11)
    ax.set_ylabel("AC Power (kW)", fontsize=11)
    if anomaly_col and anomaly_col in df.columns:
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_irradiation_vs_ac(df, anomaly_col=None, save_path=None):
    """
    Irradiation vs AC Power plot with anomalies highlighted.
    Key relationship: Under high irradiation, low AC power may indicate anomalies/shading/sensor lag.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if anomaly_col and anomaly_col in df.columns:
        normal = df[df[anomaly_col] == 1]
        anomaly = df[df[anomaly_col] == -1]
        ax.scatter(normal['IRRADIATION'], normal['AC_POWER'],
                   c='#27AE60', alpha=0.35, s=15, label='Normal (Generation aligned with Sun)')
        ax.scatter(anomaly['IRRADIATION'], anomaly['AC_POWER'],
                   c='#E74C3C', alpha=0.85, s=35, marker='x', label='Anomaly (Deviation from Solar Curve)')
    else:
        ax.scatter(df['IRRADIATION'], df['AC_POWER'], c='#27AE60', alpha=0.4, s=15)

    ax.set_title("Solar Irradiation vs AC Power Output", fontsize=14, fontweight='bold')
    ax.set_xlabel("Solar Irradiation ($W/m^2$ or normalized sensor index)", fontsize=11)
    ax.set_ylabel("AC Power (kW)", fontsize=11)
    if anomaly_col and anomaly_col in df.columns:
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_temperature_relationship(df, save_path=None):
    """
    Ambient Temperature vs Module Temperature.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df['AMBIENT_TEMPERATURE'],
        df['MODULE_TEMPERATURE'],
        c=df['IRRADIATION'],
        cmap='viridis',
        alpha=0.6,
        s=15
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Solar Irradiation", fontsize=10)
    ax.set_title("Ambient Temperature vs Module Temperature", fontsize=14, fontweight='bold')
    ax.set_xlabel("Ambient Temperature (°C)", fontsize=11)
    ax.set_ylabel("Module (Panel) Temperature (°C)", fontsize=11)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_timeseries_anomalies(df, date_col='DATE_TIME', value_col='AC_POWER', anomaly_col='anomaly_iforest', sample_days=7, save_path=None):
    """
    Time-series curve of AC power over sample days with detected anomalies marked.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    df_sorted = df.sort_values(by=date_col)
    
    if sample_days:
        start_date = df_sorted[date_col].min()
        end_date = start_date + pd.Timedelta(days=sample_days)
        df_plot = df_sorted[(df_sorted[date_col] >= start_date) & (df_sorted[date_col] <= end_date)]
    else:
        df_plot = df_sorted

    ax.plot(df_plot[date_col], df_plot[value_col], color='#2980B9', linewidth=1.2, label='AC Power (kW)', alpha=0.8)
    
    if anomaly_col in df_plot.columns:
        anom = df_plot[df_plot[anomaly_col] == -1]
        ax.scatter(anom[date_col], anom[value_col], color='#E74C3C', s=45, zorder=5, label='Detected Anomaly', edgecolors='black', linewidth=0.5)

    title_suffix = f" (Sample First {sample_days} Days)" if sample_days else ""
    ax.set_title(f"Solar Generation Over Time with Anomalies Highlighted{title_suffix}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Timestamp (15-min intervals)", fontsize=11)
    ax.set_ylabel(f"{value_col} (kW)", fontsize=11)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    plt.xticks(rotation=20)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_dbscan_clusters(df, x_col='DC_POWER', y_col='AC_POWER', cluster_col='cluster_dbscan', save_path=None):
    """
    Visualization of DBSCAN clusters and noise points (-1).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if cluster_col not in df.columns:
        return fig

    noise = df[df[cluster_col] == -1]
    clustered = df[df[cluster_col] != -1]

    # Plot clustered points
    scatter = ax.scatter(clustered[x_col], clustered[y_col], c=clustered[cluster_col], cmap='tab10', alpha=0.4, s=15, label='Normal Dense Clusters')
    # Highlight noise outliers
    ax.scatter(noise[x_col], noise[y_col], c='red', marker='D', s=60, edgecolors='black', label=f'DBSCAN Noise (Outliers, n={len(noise)})')

    ax.set_title(f"DBSCAN Density-Based Clustering ({x_col} vs {y_col})", fontsize=14, fontweight='bold')
    ax.set_xlabel(f"{x_col} (kW)", fontsize=11)
    ax.set_ylabel(f"{y_col} (kW)", fontsize=11)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_model_comparison_bars(comparison_df, save_path=None):
    """
    Bar chart comparing anomaly counts and percentages detected by Isolation Forest and DBSCAN.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Anomaly Count
    count_col = 'Anomaly Count' if 'Anomaly Count' in comparison_df.columns else ('Detected Anomalies' if 'Detected Anomalies' in comparison_df.columns else comparison_df.columns[2])
    bars1 = ax1.bar(comparison_df['Algorithm'], comparison_df[count_col], color=['#E67E22', '#3498DB'], edgecolor='black', width=0.5)
    ax1.set_title("Detected Anomalies / Noise Points (Count)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Number of Observations", fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.02 + 10), f"{int(yval):,}", ha='center', va='bottom', fontweight='bold')

    # Anomaly Percentage
    bars2 = ax2.bar(comparison_df['Algorithm'], comparison_df['Anomaly Percentage (%)'], color=['#E67E22', '#3498DB'], edgecolor='black', width=0.5)
    ax2.set_title("Detected Anomalies / Noise Points (Percentage)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Percentage of Total Observations (%)", fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.02 + 0.05), f"{yval:.4f}%", ha='center', va='bottom', fontweight='bold')

    plt.suptitle("Model Comparison: Isolation Forest vs DBSCAN", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_anomaly_distribution_pie(n_normal, n_anomaly, model_name="Isolation Forest", save_path=None):
    """
    Pie chart showing proportion of Normal vs Anomaly observations.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = ['Normal Observations', 'Detected Anomalies']
    sizes = [n_normal, n_anomaly]
    colors = ['#2ECC71', '#E74C3C']
    explode = (0, 0.1)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.2f%%', shadow=False, startangle=140,
           textprops={'fontsize': 11, 'fontweight': 'bold'},
           wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title(f"{model_name}: Normal vs Anomaly Distribution", fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig
