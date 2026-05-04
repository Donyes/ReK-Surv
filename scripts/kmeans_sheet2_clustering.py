from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from matplotlib.lines import Line2D
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "树龄",
    "树势",
    "东西冠幅",
    "南北冠幅",
    "树干高",
    "树高",
    "品种",
    "周围是否有病树",
]
NUMERIC_COLUMNS = [
    "树龄",
    "树势",
    "东西冠幅",
    "南北冠幅",
    "树干高",
    "树高",
    "周围是否有病树",
]
CATEGORICAL_COLUMNS = ["品种"]
POSITION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
FEATURE_PLOT_LABELS = {
    "树龄": "Age",
    "树势": "Vigor",
    "东西冠幅": "Canopy EW",
    "南北冠幅": "Canopy NS",
    "树干高": "Trunk H",
    "树高": "Tree H",
    "品种": "Variety",
    "周围是否有病树": "Nearby diseased",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run K-means clustering for Sheet2 trees in hlb_dataset.xlsx.")
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        default=Path("data/hlb_dataset.xlsx"),
        help="Path to the source workbook.",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        default="Sheet2",
        help="Worksheet that contains the 123 trees.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/kmeans_sheet2"),
        help="Directory used to save clustering outputs.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="Maximum K tested by the elbow method.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for K-means.",
    )
    return parser.parse_args()


def build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", build_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def select_elbow_k(ks: list[int], inertias: list[float]) -> tuple[int, list[float]]:
    if len(ks) != len(inertias):
        raise ValueError("ks and inertias must have the same length.")
    if len(ks) < 3:
        raise ValueError("At least three K values are required for elbow detection.")

    start = np.array([ks[0], inertias[0]], dtype=float)
    end = np.array([ks[-1], inertias[-1]], dtype=float)
    line_dx = end[0] - start[0]
    line_dy = end[1] - start[1]
    line_length = float(np.hypot(line_dx, line_dy))
    if line_length == 0:
        return ks[0], [0.0 for _ in ks]

    distances: list[float] = []
    for k, inertia in zip(ks, inertias):
        numerator = abs(line_dx * (start[1] - inertia) - (start[0] - k) * line_dy)
        distances.append(float(numerator / line_length))

    best_index = int(np.argmax(distances))
    return int(ks[best_index]), distances


def run_elbow_method(
    transformed_features: np.ndarray,
    max_k: int,
    random_state: int,
) -> tuple[list[int], list[float], int, list[float]]:
    upper_k = min(max_k, len(transformed_features) - 1)
    if upper_k < 3:
        raise ValueError("Not enough samples to evaluate elbow method.")

    ks = list(range(1, upper_k + 1))
    inertias: list[float] = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        model.fit(transformed_features)
        inertias.append(float(model.inertia_))

    best_k, distances = select_elbow_k(ks, inertias)
    return ks, inertias, best_k, distances


def build_cluster_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    grouped = result_df.groupby("聚类类别", sort=True)
    cluster_counts = grouped.size()
    cluster_ratios = (cluster_counts / len(result_df)).round(4)

    summary = pd.DataFrame(
        {
            "样本数": cluster_counts,
            "占比": cluster_ratios,
            "树龄均值": grouped["树龄"].mean().round(3),
            "树势均值": grouped["树势"].mean().round(3),
            "东西冠幅均值": grouped["东西冠幅"].mean().round(3),
            "南北冠幅均值": grouped["南北冠幅"].mean().round(3),
            "树干高均值": grouped["树干高"].mean().round(3),
            "树高均值": grouped["树高"].mean().round(3),
            "周围有病树占比": grouped["周围是否有病树"].mean().round(4),
            "品种众数": grouped["品种"].agg(
                lambda series: series.mode().iloc[0] if not series.mode().empty else np.nan
            ),
        }
    )

    if "观测期内发病" in result_df.columns:
        summary["观测期内发病率(仅统计)"] = grouped["观测期内发病"].mean().round(4)

    return summary.reset_index()


def build_cluster_profile_tables(result_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_profile = result_df.groupby("聚类类别", sort=True)[FEATURE_COLUMNS].mean().round(3)

    normalized_frame = result_df[FEATURE_COLUMNS].astype(float).copy()
    for column in FEATURE_COLUMNS:
        column_min = float(normalized_frame[column].min())
        column_max = float(normalized_frame[column].max())
        if column_max == column_min:
            normalized_frame[column] = 0.0
        else:
            normalized_frame[column] = (normalized_frame[column] - column_min) / (column_max - column_min)

    normalized_profile = (
        normalized_frame.assign(聚类类别=result_df["聚类类别"].to_numpy())
        .groupby("聚类类别", sort=True)[FEATURE_COLUMNS]
        .mean()
        .round(4)
    )
    return raw_profile, normalized_profile


def build_cluster_disease_table(result_df: pd.DataFrame) -> pd.DataFrame:
    disease_col = "观测期内发病"
    cluster_col = "聚类类别"
    if disease_col not in result_df.columns:
        return pd.DataFrame()

    counts = (
        result_df.groupby([cluster_col, disease_col], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )
    counts.columns = ["未发病数量", "发病数量"]
    totals = counts.sum(axis=1)
    disease_rate = (counts["发病数量"] / totals).round(4)

    comparison_df = counts.copy()
    comparison_df["总数"] = totals.astype(int)
    comparison_df["发病率"] = disease_rate
    return comparison_df.reset_index()


def run_nested_kmeans_by_primary_cluster(
    result_df: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nested_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, int | float | str]] = []

    for primary_cluster_id in sorted(result_df["聚类类别"].unique().tolist()):
        subset_df = result_df[result_df["聚类类别"] == primary_cluster_id].copy()
        subset_features = subset_df[FEATURE_COLUMNS].copy()
        subset_preprocessor = build_preprocessor()
        subset_transformed = subset_preprocessor.fit_transform(subset_features)

        nested_model = KMeans(n_clusters=2, random_state=random_state, n_init=20)
        subcluster_labels = nested_model.fit_predict(subset_transformed) + 1

        subset_df["子聚类类别"] = subcluster_labels
        subset_df["二级聚类标签"] = [
            f"{int(primary_cluster_id)}-{int(subcluster_id)}" for subcluster_id in subcluster_labels
        ]
        nested_frames.append(subset_df)

        disease_counts = (
            subset_df.groupby(["子聚类类别", "观测期内发病"], sort=True)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=[0, 1], fill_value=0)
        )

        for subcluster_id in sorted(disease_counts.index.tolist()):
            no_disease_count = int(disease_counts.loc[subcluster_id, 0])
            disease_count = int(disease_counts.loc[subcluster_id, 1])
            total_count = no_disease_count + disease_count
            summary_rows.append(
                {
                    "一级聚类类别": int(primary_cluster_id),
                    "子聚类类别": int(subcluster_id),
                    "二级聚类标签": f"{int(primary_cluster_id)}-{int(subcluster_id)}",
                    "样本数": total_count,
                    "未发病数量": no_disease_count,
                    "发病数量": disease_count,
                    "发病率": round(disease_count / total_count, 4),
                }
            )

    nested_result_df = pd.concat(nested_frames, axis=0).sort_index()
    nested_summary_df = pd.DataFrame(summary_rows).sort_values(
        ["一级聚类类别", "子聚类类别"],
        ignore_index=True,
    )
    return nested_result_df, nested_summary_df


def parse_tree_position(tree_id: object) -> tuple[float, float]:
    match = POSITION_PATTERN.fullmatch(str(tree_id))
    if match is None:
        raise ValueError(f"Tree id '{tree_id}' does not match the expected x-y coordinate pattern.")
    return float(match.group(1)), float(match.group(2))


def save_elbow_plot(ks: list[int], inertias: list[float], best_k: int, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker="o", linewidth=2)
    plt.axvline(best_k, color="tab:red", linestyle="--", label=f"Selected K = {best_k}")
    plt.xticks(ks)
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("SSE / Inertia")
    plt.title("Elbow Method for Sheet2 Tree Clustering")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_cluster_profile_heatmap(
    raw_profile: pd.DataFrame,
    normalized_profile: pd.DataFrame,
    output_path: Path,
) -> None:
    heatmap_values = normalized_profile.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    image = ax.imshow(heatmap_values, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    plot_labels = [FEATURE_PLOT_LABELS[column] for column in FEATURE_COLUMNS]
    ax.set_xticks(np.arange(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(plot_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(raw_profile.index)))
    ax.set_yticklabels([f"Cluster {cluster_id}" for cluster_id in raw_profile.index.tolist()])
    ax.set_title("Cluster Feature Heatmap (color: normalized level, text: original mean)")

    for row_index in range(heatmap_values.shape[0]):
        for col_index in range(heatmap_values.shape[1]):
            original_value = float(raw_profile.iloc[row_index, col_index])
            text_color = "white" if heatmap_values[row_index, col_index] >= 0.58 else "black"
            ax.text(
                col_index,
                row_index,
                f"{original_value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative feature level (0-1)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_cluster_profile_radar(normalized_profile: pd.DataFrame, output_path: Path) -> None:
    labels = FEATURE_COLUMNS
    plot_labels = [FEATURE_PLOT_LABELS[column] for column in labels]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    colors = plt.get_cmap("tab10", len(normalized_profile.index))

    for index, cluster_id in enumerate(normalized_profile.index.tolist()):
        values = normalized_profile.loc[cluster_id, labels].to_numpy(dtype=float).tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, color=colors(index), label=f"Cluster {cluster_id}")
        ax.fill(angles, values, color=colors(index), alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(plot_labels)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0.2, 1.0, 5))
    ax.set_yticklabels([f"{tick:.1f}" for tick in np.linspace(0.2, 1.0, 5)])
    ax.set_title("Cluster Feature Radar Chart (normalized)", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_pca_scatter_plot(
    transformed_features: np.ndarray,
    labels: np.ndarray,
    model: KMeans,
    output_path: Path,
) -> tuple[float, float]:
    pca = PCA(n_components=2)
    projected_points = pca.fit_transform(transformed_features)
    projected_centers = pca.transform(model.cluster_centers_)
    unique_labels = sorted(np.unique(labels).tolist())
    colors = plt.get_cmap("tab10", len(unique_labels))

    plt.figure(figsize=(8, 6))
    for index, cluster_id in enumerate(unique_labels):
        cluster_mask = labels == cluster_id
        plt.scatter(
            projected_points[cluster_mask, 0],
            projected_points[cluster_mask, 1],
            s=45,
            alpha=0.8,
            color=colors(index),
            label=f"Cluster {cluster_id}",
        )

    plt.scatter(
        projected_centers[:, 0],
        projected_centers[:, 1],
        s=220,
        marker="X",
        color="black",
        linewidths=1.0,
        label="Centroids",
    )
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("K-means Clusters Projected by PCA")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    explained_variance = pca.explained_variance_ratio_
    return float(explained_variance[0]), float(explained_variance[1])


def save_cluster_size_plot(labels: np.ndarray, output_path: Path) -> dict[int, int]:
    cluster_ids, cluster_counts = np.unique(labels, return_counts=True)
    cluster_count_map = {int(cluster_id): int(count) for cluster_id, count in zip(cluster_ids, cluster_counts)}

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar([f"Cluster {cluster_id}" for cluster_id in cluster_ids], cluster_counts, color=plt.cm.Set2.colors[: len(cluster_ids)])
    plt.ylabel("Tree Count")
    plt.title("Cluster Sizes")
    plt.grid(axis="y", alpha=0.25)
    for bar, count in zip(bars, cluster_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(int(count)), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    return cluster_count_map


def save_cluster_vs_disease_plot(comparison_df: pd.DataFrame, output_path: Path) -> None:
    if comparison_df.empty:
        return

    cluster_ids = comparison_df["聚类类别"].astype(int).to_list()
    healthy_counts = comparison_df["未发病数量"].astype(int).to_numpy()
    diseased_counts = comparison_df["发病数量"].astype(int).to_numpy()
    disease_rates = comparison_df["发病率"].astype(float).to_numpy()
    x = np.arange(len(cluster_ids))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(x, healthy_counts, color="#7FBF7B", label="No disease")
    ax.bar(x, diseased_counts, bottom=healthy_counts, color="#EF8A62", label="Disease")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {cluster_id}" for cluster_id in cluster_ids])
    ax.set_ylabel("Tree count")
    ax.set_title("Cluster Category vs Disease Status")
    ax.grid(axis="y", alpha=0.25)

    rate_ax = ax.twinx()
    rate_ax.plot(x, disease_rates, color="#2C3E50", marker="o", linewidth=2, label="Disease rate")
    rate_ax.set_ylim(0, 1)
    rate_ax.set_ylabel("Disease rate")

    for x_pos, rate in zip(x, disease_rates):
        rate_ax.text(x_pos, rate + 0.03, f"{rate:.1%}", ha="center", va="bottom", color="#2C3E50", fontsize=10)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = rate_ax.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_subcluster_vs_disease_plot(comparison_df: pd.DataFrame, output_path: Path) -> None:
    if comparison_df.empty:
        return

    subcluster_labels = comparison_df["二级聚类标签"].astype(str).to_list()
    healthy_counts = comparison_df["未发病数量"].astype(int).to_numpy()
    diseased_counts = comparison_df["发病数量"].astype(int).to_numpy()
    disease_rates = comparison_df["发病率"].astype(float).to_numpy()
    x = np.arange(len(subcluster_labels))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(x, healthy_counts, color="#8FD19E", label="No disease")
    ax.bar(x, diseased_counts, bottom=healthy_counts, color="#F28E6B", label="Disease")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Subcluster {label}" for label in subcluster_labels])
    ax.set_ylabel("Tree count")
    ax.set_title("Secondary Subcluster vs Disease Status")
    ax.grid(axis="y", alpha=0.25)

    for divider in [1.5, 3.5]:
        ax.axvline(divider, color="#BDBDBD", linestyle="--", linewidth=1)

    rate_ax = ax.twinx()
    rate_ax.plot(x, disease_rates, color="#1F3B5D", marker="o", linewidth=2, label="Disease rate")
    rate_ax.set_ylim(0, 1)
    rate_ax.set_ylabel("Disease rate")

    for x_pos, rate in zip(x, disease_rates):
        rate_ax.text(x_pos, rate + 0.03, f"{rate:.1%}", ha="center", va="bottom", color="#1F3B5D", fontsize=10)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = rate_ax.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_secondary_pca_plot(
    nested_result_df: pd.DataFrame,
    primary_cluster_id: int,
    output_path: Path,
) -> None:
    plot_df = nested_result_df[nested_result_df["聚类类别"] == primary_cluster_id].copy()
    if plot_df.empty:
        return

    subset_features = plot_df[FEATURE_COLUMNS].copy()
    subset_preprocessor = build_preprocessor()
    subset_transformed = subset_preprocessor.fit_transform(subset_features)
    pca = PCA(n_components=2)
    projected_points = pca.fit_transform(subset_transformed)

    plot_df["pca_x"] = projected_points[:, 0]
    plot_df["pca_y"] = projected_points[:, 1]

    subcluster_ids = sorted(plot_df["子聚类类别"].unique().tolist())
    color_map = plt.get_cmap("tab10", len(subcluster_ids))

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for index, subcluster_id in enumerate(subcluster_ids):
        color = color_map(index)
        sub_df = plot_df[plot_df["子聚类类别"] == subcluster_id]
        healthy_df = sub_df[sub_df["观测期内发病"] == 0]
        diseased_df = sub_df[sub_df["观测期内发病"] == 1]

        if not healthy_df.empty:
            ax.scatter(
                healthy_df["pca_x"],
                healthy_df["pca_y"],
                s=70,
                marker="o",
                facecolors=color,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.9,
            )
        if not diseased_df.empty:
            ax.scatter(
                diseased_df["pca_x"],
                diseased_df["pca_y"],
                s=70,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=1.6,
                alpha=0.95,
            )

        centroid = sub_df[["pca_x", "pca_y"]].mean().to_numpy(dtype=float)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=200,
            marker="X",
            color=color,
            edgecolors="black",
            linewidths=1.0,
        )

    subcluster_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor=color_map(index),
            markeredgecolor="black",
            label=f"Subcluster {primary_cluster_id}-{subcluster_id}",
        )
        for index, subcluster_id in enumerate(subcluster_ids)
    ]
    status_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor="black",
            markeredgecolor="black",
            label="Healthy",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor="black",
            label="Diseased",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="",
            markersize=9,
            markerfacecolor="black",
            markeredgecolor="black",
            label="Subcluster centroid",
        ),
    ]

    legend1 = ax.legend(handles=subcluster_handles, title="Color = subcluster", loc="upper left", frameon=True)
    ax.add_artist(legend1)
    ax.legend(handles=status_handles, title="Style = health status", loc="lower left", frameon=True)

    explained_variance = pca.explained_variance_ratio_
    ax.set_xlabel(f"PCA 1 ({explained_variance[0]:.1%})")
    ax.set_ylabel(f"PCA 2 ({explained_variance[1]:.1%})")
    ax.set_title(f"Secondary PCA Plot for Primary Cluster {primary_cluster_id}")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_secondary_scatter_plot(
    nested_result_df: pd.DataFrame,
    primary_cluster_id: int,
    output_path: Path,
) -> None:
    plot_df = nested_result_df[nested_result_df["聚类类别"] == primary_cluster_id].copy()
    if plot_df.empty:
        return

    coordinates = plot_df["编号"].map(parse_tree_position).tolist()
    plot_df["plot_x"] = [coord[0] for coord in coordinates]
    plot_df["plot_y"] = [coord[1] for coord in coordinates]

    subcluster_ids = sorted(plot_df["子聚类类别"].unique().tolist())
    color_map = plt.get_cmap("tab10", len(subcluster_ids))

    fig, ax = plt.subplots(figsize=(10, 7))
    for index, subcluster_id in enumerate(subcluster_ids):
        color = color_map(index)
        sub_df = plot_df[plot_df["子聚类类别"] == subcluster_id]
        healthy_df = sub_df[sub_df["观测期内发病"] == 0]
        diseased_df = sub_df[sub_df["观测期内发病"] == 1]

        if not healthy_df.empty:
            ax.scatter(
                healthy_df["plot_x"],
                healthy_df["plot_y"],
                s=85,
                marker="o",
                facecolors=color,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.9,
            )
        if not diseased_df.empty:
            ax.scatter(
                diseased_df["plot_x"],
                diseased_df["plot_y"],
                s=85,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=1.6,
                alpha=0.95,
            )

    subcluster_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor=color_map(index),
            markeredgecolor="black",
            label=f"Subcluster {primary_cluster_id}-{subcluster_id}",
        )
        for index, subcluster_id in enumerate(subcluster_ids)
    ]
    status_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor="black",
            markeredgecolor="black",
            label="Healthy",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor="black",
            label="Diseased",
        ),
    ]

    legend1 = ax.legend(handles=subcluster_handles, title="Color = subcluster", loc="upper left", frameon=True)
    ax.add_artist(legend1)
    ax.legend(handles=status_handles, title="Fill = status", loc="lower left", frameon=True)

    ax.set_xlabel("Orchard coordinate X")
    ax.set_ylabel("Orchard coordinate Y")
    ax.set_title(f"Secondary Scatter Plot for Primary Cluster {primary_cluster_id}")
    ax.grid(alpha=0.25, linestyle="--")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_orchard_layout_plot(result_df: pd.DataFrame, output_path: Path) -> None:
    x_positions: list[float] = []
    y_positions: list[float] = []
    for tree_id in result_df["编号"]:
        x_pos, y_pos = parse_tree_position(tree_id)
        x_positions.append(x_pos)
        y_positions.append(y_pos)

    plot_df = result_df.copy()
    plot_df["plot_x"] = x_positions
    plot_df["plot_y"] = y_positions
    has_disease_flag = "观测期内发病" in plot_df.columns

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.get_cmap("tab10", plot_df["聚类类别"].nunique())
    unique_clusters = sorted(plot_df["聚类类别"].unique().tolist())

    if has_disease_flag:
        marker_map = {0: "o", 1: "^"}
        for color_index, cluster_id in enumerate(unique_clusters):
            for disease_flag, marker in marker_map.items():
                subset = plot_df[
                    (plot_df["聚类类别"] == cluster_id) & (plot_df["观测期内发病"] == disease_flag)
                ]
                if subset.empty:
                    continue
                disease_text = "No disease in observation period" if disease_flag == 0 else "Disease in observation period"
                ax.scatter(
                    subset["plot_x"],
                    subset["plot_y"],
                    s=90,
                    marker=marker,
                    color=colors(color_index),
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.88,
                    label=f"Cluster {cluster_id} | {disease_text}",
                )
    else:
        for color_index, cluster_id in enumerate(unique_clusters):
            subset = plot_df[plot_df["聚类类别"] == cluster_id]
            ax.scatter(
                subset["plot_x"],
                subset["plot_y"],
                s=90,
                color=colors(color_index),
                edgecolors="black",
                linewidths=0.5,
                alpha=0.88,
                label=f"Cluster {cluster_id}",
            )

    ax.set_xlabel("Orchard coordinate X (left token of tree id)")
    ax.set_ylabel("Orchard coordinate Y (right token of tree id)")
    ax.set_title("Orchard Cluster Map")
    ax.grid(alpha=0.25, linestyle="--")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.input_xlsx, sheet_name=args.sheet_name)
    validate_columns(df)

    feature_frame = df[FEATURE_COLUMNS].copy()
    preprocessor = build_preprocessor()
    transformed_features = preprocessor.fit_transform(feature_frame)

    ks, inertias, best_k, distances = run_elbow_method(
        transformed_features=transformed_features,
        max_k=args.max_k,
        random_state=args.random_state,
    )

    final_model = KMeans(n_clusters=best_k, random_state=args.random_state, n_init=20)
    labels = final_model.fit_predict(transformed_features) + 1

    result_df = df.copy()
    result_df["聚类类别"] = labels

    elbow_df = pd.DataFrame(
        {
            "K": ks,
            "SSE": inertias,
            "距首尾连线距离": distances,
        }
    )
    summary_df = build_cluster_summary(result_df)
    raw_profile_df, normalized_profile_df = build_cluster_profile_tables(result_df)
    disease_comparison_df = build_cluster_disease_table(result_df)
    nested_result_df, nested_disease_summary_df = run_nested_kmeans_by_primary_cluster(
        result_df=result_df,
        random_state=args.random_state,
    )
    result_df["子聚类类别"] = nested_result_df["子聚类类别"]
    result_df["二级聚类标签"] = nested_result_df["二级聚类标签"]

    result_xlsx = args.output_dir / "sheet2_kmeans_clustering.xlsx"
    with pd.ExcelWriter(result_xlsx, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="聚类结果", index=False)
        summary_df.to_excel(writer, sheet_name="聚类概况", index=False)
        elbow_df.to_excel(writer, sheet_name="肘部法SSE", index=False)
        raw_profile_df.reset_index().to_excel(writer, sheet_name="簇原始均值", index=False)
        normalized_profile_df.reset_index().to_excel(writer, sheet_name="簇归一化画像", index=False)
        if not disease_comparison_df.empty:
            disease_comparison_df.to_excel(writer, sheet_name="簇-发病对比", index=False)
        nested_disease_summary_df.to_excel(writer, sheet_name="二级聚类发病统计", index=False)

    elbow_png = args.output_dir / "sheet2_elbow_curve.png"
    save_elbow_plot(ks=ks, inertias=inertias, best_k=best_k, output_path=elbow_png)
    pca_png = args.output_dir / "sheet2_cluster_pca.png"
    pca1_var, pca2_var = save_pca_scatter_plot(
        transformed_features=transformed_features,
        labels=labels,
        model=final_model,
        output_path=pca_png,
    )
    size_png = args.output_dir / "sheet2_cluster_sizes.png"
    cluster_counts = save_cluster_size_plot(labels=labels, output_path=size_png)
    heatmap_png = args.output_dir / "sheet2_cluster_feature_heatmap.png"
    save_cluster_profile_heatmap(
        raw_profile=raw_profile_df,
        normalized_profile=normalized_profile_df,
        output_path=heatmap_png,
    )
    radar_png = args.output_dir / "sheet2_cluster_radar.png"
    save_cluster_profile_radar(normalized_profile=normalized_profile_df, output_path=radar_png)
    orchard_png = args.output_dir / "sheet2_orchard_cluster_map.png"
    save_orchard_layout_plot(result_df=result_df, output_path=orchard_png)
    disease_png = args.output_dir / "sheet2_cluster_vs_disease.png"
    save_cluster_vs_disease_plot(comparison_df=disease_comparison_df, output_path=disease_png)
    subcluster_disease_png = args.output_dir / "sheet2_subcluster_vs_disease.png"
    save_subcluster_vs_disease_plot(
        comparison_df=nested_disease_summary_df,
        output_path=subcluster_disease_png,
    )
    secondary_scatter_paths: list[str] = []
    secondary_pca_paths: list[str] = []
    for primary_cluster_id in sorted(result_df["聚类类别"].unique().tolist()):
        secondary_scatter_path = args.output_dir / f"sheet2_secondary_scatter_primary_{int(primary_cluster_id)}.png"
        save_secondary_scatter_plot(
            nested_result_df=nested_result_df,
            primary_cluster_id=int(primary_cluster_id),
            output_path=secondary_scatter_path,
        )
        secondary_scatter_paths.append(str(secondary_scatter_path))
        secondary_pca_path = args.output_dir / f"sheet2_secondary_pca_primary_{int(primary_cluster_id)}.png"
        save_secondary_pca_plot(
            nested_result_df=nested_result_df,
            primary_cluster_id=int(primary_cluster_id),
            output_path=secondary_pca_path,
        )
        secondary_pca_paths.append(str(secondary_pca_path))

    metrics = {
        "input_xlsx": str(args.input_xlsx),
        "sheet_name": args.sheet_name,
        "row_count": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "preprocessing": {
            "numeric_columns_scaled": NUMERIC_COLUMNS,
            "categorical_columns_one_hot": CATEGORICAL_COLUMNS,
        },
        "tested_k_values": ks,
        "sse": inertias,
        "elbow_distances": distances,
        "selected_k": int(best_k),
        "cluster_counts": cluster_counts,
        "pca_explained_variance_ratio": {
            "pca1": pca1_var,
            "pca2": pca2_var,
            "total": pca1_var + pca2_var,
        },
        "nested_cluster_disease_summary": nested_disease_summary_df.to_dict(orient="records"),
        "secondary_scatter_plots": secondary_scatter_paths,
        "secondary_pca_plots": secondary_pca_paths,
        "orchard_coordinate_rule": "编号 is parsed as x-y, where x is the left side and y is the right side of '-'.",
        "random_state": int(args.random_state),
    }
    metrics_path = args.output_dir / "sheet2_kmeans_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Selected K: {best_k}")
    print(f"Results workbook: {result_xlsx}")
    print(f"Elbow plot: {elbow_png}")
    print(f"PCA scatter plot: {pca_png}")
    print(f"Cluster size plot: {size_png}")
    print(f"Feature heatmap: {heatmap_png}")
    print(f"Feature radar chart: {radar_png}")
    print(f"Orchard cluster map: {orchard_png}")
    print(f"Cluster vs disease plot: {disease_png}")
    print(f"Subcluster vs disease plot: {subcluster_disease_png}")
    for secondary_scatter_path in secondary_scatter_paths:
        print(f"Secondary scatter plot: {secondary_scatter_path}")
    for secondary_pca_path in secondary_pca_paths:
        print(f"Secondary PCA plot: {secondary_pca_path}")
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
