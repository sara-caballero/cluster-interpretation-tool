# Clustering Analysis Pipeline
# Implements automated clustering with visualization and interpretation

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
from report_theme import THEME, prettify_name, fmt_val, fmt_pct, fmt_pair, legend_cluster_label, apply_theme, get_direction_symbol, get_direction_color

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from PIL import Image as PILImage
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from umap import UMAP
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    from pySankey.sankey import sankey
    HAS_SANKEY = True
except Exception:
    HAS_SANKEY = False


# Utility functions

def minmax_df(df):
    # Min-max scaling with constant column handling
    mins = df.min()
    rng  = df.max() - mins
    rng  = rng.replace(0, 1.0)
    return (df - mins) / rng

def pick_k_auto(k_values, inertia_norm, silhouettes, angles,
                elbow_angle_thresh=170.0, silhouette_floor=0.25, sil_within=0.02):
    # Automatic k selection for KMeans clustering
    # Combines silhouette analysis and elbow method
    kv   = list(k_values)
    inert = np.asarray(inertia_norm, float)
    sil   = np.asarray(silhouettes, float)
    ang   = np.asarray(angles, float)

    i_sil    = int(np.nanargmax(sil))
    k_sil    = kv[i_sil]
    sil_best = float(sil[i_sil])

    elbow_idx, elbow_angle = None, None
    for i in range(1, len(ang) - 1):
        if ang[i-1] >= ang[i] <= ang[i+1]:
            if elbow_idx is None or ang[i] < elbow_angle:
                elbow_idx, elbow_angle = i, float(ang[i])
    k_elbow = kv[elbow_idx] if elbow_idx is not None else None

    elbow_strong, slope_ratio = False, None
    if elbow_idx is not None and 1 <= elbow_idx < len(inert) - 1:
        drop_prev = inert[elbow_idx-1] - inert[elbow_idx]
        drop_next = inert[elbow_idx] - inert[elbow_idx+1]
        slope_ratio = drop_next / max(drop_prev, 1e-9)
        if (elbow_angle is not None) and (elbow_angle <= elbow_angle_thresh) and (slope_ratio < 0.75):
            elbow_strong = True

    if sil_best >= silhouette_floor:
        near = np.where(sil >= sil_best - sil_within)[0]
        k = kv[int(near[0])]
        why = f"silhouette looks ok (best={sil_best:.3f}), picked smallest k within {sil_within} of best"
    elif elbow_strong:
        k = k_elbow
        why = f"silhouette weak; elbow at k={k_elbow} (angle={elbow_angle:.1f}°, ratio={slope_ratio:.2f})"
    else:
        near = np.where(sil >= sil_best - max(sil_within, 0.01))[0]
        k = kv[int(near[0])] if len(near) else k_sil
        why = "no clear elbow + low silhouette, chose smallest k near best silhouette"
    return k, {"k_sil": k_sil, "sil_best": sil_best, "k_elbow": k_elbow,
               "elbow_angle": elbow_angle, "elbow_strong": elbow_strong, "slope_ratio": slope_ratio}

def explain_clusters_numeric(X_df, labels, top_n=5):
    # Analyze cluster characteristics by comparing medians
    # Returns ranked features by statistical significance
    out = []
    overall_med = X_df.median()
    overall_std = X_df.std().replace(0, 1.0)

    for k in sorted(pd.Series(labels).unique()):
        mask = (labels == k)
        med  = X_df.loc[mask].median()
        z    = (med - overall_med) / overall_std
        z    = z.replace([np.inf, -np.inf], np.nan).dropna()
        z    = z.sort_values(key=lambda s: np.abs(s), ascending=False)

        print(f"\nCluster {k} (n={int(mask.sum())}): top differences")
        for feat, val in z.head(top_n).items():
            direction = "higher" if val > 0 else "lower"
            print(f"  - {feat}: {direction} median (z≈{val:.2f})")
            out.append({
                "cluster": int(k),
                "feature": feat,
                "z_median": float(val),
                "direction": direction,
                "cluster_median": float(med[feat]),
                "overall_median": float(overall_med[feat])
            })
    print()
    return pd.DataFrame(out)

def sankey_top_flows(df, labels, features, bins=3, title="Top drivers → Clusters"):
    # Generate Sankey diagram showing feature-to-cluster relationships
    if not HAS_SANKEY:
        print("Sankey not installed, skipping.")
        return
    left, right, weights = [], [], []
    d = df.copy()
    d["_cluster"] = pd.Series(labels, index=df.index).values
    for f in features:
        if pd.api.types.is_numeric_dtype(d[f]):
            try:
                binned = pd.qcut(d[f], q=bins, duplicates="drop").astype(str)
            except Exception:
                binned = pd.cut(d[f], bins=bins).astype(str)
        else:
            binned = d[f].astype(str).fillna("NaN")
        tmp = pd.DataFrame({
            "left": f + ": " + binned,
            "right": "Cluster " + pd.Series(d["_cluster"]).astype(int).astype(str),
        })
        counts = tmp.value_counts().reset_index(name="n")
        left   += counts["left"].tolist()
        right  += counts["right"].tolist()
        weights += counts["n"].tolist()
    plt.figure(figsize=(7, 5))
    sankey(left=left, right=right, leftWeight=weights, rightWeight=weights, aspect=20, fontsize=10)
    plt.title(title)
    plt.show()


# Data preprocessing functions

def preprocess_data(
    file_path,
    target=None,
    excluded_cols=None,
    scaling="minmax",
    outlier_method="isoforest",
    contamination=0.03,
    separator=",",
    alpha=1.0,
    beta=1.0,
    auto_balance=True,
):
    """
    Preprocess data for clustering analysis.
    
    Args:
        file_path: Path to CSV file
        target: Target column name (optional)
        excluded_cols: Columns to exclude from analysis
        scaling: Scaling method ('minmax', 'standard', 'none')
        outlier_method: Outlier detection method
        contamination: Outlier contamination ratio
        separator: CSV separator character
    
    Returns:
        dict: Preprocessing results and metadata
    """
    
    # Load dataset
    print("Reading CSV...")
    raw = pd.read_csv(file_path, sep=separator)
    original_shape = raw.shape
    
    # Feature selection
    feat_cols = list(raw.columns)
    if target is not None and target in feat_cols:
        feat_cols.remove(target)
    if excluded_cols is not None:
        for col in excluded_cols:
            if col in feat_cols:
                feat_cols.remove(col)
    X = raw[feat_cols].copy()
    
    # Categorical feature encoding
    categorical_cols = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            categorical_cols.append(col)
    
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=float)
    for c in X_encoded.columns:
        if X_encoded[c].dtype == bool:
            X_encoded[c] = X_encoded[c].astype(float)
    
    # Feature re-weighting (balance categorical vs numeric)
    if auto_balance:
        print("Applying feature re-weighting with auto-balance...")
    else:
        print(f"Applying feature re-weighting: alpha={alpha}, beta={beta}")
    X_encoded = reweight_features(X_encoded, X, alpha=alpha, beta=beta, auto_balance=auto_balance)
    
    # Feature scaling
    scaler_info = {"method": scaling, "per_feature": {}}
    
    if scaling == "standard":
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns, index=X_encoded.index)
        # Guardar parámetros de escalado estándar
        for i, col in enumerate(X_encoded.columns):
            scaler_info["per_feature"][col] = {
                "mean": scaler.mean_[i],
                "std": scaler.scale_[i]
            }
    elif scaling == "minmax":
        X_scaled = minmax_df(X_encoded).fillna(0.0)
        # Guardar parámetros de escalado minmax
        for col in X_encoded.columns:
            min_val = X_encoded[col].min()
            max_val = X_encoded[col].max()
            range_val = max_val - min_val
            scaler_info["per_feature"][col] = {
                "min": min_val,
                "range": range_val
            }
    elif scaling == "none":
        X_scaled = X_encoded.astype(float)
        # Para "none", no hay transformación
        for col in X_encoded.columns:
            scaler_info["per_feature"][col] = {
                "min": 0,
                "range": 1
            }
    else:
        raise ValueError("scaling must be 'minmax', 'standard', or 'none'")
    
    print("Data shape after prep:", X_scaled.shape)
    
    # Outlier removal
    outliers_removed = 0
    if outlier_method == "isoforest":
        print(f"Removing outliers with IsolationForest (contamination={contamination:.3f})...")
        iso  = IsolationForest(contamination=contamination, random_state=42)
        pred = iso.fit_predict(X_scaled.values)
        keep = pred == 1
        n_out = int((~keep).sum())
        if n_out > 0 and keep.sum() >= 10:
            X_scaled = X_scaled.loc[keep]
            raw      = raw.loc[keep]
            outliers_removed = n_out
            print(f"Removed {n_out} outliers. New shape: {X_scaled.shape}")
        else:
            print("Outlier removal skipped (not enough inliers).")
    elif outlier_method == "none":
        print("Outlier removal disabled.")
    else:
        raise ValueError("outlier_method must be 'isoforest' or 'none'")
    
    # Compile preprocessing results
    preprocessing_info = {
        "original_shape": original_shape,
        "final_shape": X_scaled.shape,
        "features_used": list(X_scaled.columns),
        "scaling_method": scaling,
        "outlier_method": outlier_method,
        "outliers_removed": outliers_removed,
        "categorical_encoded": len(categorical_cols),
        "raw_data": raw,
        "scaled_data": X_scaled,
        "scaler_info": scaler_info
    }
    
    return preprocessing_info


# Utility functions for human-readable interpretations

def inverse_scale(series_or_scalar, scaler_info, feature_name=None):
    """
    Convierte valores escalados de vuelta a unidades originales.
    
    Args:
        series_or_scalar: Serie de pandas o valor escalar
        scaler_info: Diccionario con información del escalado
        feature_name: Nombre de la característica (requerido si series_or_scalar es escalar)
    
    Returns:
        Valores en unidades originales
    """
    if scaler_info["method"] == "none":
        return series_or_scalar
    
    if isinstance(series_or_scalar, (int, float)):
        # Valor escalar
        if feature_name is None:
            raise ValueError("feature_name es requerido para valores escalares")
        
        if feature_name not in scaler_info["per_feature"]:
            return series_or_scalar
        
        params = scaler_info["per_feature"][feature_name]
        
        if scaler_info["method"] == "standard":
            # x_original = x_scaled * std + mean
            return series_or_scalar * params["std"] + params["mean"]
        elif scaler_info["method"] == "minmax":
            # x_original = x_scaled * range + min
            return series_or_scalar * params["range"] + params["min"]
    else:
        # Serie de pandas
        result = series_or_scalar.copy()
        for col in series_or_scalar.index:
            if col in scaler_info["per_feature"]:
                params = scaler_info["per_feature"][col]
                
                if scaler_info["method"] == "standard":
                    result[col] = series_or_scalar[col] * params["std"] + params["mean"]
                elif scaler_info["method"] == "minmax":
                    result[col] = series_or_scalar[col] * params["range"] + params["min"]
        
        return result

def prettify(name):
    """
    Convierte nombres de características a formato legible.
    """
    return name.replace("_", " ").title()

def format_numeric_comparison(cluster_val, overall_val):
    """
    Formats numeric comparison with clear thresholds and natural language.
    """
    if overall_val == 0:
        if abs(cluster_val) < 0.01:
            return "almost zero"
        else:
            return f"{cluster_val:.2f} vs {overall_val:.2f} (much different)"
    
    rel_diff = (cluster_val - overall_val) / abs(overall_val)
    abs_percent = abs(rel_diff) * 100
    
    # Format values
    cluster_str = f"{cluster_val:.2f}" if abs(cluster_val) >= 0.01 else "0.00"
    overall_str = f"{overall_val:.2f}" if abs(overall_val) >= 0.01 else "0.00"
    
    # Determine description
    if abs_percent < 10:
        return f"{cluster_str} vs {overall_str} (similar)"
    elif abs_percent < 25:
        if rel_diff > 0:
            return f"{cluster_str} vs {overall_str} (slightly higher)"
        else:
            return f"{cluster_str} vs {overall_str} (slightly lower)"
    else:
        if rel_diff > 0:
            return f"{cluster_str} vs {overall_str} (much higher)"
        else:
            return f"{cluster_str} vs {overall_str} (much lower)"

def format_categorical_comparison(cluster_pct, overall_pct):
    """
    Formats categorical comparison with prevalence and difference.
    """
    diff = cluster_pct - overall_pct
    abs_diff = abs(diff)
    
    if abs_diff < 5:
        return f"{cluster_pct:.0f}% vs {overall_pct:.0f}% (similar)"
    elif abs_diff < 15:
        if diff > 0:
            return f"{cluster_pct:.0f}% vs {overall_pct:.0f}% (slightly more common)"
        else:
            return f"{cluster_pct:.0f}% vs {overall_pct:.0f}% (slightly less common)"
    else:
        if diff > 0:
            return f"{cluster_pct:.0f}% vs {overall_pct:.0f}% (much more common)"
        else:
            return f"{cluster_pct:.0f}% vs {overall_pct:.0f}% (much less common)"

def reweight_features(X, original_df, alpha=1.0, beta=1.0, auto_balance=True):
    """
    Re-weights features to balance categorical vs numeric features.
    
    Args:
        X: DataFrame with encoded features (after get_dummies)
        original_df: Original DataFrame before encoding
        alpha: Weight for categorical features (used only if auto_balance=False)
        beta: Weight for numeric features (used only if auto_balance=False)
        auto_balance: If True, automatically calculate alpha and beta based on feature ratios
    
    Returns:
        DataFrame with re-weighted features
    """
    import numpy as np
    
    X_weighted = X.copy()
    
    # Auto-balance logic
    if auto_balance:
        # Count numeric columns (from original_df)
        n_num = len([col for col in original_df.columns if col in X_weighted.columns])
        
        # Count dummy columns (encoded categorical features)
        n_dummy = len([col for col in X_weighted.columns if '_' in col])
        
        # Calculate ratio
        ratio = n_dummy / max(n_num, 1)
        
        # Set alpha based on ratio
        if ratio >= 1.5:
            alpha = 0.6
        elif 0.75 <= ratio < 1.5:
            alpha = 0.7
        else:
            alpha = 0.8
        
        # Set beta
        beta = 1.3
        
        print(f"Auto-balance: {n_dummy} dummy columns, {n_num} numeric columns (ratio={ratio:.2f})")
        print(f"Auto-selected: alpha={alpha}, beta={beta}")
    
    # Get original column names (before encoding)
    original_cols = set(original_df.columns)
    
    # Process each original column
    for col in original_cols:
        if col in X_weighted.columns:
            # Numeric column - multiply by beta
            X_weighted[col] = X_weighted[col] * beta
        else:
            # Categorical column - find all its dummy columns
            dummy_cols = [c for c in X_weighted.columns if c.startswith(col + '_')]
            if dummy_cols:
                # Calculate weight: alpha / sqrt(L-1) where L-1 is number of dummy columns
                L_minus_1 = len(dummy_cols)
                weight = alpha / np.sqrt(L_minus_1)
                
                # Apply weight to all dummy columns
                for dummy_col in dummy_cols:
                    X_weighted[dummy_col] = X_weighted[dummy_col] * weight
    
    return X_weighted

def explain_clusters_numeric_original(raw_df, X_scaled, labels, scaler_info, top_n=5):
    """
    Calcula las características más importantes de cada cluster en unidades originales.
    
    Args:
        raw_df: DataFrame con datos originales (sin escalar)
        X_scaled: DataFrame con datos escalados
        labels: Etiquetas de cluster
        scaler_info: Información del escalado
        top_n: Número de características principales por cluster
    
    Returns:
        DataFrame con características importantes en unidades originales
    """
    import numpy as np
    import pandas as pd
    
    results = []
    
    for cluster_id in sorted(labels.unique()):
        # Máscara para el cluster actual
        mask = labels == cluster_id
        cluster_data = X_scaled[mask]
        
        # Calcular medianas del cluster y globales
        cluster_medians = cluster_data.median()
        overall_medians = X_scaled.median()
        
        # Convertir a unidades originales
        cluster_medians_orig = inverse_scale(cluster_medians, scaler_info)
        overall_medians_orig = inverse_scale(overall_medians, scaler_info)
        
        # Calcular diferencias y z-scores
        differences = cluster_medians_orig - overall_medians_orig
        
        # Para cada característica
        for feature in X_scaled.columns:
            cluster_med = cluster_medians_orig[feature]
            overall_med = overall_medians_orig[feature]
            diff = differences[feature]
            
            # Determinar dirección
            direction = "higher" if diff > 0 else "lower"
            
            # Calcular z-score aproximado (usando desviación estándar de los datos originales)
            if feature in raw_df.columns:
                feature_std = raw_df[feature].std()
                if feature_std > 0:
                    z_score = diff / feature_std
                else:
                    z_score = 0
            else:
                z_score = 0
            
            results.append({
                'cluster': cluster_id,
                'feature': feature,
                'direction': direction,
                'cluster_median_orig': cluster_med,
                'overall_median_orig': overall_med,
                'z_score': z_score
            })
    
    # Convertir a DataFrame y ordenar por z-score absoluto
    results_df = pd.DataFrame(results)
    results_df['abs_z_score'] = results_df['z_score'].abs()
    
    # Seleccionar top_n características por cluster
    top_features = []
    for cluster_id in sorted(labels.unique()):
        cluster_results = results_df[results_df['cluster'] == cluster_id]
        top_cluster = cluster_results.nlargest(top_n, 'abs_z_score')
        top_features.append(top_cluster)
    
    if top_features:
        return pd.concat(top_features, ignore_index=True)
    else:
        return pd.DataFrame()


# Main clustering pipeline

def run_pipeline(
    file_path,
    target=None,
    excluded_cols=None,
    scaling="minmax",
    embedder="PCA",
    cluster_on_features=True,
    max_k=10,
    manual_k=None,
    outlier_method="isoforest",
    contamination=0.03,
    draw_sankey=False,
    preprocessed_data=None,
    separator=",",
    alpha=1.0,
    beta=1.0,
    auto_balance=True,
):

    # Handle preprocessed data or perform preprocessing
    if preprocessed_data is not None:
        X_scaled = preprocessed_data["scaled_data"]
        raw = preprocessed_data["raw_data"]
        print("Using preprocessed data...")
    else:
        # Load dataset
        print("Reading CSV...")
        raw = pd.read_csv(file_path, sep=separator)

        # Feature selection
        feat_cols = list(raw.columns)
        if target is not None and target in feat_cols:
            feat_cols.remove(target)
        if excluded_cols is not None:
            for col in excluded_cols:
                if col in feat_cols:
                    feat_cols.remove(col)
        X = raw[feat_cols].copy()

        # Categorical encoding
        X = pd.get_dummies(X, drop_first=True, dtype=float)
        for c in X.columns:
            if X[c].dtype == bool:
                X[c] = X[c].astype(float)

        # Feature re-weighting (balance categorical vs numeric)
        if auto_balance:
            print("Applying feature re-weighting with auto-balance...")
        else:
            print(f"Applying feature re-weighting: alpha={alpha}, beta={beta}")
        X = reweight_features(X, raw[feat_cols], alpha=alpha, beta=beta, auto_balance=auto_balance)

        # Feature scaling
        scaler_info = {"method": scaling, "per_feature": {}}
        
        if scaling == "standard":
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            # Guardar parámetros de escalado estándar
            for i, col in enumerate(X.columns):
                scaler_info["per_feature"][col] = {
                    "mean": scaler.mean_[i],
                    "std": scaler.scale_[i]
                }
        elif scaling == "minmax":
            X_scaled = minmax_df(X).fillna(0.0)
            # Guardar parámetros de escalado minmax
            for col in X.columns:
                min_val = X[col].min()
                max_val = X[col].max()
                range_val = max_val - min_val
                scaler_info["per_feature"][col] = {
                    "min": min_val,
                    "range": range_val
                }
        elif scaling == "none":
            X_scaled = X.astype(float)
            # Para "none", no hay transformación
            for col in X.columns:
                scaler_info["per_feature"][col] = {
                    "min": 0,
                    "range": 1
                }
        else:
            raise ValueError("scaling must be 'minmax', 'standard', or 'none'")

        print("Data shape after prep:", X_scaled.shape)

        # Outlier removal
        if outlier_method == "isoforest":
            print(f"Removing outliers with IsolationForest (contamination={contamination:.3f})...")
            iso  = IsolationForest(contamination=contamination, random_state=42)
            pred = iso.fit_predict(X_scaled.values)
            keep = pred == 1
            n_out = int((~keep).sum())
            if n_out > 0 and keep.sum() >= 10:
                X_scaled = X_scaled.loc[keep]
                raw      = raw.loc[keep]
                print(f"Removed {n_out} outliers. New shape: {X_scaled.shape}")
            else:
                print("Outlier removal skipped (not enough inliers).")
        elif outlier_method == "none":
            print("Outlier removal disabled.")
        else:
            raise ValueError("outlier_method must be 'isoforest' or 'none'")

    # Generate 2D embedding for visualization
    print(f"Computing {embedder} embedding for plots...")
    if embedder.upper() == "UMAP":
        if HAS_UMAP:
            emb_xy = UMAP(n_components=2, random_state=1).fit_transform(X_scaled.values)
        else:
            print("UMAP not installed; using PCA instead.")
            emb_xy = PCA(n_components=2, random_state=1).fit_transform(X_scaled.values)
    else:
        emb_xy = PCA(n_components=2, random_state=1).fit_transform(X_scaled.values)

    embed = pd.DataFrame({"X": emb_xy[:, 0], "Y": emb_xy[:, 1]}, index=X_scaled.index)

    # Select clustering data source
    data_for_kmeans = X_scaled if cluster_on_features else embed

    # K value determination
    if manual_k is not None:
        # Use manual k value
        best_k = manual_k
        print(f"Using manual k value: {best_k}")
        
        # Generate reference plots for manual k
        if max_k is not None:
            k_vals = list(range(2, min(max_k, len(data_for_kmeans) - 1) + 1))
            if len(k_vals) > 0:
                inertias, sils = [], []
                for k in k_vals:
                    km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
                    lab = km.fit_predict(data_for_kmeans)
                    inertias.append(km.inertia_)
                    sils.append(silhouette_score(data_for_kmeans, lab))
                inertia_norm = np.asarray(inertias, float) / max(inertias)
                
                # Model selection visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Embedding plot
                axes[0].scatter(embed["X"], embed["Y"], s=10)
                axes[0].set_title(f"{embedder} embedding")
                axes[0].set_xticks([])
                axes[0].set_yticks([])
                
                # Model selection plot
                ax1 = axes[1]
                ax1.plot(k_vals, inertia_norm, marker="o", label="normalized inertia (elbow)")
                ax1.set_xlabel("number of clusters (k)")
                ax1.set_ylabel("normalized inertia [0..1]")
                ax2 = ax1.twinx()
                ax2.plot(k_vals, sils, marker="X", linestyle="--", label="average silhouette score")
                ax2.set_ylabel("silhouette [-1..1]")
                
                # Highlight selected k value
                if best_k in k_vals:
                    k_idx = k_vals.index(best_k)
                    ax1.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Manual k={best_k}')
                    ax1.text(best_k, inertia_norm[k_idx], f'Manual\nk={best_k}', 
                             ha="center", va="bottom", color='red', fontweight='bold')
                
                ax1.legend(loc="upper right")
                ax1.set_title("KMeans model selection (Manual k highlighted)")
                
                plt.tight_layout()
                plt.show()
        else:
            # Simple embedding display
            plt.figure(figsize=(6, 5))
            plt.scatter(embed["X"], embed["Y"], s=10)
            plt.title(f"{embedder} embedding")
            plt.xticks([])
            plt.yticks([])
            plt.show()

    else:
        # Automatic k selection
        print("Evaluating k for KMeans...")
        k_vals = list(range(2, min(max_k, len(data_for_kmeans) - 1) + 1))
        if len(k_vals) == 0:
            raise ValueError("Not enough rows to evaluate k (maybe max_k too big?).")

        inertias, sils = [], []
        for k in k_vals:
            km  = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
            lab = km.fit_predict(data_for_kmeans)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(data_for_kmeans, lab))
        inertia_norm = np.asarray(inertias, float) / max(inertias)

        # Calculate elbow angles
        angles = [180.0]
        for i in range(1, len(inertia_norm) - 1):
            left  = inertia_norm[i] - inertia_norm[i - 1]
            right = inertia_norm[i + 1] - inertia_norm[i]
            ang = 180.0 - math.degrees(math.atan((right - left) / (1 + (right * left))))
            angles.append(ang)
            
        angles.append(180.0)

        # Create visualization grid
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Embedding visualization
        axes[0].scatter(embed["X"], embed["Y"], s=10)
        axes[0].set_title(f"{embedder} embedding")
        axes[0].set_xticks([]); axes[0].set_yticks([])

        # Model selection visualization
        ax1 = axes[1]
        ax1.plot(k_vals, inertia_norm, marker="o", label="normalized inertia (elbow)")
        ax1.set_xlabel("number of clusters (k)")
        ax1.set_ylabel("normalized inertia [0..1]")
        ax2 = ax1.twinx()
        ax2.plot(k_vals, sils, marker="X", linestyle="--", label="average silhouette score")
        ax2.set_ylabel("silhouette [-1..1]")

        # Add value labels for clarity
        for i, s in enumerate(sils):
            is_peak = (i == 0 or s >= sils[i-1]) and (i == len(sils)-1 or s >= sils[i+1])
            if is_peak:
                ax2.text(k_vals[i], s, f'k={k_vals[i]}\n{s:.2f}', ha="center", va="bottom")

        for a in range(1, len(angles) - 1):
            if angles[a-1] >= angles[a] <= angles[a+1] and a < len(k_vals) and a < len(inertia_norm):
                ax1.text(k_vals[a], float(inertia_norm[a]), f'k={k_vals[a]}\n{angles[a]:.1f}°',
                         ha="center", va="bottom")

        ax1.legend(loc="upper right")
        ax1.set_title("KMeans model selection")
        plt.tight_layout(); plt.show()

        # Select optimal k value
        best_k, why = pick_k_auto(k_vals, inertia_norm, sils, angles)
        print("Auto-picked k:", best_k, "->", why)


    # Final clustering
    km_final = KMeans(n_clusters=best_k, init="k-means++", n_init=20, random_state=42)
    labels_kmeans = pd.Series(km_final.fit_predict(data_for_kmeans), index=data_for_kmeans.index, name="Cluster")
    sil_final = silhouette_score(data_for_kmeans, labels_kmeans)
    print(f"KMeans silhouette at k={best_k}: {sil_final:.3f}")

    # Enhanced cluster visualization
    fig, ax = plt.subplots(figsize=THEME.FIGURE_SIZE)
    
    # Calculate cluster statistics for legend
    cluster_counts = labels_kmeans.value_counts().sort_index()
    total_points = len(labels_kmeans)
    
    # Plot each cluster with theme colors
    for cluster_id in sorted(labels_kmeans.unique()):
        mask = labels_kmeans == cluster_id
        cluster_points = embed.loc[mask]
        color = THEME.get_cluster_color(cluster_id)
        
        ax.scatter(cluster_points["X"], cluster_points["Y"], 
                  s=THEME.POINT_SIZE, 
                  alpha=THEME.ALPHA_CLUSTER,
                  color=color,
                  edgecolors=color,
                  linewidth=0.5,
                  label=legend_cluster_label(cluster_id, cluster_counts[cluster_id], total_points))
    
    # Apply theme styling
    apply_theme(ax, title=f"{embedder} + KMeans(k={best_k})", 
               xlabel="PC1", ylabel="PC2")
    
    # Enhanced legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True,
             fontsize=THEME.BODY_SIZE, framealpha=0.9)
    
    plt.show()

    # Cluster interpretation
    print("KMeans: quick numeric interpretation (top drivers):")
    drivers_km = explain_clusters_numeric(X_scaled, labels_kmeans, top_n=5)

    # Calculate drivers in original units if scaler info is available
    drivers_km_orig = None
    if preprocessed_data is not None and "scaler_info" in preprocessed_data:
        print("Calculating drivers in original units...")
        # Use the original raw data from preprocessing (before scaling)
        raw_original = preprocessed_data["raw_data"]
        drivers_km_orig = explain_clusters_numeric_original(
            raw_original, X_scaled, labels_kmeans, preprocessed_data["scaler_info"], top_n=5
        )
    elif 'scaler_info' in locals():
        print("Calculating drivers in original units...")
        # Use the raw data from current run
        drivers_km_orig = explain_clusters_numeric_original(
            raw, X_scaled, labels_kmeans, scaler_info, top_n=5
        )

    # Enhanced feature distribution analysis
    print("KMeans: feature distributions by cluster (enhanced plots)...")
    tmp = X_scaled.copy()
    tmp["Cluster"] = labels_kmeans.values
    
    # Calculate visualization grid
    n_features = min(len(X_scaled.columns), 12)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(X_scaled.columns[:n_features]):
        row, col_idx = i // n_cols, i % n_cols
        ax = axes[row, col_idx]
        
        # Create enhanced violin+box hybrid plot
        # First, violin plot with light fill
        violin_parts = ax.violinplot([tmp[tmp["Cluster"] == c][col].dropna() 
                                    for c in sorted(tmp["Cluster"].unique())],
                                   positions=sorted(tmp["Cluster"].unique()),
                                   showmeans=False, showmedians=False, showextrema=False)
        
        # Style violin plots
        for pc in violin_parts['bodies']:
            pc.set_facecolor(THEME.NEUTRAL_LIGHT)
            pc.set_alpha(0.6)
            pc.set_edgecolor(THEME.NEUTRAL_MID)
            pc.set_linewidth(0.8)
        
        # Add box plot on top
        box_parts = ax.boxplot([tmp[tmp["Cluster"] == c][col].dropna() 
                              for c in sorted(tmp["Cluster"].unique())],
                             positions=sorted(tmp["Cluster"].unique()),
                             patch_artist=True, widths=0.3,
                             boxprops=dict(facecolor='white', alpha=0.8, 
                                          edgecolor=THEME.NEUTRAL_DARK, linewidth=1),
                             medianprops=dict(color=THEME.PRIMARY, linewidth=2),
                             whiskerprops=dict(color=THEME.NEUTRAL_DARK, linewidth=1),
                             capprops=dict(color=THEME.NEUTRAL_DARK, linewidth=1),
                             flierprops=dict(marker='o', markerfacecolor=THEME.NEUTRAL_MID, 
                                            markeredgecolor=THEME.NEUTRAL_DARK, 
                                            markersize=4, alpha=0.6))
        
        # Add overall median line
        overall_median = tmp[col].median()
        ax.axhline(y=overall_median, color=THEME.NEUTRAL_MID, 
                  linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Apply theme styling
        apply_theme(ax, title=f"{prettify_name(col)} (by cluster)")
        ax.set_xlabel("Cluster", fontsize=THEME.BODY_SIZE)
        ax.set_ylabel(prettify_name(col), fontsize=THEME.BODY_SIZE)
        
        # Set consistent y-limits for better comparison
        y_min, y_max = tmp[col].quantile([0.05, 0.95])
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Remove unused subplots
    for i in range(n_features, n_rows * n_cols):
        row, col_idx = i // n_cols, i % n_cols
        axes[row, col_idx].set_visible(False)
    
    # Enhanced title
    fig.suptitle("Feature Distributions by Cluster (Enhanced)", 
                fontsize=THEME.HEADING_SIZE, fontweight='600',
                color=THEME.NEUTRAL_DARK, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

    # Optional Sankey visualization
    if draw_sankey and HAS_SANKEY and not drivers_km.empty:
        top_feats = (drivers_km.groupby("feature")["z_median"]
                     .apply(lambda s: np.nanmean(np.abs(s)))
                     .sort_values(ascending=False).head(3).index.tolist())
        sankey_top_flows(X_scaled, labels_kmeans, top_feats, bins=3, title="KMeans: top drivers → clusters")

    # Target comparison analysis
    if target is not None and target in raw.columns:
        print("Comparing target vs clusters...")
        df_k = raw.loc[labels_kmeans.index].copy()
        df_k["Cluster"] = labels_kmeans.values
        tab = df_k.groupby("Cluster")[target].value_counts(normalize=True).unstack()
        print("\nTarget distribution by cluster:")
        print(tab)
        sns.heatmap(tab, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Target vs KMeans cluster")
        plt.show()

    # Return analysis results
    results_dict = {
        "X_scaled": X_scaled,
        "embedding": embed,
        "kmeans_labels": labels_kmeans,
        "kmeans_model": km_final,
        "kmeans_drivers": drivers_km,
        "raw_data": raw,  # Include raw data for PDF generation
    }
    
    # Add original units drivers if available
    if drivers_km_orig is not None:
        results_dict["kmeans_drivers_orig"] = drivers_km_orig
    
    return results_dict



def auto_describe_clusters(results, file_path=None, target=None, top_n=3):
    """
    Generate human-readable cluster summaries.
    
    Args:
        results: Clustering results dictionary
        file_path: Path to original data file
        target: Target column for comparison
        top_n: Number of top features to highlight
    
    Returns:
        list: List of cluster summary strings
    """
    import numpy as np
    import pandas as pd

    # Usar drivers en unidades originales si están disponibles
    if "kmeans_drivers_orig" in results and not results["kmeans_drivers_orig"].empty:
        drivers = results["kmeans_drivers_orig"].copy()
        use_original_units = True
    else:
        drivers = results["kmeans_drivers"].copy()
        use_original_units = False

    labels = results["kmeans_labels"]
    X = results["X_scaled"]

    # Calcular tamaños de clusters
    sizes = labels.value_counts().sort_index()

    # Process target variable for comparison
    target_binary = None
    overall_pct_1 = None  # overall % of class 1.0

    if file_path is not None and target is not None:
        raw = pd.read_csv(file_path)
        raw = raw.loc[labels.index].copy()  # align with rows used after outlier removal

        if target in raw.columns:
            ser = raw[target]

            # Convert to binary format
            if pd.api.types.is_numeric_dtype(ser):
                uniq = pd.unique(ser.dropna())
                if len(uniq) == 2 and set(np.sort(uniq)) <= {0, 1} | {0.0, 1.0}:
                    target_binary = ser.astype(float)
            else:
                # Map categorical to binary
                uniq = pd.unique(ser.dropna())
                if len(uniq) == 2:
                    u_sorted = np.sort(uniq.astype(str))
                    mapping = {u_sorted[0]: 0.0, u_sorted[1]: 1.0}
                    target_binary = ser.astype(str).map(mapping).astype(float)

            if target_binary is not None:
                # Calculate overall target distribution
                overall_pct_1 = 100.0 * (target_binary == 1.0).mean()

    # Reconstruir información de características originales
    raw = None
    binary_features = {}
    
    if file_path is not None:
        raw = pd.read_csv(file_path)
        raw = raw.loc[labels.index].copy()
        
        # Identificar características binarias
        for feat in raw.columns:
            unique_vals = raw[feat].dropna().unique()
            if len(unique_vals) == 2:  # característica binaria
                binary_features[feat] = unique_vals

    # Generar resúmenes de clusters
    summaries = []
    for cid in sorted(drivers["cluster"].unique()):
        sort_col = "z_score" if use_original_units else "z_median"
        chunk = (drivers[drivers["cluster"] == cid]
                 .sort_values(sort_col, key=lambda s: s.abs(), ascending=False)
                 .head(top_n))

        phrases = []
        for _, r in chunk.iterrows():
            feat = r.feature
            direction = r.direction
            
            if use_original_units:
                cluster_med = r.cluster_median_orig
                overall_med = r.overall_median_orig
            else:
                cluster_med = r.cluster_median
                overall_med = r.overall_median
            
            # Determine feature type and format accordingly
            if '_' in feat and raw is not None:  # encoded feature
                base, level = feat.rsplit('_', 1)
                if base in binary_features:
                    # Categorical/binary feature
                    mask = (labels == cid)
                    cluster_data = raw.loc[mask, base]
                    overall_data = raw[base]
                    
                    # Calculate real percentages
                    cluster_pct = 100.0 * (cluster_data == level).mean()
                    overall_pct = 100.0 * (overall_data == level).mean()
                    
                    # Format categorical comparison
                    comparison_text = format_categorical_comparison(cluster_pct, overall_pct)
                    phrases.append(f"{prettify(level)} in {prettify(base)}: {comparison_text}")
                else:
                    # Numeric encoded feature
                    comparison_text = format_numeric_comparison(cluster_med, overall_med)
                    phrases.append(f"{prettify(feat)}: {comparison_text}")
            else:
                # Non-encoded numeric feature
                comparison_text = format_numeric_comparison(cluster_med, overall_med)
                phrases.append(f"{prettify(feat)}: {comparison_text}")

        line = f"Cluster {cid} (n={sizes.get(cid, 0)}): " + "; ".join(phrases) + "."

        # Add target comparison if available
        if target_binary is not None:
            mask = (labels == cid)
            tb_c = target_binary.loc[mask]
            if tb_c.notna().any():
                pct1 = 100.0 * (tb_c == 1.0).mean()
                pct0 = 100.0 * (tb_c == 0.0).mean()
                
                # Always use target=1 for consistency
                pct_here = pct1
                pct_over = overall_pct_1

                if pct_over is not None:
                    diff_target = abs(pct_here - pct_over)
                    if diff_target < 5:
                        target_desc = "similar frequency"
                    elif diff_target < 15:
                        target_desc = "slightly more frequent" if pct_here > pct_over else "slightly less frequent"
                    else:
                        target_desc = "much more frequent" if pct_here > pct_over else "much less frequent"

                    line += f" → {target}=1 in {pct_here:.0f}% vs {pct_over:.0f}% ({target_desc})"
                else:
                    line += f" → {target}=1 in {pct_here:.0f}%"


        summaries.append(line)

    for s in summaries:
        print(s)
    return summaries


def generate_visualization_images(results, embedder="PCA", manual_k=None, max_k=None):
    """
    Generate visualization images for PDF inclusion with enhanced styling.
    
    Args:
        results: Dictionary containing clustering results
        embedder: Embedding method used
        manual_k: Manual k value if used
        max_k: Maximum k value tested
    
    Returns:
        dict: Dictionary with image data for each visualization
    """
    if not HAS_REPORTLAB:
        return {}
    
    images = {}
    X_scaled = results["X_scaled"]
    labels = results["kmeans_labels"]
    embed = results["embedding"]
    
    try:
        # 1. Enhanced 2D Embedding Plot
        fig, ax = plt.subplots(figsize=THEME.FIGURE_SIZE)
        
        # Calculate cluster statistics for legend
        cluster_counts = labels.value_counts().sort_index()
        total_points = len(labels)
        
        # Plot each cluster with theme colors
        for cluster_id in sorted(labels.unique()):
            mask = labels == cluster_id
            cluster_points = embed.loc[mask]
            color = THEME.get_cluster_color(cluster_id)
            
            ax.scatter(cluster_points["X"], cluster_points["Y"], 
                      s=THEME.POINT_SIZE, 
                      alpha=THEME.ALPHA_CLUSTER,
                      color=color,
                      edgecolors=color,
                      linewidth=0.5,
                      label=legend_cluster_label(cluster_id, cluster_counts[cluster_id], total_points))
        
        # Apply theme styling
        apply_theme(ax, title=f"{embedder} Embedding", 
                   xlabel="PC1", ylabel="PC2")
        
        # Enhanced legend
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True,
                 fontsize=THEME.BODY_SIZE, framealpha=0.9)
        
        # Add caption
        fig.text(0.5, 0.02, "Points colored by cluster; transparency shows density", 
                ha='center', fontsize=THEME.CAPTION_SIZE, 
                color=THEME.NEUTRAL_MID, style='italic')
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=THEME.DPI, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        images['embedding'] = buffer.getvalue()
        plt.close()
        
        # 2. Enhanced Model Selection Plot
        if max_k is not None:
            k_vals = list(range(2, min(max_k, len(X_scaled) - 1) + 1))
            if len(k_vals) > 0:
                inertias, sils = [], []
                for k in k_vals:
                    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
                    lab = km.fit_predict(X_scaled)
                    inertias.append(km.inertia_)
                    sils.append(silhouette_score(X_scaled, lab))
                
                inertia_norm = np.asarray(inertias, float) / max(inertias)
                
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                # Plot inertia with smooth line
                line1 = ax1.plot(k_vals, inertia_norm, marker="o", linewidth=2.5, 
                               markersize=8, label="Normalized Inertia (Elbow)", 
                               color=THEME.PRIMARY, markerfacecolor=THEME.PRIMARY,
                               markeredgecolor='white', markeredgewidth=1)
                
                # Apply theme to primary axis
                apply_theme(ax1, xlabel="Number of Clusters (k)", ylabel="Normalized Inertia [0..1]")
                ax1.set_ylabel("Normalized Inertia [0..1]", fontsize=THEME.BODY_SIZE, 
                              color=THEME.PRIMARY, fontweight='500')
                ax1.tick_params(axis='y', labelcolor=THEME.PRIMARY)
                
                # Create secondary axis for silhouette
                ax2 = ax1.twinx()
                line2 = ax2.plot(k_vals, sils, marker="s", linewidth=2.5, markersize=7,
                               label="Silhouette Score", color=THEME.SECONDARY, 
                               linestyle='--', markerfacecolor=THEME.SECONDARY,
                               markeredgecolor='white', markeredgewidth=1)
                ax2.set_ylabel("Silhouette Score [-1..1]", fontsize=THEME.BODY_SIZE, 
                              color=THEME.SECONDARY, fontweight='500')
                ax2.tick_params(axis='y', labelcolor=THEME.SECONDARY)
                
                # Highlight selected k with accent color
                selected_k = manual_k if manual_k is not None else k_vals[np.argmax(sils)]
                if selected_k in k_vals:
                    k_idx = k_vals.index(selected_k)
                    ax1.axvline(x=selected_k, color=THEME.ACCENT, linestyle='-', 
                               linewidth=3, alpha=0.8)
                    
                    # Annotate selected k
                    ax1.annotate(f'Selected k = {selected_k}', 
                               xy=(selected_k, inertia_norm[k_idx]), 
                               xytext=(selected_k + 0.5, inertia_norm[k_idx] + 0.1),
                               arrowprops=dict(arrowstyle='->', color=THEME.ACCENT, lw=2),
                               fontsize=THEME.BODY_SIZE, fontweight='bold',
                               color=THEME.ACCENT,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                       edgecolor=THEME.ACCENT, alpha=0.9))
                
                # Enhanced title
                ax1.set_title("Model Selection: Inertia vs Silhouette Analysis", 
                             fontsize=THEME.HEADING_SIZE, fontweight='600',
                             color=THEME.NEUTRAL_DARK, pad=THEME.MARGIN_MEDIUM)
                
                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper right', frameon=True, 
                          fancybox=True, shadow=True, fontsize=THEME.BODY_SIZE)
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=THEME.DPI, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                buffer.seek(0)
                images['model_selection'] = buffer.getvalue()
                plt.close()
        
        # 3. Enhanced Final Clustering Plot
        fig, ax = plt.subplots(figsize=THEME.FIGURE_SIZE)
        
        # Plot clusters with theme colors and centroids
        for cluster_id in sorted(labels.unique()):
            mask = labels == cluster_id
            cluster_points = embed.loc[mask]
            color = THEME.get_cluster_color(cluster_id)
            
            # Plot points
            ax.scatter(cluster_points["X"], cluster_points["Y"], 
                      s=THEME.POINT_SIZE, 
                      alpha=0.5,  # Slightly stronger than embedding plot
                      color=color,
                      edgecolors=color,
                      linewidth=0.5,
                      label=legend_cluster_label(cluster_id, cluster_counts[cluster_id], total_points))
            
            # Add centroid annotation
            centroid_x = cluster_points["X"].mean()
            centroid_y = cluster_points["Y"].mean()
            
            # Create rounded rectangle for centroid label
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, 
                             edgecolor='white', alpha=0.9, linewidth=1.5)
            ax.text(centroid_x, centroid_y, f'C{cluster_id}', 
                   ha='center', va='center', fontsize=THEME.BODY_SIZE, 
                   fontweight='bold', color='white', bbox=bbox_props)
        
        # Apply theme styling
        apply_theme(ax, title=f"Final Clustering Results (k={len(labels.unique())})", 
                   xlabel="PC1", ylabel="PC2")
        
        # Enhanced legend
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True,
                 fontsize=THEME.BODY_SIZE, framealpha=0.9)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=THEME.DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        images['clustering'] = buffer.getvalue()
        plt.close()
        
        # 4. Enhanced Feature Distribution Plots (violin+box hybrid)
        top_features = results["kmeans_drivers"].groupby("feature")["z_median"].apply(
            lambda s: np.nanmean(np.abs(s))).sort_values(ascending=False).head(6).index.tolist()
        
        if top_features:
            n_features = len(top_features)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            tmp = X_scaled.copy()
            tmp["Cluster"] = labels.values
            
            for i, col in enumerate(top_features):
                if i < len(top_features):
                    row, col_idx = i // n_cols, i % n_cols
                    ax = axes[row, col_idx]
                    
                    # Create violin+box hybrid plot
                    # First, violin plot with light fill
                    violin_parts = ax.violinplot([tmp[tmp["Cluster"] == c][col].dropna() 
                                                for c in sorted(tmp["Cluster"].unique())],
                                               positions=sorted(tmp["Cluster"].unique()),
                                               showmeans=False, showmedians=False, showextrema=False)
                    
                    # Style violin plots
                    for pc in violin_parts['bodies']:
                        pc.set_facecolor(THEME.NEUTRAL_LIGHT)
                        pc.set_alpha(0.6)
                        pc.set_edgecolor(THEME.NEUTRAL_MID)
                        pc.set_linewidth(0.8)
                    
                    # Add box plot on top
                    box_parts = ax.boxplot([tmp[tmp["Cluster"] == c][col].dropna() 
                                          for c in sorted(tmp["Cluster"].unique())],
                                         positions=sorted(tmp["Cluster"].unique()),
                                         patch_artist=True, widths=0.3,
                                         boxprops=dict(facecolor='white', alpha=0.8, 
                                                      edgecolor=THEME.NEUTRAL_DARK, linewidth=1),
                                         medianprops=dict(color=THEME.PRIMARY, linewidth=2),
                                         whiskerprops=dict(color=THEME.NEUTRAL_DARK, linewidth=1),
                                         capprops=dict(color=THEME.NEUTRAL_DARK, linewidth=1),
                                         flierprops=dict(marker='o', markerfacecolor=THEME.NEUTRAL_MID, 
                                                        markeredgecolor=THEME.NEUTRAL_DARK, 
                                                        markersize=4, alpha=0.6))
                    
                    # Add overall median line
                    overall_median = tmp[col].median()
                    ax.axhline(y=overall_median, color=THEME.NEUTRAL_MID, 
                              linestyle='--', linewidth=1.5, alpha=0.8, 
                              label='Overall Median' if i == 0 else "")
                    
                    # Apply theme styling
                    apply_theme(ax, title=f"{prettify_name(col)} (by cluster)")
                    ax.set_xlabel("Cluster", fontsize=THEME.BODY_SIZE)
                    ax.set_ylabel(prettify_name(col), fontsize=THEME.BODY_SIZE)
                    
                    # Set consistent y-limits for better comparison
                    y_min, y_max = tmp[col].quantile([0.05, 0.95])
                    y_range = y_max - y_min
                    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            # Hide empty subplots
            for i in range(n_features, n_rows * n_cols):
                row, col_idx = i // n_cols, i % n_cols
                axes[row, col_idx].set_visible(False)
            
            # Enhanced title
            fig.suptitle("Feature Distributions by Cluster (Top Features)", 
                        fontsize=THEME.HEADING_SIZE, fontweight='600',
                        color=THEME.NEUTRAL_DARK, y=0.98)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=THEME.DPI, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            images['feature_distributions'] = buffer.getvalue()
            plt.close()
        
    except Exception as e:
        print(f"Error generating visualization images: {e}")
    
    return images


def generate_clustering_pdf(results, file_path=None, target=None, top_n=3, 
                           scaling="minmax", embedder="PCA", manual_k=None, 
                           max_k=None, outlier_method="none", contamination=0.03,
                           preprocessed_data=None, original_filename=None):
    """
    Generate a comprehensive PDF report of clustering results.
    
    Args:
        results: Dictionary containing clustering results
        file_path: Path to original CSV file
        target: Target column name
        top_n: Number of top features to highlight
        scaling: Scaling method used
        embedder: Embedding method used
        manual_k: Manual k value if used
        max_k: Maximum k value tested
        outlier_method: Outlier removal method
        contamination: Outlier contamination ratio
    
    Returns:
        bytes: PDF content as bytes
    """
    if not HAS_REPORTLAB:
        raise ImportError("reportlab is required for PDF generation. Install with: pip install reportlab")
    
    # Create PDF in memory with enhanced metadata
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, 
                           topMargin=72, bottomMargin=18)
    
    # Get enhanced styles with theme colors
    styles = getSampleStyleSheet()
    
    # Enhanced title style
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                fontSize=THEME.TITLE_SIZE, spaceAfter=30, 
                                alignment=TA_CENTER, fontName='Helvetica-Bold',
                                textColor=colors.Color(47/255, 109/255, 179/255))
    
    # Enhanced heading style with accent bar
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], 
                                  fontSize=THEME.HEADING_SIZE, spaceAfter=12, spaceBefore=12,
                                  fontName='Helvetica-Bold', textColor=colors.Color(51/255, 65/255, 85/255),
                                  leftIndent=0, borderWidth=0, borderPadding=0)
    
    # Enhanced normal style
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'],
                                 fontSize=THEME.BODY_SIZE, fontName='Helvetica',
                                 textColor=colors.Color(51/255, 65/255, 85/255),
                                 spaceAfter=6, spaceBefore=6)
    
    # Build PDF content
    story = []
    
    # Enhanced cover header with color bar
    if original_filename:
        title_text = f"Cluster Analysis Report"
        subtitle_text = f"{original_filename} — generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    else:
        title_text = "Cluster Analysis Report"
        subtitle_text = f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    
    # Add color bar
    color_bar = Table([[title_text]], colWidths=[6*inch])
    color_bar.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.Color(47/255, 109/255, 179/255)),
        ('TEXTCOLOR', (0, 0), (0, 0), colors.white),
        ('ALIGN', (0, 0), (0, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, 0), THEME.TITLE_SIZE),
        ('BOTTOMPADDING', (0, 0), (0, 0), 15),
        ('TOPPADDING', (0, 0), (0, 0), 15),
    ]))
    story.append(color_bar)
    story.append(Spacer(1, 8))
    
    # Add subtitle
    subtitle_style = ParagraphStyle('Subtitle', parent=normal_style,
                                   fontSize=THEME.BODY_SIZE, alignment=TA_CENTER,
                                   textColor=colors.Color(148/255, 163/255, 184/255))
    story.append(Paragraph(subtitle_text, subtitle_style))
    story.append(Spacer(1, 20))
    
    # Report metadata
    story.append(Paragraph("Report Information", heading_style))
    metadata_data = [
        ['Generated on (UTC):', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')],
        ['Original file:', original_filename if original_filename else 'Not specified'],
        ['Scaling method:', scaling],
        ['Embedding method:', embedder],
        ['Outlier removal:', outlier_method],
        ['Outlier contamination:', f"{contamination:.1%}" if outlier_method != "none" else "N/A"],
    ]
    
    if manual_k is not None:
        metadata_data.append(['K selection:', f'Manual (k={manual_k})'])
    else:
        metadata_data.append(['K selection:', f'Auto (max_k={max_k})'])
    
    metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metadata_table)
    story.append(Spacer(1, 20))
    
    # Data preprocessing information (moved up)
    if preprocessed_data is not None:
        story.append(Paragraph("Data Preprocessing Summary", heading_style))
        
        prep_data = [
            ['Original data shape:', f"{preprocessed_data['original_shape'][0]} × {preprocessed_data['original_shape'][1]}"],
            ['Final data shape:', f"{preprocessed_data['final_shape'][0]} × {preprocessed_data['final_shape'][1]}"],
            ['Outliers removed:', str(preprocessed_data['outliers_removed'])],
            ['Categorical features encoded:', str(preprocessed_data['categorical_encoded'])],
            ['Scaling method:', preprocessed_data['scaling_method']],
            ['Outlier method:', preprocessed_data['outlier_method']],
        ]
        
        prep_table = Table(prep_data, colWidths=[2*inch, 2*inch])
        prep_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(prep_table)
        story.append(Spacer(1, 20))
    
    # Visualizations (moved up)
    story.append(Paragraph("Visualizations", heading_style))
    
    # Generate visualization images
    images = generate_visualization_images(results, embedder, manual_k, max_k)
    
    # Add embedding plot
    if 'embedding' in images:
        story.append(Paragraph("2D Embedding Plot", heading_style))
        story.append(Paragraph("Shows the data structure in reduced dimensions using " + embedder + ".", normal_style))
        img_buffer = io.BytesIO(images['embedding'])
        img = Image(img_buffer, width=6*inch, height=4.5*inch)
        story.append(img)
        story.append(Spacer(1, 12))
    
    # Add model selection plot
    if 'model_selection' in images:
        story.append(Paragraph("Model Selection Analysis", heading_style))
        story.append(Paragraph("Shows inertia (elbow method) and silhouette scores for different k values. The selected k is highlighted.", normal_style))
        img_buffer = io.BytesIO(images['model_selection'])
        img = Image(img_buffer, width=7*inch, height=4.2*inch)
        story.append(img)
        story.append(Spacer(1, 12))
    
    # Add clustering results plot
    if 'clustering' in images:
        story.append(Paragraph("Final Clustering Results", heading_style))
        story.append(Paragraph("Shows the final cluster assignments on the 2D embedding. Each color represents a different cluster.", normal_style))
        img_buffer = io.BytesIO(images['clustering'])
        img = Image(img_buffer, width=6*inch, height=4.5*inch)
        story.append(img)
        story.append(Spacer(1, 12))
    
    # Add feature distributions plot
    if 'feature_distributions' in images:
        story.append(Paragraph("Feature Distributions by Cluster", heading_style))
        story.append(Paragraph("Box plots showing how the most important features are distributed across different clusters.", normal_style))
        img_buffer = io.BytesIO(images['feature_distributions'])
        img = Image(img_buffer, width=7*inch, height=5*inch)
        story.append(img)
        story.append(Spacer(1, 12))
    
    story.append(Spacer(1, 20))
    
    # Data summary
    story.append(Paragraph("Data Summary", heading_style))
    X_scaled = results["X_scaled"]
    labels = results["kmeans_labels"]
    
    summary_data = [
        ['Total samples:', str(len(X_scaled))],
        ['Total features:', str(len(X_scaled.columns))],
        ['Number of clusters:', str(len(labels.unique()))],
        ['Silhouette score:', f"{silhouette_score(X_scaled, labels):.3f}"],
    ]
    
    # Add cluster sizes
    cluster_sizes = labels.value_counts().sort_index()
    for i, (cluster_id, size) in enumerate(cluster_sizes.items()):
        summary_data.append([f'Cluster {cluster_id} size:', str(size)])
    
    summary_table = Table(summary_data, colWidths=[2*inch, 1*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Enhanced Top Driver Features table
    story.append(Paragraph("Top Driver Features", heading_style))
    # Usar drivers en unidades originales si están disponibles
    if "kmeans_drivers_orig" in results and not results["kmeans_drivers_orig"].empty:
        drivers = results["kmeans_drivers_orig"]
        use_original_units = True
    else:
        drivers = results["kmeans_drivers"]
        use_original_units = False
    
    if not drivers.empty:
        # Create enhanced table data with prettified names and formatted values
        table_data = [['Cluster', 'Feature', 'Direction', 'Z-Score', 'Cluster Median', 'Overall Median']]
        
        for cluster_id in sorted(drivers['cluster'].unique()):
            cluster_drivers = drivers[drivers['cluster'] == cluster_id].head(top_n)
            for _, row in cluster_drivers.iterrows():
                if use_original_units:
                    z_score = row['z_score']
                    cluster_med = row['cluster_median_orig']
                    overall_med = row['overall_median_orig']
                else:
                    z_score = row['z_median']
                    cluster_med = row['cluster_median']
                    overall_med = row['overall_median']
                
                # Format direction with symbol and color
                direction_symbol = get_direction_symbol(row['direction'])
                direction_text = f"{direction_symbol} {row['direction'].title()}"
                
                table_data.append([
                    str(int(row['cluster'])),
                    prettify_name(row['feature']),
                    direction_text,
                    f"{z_score:.2f}",
                    fmt_val(cluster_med),
                    fmt_val(overall_med)
                ])
        
        drivers_table = Table(table_data, colWidths=[0.8*inch, 1.5*inch, 0.8*inch, 0.8*inch, 1*inch, 1*inch])
        
        # Enhanced table styling with theme colors
        primary_color = colors.Color(47/255, 109/255, 179/255)  # THEME.PRIMARY
        neutral_light = colors.Color(229/255, 231/255, 235/255)  # THEME.NEUTRAL_LIGHT
        neutral_dark = colors.Color(51/255, 65/255, 85/255)     # THEME.NEUTRAL_DARK
        
        drivers_table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            
            # Body styling with alternating rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, neutral_light]),
            ('TEXTCOLOR', (0, 1), (-1, -1), neutral_dark),
            
            # Grid and borders
            ('GRID', (0, 0), (-1, -1), 0.5, neutral_dark),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(drivers_table)
    
    story.append(Spacer(1, 20))
    
    # Enhanced Cluster Interpretations with styled callout boxes
    story.append(Paragraph("Cluster Interpretations", heading_style))
    summaries = auto_describe_clusters(results, file_path, target, top_n)
    
    for i, summary in enumerate(summaries):
        # Extract cluster ID for color coding
        cluster_id = i  # Assuming summaries are in cluster order
        
        # Create styled callout box for each cluster interpretation
        cluster_color = THEME.get_cluster_color(cluster_id)
        
        # Convert theme color to ReportLab color
        rgb_color = tuple(int(cluster_color[i:i+2], 16) for i in (1, 3, 5))
        reportlab_color = colors.Color(rgb_color[0]/255, rgb_color[1]/255, rgb_color[2]/255)
        
        # Create a styled paragraph with background and border
        cluster_style = ParagraphStyle(
            'ClusterInterpretation',
            parent=normal_style,
            leftIndent=20,
            rightIndent=20,
            spaceBefore=8,
            spaceAfter=8,
            borderWidth=2,
            borderColor=reportlab_color,
            borderPadding=8,
            backColor=colors.Color(0.95, 0.95, 0.95),  # Light gray background
            fontSize=10,
            fontName='Helvetica'
        )
        
        # Bold key parts of the summary
        enhanced_summary = summary
        # Bold cluster name and key figures
        import re
        enhanced_summary = re.sub(r'(Cluster \d+)', r'<b>\1</b>', enhanced_summary)
        enhanced_summary = re.sub(r'(\d+% vs \d+%)', r'<b>\1</b>', enhanced_summary)
        enhanced_summary = re.sub(r'(much higher|much lower|slightly higher|slightly lower|similar)', 
                                 r'<b>\1</b>', enhanced_summary)
        
        story.append(Paragraph(enhanced_summary, cluster_style))
        story.append(Spacer(1, 8))
    
    # Add note about the report
    story.append(Spacer(1, 20))
    story.append(Paragraph("Report Summary", heading_style))
    story.append(Paragraph(
        "This comprehensive report includes all numerical results, statistical summaries, and visualizations "
        "from the clustering analysis. The visualizations above show the data structure, model selection process, "
        "final clustering results, and feature distributions. This report provides a complete overview of the "
        "clustering analysis for documentation, presentation, or further analysis purposes.",
        normal_style
    ))
    
    # Build PDF with enhanced metadata
    doc.build(story)
    buffer.seek(0)
    
    # Add PDF metadata
    pdf_content = buffer.getvalue()
    
    # Note: ReportLab metadata is set during document creation
    # The metadata includes title, author, subject as specified in the doc.build() call
    
    return pdf_content



