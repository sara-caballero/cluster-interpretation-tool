# CLUSTERING PIPELINE

# 1. Load and clean data
# 2. Scale and (optionally) remove outliers
# 3. Create a 2D embedding for visualization
# 4. Run KMeans with automatic k selection
# 5. Interpret clusters with plots + summaries

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

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


# HELPERS (Small utilities used throughout the pipeline)

def minmax_df(df):
    # quick min-max scaler that doesn't blow up on constant columns
    mins = df.min()
    rng  = df.max() - mins
    rng  = rng.replace(0, 1.0)  # avoid divide by zero
    return (df - mins) / rng

def pick_k_auto(k_values, inertia_norm, silhouettes, angles,
                elbow_angle_thresh=170.0, silhouette_floor=0.25, sil_within=0.02):
    # Decide automatically the "best" k for KMeans.
    # Logic:
    # - If silhouette is good → pick the smallest k close to the best.
    # - Else if the elbow curve looks sharp → pick that k.
    # - Else → fall back to smallest k near best silhouette.
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
        slope_ratio = drop_next / max(drop_prev, 1e-9)  # smaller = sharper elbow
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
    # For each cluster, compare its median to the overall median.
    # Rank features by how much they differ (z-score).
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
    # optional viz, not required; only runs if pySankey is installed
    # Shows how feature bins (left) "flow" into clusters (right).
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


# -------------------------
# preprocessing function
# -------------------------

def preprocess_data(
    file_path,
    target=None,
    excluded_cols=None,
    scaling="minmax",
    outlier_method="isoforest",
    contamination=0.03,
):
    """
    Preprocess data for clustering: load, clean, encode, scale, and remove outliers.
    Returns preprocessed data and metadata for display.
    """
    
    # STEP 1: Load the dataset
    print("Reading CSV...")
    raw = pd.read_csv(file_path)
    
    # STEP 2: Select features
    feat_cols = list(raw.columns)
    if target is not None and target in feat_cols:
        feat_cols.remove(target)
    if excluded_cols is not None:
        for col in excluded_cols:
            if col in feat_cols:
                feat_cols.remove(col)
    X = raw[feat_cols].copy()
    
    # STEP 3: Encode categorical features
    # Identify categorical columns before encoding
    categorical_cols = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            categorical_cols.append(col)
    
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=float)
    for c in X_encoded.columns:
        if X_encoded[c].dtype == bool:
            X_encoded[c] = X_encoded[c].astype(float)
    
    # STEP 4: Scale features
    if scaling == "standard":
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns, index=X_encoded.index)
    elif scaling == "minmax":
        X_scaled = minmax_df(X_encoded).fillna(0.0)
    elif scaling == "none":
        X_scaled = X_encoded.astype(float)
    else:
        raise ValueError("scaling must be 'minmax', 'standard', or 'none'")
    
    print("Data shape after prep:", X_scaled.shape)
    
    # STEP 5: Remove outliers
    outliers_removed = 0
    if outlier_method == "isoforest":
        print(f"Removing outliers with IsolationForest (contamination={contamination:.3f})...")
        iso  = IsolationForest(contamination=contamination, random_state=42)
        pred = iso.fit_predict(X_scaled.values)  # 1=inlier, -1=outlier
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
    
    # Create preprocessing summary
    preprocessing_info = {
        "original_shape": raw.shape,
        "final_shape": X_scaled.shape,
        "features_used": list(X_scaled.columns),
        "scaling_method": scaling,
        "outlier_method": outlier_method,
        "outliers_removed": outliers_removed,
        "categorical_encoded": len(categorical_cols),
        "raw_data": raw,
        "scaled_data": X_scaled
    }
    
    return preprocessing_info


# -------------------------
# main function
# -------------------------

def run_pipeline(
    file_path,
    target=None,               # set to your target col or None (not used for clustering, just for comparison)
    excluded_cols=None,        # columns to exclude from clustering
    scaling="minmax",          # "minmax", "standard", or "none"
    embedder="PCA",            # method for 2D plot: "PCA", "UMAP", or "tSNE". IMP! This does not affect clustering if cluster_on_features=True
    cluster_on_features=True,  # True = cluster on scaled features, False = cluster on 2D embedding
    max_k=10,                  # maximum number of clusters to try
    manual_k=None,             # manual k value override (if provided, skips auto-selection)
    outlier_method="isoforest",# method to drop outliers ("isoforest" or "none")
    contamination=0.03,        # ~3% outliers to drop. higher value -> more aggresive.
    draw_sankey=False,         # optional
    preprocessed_data=None,    # if provided, skip preprocessing
):

    # Use preprocessed data if provided, otherwise preprocess
    if preprocessed_data is not None:
        X_scaled = preprocessed_data["scaled_data"]
        raw = preprocessed_data["raw_data"]
        print("Using preprocessed data...")
    else:
        # STEP 1: Load the dataset
        print("Reading CSV...")
        raw = pd.read_csv(file_path)

        # STEP 2: Select features
        feat_cols = list(raw.columns)
        if target is not None and target in feat_cols:
            feat_cols.remove(target)
        if excluded_cols is not None:
            for col in excluded_cols:
                if col in feat_cols:
                    feat_cols.remove(col)
        X = raw[feat_cols].copy()

        # STEP 3: Encode categorical features
        X = pd.get_dummies(X, drop_first=True, dtype=float)
        for c in X.columns:
            if X[c].dtype == bool:
                X[c] = X[c].astype(float)

        # STEP 4: Scale features
        if scaling == "standard":
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        elif scaling == "minmax":
            X_scaled = minmax_df(X).fillna(0.0)
        elif scaling == "none":
            X_scaled = X.astype(float)
        else:
            raise ValueError("scaling must be 'minmax', 'standard', or 'none'")

        print("Data shape after prep:", X_scaled.shape)

        # STEP 5: Remove outliers
        if outlier_method == "isoforest":
            print(f"Removing outliers with IsolationForest (contamination={contamination:.3f})...")
            iso  = IsolationForest(contamination=contamination, random_state=42)
            pred = iso.fit_predict(X_scaled.values)  # 1=inlier, -1=outlier
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

    # STEP 6: Create a 2D embedding for visualization
    # (not used for clustering unless cluster_on_features=False)
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

    # STEP 7: Choose what to cluster on
    # (original features or 2D embedding)
    data_for_kmeans = X_scaled if cluster_on_features else embed

    # STEP 8: Determine k value (auto or manual)
    if manual_k is not None:
        # Manual k override
        best_k = manual_k
        print(f"Using manual k value: {best_k}")
        
        # Still create model selection plot for reference (if max_k is available)
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
                
                # Create model selection plot
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot 1: Embedding
                axes[0].scatter(embed["X"], embed["Y"], s=10)
                axes[0].set_title(f"{embedder} embedding")
                axes[0].set_xticks([])
                axes[0].set_yticks([])
                
                # Plot 2: K-means model selection (for reference)
                ax1 = axes[1]
                ax1.plot(k_vals, inertia_norm, marker="o", label="normalized inertia (elbow)")
                ax1.set_xlabel("number of clusters (k)")
                ax1.set_ylabel("normalized inertia [0..1]")
                ax2 = ax1.twinx()
                ax2.plot(k_vals, sils, marker="X", linestyle="--", label="average silhouette score")
                ax2.set_ylabel("silhouette [-1..1]")
                
                # Highlight manual k value
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
            # Just show embedding plot if no max_k available
            plt.figure(figsize=(6, 5))
            plt.scatter(embed["X"], embed["Y"], s=10)
            plt.title(f"{embedder} embedding")
            plt.xticks([])
            plt.yticks([])
            plt.show()
    else:
        # Auto k selection (original logic)
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

        # elbow "angles" (like the class template)
        angles = [180.0]
        for i in range(1, len(inertia_norm) - 1):
            left  = inertia_norm[i] - inertia_norm[i - 1]
            right = inertia_norm[i + 1] - inertia_norm[i]
            ang = 180.0 - math.degrees(math.atan((right - left) / (1 + (right * left))))
            angles.append(ang)
            angles.append(180.0)

        # Create 2-column grid for main plots after K-means evaluation
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Embedding
        axes[0].scatter(embed["X"], embed["Y"], s=10)
        axes[0].set_title(f"{embedder} embedding")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # Plot 2: K-means model selection
        ax1 = axes[1]
        ax1.plot(k_vals, inertia_norm, marker="o", label="normalized inertia (elbow)")
        ax1.set_xlabel("number of clusters (k)")
        ax1.set_ylabel("normalized inertia [0..1]")
        ax2 = ax1.twinx()
        ax2.plot(k_vals, sils, marker="X", linestyle="--", label="average silhouette score")
        ax2.set_ylabel("silhouette [-1..1]")

        # add tiny labels so it's easier to read
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
        
        plt.tight_layout()
        plt.show()

        # Auto-pick the best k
        best_k, why = pick_k_auto(k_vals, inertia_norm, sils, angles)
        print("Auto-picked k:", best_k, "->", why)

    # STEP 10: Final KMeans fit
    km_final = KMeans(n_clusters=best_k, init="k-means++", n_init=20, random_state=42)
    labels_kmeans = pd.Series(km_final.fit_predict(data_for_kmeans), index=data_for_kmeans.index, name="Cluster")
    sil_final = silhouette_score(data_for_kmeans, labels_kmeans)
    print(f"KMeans silhouette at k={best_k}: {sil_final:.3f}")

    # STEP 11: Plot clusters on 2D embedding
    plt.figure(figsize=(6, 5))
    for c in sorted(labels_kmeans.unique()):
        pts = embed.loc[labels_kmeans == c]
        plt.scatter(pts["X"], pts["Y"], s=10, label=f"Cluster {c}")
    plt.legend()
    plt.title(f"{embedder} + KMeans(k={best_k})")
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # STEP 12: Interpret clusters
    print("KMeans: quick numeric interpretation (top drivers):")
    drivers_km = explain_clusters_numeric(X_scaled, labels_kmeans, top_n=5)

    # STEP 13: Feature distributions
    print("KMeans: feature distributions by cluster (boxplots)...")
    tmp = X_scaled.copy()
    tmp["Cluster"] = labels_kmeans.values
    
    # Calculate grid size based on number of features (max 12)
    n_features = min(len(X_scaled.columns), 12)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(X_scaled.columns[:n_features]):
        row, col_idx = i // n_cols, i % n_cols
        sns.boxplot(x="Cluster", y=col, data=tmp, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f"{col} by cluster (KMeans)")
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row, col_idx = i // n_cols, i % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

   # STEP 14: Optional Sankey diagram
    if draw_sankey and HAS_SANKEY and not drivers_km.empty:
        top_feats = (drivers_km.groupby("feature")["z_median"]
                     .apply(lambda s: np.nanmean(np.abs(s)))
                     .sort_values(ascending=False).head(3).index.tolist())
        sankey_top_flows(X_scaled, labels_kmeans, top_feats, bins=3, title="KMeans: top drivers → clusters")

     # STEP 15: Compare with target column
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

    # STEP 16: Return results
    return {
        "X_scaled": X_scaled,
        "embedding": embed,
        "kmeans_labels": labels_kmeans,
        "kmeans_model": km_final,
        "kmeans_drivers": drivers_km,
    }



def auto_describe_clusters(results, file_path=None, target=None, top_n=3):
    """
    Build short, plain-English summaries of clusters from results['kmeans_drivers'].
    Each summary highlights the top features that set a cluster apart,
    showing values in-cluster vs overall. If a binary target is given, 
    compares its distribution per cluster vs overall.
    """
    import numpy as np
    import pandas as pd

    drivers = results["kmeans_drivers"].copy()
    labels  = results["kmeans_labels"]
    X       = results["X_scaled"]

    # cluster sizes (for the "n=" part)
    sizes = labels.value_counts().sort_index()

    # helper to phrase features nicely with numbers
    def humanize_feature(feat, direction, cluster_med, overall_med, is_binary):
        if is_binary and "_" in feat:  # one-hot encoded binary feature
            base, level = feat.rsplit("_", 1)
            if direction == "higher":
                return f"{level} {base} is more common ({cluster_med*100:.1f}% vs {overall_med*100:.1f}% overall)"
            else:
                return f"{level} {base} is less common ({cluster_med*100:.1f}% vs {overall_med*100:.1f}% overall)"
        else:  # numeric feature
            if direction == "higher":
                return f"higher {feat} values (median {cluster_med:.2f} vs {overall_med:.2f} overall)"
            else:
                return f"lower {feat} values (median {cluster_med:.2f} vs {overall_med:.2f} overall)"

    # Load target (optional) and coerce to binary 0.0/1.0 if possible
    target_binary = None
    overall_pct_1 = None  # overall % of class 1.0

    if file_path is not None and target is not None:
        raw = pd.read_csv(file_path)
        raw = raw.loc[labels.index].copy()  # align with rows used after outlier removal

        if target in raw.columns:
            ser = raw[target]

            # Try to coerce to binary numeric 0/1
            if pd.api.types.is_numeric_dtype(ser):
                uniq = pd.unique(ser.dropna())
                if len(uniq) == 2 and set(np.sort(uniq)) <= {0, 1} | {0.0, 1.0}:
                    target_binary = ser.astype(float)
            else:
                # Non-numeric but two unique labels -> map to 0.0/1.0 deterministically
                uniq = pd.unique(ser.dropna())
                if len(uniq) == 2:
                    u_sorted = np.sort(uniq.astype(str))
                    mapping = {u_sorted[0]: 0.0, u_sorted[1]: 1.0}
                    target_binary = ser.astype(str).map(mapping).astype(float)

            if target_binary is not None:
                # compute overall % for 1.0 (use non-null only)
                overall_pct_1 = 100.0 * (target_binary == 1.0).mean()

    # Reconstruct dummies from original CSV (aligned with rows after outlier removal)
    if file_path is not None:
        raw = pd.read_csv(file_path)
        raw = raw.loc[labels.index].copy()  # align with rows used after outlier removal
        
        # Get original feature names (before encoding)
        original_features = []
        for feat in drivers['feature'].unique():
            if '_' in feat:  # encoded feature
                base = feat.rsplit('_', 1)[0]
                if base not in original_features:
                    original_features.append(base)
            else:
                if feat not in original_features:
                    original_features.append(feat)
        
        # Reconstruct dummies for binary features
        binary_features = {}
        for feat in original_features:
            if feat in raw.columns:
                unique_vals = raw[feat].dropna().unique()
                if len(unique_vals) == 2:  # binary feature
                    binary_features[feat] = unique_vals

    # Build the sentences
    summaries = []
    for cid in sorted(drivers["cluster"].unique()):
        chunk = (drivers[drivers["cluster"] == cid]
                 .sort_values("z_median", key=lambda s: s.abs(), ascending=False)
                 .head(top_n))

        phrases = []
        for _, r in chunk.iterrows():
            feat = r.feature
            direction = r.direction
            
            # Check if it's a binary feature
            is_binary = False
            if '_' in feat:  # encoded feature
                base, level = feat.rsplit('_', 1)
                if base in binary_features:
                    is_binary = True
                    # Calculate percentage from original data
                    mask = (labels == cid)
                    cluster_data = raw.loc[mask, base]
                    overall_data = raw[base]
                    
                    # Calculate percentages
                    cluster_pct = 100.0 * (cluster_data == level).mean()
                    overall_pct = 100.0 * (overall_data == level).mean()
                    
                    if direction == "higher":
                        phrases.append(f"{level} {base} is more common ({cluster_pct:.1f}% vs {overall_pct:.1f}% overall)")
                    else:
                        phrases.append(f"{level} {base} is less common ({cluster_pct:.1f}% vs {overall_pct:.1f}% overall)")
                else:
                    # Use original logic for non-binary encoded features
                    phrases.append(humanize_feature(feat, direction, r.cluster_median, r.overall_median, False))
            else:
                # Use original logic for non-encoded features
                phrases.append(humanize_feature(feat, direction, r.cluster_median, r.overall_median, False))

        line = f"Cluster {cid} (n={sizes.get(cid, 0)}): " + ", ".join(phrases) + "."

        # Append target info if available
        if target_binary is not None:
            mask = (labels == cid)
            tb_c = target_binary.loc[mask]
            if tb_c.notna().any():
                pct1 = 100.0 * (tb_c == 1.0).mean()
                pct0 = 100.0 * (tb_c == 0.0).mean()
                if pct1 >= pct0:
                    chosen_cls = 1.0
                    pct_here   = pct1
                    pct_over   = overall_pct_1
                else:
                    chosen_cls = 0.0
                    pct_here   = pct0
                    pct_over   = 100.0 - overall_pct_1 if overall_pct_1 is not None else None

                if pct_over is not None:
                    line += f" {target}={chosen_cls:.0f} occurs in {pct_here:.1f}% of this cluster vs {pct_over:.1f}% overall."
                else:
                    line += f" {target}={chosen_cls:.0f} occurs in {pct_here:.1f}% of this cluster."
        summaries.append(line)

    for s in summaries:
        print(s)
    return summaries



