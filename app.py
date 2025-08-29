import io
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import contextlib

# your pipeline + auto_describe live here:
from pipeline import run_pipeline, auto_describe_clusters, preprocess_data

st.set_page_config(page_title="Cluster Interpretation Tool", layout="wide")
st.title("🔍 Cluster Interpretation Tool")

st.markdown(
    """
Upload a CSV, preprocess the data, and get:
- automatic K selection (silhouette + elbow),
- 2D embedding plot,
- KMeans clusters,
- plain-English, numeric **cluster summaries** (in-cluster vs **overall**),
- optional target vs cluster comparison.
"""
)

# ---- helper: route plt.show() into Streamlit ----
@contextlib.contextmanager
def streamlit_matplotlib():
    orig_show = plt.show

    def st_show(*args, **kwargs):
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    plt.show = st_show
    try:
        yield
    finally:
        plt.show = orig_show

# ---- Upload CSV ----
file = st.file_uploader("Upload a CSV file", type=["csv"])

if file:
    # persist as a temp path because your pipeline expects a filesystem path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    # quick peek (limits huge files)
    df_preview = pd.read_csv(tmp_path, nrows=500)
    with st.expander("🔎 Preview (first 500 rows)"):
        st.dataframe(df_preview)

    # ---- Controls ----
    st.sidebar.header("⚙️ Settings")
    all_cols = list(df_preview.columns)
    target = st.sidebar.selectbox("Target column (optional)", [None] + all_cols, index=0)
    scaling = st.sidebar.radio("Scaling", ["minmax", "standard", "none"], index=0)
    embedder = st.sidebar.radio("Embedding (for plots)", ["PCA", "UMAP"], index=0)
    cluster_on_features = st.sidebar.checkbox("Cluster on full features (recommended)", value=True)
    max_k = st.sidebar.slider("Max k to try", 2, 15, 8)
    outlier_method = st.sidebar.radio("Outlier removal", ["isoforest", "none"], index=0)
    contamination = st.sidebar.slider("Outlier contamination", 0.0, 0.10, 0.03, 0.01)
    draw_sankey = st.sidebar.checkbox("Draw Sankey (requires pySankey)", value=False)

    # Initialize session state for preprocessed data
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None

    # ---- Preprocess Data Button ----
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🔧 Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                preprocessed_info = preprocess_data(
                    file_path=tmp_path,
                    target=target,
                    scaling=scaling,
                    outlier_method=outlier_method,
                    contamination=contamination,
                )
                st.session_state.preprocessed_data = preprocessed_info
            st.success("Data preprocessed successfully!")

    with col2:
        if st.session_state.preprocessed_data is not None:
            st.success("✅ Data is preprocessed and ready for clustering!")
        else:
            st.info("Click 'Preprocess Data' to prepare your data for clustering")

    # ---- Show Preprocessed Data ----
    if st.session_state.preprocessed_data is not None:
        preprocessed_info = st.session_state.preprocessed_data
        
        st.subheader("📊 Preprocessed Data Summary")
        
        # Create a nice summary display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original Shape", f"{preprocessed_info['original_shape'][0]} × {preprocessed_info['original_shape'][1]}")
        
        with col2:
            st.metric("Final Shape", f"{preprocessed_info['final_shape'][0]} × {preprocessed_info['final_shape'][1]}")
        
        with col3:
            st.metric("Outliers Removed", preprocessed_info['outliers_removed'])
        
        with col4:
            st.metric("Categorical Features Encoded", preprocessed_info['categorical_encoded'])
        
        # Show preprocessing details
        with st.expander("🔍 Preprocessing Details"):
            st.write(f"**Scaling Method:** {preprocessed_info['scaling_method']}")
            st.write(f"**Outlier Method:** {preprocessed_info['outlier_method']}")
            if preprocessed_info['outlier_method'] == 'isoforest':
                st.write(f"**Outlier Contamination:** {contamination}")
            
            st.write("**Features Used:**")
            features_df = pd.DataFrame({
                'Feature': preprocessed_info['features_used'],
                'Type': ['Encoded' if '_' in f else 'Original' for f in preprocessed_info['features_used']]
            })
            st.dataframe(features_df, use_container_width=True)
        
        # Show sample of preprocessed data
        with st.expander("📋 Sample of Preprocessed Data (first 10 rows)"):
            sample_data = preprocessed_info['scaled_data'].head(10)
            st.dataframe(sample_data, use_container_width=True)
        
        # Show data distribution
        with st.expander("📈 Data Distribution"):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Feature distributions
            sample_features = preprocessed_info['features_used'][:4]  # Show first 4 features
            for i, feature in enumerate(sample_features):
                row, col = i // 2, i % 2
                axes[row, col].hist(preprocessed_info['scaled_data'][feature], bins=20, alpha=0.7)
                axes[row, col].set_title(f'{feature} Distribution')
                axes[row, col].set_xlabel('Value')
                axes[row, col].set_ylabel('Frequency')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ---- Run Clustering Button ----
    if st.session_state.preprocessed_data is not None:
        if st.button("🚀 Run clustering"):
            with st.spinner("Running pipeline…"):
                # capture all matplotlib figures that your pipeline .show() calls produce
                with streamlit_matplotlib():
                    results = run_pipeline(
                        file_path=tmp_path,
                        target=target,
                        scaling=scaling,
                        embedder=embedder,
                        cluster_on_features=cluster_on_features,
                        max_k=max_k,
                        outlier_method=outlier_method,
                        contamination=contamination,
                        draw_sankey=draw_sankey,
                        preprocessed_data=st.session_state.preprocessed_data,
                    )

            st.success("Done!")

            # ---- Summaries ----
            st.subheader("🧩 Cluster summaries")
            summaries = auto_describe_clusters(
                results,
                file_path=tmp_path,
                target=target,
                top_n=3
            )
            for s in summaries:
                st.write("• " + s)

            # ---- Download labels ----
            labels = results["kmeans_labels"].rename("Cluster").to_frame()
            st.download_button(
                "⬇️ Download cluster assignments (CSV)",
                data=labels.to_csv(index=True).encode("utf-8"),
                file_name="cluster_assignments.csv",
                mime="text/csv",
            )

            # ---- Drivers table ----
            with st.expander("📄 Top driver features (table)"):
                st.dataframe(results["kmeans_drivers"])
    else:
        st.info("Please preprocess your data first before running clustering.")
else:
    st.info("Upload a CSV to get started.")
