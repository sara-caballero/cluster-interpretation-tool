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

st.markdown("Discover patterns and insights in your data through automated clustering analysis.")

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
    
    # Documentation section at the bottom of settings
    st.sidebar.markdown("---")
    with st.sidebar.expander("📚 Documentation"):
        st.markdown("**User Guide**: [How to use this tool](docs/USER_GUIDE.md)")
        st.markdown("**Technical Guide**: [Algorithms and concepts](docs/TECHNICAL_GUIDE.md)")
        st.markdown("---")
        st.markdown("*Both guides are in development*")

    # Initialize session state for preprocessed data
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None

    # ---- Preprocess Data Button ----
    # Only show preprocessing button if outlier removal is enabled
    if outlier_method != "none":
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

        with col2:
            if st.session_state.preprocessed_data is not None:
                st.success("Data is preprocessed and ready for clustering!")
    else:
        # Clear any existing preprocessed data when outlier removal is disabled
        if 'preprocessed_data' in st.session_state:
            del st.session_state.preprocessed_data

    # ---- Show Preprocessed Data ----
    if 'preprocessed_data' in st.session_state and st.session_state.preprocessed_data is not None:
        preprocessed_info = st.session_state.preprocessed_data
        
        st.subheader("Preprocessed Data Summary")
        
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
        with st.expander("Preprocessing Details"):
            st.write(f"**Scaling Method:** {preprocessed_info['scaling_method']}")
            st.write(f"**Outlier Method:** {preprocessed_info['outlier_method']}")
            if preprocessed_info['outlier_method'] == 'isoforest':
                st.write(f"**Outlier Contamination:** {contamination}")
            
            st.write("**Features Used:**")
            features_df = pd.DataFrame({
                'Feature': preprocessed_info['features_used'],
                'Type': ['Encoded' if '_' in f else 'Original' for f in preprocessed_info['features_used']]
            })
            st.dataframe(features_df, width='stretch')
        
        # Show sample of preprocessed data
        with st.expander("Sample of Preprocessed Data (first 10 rows)"):
            sample_data = preprocessed_info['scaled_data'].head(10)
            st.dataframe(sample_data, width='stretch')
        
        # Show data distribution
        with st.expander("Data Distribution"):
            # Calculate grid size based on number of features (max 10)
            n_features = min(len(preprocessed_info['features_used']), 10)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            # Feature distributions
            sample_features = preprocessed_info['features_used'][:n_features]
            for i, feature in enumerate(sample_features):
                row, col = i // n_cols, i % n_cols
                axes[row, col].hist(preprocessed_info['scaled_data'][feature], bins=20, alpha=0.7)
                axes[row, col].set_title(f'{feature} Distribution')
                axes[row, col].set_xlabel('Value')
                axes[row, col].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(n_features, n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ---- Run Clustering Button ----
    # Show clustering button if we have preprocessed data OR if outlier removal is disabled
    show_clustering_button = (
        ('preprocessed_data' in st.session_state and st.session_state.preprocessed_data is not None) or
        outlier_method == "none"
    )
    
    if show_clustering_button:
        if st.button("🚀 Run clustering"):
            with st.spinner("Running pipeline…"):
                # capture all matplotlib figures that your pipeline .show() calls produce
                with streamlit_matplotlib():
                    # Pass preprocessed_data only if it exists
                    preprocessed_data = None
                    if 'preprocessed_data' in st.session_state:
                        preprocessed_data = st.session_state.preprocessed_data
                    
                    results = run_pipeline(
                        file_path=tmp_path,
                        target=target,
                        scaling=scaling,
                        embedder=embedder,
                        cluster_on_features=cluster_on_features,
                        max_k=max_k,
                        outlier_method=outlier_method,
                        contamination=contamination,
                        preprocessed_data=preprocessed_data,
                    )

            st.success("Done!")

            # ---- Summaries ----
            st.subheader("Cluster summaries")
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
            with st.expander("Top driver features (table)"):
                st.dataframe(results["kmeans_drivers"])
    else:
        st.info("Please preprocess your data first before running clustering.")
else:
    st.info("Upload a CSV to get started.")
