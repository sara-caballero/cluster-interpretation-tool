import io
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import contextlib


# Health check endpoint for monitoring
try:
    # Streamlit >= 1.30
    qp = st.query_params
except Exception:
    # Fallback for older versions
    qp = st.experimental_get_query_params()

if (qp.get("ping") == ["1"]) or (qp.get("ping") == "1"):
    st.write("ok")
    st.stop()


# Import pipeline functions
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

# Context manager to redirect matplotlib plots to Streamlit
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

# Application settings and documentation
st.sidebar.header("⚙️ Settings")

# Global variables initialization
target = None
all_cols = []

# File upload section
file = st.file_uploader("Upload a CSV file", type=["csv"])

if file:
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    # Column separator selection
    st.subheader("📁 File Configuration")
    separator = st.selectbox(
        "Column separator",
        ["comma", "semicolon", "tab", "space"],
        index=0,
        help="Select the character that separates columns in the CSV file"
    )
    
    # Map separator names to actual characters
    separator_map = {
        "comma": ",",
        "semicolon": ";", 
        "tab": "\t",
        "space": " "
    }
    sep_char = separator_map[separator]
    
    # Preview data with selected separator
    df_preview = pd.read_csv(tmp_path, sep=sep_char, nrows=500)
    all_cols = list(df_preview.columns)
    
    with st.expander("🔎 Preview (first 500 rows)"):
        st.dataframe(df_preview)

# Configuration options
target = st.sidebar.selectbox("Target column (optional)", [None] + all_cols, index=0)

# Feature selection settings
excluded_cols = st.sidebar.multiselect(
    "Exclude columns from clustering (optional)", 
    all_cols if all_cols else [],
    help="Select columns to exclude from analysis (e.g., IDs, timestamps)"
)

scaling = st.sidebar.radio("Scaling", ["minmax", "standard", "none"], index=0)
embedder = st.sidebar.radio("Embedding (for plots)", ["PCA", "UMAP"], index=0)
cluster_on_features = st.sidebar.checkbox("Cluster on full features (recommended)", value=True)
# Clustering configuration
k_selection = st.sidebar.radio("K selection method", ["Auto (silhouette + elbow)", "Manual override"], index=0)

if k_selection == "Auto (silhouette + elbow)":
    max_k = st.sidebar.slider("Max k to try", 2, 15, 8)
    manual_k = None
else:
    manual_k = st.sidebar.slider("Manual k value", 2, 15, 3)
    max_k = None

outlier_method = st.sidebar.radio("Outlier removal", ["isoforest", "none"], index=0)
contamination = st.sidebar.slider("Outlier contamination", 0.0, 0.10, 0.03, 0.01)

# Documentation access
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Documentation")
if st.sidebar.button("📖 User Guide"):
    st.session_state.show_user_guide_modal = True
if st.sidebar.button("🔬 Technical Guide"):
    st.session_state.show_technical_guide_modal = True

# Modal state management
if 'show_user_guide_modal' not in st.session_state:
    st.session_state.show_user_guide_modal = False
if 'show_technical_guide_modal' not in st.session_state:
    st.session_state.show_technical_guide_modal = False

# User guide display
if st.session_state.show_user_guide_modal:
    st.markdown("---")
    try:
        with open("docs/USER_GUIDE.md", "r", encoding="utf-8") as f:
            user_guide_content = f.read()
        st.markdown(user_guide_content)
    except FileNotFoundError:
        st.error("User guide file not found. Ensure docs/USER_GUIDE.md exists.")
    except Exception as e:
        st.error(f"Error loading user guide: {str(e)}")
    
    if st.button("Close User Guide"):
        st.session_state.show_user_guide_modal = False
        st.rerun()

# Technical guide display
if st.session_state.show_technical_guide_modal:
    st.markdown("---")
    try:
        with open("docs/TECHNICAL_GUIDE.md", "r", encoding="utf-8") as f:
            technical_guide_content = f.read()
        st.markdown(technical_guide_content)
    except FileNotFoundError:
        st.error("Technical guide file not found. Ensure docs/TECHNICAL_GUIDE.md exists.")
    except Exception as e:
        st.error(f"Error loading technical guide: {str(e)}")
    
    if st.button("Close Technical Guide"):
        st.session_state.show_technical_guide_modal = False
        st.rerun()

# Main application logic
if file:
    # Session state initialization
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None

    # Data preprocessing section
    # Show preprocessing controls when needed
    if outlier_method != "none":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("🔧 Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    preprocessed_info = preprocess_data(
                        file_path=tmp_path,
                        target=target,
                        excluded_cols=excluded_cols,
                        scaling=scaling,
                        outlier_method=outlier_method,
                        contamination=contamination,
                        separator=sep_char,
                    )
                    st.session_state.preprocessed_data = preprocessed_info

        with col2:
            if st.session_state.preprocessed_data is not None:
                st.success("Data is preprocessed and ready for clustering!")
    else:
        # Clean up preprocessed data when not needed
        if 'preprocessed_data' in st.session_state:
            del st.session_state.preprocessed_data

    # Display preprocessed data information
    if 'preprocessed_data' in st.session_state and st.session_state.preprocessed_data is not None:
        preprocessed_info = st.session_state.preprocessed_data
        
        st.subheader("Preprocessed Data Summary")
        
        # Create summary metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original Shape", f"{preprocessed_info['original_shape'][0]} × {preprocessed_info['original_shape'][1]}")
        
        with col2:
            st.metric("Final Shape", f"{preprocessed_info['final_shape'][0]} × {preprocessed_info['final_shape'][1]}")
        
        with col3:
            st.metric("Outliers Removed", preprocessed_info['outliers_removed'])
        
        with col4:
            st.metric("Categorical Features Encoded", preprocessed_info['categorical_encoded'])
        
        # Display preprocessing configuration
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
        
        # Display data sample
        with st.expander("Sample of Preprocessed Data (first 10 rows)"):
            sample_data = preprocessed_info['scaled_data'].head(10)
            st.dataframe(sample_data, width='stretch')
        
        # Display feature distributions
        with st.expander("Data Distribution"):
            # Calculate subplot grid dimensions
            n_features = min(len(preprocessed_info['features_used']), 10)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            # Plot feature histograms
            sample_features = preprocessed_info['features_used'][:n_features]
            for i, feature in enumerate(sample_features):
                row, col = i // n_cols, i % n_cols
                axes[row, col].hist(preprocessed_info['scaled_data'][feature], bins=20, alpha=0.7)
                axes[row, col].set_title(f'{feature} Distribution')
                axes[row, col].set_xlabel('Value')
                axes[row, col].set_ylabel('Frequency')
            
            # Remove unused subplots
            for i in range(n_features, n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Clustering execution section
    # Determine when to show clustering button
    show_clustering_button = (
        ('preprocessed_data' in st.session_state and st.session_state.preprocessed_data is not None) or
        outlier_method == "none"
    )
    
    if show_clustering_button:
        if st.button("🚀 Run clustering"):
            with st.spinner("Running pipeline…"):
                # Capture matplotlib output for Streamlit display
                with streamlit_matplotlib():
                    # Use preprocessed data if available
                    preprocessed_data = None
                    if 'preprocessed_data' in st.session_state:
                        preprocessed_data = st.session_state.preprocessed_data
                    
                    results = run_pipeline(
                        file_path=tmp_path,
                        target=target,
                        excluded_cols=excluded_cols,
                        scaling=scaling,
                        embedder=embedder,
                        cluster_on_features=cluster_on_features,
                        max_k=max_k,
                        manual_k=manual_k,
                        outlier_method=outlier_method,
                        contamination=contamination,
                        preprocessed_data=preprocessed_data,
                        separator=sep_char,
                    )

            st.success("Done!")

            # Display cluster summaries
            st.subheader("Cluster summaries")
            summaries = auto_describe_clusters(
                results,
                file_path=tmp_path,
                target=target,
                top_n=3
            )
            for s in summaries:
                st.write("• " + s)

            # Download comprehensive PDF report
            st.subheader("📥 Download Results")
            try:
                from pipeline import generate_clustering_pdf
                
                pdf_data = generate_clustering_pdf(
                    results=results,
                    file_path=tmp_path,
                    target=target,
                    top_n=3,
                    scaling=scaling,
                    embedder=embedder,
                    manual_k=manual_k,
                    max_k=max_k,
                    outlier_method=outlier_method,
                    contamination=contamination
                )
                
                st.download_button(
                    "📄 Download full report (PDF)",
                    data=pdf_data,
                    file_name="clustering_analysis_report.pdf",
                    mime="application/pdf",
                )
            except ImportError:
                st.error("PDF generation requires reportlab. Install with: pip install reportlab")
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")

            # Display feature importance table
            with st.expander("Top driver features (table)"):
                st.dataframe(results["kmeans_drivers"])
    else:
        st.info("Preprocess your data first before running clustering.")
else:
    st.info("Upload a CSV to get started.")
