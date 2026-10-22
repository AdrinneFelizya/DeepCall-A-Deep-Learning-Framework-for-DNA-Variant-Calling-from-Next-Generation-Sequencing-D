"""
DeepCall Frontend — Streamlit Web Interface
DNA Variant Calling from NGS Pileup Images
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import torch
import os
import plotly.graph_objects as go
from PIL import Image

# Page config
st.set_page_config(
    page_title="DeepCall — DNA Variant Calling",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1e3a8a; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #64748b; text-align: center; margin-bottom: 2rem; }
    .prediction-box { background: #eff6ff; padding: 1.5rem; border-radius: 0.75rem; border: 2px solid #3b82f6; margin: 1rem 0; }
    .warning-box { background: #fef3c7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b; margin: 1rem 0; }
    .success-box { background: #f0fdf4; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #10b981; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">🧬 DeepCall</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Deep Learning DNA Variant Calling from NGS Pileup Images</h3>', unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "🔮 Predict Variant (Image)", "🧬 Real-Time (BAM File)", "📊 Train Model", "📈 Model Comparison", "📄 About"]
    )

    if page == "🏠 Home":
        show_home()
    elif page == "🔮 Predict Variant (Image)":
        show_predict_image()
    elif page == "🧬 Real-Time (BAM File)":
        show_real_time()
    elif page == "📊 Train Model":
        show_train()
    elif page == "📈 Model Comparison":
        show_comparison()
    elif page == "📄 About":
        show_about()


def show_home():
    st.markdown("## Welcome to DeepCall")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **DeepCall** identifies genetic mutations from next-generation sequencing (NGS) data
        by treating DNA pileup data as multi-channel images.

        ### Input Modes
        - **Image mode:** Upload pileup images (`.npy`, `.png`, `.jpg`)
        - **Real-time mode:** Upload BAM/CRAM files for real genomic variant calling

        ### Outputs
        - Variant class + confidence score
        - Standard VCF v4.2 file
        - Model comparison charts

        ### Real-life applications
        - Clinical diagnostics for rare diseases
        - Precision oncology (somatic mutations)
        - Pharmacogenomics screening
        - Infectious disease surveillance
        """)
    with col2:
        st.metric("Models Available", "4")
        st.metric("Variant Classes", "4")
        st.metric("Input Format", "Image or BAM")
        st.metric("Output Format", "VCF v4.2")

        st.markdown("### Model Architectures")
        for arch in ["CNN", "ResNet18", "VGG16", "Transformer"]:
            st.write(f"• {arch}")


def show_predict_image():
    st.markdown("## 🔮 Predict Variant from Pileup Image")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Model Settings")
        architecture = st.selectbox(
            "Select Model Architecture",
            ["cnn", "resnet18", "vgg16", "transformer"],
            index=0,
        )
        model_path = st.text_input(
            "Model Checkpoint Path",
            value=f"saved_models/{architecture}_best.pth",
        )
        if st.button("Load Model", type="primary"):
            try:
                from backend.predict import ModelPredictor
                st.session_state.predictor = ModelPredictor(
                    config_path="config.yaml",
                    model_path=model_path,
                )
                st.session_state.model_loaded = True
                st.session_state.model_name = architecture
                st.success(f"✅ {architecture} model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load model: {str(e)}")
                st.session_state.model_loaded = False

    with col2:
        st.markdown("### Upload Pileup Image")
        uploaded_file = st.file_uploader(
            "Choose a pileup image (PNG, JPG, NPY)",
            type=["png", "jpg", "jpeg", "npy"],
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".npy"):
                    image = np.load(uploaded_file)
                else:
                    img = Image.open(uploaded_file).convert("RGB")
                    image = np.array(img).transpose(2, 0, 1) / 255.0

                if image.ndim == 3 and image.shape[0] in [3, 5, 8]:
                    pass
                elif image.ndim == 3 and image.shape[-1] in [3, 5, 8]:
                    image = image.transpose(2, 0, 1)
                else:
                    st.warning("Image shape may not match expected format (channels, height, width).")

                st.session_state.uploaded_image = image
                if uploaded_file.name.endswith(".npy"):
                    st.info("NPY pileup loaded (shape: {})".format(tuple(image.shape)))
                else:
                    st.image(
                        Image.open(uploaded_file),
                        caption="Uploaded Pileup Image",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

    if st.session_state.get("model_loaded", False) and st.session_state.get("uploaded_image") is not None:
        if st.button("🔍 Run Prediction", type="primary"):
            with st.spinner("Analyzing pileup image..."):
                try:
                    result = st.session_state.predictor.predict(
                        st.session_state.uploaded_image
                    )
                    st.session_state.prediction_result = result

                    st.markdown("### Prediction Result")
                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>Predicted Class: {result['class_name']}</h3>
                            <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                            <p><strong>Class Index:</strong> {result['class']}</p>
                            <p><strong>Interpretation:</strong> {__import__('reporting').describe_prediction(result['class'], result['confidence'], result['class_name'])}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with result_col2:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(result["probabilities"].keys()),
                                y=list(result["probabilities"].values()),
                                marker_color=["#94a3b8", "#ef4444", "#f59e0b", "#10b981"],
                            )
                        ])
                        fig.update_layout(
                            title="Class Probabilities",
                            yaxis_title="Probability",
                            yaxis=dict(range=[0, 1]),
                            height=300,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### VCF Output Preview")
                    vcf_line = f"chr1\t1000\t.\tA\tG\t{int(result['confidence']*100)}\tPASS\tDP=50;TYPE=SNP\tGT:GQ\t0/1:{int(result['confidence']*100)}"
                    st.code(vcf_line, language="text")

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

    elif not st.session_state.get("model_loaded", False):
        st.warning("⚠️ Please load a model first.")
    elif st.session_state.get("uploaded_image") is None:
        st.info("📤 Please upload a pileup image.")


def show_real_time():
    st.markdown("## 🧬 Real-Time Variant Calling from BAM Files")

    try:
        import pysam
        pysam_available = True
    except ImportError:
        pysam_available = False

    if not pysam_available:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ pysam not installed</h4>
            <p>Real BAM processing requires <code>pysam</code>.</p>
            <p>Install via conda: <code>conda install -c conda-forge -c bioconda pysam</code></p>
            <p><strong>Demo mode</strong> is active below — shows the interface with synthetic predictions.</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### BAM File Settings")
        architecture = st.selectbox(
            "Model",
            ["cnn", "resnet18", "vgg16", "transformer"],
            index=0,
            key="real_arch",
        )
        model_path = st.text_input(
            "Checkpoint",
            value=f"saved_models/{architecture}_best.pth",
            key="real_model",
        )
        bam_file = st.file_uploader("Upload BAM file", type=["bam", "cram"])
        chrom = st.text_input("Chromosome", value="chr22")
        start_pos = st.number_input("Start Position", value=1000000, min_value=1)
        end_pos = st.number_input("End Position", value=10001000, min_value=1)
        step = st.number_input("Step (bp)", value=1000, min_value=1)

        if st.button("Load Model", type="primary", key="load_real_model"):
            try:
                from backend.real_predict import RealTimeVariantCaller
                st.session_state.real_caller = RealTimeVariantCaller(
                    config_path="config.yaml",
                    model_path=model_path,
                )
                st.session_state.real_loaded = True
                st.session_state.real_loaded_arch = architecture
                st.success(f"✅ {architecture} model ready!")
            except Exception as e:
                st.error(f"❌ {str(e)}")

    with col2:
        st.markdown("### Real-Time Results")
        if st.session_state.get("real_loaded") and st.button(
            "🚀 Scan Region for Variants", type="primary", key="scan_variants"
        ):
            if bam_file is None:
                st.warning("Please upload a BAM file first.")
            elif not pysam_available:
                st.info("Demo mode: showing synthetic predictions (pysam not available)")
                import random
                random.seed(42)
                results = []
                variant_classes = [
                    (1, "SNP", "G"),
                    (2, "Insertion", "AG"),
                    (3, "Deletion", "A"),
                ]
                for pos in range(int(start_pos), int(end_pos), int(step)):
                    if random.random() < 0.05:
                        cls, cname, alt = random.choice(variant_classes)
                        conf = round(random.uniform(0.85, 0.99), 3)
                        results.append({
                            "chrom": chrom,
                            "pos": pos,
                            "ref": "A",
                            "alt": alt,
                            "class": cls,
                            "class_name": cname,
                            "confidence": conf,
                            "is_variant": True,
                            "vcf_line": f"{chrom}\t{pos}\t.\tA\t{alt}\t{int(conf*100)}\tPASS\tDP=50;TYPE={cname}\tGT:GQ\t0/1:{int(conf*100)}",
                        })
                st.session_state.real_results = results
                st.success(f"Found {len(results)} variants (demo mode)")
            else:
                try:
                    temp_bam = f"temp_{bam_file.name}"
                    with open(temp_bam, "wb") as f:
                        f.write(bam_file.getbuffer())

                    from backend.real_predict import RealTimeVariantCaller, write_vcf
                    caller = RealTimeVariantCaller(
                        config_path="config.yaml",
                        model_path=model_path,
                    )
                    results = caller.batch_predict_from_positions(
                        temp_bam,
                        [(chrom, p, "A", "G") for p in range(int(start_pos), int(end_pos), int(step))],
                    )
                    variants = [r for r in results if r.get("is_variant")]
                    st.session_state.real_results = variants
                    write_vcf(variants, "output/real_time_calls.vcf")
                    st.success(f"Found {len(variants)} variants in region")
                    os.remove(temp_bam)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.session_state.get("real_results"):
        st.markdown("### Variant Calls")
        st.text(__import__("reporting").summarize_variants(st.session_state.real_results))
        table = [
            {
                "Chrom": r["chrom"],
                "Pos": r["pos"],
                "Ref": r["ref"],
                "Alt": r["alt"],
                "Class": r["class_name"],
                "Description": __import__("reporting").describe_prediction(
                    r["class"], r["confidence"], r["class_name"],
                    position=r["pos"], ref=r["ref"], alt=r["alt"],
                ),
                "Confidence": f"{r['confidence']:.2%}",
            }
            for r in st.session_state.real_results
        ]
        st.table(table)

        if os.path.exists("output/real_time_calls.vcf"):
            with open("output/real_time_calls.vcf", "r") as f:
                vcf_content = f.read()
            st.download_button(
                label="📥 Download VCF",
                data=vcf_content,
                file_name="real_time_calls.vcf",
                mime="text/plain",
            )


def show_train():
    st.markdown("## 📊 Train Model")
    col1, col2 = st.columns(2)

    with col1:
        architecture = st.selectbox(
            "Select Architecture",
            ["cnn", "resnet18", "vgg16", "transformer"],
            index=0,
            key="train_arch",
        )
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=15)
        save_dir = st.text_input("Save Directory", value="saved_models")

    with col2:
        st.markdown("### Training Info")
        st.write(f"Architecture: {architecture}")
        st.write(f"Epochs: {epochs}")
        st.write(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    if st.button("🚀 Start Training", type="primary"):
        with st.spinner(f"Training {architecture}... This may take several minutes."):
            try:
                from backend.train_model import train_model
                results = train_model(
                    config_path="config.yaml",
                    architecture=architecture,
                    epochs=int(epochs),
                    save_dir=save_dir,
                )
                st.session_state.training_results = results
                st.success("✅ Training complete!")

                st.markdown("### Training Results")
                st.metric("Test Accuracy", f"{results['accuracy']:.4f}")

                table = [
                    {
                        "Class": results["class_names"][i],
                        "Precision": round(results["precision"][i], 4),
                        "Recall": round(results["recall"][i], 4),
                        "F1-Score": round(results["f1"][i], 4),
                    }
                    for i in range(len(results["class_names"]))
                ]
                st.table(table)

                if os.path.exists(results["cm_path"]):
                    st.markdown("### Confusion Matrix")
                    st.image(results["cm_path"], caption="Confusion Matrix")

            except Exception as e:
                st.error(f"Training failed: {str(e)}")


def show_comparison():
    st.markdown("## 📈 Model Comparison")
    st.markdown(
        "Train and compare multiple architectures. Results will be displayed in a table."
    )

    if st.button("🔄 Run Comparison (CNN, ResNet18, VGG16)", type="primary"):
        from backend.train_model import train_model

        architectures = ["cnn", "resnet18", "vgg16"]
        results = []

        for arch in architectures:
            with st.spinner(f"Training {arch}..."):
                try:
                    r = train_model(
                        config_path="config.yaml",
                        architecture=arch,
                        epochs=10,
                        save_dir="saved_models",
                    )
                    results.append(r)
                except Exception as e:
                    st.error(f"{arch} failed: {str(e)}")

        if results:
            st.session_state.comparison_results = results

            import pandas as pd
            df = pd.DataFrame([
                {
                    "Model": r["architecture"].upper(),
                    "Accuracy": f"{r['accuracy']:.4f}",
                    "SNP F1": f"{r['f1'][1]:.4f}",
                    "Insertion F1": f"{r['f1'][2]:.4f}",
                    "Deletion F1": f"{r['f1'][3]:.4f}",
                }
                for r in results
            ])
            st.dataframe(df, use_container_width=True)

            fig = go.Figure()
            for _, row in df.iterrows():
                fig.add_trace(
                    go.Bar(
                        x=[row["Model"]],
                        y=[float(row["Accuracy"])],
                        name=row["Model"],
                    )
                )
            fig.update_layout(
                title="Model Accuracy Comparison",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1]),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


def show_about():
    st.markdown("## 📄 About DeepCall")
    st.markdown("""
    ### Project Overview
    DeepCall is a deep learning framework for DNA variant calling from next-generation
    sequencing (NGS) data. It treats aligned sequencing reads as multi-channel images
    and uses deep neural networks to classify genetic variants.

    ### Input Modes
    - **Image:** Upload `.npy`, `.png`, `.jpg` pileup images
    - **Real-time:** Upload `.bam` / `.cram` files for real genomic analysis

    ### Output
    - Variant class, confidence, VCF v4.2

    ### Conference Paper
    Suitable for submission to: ISMB, RECOMB, IEEE BIBM, PSB

    ### Ethics
    Research prototype. Not for clinical diagnosis without validation.
    """)


if __name__ == "__main__":
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "predictor" not in st.session_state:
        st.session_state.predictor = None
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "real_loaded" not in st.session_state:
        st.session_state.real_loaded = False
    if "real_caller" not in st.session_state:
        st.session_state.real_caller = None
    if "real_results" not in st.session_state:
        st.session_state.real_results = []
    if "training_results" not in st.session_state:
        st.session_state.training_results = None

    main()