import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from esm import pretrained
from transformer_predictor import TransformerDistancePredictor
from pdb_converter import distmap_to_coords, coords_to_pdb
import tempfile
import os
import streamlit.components.v1 as components
import base64
import glob


# ========= Load background image ==========
def set_background():
    with open("/Users/lgraese/Desktop/WBS Coding School/DS Bootcamp/10_Final_project/NanoFolder_app/images/background.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# ========= Set custom title ==========
def set_custom_title():
    st.markdown("""
        <style>
        .custom-title {
            font-size: 80px !important;
            font-weight: bold;
            background: linear-gradient(90deg, #990824 0%, #A13725 3%, #B55A36 10%,#F3AE65 15%,#A3A177 20%, #828B88 30%, #89A3AD 35%, #4F799E 45%, #2F4F79 55%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: left;
            margin-bottom: 1rem;
        }
        </style>
        <h1 class="custom-title">NanoFolder 🦙</h1>
    """, unsafe_allow_html=True)


# ========= Load ESM2 model ==========
@st.cache_resource
def load_esm2_model():
    model, alphabet = pretrained.esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

# ========= Generate embedding from sequence ==========
def generate_embedding(sequence, model, batch_converter, device):
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    embedding = token_representations[0, 1:-1]  # Remove BOS/EOS
    return embedding

# ========= Predict distance map ==========
def predict_distance_map_ensemble(embedding, models, device):
    embedding = embedding.to(device)
    preds = []

    with torch.no_grad():
        for model in models:
            pred = model(embedding).cpu().numpy()
            preds.append(pred)

    # Average predictions across models
    avg_pred = np.mean(preds, axis=0)
    return avg_pred


# ========= Plot distance map ==========
def plot_distance_map(dist_map, title):
    fig, ax = plt.subplots()
    cax = ax.imshow(dist_map, cmap='viridis')
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Distance (Å)')
    ax.set_title(title)
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Residue Index')    
    st.pyplot(fig)

# ========= Visualize PDB structure ==========
def show_pdb(pdb_str):
    pdb_block = pdb_str.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    html = f"""
    <html>
    <head>
        <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    </head>
    <body>
        <div id="viewer" style="width: 100%; height: 400px;"></div>
        <script>
            let viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
            viewer.addModel("{pdb_block}", "pdb");
            viewer.setStyle({{model: -1}}, {{stick: {{colorscheme: "spectrum"}}}});
            viewer.zoomTo();
            viewer.zoom(2);
            viewer.render();
        </script>
    </body>
    </html>
    """
    components.html(html, height=420)



# ========= Main App ==========
def main():
    st.set_page_config(page_title="NanoFolder", layout="centered")
    set_background()
    set_custom_title()

    st.markdown("<h3 style='color: white;'>An sdAb Structure Predictor</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: white; font-size: 16px;'>
    Enter a <strong>protein sequence</strong> to generate ESM2 embeddings, predict a <strong>distance map</strong>, 
    reconstruct a <strong>3D structure (PDB)</strong>, and visualize it.
    </p>
    """, unsafe_allow_html=True)

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm2_model, batch_converter = load_esm2_model()
    esm2_model = esm2_model.to(device)

    # Load ensemble of models
    model_dir = "NanoFolder_app/model"
    model_paths = glob.glob(os.path.join(model_dir, "*.pt"))
    model_paths.sort()

    ensemble = []
    for path in model_paths:
        model = TransformerDistancePredictor()
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        model.eval()
        ensemble.append(model)

    # Session state setup
    if "distmap" not in st.session_state:
        st.session_state["distmap"] = None
    if "pdb_content" not in st.session_state:
        st.session_state["pdb_content"] = None
    if "distmap_path" not in st.session_state:
        st.session_state["distmap_path"] = None
    if "pdb_path" not in st.session_state:
        st.session_state["pdb_path"] = None

    # Input
    sequence = st.text_area("Enter Protein Sequence:", height=200)
    if st.button("Predict Structure"):
        if not sequence.strip():
            st.warning("⚠️ Please enter a protein sequence.")
            return

        with st.spinner("Generating structure..."):
            embedding = generate_embedding(sequence, esm2_model, batch_converter, device)
            pred_dist_map = predict_distance_map_ensemble(embedding, ensemble, device)
            sym_dist_map = (pred_dist_map + pred_dist_map.T) / 2
            np.fill_diagonal(sym_dist_map, 0)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp_dist:
                np.save(tmp_dist.name, pred_dist_map)
                st.session_state["distmap_path"] = tmp_dist.name

            coords = distmap_to_coords(sym_dist_map)
            tmp_pdb_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb").name
            coords_to_pdb(coords, sequence, tmp_pdb_path)
            
            with open(tmp_pdb_path, "r") as f:
                pdb_content = f.read()



            st.session_state["distmap"] = pred_dist_map
            st.session_state["pdb_path"] = tmp_pdb_path
            st.session_state["pdb_content"] = pdb_content


    # Display if available
    if st.session_state["distmap"] is not None and st.session_state["pdb_content"] is not None:
        st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

        st.markdown("<h3 style='color: white;'>Predicted Distance Map</h3>", unsafe_allow_html=True)
        plot_distance_map(st.session_state["distmap"], "Predicted Distance Map")

        st.markdown("<hr style='margin-top: 50px; margin-bottom: 50px; border: none; height: 2px; background-color: white;' />", unsafe_allow_html=True)

        st.markdown("<h3 style='color: white;'>Predicted 3D Structure</h3>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color: grey; font-size: 14px; margin-top: -10px;'>
        The 3D coordinates shown here are reconstructed from the predicted pairwise distance map using multidimensional scaling (MDS) and idealized backbone geometry. No physical energy terms, stereochemistry, or atom-level constraints are applied.
        </p>
        """, unsafe_allow_html=True)

        show_pdb(st.session_state["pdb_content"])
    
        st.markdown("""
        <div style='background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-left: 4px solid #FFA500; color: white; margin-top: 30px; font-size: 14px;'>
        <b>Note:</b> This tool was developed as a learning project to explore how distance-based protein structure prediction works. The output structures are approximate and lack physical refinement or atomic detail. Use them for educational insight, not scientific conclusions.
        </div>
        """, unsafe_allow_html=True)


        st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

        st.download_button(
            label="📥 Download Distance Map (.npy)",
            data=open(st.session_state["distmap_path"], "rb").read(),
            file_name="predicted_distance_map.npy"
        )
        st.download_button(
            label="📥 Download PDB File (.pdb)",
            data=open(st.session_state["pdb_path"], "rb").read(),
            file_name="predicted_structure.pdb"
        )

    # Footer
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: rgba(255, 255, 255, 0.0);  /* transparent background */
            text-align: center;
            padding: 10px 0;
            font-size: 0.85em;
            color: gray;
            z-index: 100;
        }
        </style>
        <div class="footer">
            © 2025 Dr. Lareen Gräser. All rights reserved.
            <br> <small>
            Background structure: Pompidor et al., <em>Engineered nanobodies with a lanthanide binding motif for crystallographic phasing</em>,  
            PDB ID: 6XYF. <a href="https://www.rcsb.org/structure/6XYF" target="_blank">RCSB PDB</a>
            </small>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
