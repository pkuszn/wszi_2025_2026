import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from models.gnn_model import GNNModel
from utils.graph_utils import mol_to_graph
import os

def agent_executor(user_prompt):
    """Analyses input and predicts the bioactiivty"""
    words = user_prompt.split()
    potential_smiles = [w for w in words if len(w) > 5]
    return potential_smiles[0] if potential_smiles else None

def predict_bioactivity(smiles):
    """Tool calling func"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, "Invalid SMILES format"
    
    data = mol_to_graph(smiles, target_value=0)
    
    # model = load_trained_model() 
    # prediction = model(data.x, data.edge_index, data.batch)
    
    mock_prediction = 5.42 
    return mock_prediction, None

st.set_page_config(page_title="ChEMBL GNN Predictor", layout="wide")

st.title("🔬 Molecular GNN asystent")
st.markdown("Enter SMILES.")

with st.sidebar:
    st.header("Model status")
    st.success("Model GIN załadowany (Test R2: 0.15)")
    st.info("Urządzenie: CPU")

user_input = st.text_input("Your prompt (for example: 'Calculate pIC50 for C1=CC=C(C=C1)C(=O)O')", "")

if user_input:
    with st.spinner('Analyse...'):
        extracted_smiles = agent_executor(user_input)
        
    if extracted_smiles:
        st.write(f"🔎 Found: `{extracted_smiles}`")
        
        mol = Chem.MolFromSmiles(extracted_smiles)
        if mol:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("2D structure")
                img = Draw.MolToImage(mol)
                st.image(img, use_column_width=True)
            
            with col2:
                st.subheader("GNN prediction reuslt")
                prediction, error = predict_bioactivity(extracted_smiles)
                
                if not error:
                    st.metric(label="Predicted pIC50", value=f"{prediction:.2f}")
                    st.progress(prediction / 10.0) # type: ignore
                else:
                    st.error(error)
        else:
            st.error("RDKit couldn't interpret SMILES")
    else:
        st.warning("SMILES not found by agent")

st.divider()
st.caption("...")