
import re
from typing import Optional, Tuple

from langgraph.prebuilt import ToolNode, tools_condition
import numpy as np
import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from models.hybrid_model import HybridModel
from utils.graph_utils import mol_to_graph
import networkx as nx
import matplotlib.pyplot as plt
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors, Lipinski, QED
from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import pubchempy as pcp
MODEL_PATH = "/home/pkuszn/repos/WSzI/src/notebooks/model_Hybrid_tune_0_1781204396_weights.pth"
NUM_FEATURES = 10 
NUM_EXTRA_FEATURES = 8 
HIDDEN_CHANNELS = 128


def sanitize_smiles(smiles: str) -> str:
    return re.sub(r'[.=,;!\-\+]+$', '', smiles.strip())

def extract_smiles(text: str) -> str:
    """Extracts a valid SMILES string. Use this when the user mentions a molecule."""
    words = text.split()
    for word in words:
        clean = sanitize_smiles(word)
        mol = Chem.MolFromSmiles(clean)
        if mol:
            return clean
    return "No valid SMILES found"

def calculate_descriptors(smiles):
    """Calculates descriptors using rdkit

    Args:
        smiles: Smiles
    """
    mol = Chem.MolFromSmiles(smiles)

    alogp = Descriptors.MolLogP(mol) # type: ignore
    psa = Descriptors.TPSA(mol) # type: ignore
    hba = Lipinski.NumHAcceptors(mol) # type: ignore
    hbd = Lipinski.NumHDonors(mol) # type: ignore

    num_ro5_violations = (
        int(alogp > 5)
        + int(Descriptors.MolWt(mol) > 500) # type: ignore
        + int(hba > 10)
        + int(hbd > 5)
    )

    qed_weighted = QED.qed(mol)

    logP_over_PSA = alogp / (psa + 1e-6)

    HBA_HBD_sum = hba + hbd

    return np.array([
        alogp,
        psa,
        hba,
        hbd,
        num_ro5_violations,
        qed_weighted,
        logP_over_PSA,
        HBA_HBD_sum
    ], dtype=np.float32)

def predict_bioactivity(smiles) -> Tuple[Optional[float], Optional[str]]:
    """Tool calling func"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, "Invalid SMILES format"

    model, global_scaler, atom_scaler = load_trained_model() 

    extra_features = calculate_descriptors(smiles)

    scaled_extra = global_scaler.transform(extra_features.reshape(1, -1))

    data = mol_to_graph(smiles=smiles, target_value=0, extra_features_vector=scaled_extra, atom_scaler=atom_scaler)
    
    for name, module in model.named_modules():
        print(name, type(module))
    with torch.no_grad():
        if not hasattr(data, "batch"):
            data.batch = torch.zeros( # type: ignore
                data.x.size(0), # type: ignore
                dtype=torch.long
            )

        prediction = model(
            data.x, # type: ignore
            data.edge_index, # type: ignore
            data.batch, # type: ignore
            data.extra_features # type: ignore
        )
        
    return float(prediction.item()), None

def visualize_structure(smiles: str) -> str:
    """
    Call this to render a 2D image of the molecule structure.
    Input must be a valid SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol)
        img.save("structure.png")
        return "2D structure rendered and saved as structure.png"
    return "Failed to render structure"

def visualize_gnn(smiles: str) -> str:
    """
    Call this to generate and plot the GNN graph topology.
    Input must be a valid SMILES string.
    """
    fig, _ = visualize_gnn_graph(smiles) # type: ignore
    if fig:
        fig.savefig("gnn_graph.png")
        plt.close(fig)
        return "GNN graph topology plotted and saved as gnn_graph.png"
    return "Failed to visualize GNN graph"


def visualize_gnn_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        return None
    
    _, global_scaler, atom_scaler = load_trained_model() 

    extra_features = calculate_descriptors(smiles)

    scaled_extra = global_scaler.transform(extra_features.reshape(1, -1))

    data = mol_to_graph(smiles=smiles, target_value=0, extra_features_vector=scaled_extra, atom_scaler=atom_scaler)
    atom_labels = {idx: atom.GetSymbol() for idx, atom in enumerate(mol.GetAtoms())}

    G = nx.Graph()

    if data.edge_index.shape[1] > 0: # type: ignore
        edges = data.edge_index.t().tolist() # type: ignore
        G.add_edges_from(edges)
    else:
        for i in range(data.x.shape[0]): # type: ignore
            G.add_node(i)

    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    pos = nx.spring_layout(G, seed=42)

    nx.draw(
            G, pos,
            labels=atom_labels,
            with_labels=True,
            node_color="#4A90E2",
            node_size=800,
            font_size=12,
            font_color="white",
            font_weight="bold",
            edge_color="#888888",
            width=2,
            ax=ax)
    
    return fig, data

@tool("BioactivityPredictor")
def bioactivity_predictor(text: str) -> str:
    """
    Predict pIC50 for a molecule represented by SMILES.
    """
    smiles = extract_smiles(text)
    if smiles == "No valid SMILES found":
        return "No valid SMILES found"
    pred, error = predict_bioactivity(smiles)

    if error:
        return error

    return f"Predicted pIC50 = {pred:.2f}"

@tool("Plot2DStructure")
def plot_2d_structure(text: str):
    """Prints 2d structure"""
    smiles = extract_smiles(text)

    if smiles == "No valid SMILES found":
        return "No valid SMILES found"

    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        return "Invalid SMILES"

    return f"STRUCTURE_DATA:{smiles}"

@tool("IupacName")
def get_iupac_name(text: str):
    """Gets the iupac name of smiles.

    Args:
        text (str): Smiles
    """
    smiles = extract_smiles(text)

    if smiles == "No valid SMILES found":
        return "No valid SMILES found"
        
    compounds = pcp.get_compounds(smiles, "smiles")

    if compounds: # type: ignore
        return compounds[0].iupac_name

@st.cache_resource
def load_trained_model():
    model = HybridModel(
        num_node_features=NUM_FEATURES,
        num_extra_features=NUM_EXTRA_FEATURES,
        hidden_channels=HIDDEN_CHANNELS
    )
    
    checkpoint = torch.load(
        MODEL_PATH, 
        map_location=torch.device('cpu'),
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model, checkpoint["global_scaler"], checkpoint["atom_scaler"]
    

def get_molecule_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "Unknown molecule"
    
    formula = rdMolDescriptors.CalcMolFormula(mol)
    weight = rdMolDescriptors.CalcExactMolWt(mol)
    
    return f"Formula: {formula} | Exact Mass: {weight:.2f} g/mol"

tools = [bioactivity_predictor, plot_2d_structure, get_iupac_name]

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
).bind_tools(tools)

features = [
    'alogp', 'psa', 'hba', 'hbd', 'num_ro5_violations', 'qed_weighted',
    'logP_over_PSA', 'HBA_HBD_sum'
]

sys_msg = SystemMessage(content="""
You are a chemistry assistant.

Available tools:
- BioactivityPredictor
- Plot2DStructure
- IupacName

CRITICAL RULES:

1. NEVER output SMILES directly in final answer.
2. ALWAYS call IupacName tool and print the name for user.
3. Use BioactivityPredictor exactly once for a prediction request.
4. ALWAYS pass SMILES output to next tool without modification.
5. If user asks ANY of the following:
   - structure
   - image
   - draw
   - visualize
   - show molecule
   → you MUST call Plot2DStructure tool

6. If user asks:
   - pIC50
   - activity
   → use BioactivityPredictor

7. You may chain tools, but MUST NOT skip tool execution.

8. Final answer must be based ONLY on tool outputs.
                        
9. If user asks:
    - compound name
    - name
    → use IupacName                 
                        
10. DO NOT try to draw the molecule with text
""")

def assistant(state: MessagesState):
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)
builder.add_edge("tools", "assistant")
graph = builder.compile()


st.set_page_config(page_title="ChEMBL GNN+MLP Predictor", layout="wide")

st.title("🔬 GNN+MLP Bioactivity Prediction")
st.markdown("Enter SMILES.")

with st.sidebar:
    st.header("Model status")
    st.success(f"Model: {MODEL_PATH}")
    st.info("Device: CPU")

user_input = st.text_input("Your prompt (for example: 'Calculate pIC50 for C1=CC=C(C=C1)C(=O)O')", "")

if user_input:
    with st.spinner('Analyzing...'):
        messages = [HumanMessage(content=user_input)]
        result = graph.invoke({"messages": messages}) # type: ignore
        prediction = None
        img = None
        iupac = None
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                st.caption(f"⚙️ Tool executed: {msg.name}")

                if msg.name == "BioactivityPredictor":
                    prediction = msg.content

                if msg.name == "Plot2DStructure" and "STRUCTURE_DATA:" in msg.content:
                    smiles_str = msg.content.split("STRUCTURE_DATA:")[1] # type: ignore
                    mol = Chem.MolFromSmiles(smiles_str)
                    if mol:
                        img = Draw.MolToImage(mol)

                if msg.name == "IupacName":
                    iupac = msg.content

            elif isinstance(msg, HumanMessage):
                st.markdown(f"**You:** {msg.content}")

            elif isinstance(msg, AIMessage) and msg.content:
                st.markdown(f"**Assistant:** {msg.content}")

        if prediction:
            st.success(prediction)

        if img is not None:
            st.image(img, caption="Molecule Structure", use_container_width=True)

        if iupac:
            st.info(iupac)
st.divider()