
import re

from langgraph.prebuilt import ToolNode, tools_condition
import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from models.gnn_model import GNNModel
from utils.graph_utils import mol_to_graph
import networkx as nx
import matplotlib.pyplot as plt
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from rdkit.Chem import rdMolDescriptors
from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
MODEL_PATH = "/home/pkuszn/repos/WSzI/src/notebooks/model_GCN_weights.pth"
NUM_FEATURES = 4  
HIDDEN_CHANNELS = 64

def sanitize_smiles(smiles: str) -> str:
    return re.sub(r'[.=,;!\-\+]+$', '', smiles.strip())

@tool("ExtractSmiles")
def extract_smiles(text: str) -> str:
    """Extracts a valid SMILES string. Use this when the user mentions a molecule."""
    # Szukamy słów, które wyglądają jak SMILES
    words = text.split()
    for word in words:
        clean = sanitize_smiles(word)
        mol = Chem.MolFromSmiles(clean)
        if mol:
            return clean
    return "No valid SMILES found"

@tool("BioactivityPredictor")
def bioactivity_predictor(smiles: str) -> str:
    """
    Predict pIC50 for a molecule represented by SMILES.
    """

    pred, error = predict_bioactivity(smiles)

    if error:
        return error

    return f"Predicted pIC50 = {pred:.2f}"

@tool("Plot2DStructure")
def plot_2d_structure(smiles: str) -> str:
    """Prints 2d structure"""
    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        return "Invalid SMILES"

    img = Draw.MolToImage(mol)
    img.save("structure.png")

    return "Saved structure.png"

def predict_bioactivity(smiles):
    """Tool calling func"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, "Invalid SMILES format"
    
    data = mol_to_graph(smiles, target_value=0)
    
    model = load_trained_model() 
    with torch.no_grad():
        prediction = model(data.x, data.edge_index, data.batch) # type: ignore
        
    return prediction.item(), None

def visualize_structure(smiles: str) -> str:
    """
    Call this to render a 2D image of the molecule structure.
    Input must be a valid SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol)
        # We save it to a temporary path or just return success
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
    
    data = mol_to_graph(smiles, target_value=0)
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

@st.cache_resource
def load_trained_model():
    model = GNNModel(num_node_features=NUM_FEATURES, hidden_channels=HIDDEN_CHANNELS)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    return model
def get_molecule_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "Unknown molecule"
    
    formula = rdMolDescriptors.CalcMolFormula(mol)
    weight = rdMolDescriptors.CalcExactMolWt(mol)
    
    return f"Formula: {formula} | Exact Mass: {weight:.2f} g/mol"


def run_pipeline(user_input: str):
    smiles = extract_smiles.invoke(user_input)

    if smiles == "No valid SMILES found":
        return {
            "smiles": None,
            "prediction": None,
            "error": "No valid SMILES found"
        }

    prediction = bioactivity_predictor.invoke(smiles)

    return {
        "smiles": smiles,
        "prediction": prediction
    }


tools = [extract_smiles, bioactivity_predictor, plot_2d_structure]

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
).bind_tools(tools)


sys_msg = SystemMessage(content="""
You are a chemistry assistant.

Available tools:
- ExtractSmiles
- BioactivityPredictor
- Plot2DStructure

CRITICAL RULES:

1. NEVER output SMILES directly in final answer.
2. ALWAYS use ExtractSmiles if SMILES is needed.
3. ALWAYS pass SMILES output to next tool without modification.
4. If user asks ANY of the following:
   - structure
   - image
   - draw
   - visualize
   - show molecule
   → you MUST call Plot2DStructure tool

5. If user asks:
   - pIC50
   - activity
   → use BioactivityPredictor

6. You may chain tools, but MUST NOT skip tool execution.

7. Final answer must be based ONLY on tool outputs.
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


st.set_page_config(page_title="ChEMBL GNN Predictor", layout="wide")

st.title("🔬 Molecular GNN Assistant")
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
        
        
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                st.markdown(f"**You:** {msg.content}")
            
            elif isinstance(msg, AIMessage) and msg.content:
                st.markdown(f"**Assistant:** {msg.content}")
            
            elif isinstance(msg, ToolMessage):
                st.caption(f"⚙️ Tool executed: {msg.name}")
\\
st.divider()