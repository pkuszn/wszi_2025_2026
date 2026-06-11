import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem

HYB_MAP = {
    'UNSPECIFIED': 0, 'S': 1, 'SP': 2, 'SP2': 3, 
    'SP3': 4, 'SP3D': 5, 'SP3D2': 6
}

def smiles_to_graph(smiles, y_val):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    atomic_number = []
    for atom in mol.GetAtoms():
        atomic_number.append([atom.GetAtomicNum()])
    x = torch.tensor(atomic_number, dtype=torch.float)
    
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index, y=torch.tensor([y_val], dtype=torch.float))


def mol_to_graph(smiles, target_value, extra_features_vector, atom_scaler):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    AllChem.ComputeGasteigerCharges(mol) # type: ignore
    
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            1.0 if atom.IsInRing() else 0.0,
            float(atom.GetIsAromatic()),
            float(atom.GetTotalNumHs()),
            float(atom.GetTotalValence()),
            float(atom.GetMass()),
            HYB_MAP.get(str(atom.GetHybridization()), 0),
            float(atom.GetProp('_GasteigerCharge'))
        ])
    
    node_feats_array = np.array(node_feats)
    scaled_feats = atom_scaler.transform(node_feats_array)
    x = torch.tensor(scaled_feats, dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=torch.tensor([target_value], dtype=torch.float))
    
    data.extra_features = torch.tensor(extra_features_vector, dtype=torch.float).view(1, -1)
    
    return data