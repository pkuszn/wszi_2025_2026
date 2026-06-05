import torch
from torch_geometric.data import Data
from rdkit import Chem

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


def mol_to_graph(smiles, target_value, extra_features_vector):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None

    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            1.0 if atom.IsInRing() else 0.0
        ])
    x = torch.tensor(node_feats, dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=torch.tensor([target_value], dtype=torch.float))
    
    data.extra_features = torch.tensor(extra_features_vector, dtype=torch.float)

    return data