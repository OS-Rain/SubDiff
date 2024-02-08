import torch
import random
import os
import pickle
import logging
import numpy as np
import os.path as path
import torch.nn.functional as F
from rdkit import Chem
from arguments import parse_arguments
def adjust_mols(charges, positions):
    num_mols = charges.shape[0]
    new_charges = []
    new_positions = []
    # new_one_hots = []
    for i in range(num_mols):
        charge = charges[i]
        position = positions[i]
        # one_hot = one_hots[i]
        
        non_one_elements = charge > 1
        one_elements = charge <= 1
        
        charge = torch.cat((charge[non_one_elements], charge[one_elements]))
        position = torch.cat((position[non_one_elements], position[one_elements]))
        # one_hot  = torch.cat((one_hot[non_one_elements], one_hot[one_elements]))

        new_charges.append(charge)
        new_positions.append(position)
        # new_one_hots.append(one_hot)
    return torch.stack(new_charges), torch.stack(new_positions)

def pad_edges(max_edges, edges_index):
    data_tensor = torch.zeros([1, 2, max_edges], dtype=torch.long)
    for data in edges_index:
        data = torch.stack(data)
        raw_edges = data.shape[1]
        expanded_size = (0, max_edges-raw_edges)
        data = F.pad(data, expanded_size)
        data = torch.unsqueeze(data, dim=0)
        data_tensor = torch.cat([data_tensor, data], dim=0)
    
    return data_tensor[1:]

def get_edges_index(smiles_list):
    edges_index = []
    for smile in smiles_list:
        mol_fromsmile = Chem.MolFromSmiles(smile)
        mol_fromsmile = Chem.AddHs(mol_fromsmile)
        adjacency_matrix = Chem.GetAdjacencyMatrix(mol_fromsmile)
        num_nodes = adjacency_matrix.shape[0]
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adjacency_matrix[i][j] == 1:
                    edges.append([i, j])
                    edges.append([j, i])
        edges = torch.tensor(edges).T
        edges_index.append([torch.LongTensor(edges[0]), torch.LongTensor(edges[1])])
    return edges_index

def get_subgraphs_edges_index(motif_smiles_list, smiles_list, max_atoms, charges):
    num_charges = len(charges)
    motif_mols = [Chem.MolFromSmiles(m) for m in motif_smiles_list]
    mols = [Chem.MolFromSmiles(m) for m in smiles_list]
    edges_index = []
    num_mols = len(mols)
    # match_subgraphs = -torch.ones([num_mols, max_atoms], dtype=torch.long)
    match_subgraphs = -np.ones([num_mols, max_atoms],dtype=np.longlong)
    mask_subgraphs = np.zeros([num_mols, max_atoms],dtype=np.longlong)
    # print(" The shape of match_subgraphs is :", match_subgraphs.shape)
    for i, mol in enumerate(mols):
        mol_with_h = Chem.AddHs(mol)    
        adjacency_matrix = Chem.GetAdjacencyMatrix(mol_with_h)
        matches_set = set()
        count = 0
        for m, motif in enumerate(motif_mols):
            cur_matches = mol.GetSubstructMatches(motif)
            cur_matches_set = set()
            for match in cur_matches:
                isnot_overl = True
                for id in match:
                    if(id in matches_set or id in cur_matches_set):
                        isnot_overl = False
                        break
                if isnot_overl:
                    count += 1
                    for id in match:
                        mask_subgraphs[i][id] = count
                        matches_set.add(id)
                        cur_matches_set.add(id)
            atoms_in_substructure = [mol_with_h.GetAtomWithIdx(idx) for idx in cur_matches_set]
            for atom in atoms_in_substructure:
                left_id = atom.GetIdx()
                match_subgraphs[i][left_id] = m + num_charges
                for neighbor in atom.GetNeighbors():
                    right_id = neighbor.GetIdx()
                    if(str(neighbor.GetSymbol())!='H' and right_id not in cur_matches_set):
                        adjacency_matrix[left_id][right_id] = 0
                        adjacency_matrix[right_id][left_id] = 0
                    elif(str(neighbor.GetSymbol())=='H'):
                        matches_set.add(right_id)
                        match_subgraphs[i][right_id] = m + num_charges
                        mask_subgraphs[i][right_id] = mask_subgraphs[i][left_id]
        for atom in mol_with_h.GetAtoms():
            atom_idx = atom.GetIdx()
            if(match_subgraphs[i][atom_idx]==-1):
                atom_type = atom.GetSymbol()
                charges_id = charges[str(atom_type)]
                match_subgraphs[i][atom_idx] = charges_id
                count += 1
                mask_subgraphs[i][atom_idx] = count
        edges = []
        num_nodes = adjacency_matrix.shape[0]
        for k in range(num_nodes):
            for j in range(k + 1, num_nodes):
                if adjacency_matrix[k][j] == 1:
                    edges.append([k, j])
                    edges.append([j, k])
        edges = torch.tensor(edges).T
        if(edges.shape[0] == 0):
            edges_index.append(edges)
            continue
        edges_index.append([torch.LongTensor(edges[0]), torch.LongTensor(edges[1])])
    return edges_index, match_subgraphs, mask_subgraphs

def count_elements(numbers):
    count_dict = {}     
    for num in numbers:
        if num in count_dict:
            count_dict[num] += 1  
        else:
            count_dict[num] = 1
    return count_dict


def molecules_match_subgraphs(dataset_dir, file_path_subgraphs, charges):
    splits=None
    split_names = splits.keys() if splits is not None else [
            'train', 'valid', 'test']
    datafiles = {split: os.path.join(
            *(dataset_dir, split + '.npz')) for split in split_names}
    datasets = {}
    for split, datafile in datafiles.items():
        with np.load(datafile) as f:                
            datasets[split] = {key: val if type(val[0])==type(np.array([''])[0]) else torch.from_numpy(
                val) for key, val in f.items()}
    # included_species = torch.cat([dataset['charges'].unique()
    #                          for dataset in datasets.values()]).unique(sorted=True)
    # num_charges = len(charges)

    # file_path_subgraphs = "preprocess/merging_operation.txt" 
    motif_smiles_list = []
    with open(file_path_subgraphs, "r") as file:
        for line in file:
            line = line.rstrip("\n")
            motif_smiles_list.append(line)
    
    logging.info('Saving edges data and adjusted data:')
    for split in split_names:
        smiles_list = []
        with np.load(os.path.join(
            *(dataset_dir, split + '.smiles.npz'))) as f:
            for _, val in f.items():
                for smile in val:
                    smiles_list.append(smile)
        edges_index = get_edges_index(smiles_list)
        with open(os.path.join(dataset_dir, split+'.edges.pkl'), 'wb') as f:
            pickle.dump(edges_index, f)

        max_atoms = datasets['train']['charges'].shape[1]
        subgraphs_edges_index, match_subgraphs, mask_subgraphs = get_subgraphs_edges_index(motif_smiles_list, smiles_list, max_atoms, charges)
        with open(os.path.join(dataset_dir, split+'.subgraphedges.pkl'), 'wb') as f:
            pickle.dump(subgraphs_edges_index, f)
        
        datasets[split]['charges'], datasets[split]['positions'] = adjust_mols(
            datasets[split]['charges'], datasets[split]['positions'])
        datasets[split].update({'subgraph_charges': match_subgraphs + 1})

        # datasets[split]['edges'] = pad_edges(56, subgraphs_edges_index)
        datasets[split]['subgraph_masks'] = mask_subgraphs
        # datasets[split].update({'subgraphs_edges': subgraphs_edges_index})
        
        np.savez_compressed(os.path.join(
            *(dataset_dir, split + '.npz')), **datasets[split])
        
        
    return datasets


def get_subgraphs_info(dataset_dir, datasets, file_path, dic):
    dataset_info = {}

    num_subgraphs = []
    for cur in datasets['train']['subgraph_masks']:
        num_subgraphs.append(int(cur.max()))
    result = count_elements(num_subgraphs)
    dataset_info.update({'num_subgraphs': result})

    count_dic = {}
    for i in range(100):
        count_dic.update({i:[0]})
    for mol_id in range(datasets['train']['subgraph_charges'].shape[0]):
        for item in datasets['train']['subgraph_masks'][mol_id]:
            if(int(item)==0):
                break
            index = datasets['train']['subgraph_masks'][mol_id] == item
            num_count = (datasets['train']['subgraph_charges'][mol_id][index]).shape[0]
            cur_atom = datasets['train']['subgraph_charges'][mol_id][index][0]
            count_dic[int(cur_atom)].append(int(num_count))
    dataset_info.update({'num_nodes_in_subgraph': count_dic})
    subgraph_decoder = ['H', 'C', 'N', 'O', 'F']
    num_atoms = len(dic) + 1
    motif_smiles_list = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.rstrip("\n")
            motif_smiles_list.append(line)
    for i, motif in enumerate(motif_smiles_list):
        dic.update({i+num_atoms : motif})    
        subgraph_decoder.append(motif) 
    
    dataset_info.update({'subgraph_macher': dic})
    dataset_info.update({'subgraph_decoder': subgraph_decoder})

    with open(os.path.join(dataset_dir, 'dataset_info.pkl'), "wb") as file:
        pickle.dump(dataset_info, file)

if __name__ == "__main__":
    args = parse_arguments()

    if(args.dataset == 'QM9'):
        charges = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        dic = {1:'H', 2:'C',  3:'N', 4 :'O',  5:'F'}
    else:
        charges = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7, 'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}     
        dic = {1:'H', 2:'B', 3:'C', 4:'N', 5:'O', 6:'F', 7:'Al', 8:'Si', 9:'P', 10:'S', 11:'Cl', 12:'As', 13:'Br', 14:'I', 15:'Hg', 16:'Bi'}  
    datasets = molecules_match_subgraphs(dataset_dir = "qm9/temp/qm9",
                        file_path_subgraphs = path.join("preprocess/", "merging_operation.txt"),
                        charges = charges
                        )
    
    get_subgraphs_info(dataset_dir= "qm9/temp/qm9", datasets=datasets, file_path = path.join("preprocess/", "merging_operation.txt"), dic=dic)