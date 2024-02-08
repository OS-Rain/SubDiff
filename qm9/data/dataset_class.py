import torch
from torch.utils.data import Dataset

import os
from itertools import islice
from math import inf
from torch_geometric.nn import global_mean_pool

import logging

class ProcessedDataset(Dataset):
    """
    Data structure for a pre-processed cormorant dataset.  Extends PyTorch Dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    included_species : tensor of scalars, optional
        Atomic species to include in ?????.  If None, uses all species.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    normalize : bool, optional
        ????? IS THIS USED?
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.

    """
    def __init__(self, data, included_species=None, all_species_subgraphs=None, num_pts=-1, normalize=True, shuffle=True, subtract_thermo=True):

        self.data = data

        if num_pts < 0:
            self.num_pts = len(data['charges'])
        else:
            if num_pts > len(data['charges']):
                logging.warning('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(data['charges'])))
                self.num_pts = len(data['charges'])
            else:
                self.num_pts = num_pts

        # If included species is not specified
        if included_species is None:
            included_species = torch.unique(self.data['charges'], sorted=True)
            if included_species[0] == 0:
                included_species = included_species[1:]

        if subtract_thermo:
            thermo_targets = [key.split('_')[0] for key in data.keys() if key.endswith('_thermo')]
            if len(thermo_targets) == 0:
                logging.warning('No thermochemical targets included! Try reprocessing dataset with --force-download!')
            else:
                logging.info('Removing thermochemical energy from targets {}'.format(' '.join(thermo_targets)))
            for key in thermo_targets:
                data[key] -= data[key + '_thermo'].to(data[key].dtype)

        self.included_species = included_species
        self.all_species_subgraphs = all_species_subgraphs

        self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
        self.data['subgraph_one_hot'] = self.data['subgraph_charges'].unsqueeze(-1) == all_species_subgraphs.unsqueeze(0).unsqueeze(0)
        self.num_species = len(included_species)
        self.max_charge = max(included_species)

        subgraph_positions = torch.zeros_like(self.data['positions'], device=self.data['positions'].device)
        subgraph_one_hot = torch.zeros_like(self.data['subgraph_one_hot'], device=self.data['subgraph_one_hot'].device)
        subgraph_charges = torch.zeros_like(self.data['subgraph_charges'], device=self.data['subgraph_charges'].device)
        
        for i, node_x in enumerate(data['positions']):
            indice = data['subgraph_masks'][i]
            max_size = int(indice.max())
            subgraph_positions[i][:max_size] = global_mean_pool(node_x, indice)[1:]
            subgraph_one_hot[i][:max_size] = global_mean_pool(self.data['subgraph_one_hot'][i], indice)[1:]
            subgraph_charges[i][:max_size] = global_mean_pool(self.data['subgraph_charges'][i], indice)[1:]
        
        self.data['subgraph_positions'] = subgraph_positions
        self.data['subgraph_one_hot'] = subgraph_one_hot
        self.data['subgraph_charges'] = subgraph_charges

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}

        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()

        if shuffle:
            self.perm = torch.randperm(len(data['charges']))[:self.num_pts]
        else:
            self.perm = None

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        return {key: val[idx] for key, val in self.data.items()}
