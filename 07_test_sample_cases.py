import os
import pickle
from unicodedata import normalize
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from transvae.sampling import reconstructing, sampling, sampling_analysis
from transvae.training_mol import TransVAE
from transvae.parsers import device_init, model_init, sample_parser_mol
from transvae.tvae_util import *
from itertools import product
from time import perf_counter

from rdkit.Chem import rdmolops

def ring_relative_positions(mol, query_atom_idx):
    ri = mol.GetRingInfo()
    positions = {"ortho": [], "meta": [], "para": []}
    for ring in ri.AtomRings():
        if query_atom_idx not in ring:
            continue

        ring_atoms = list(ring)
        for atom_idx in ring_atoms:
            if atom_idx == query_atom_idx:
                continue

            # shortest path length
            path = rdmolops.GetShortestPath(mol, query_atom_idx, atom_idx)
            dist = len(path) - 1
            if dist == 1:
                positions["ortho"].append(atom_idx)
            elif dist == 2:
                positions["meta"].append(atom_idx)
            elif dist == 3:
                positions["para"].append(atom_idx)

    return positions

# prompt 1
def check_prompt_1(smiles):
    flag = False
    mol = Chem.MolFromSmiles(smiles)
    # display(mol)
    # template = Chem.MolFromSmarts(prompt1)
    # display(template)
    # prompt2_sub_level1 = "Oc1cc(*)c(*)cc(N(*)*)1"
    prompt2_sub_level2 = "[O;D1]c1ccccc(CN)1"
    # template_level1 = Chem.MolFromSmarts(prompt2_sub_level1)
    template_level2 = Chem.MolFromSmarts(prompt2_sub_level2)
    # display(template_level1)
    # display(template_level2)
    if mol is None or template_level2 is None:
        return False

    # matches_level1 = mol.GetSubstructMatches(template_level1)
    matches_level2 = mol.GetSubstructMatches(template_level2)
    # print(matches_level1)
    # print(matches_level2)
    if len(matches_level2) == 0:
        return False
    
    matches_level2_pair = []
    # create pairs by combine matches_level2
    for i in range(len(matches_level2)):
        for j in range(i+1, len(matches_level2)):
            matches_level2_pair.append(matches_level2[i] + matches_level2[j])


    final_flag = False
    if len(matches_level2) >= 2:
        for match in matches_level2_pair:
            flag = False
            n_sub = []
            n_meta = []
            o_ortho = []
            o_ortho_n_meta = []
            c_sub = []
            o_para = []
            bonds_to_break = set()
            for a in match:
                atom = mol.GetAtomWithIdx(a)
                for bond in atom.GetBonds():
                    if bond.GetBeginAtomIdx() == a and atom.GetSymbol() == "N" and bond.GetEndAtomIdx() not in match:
                        bonds_to_break.add(bond.GetIdx())
                        n_sub.append(a)
                    elif bond.GetEndAtomIdx() == a and atom.GetSymbol() == "N" and bond.GetBeginAtomIdx() not in match:
                        bonds_to_break.add(bond.GetIdx())
                        n_sub.append(a)
                    elif atom.GetSymbol() == "N":
                        atom_to_n = bond.GetBeginAtomIdx() if bond.GetBeginAtomIdx() != a else bond.GetEndAtomIdx()
                        if mol.GetAtomWithIdx(atom_to_n).GetSymbol() == "C" and atom_to_n in match:
                            atom_to_c_in_ring_bonds = mol.GetAtomWithIdx(atom_to_n).GetBonds()
                            for b in atom_to_c_in_ring_bonds:
                                atom_to_c_in_ring = b.GetBeginAtomIdx() if b.GetBeginAtomIdx() != atom_to_n else b.GetEndAtomIdx()
                                if mol.GetAtomWithIdx(atom_to_c_in_ring).GetSymbol() == "C" and mol.GetAtomWithIdx(atom_to_c_in_ring).IsInRing() and atom_to_c_in_ring in match:
                                    pos = ring_relative_positions(mol, atom_to_c_in_ring)
                                    if 'meta' in pos: n_meta.extend(pos['meta'])
                    elif bond.GetBeginAtomIdx() == a and atom.GetSymbol() == "C" and atom.IsInRing() and bond.GetEndAtomIdx() not in match:
                        bonds_to_break.add(bond.GetIdx())
                        c_sub.append(a)
                    elif bond.GetEndAtomIdx() == a and atom.GetSymbol() == "C" and atom.IsInRing() and bond.GetBeginAtomIdx() not in match:
                        bonds_to_break.add(bond.GetIdx())
                        c_sub.append(a)
                    elif atom.GetSymbol() == "O":
                        atom_to_o = bond.GetBeginAtomIdx() if bond.GetBeginAtomIdx() != a else bond.GetEndAtomIdx() 
                        if mol.GetAtomWithIdx(atom_to_o).GetSymbol() == "C" and mol.GetAtomWithIdx(atom_to_o).IsInRing() and atom_to_o in match:
                            pos = ring_relative_positions(mol, atom_to_o)
                            if 'para' in pos: o_para.extend(pos['para'])
                            if 'ortho' in pos: o_ortho.extend(pos['ortho'])

            # print("n_sub:", n_sub)
            # print("n_meta:", n_meta)
            # print("c_sub:", c_sub)
            # print("o_para:", o_para)
            # print("o_ortho:", o_ortho)
            o_ortho_n_meta = list(set(o_ortho) & set(n_meta)) # find the intersection of o_ortho and n_meta


            bonds_to_break = list(set(bonds_to_break))
            # print(bonds_to_break)
        
            fragments = Chem.FragmentOnBonds(mol, bonds_to_break, addDummies=True)
            # display(fragments)
            flag_core = []
            flag_n_sub = []
            flag_n_bridge = []
            flag_c_sub_o_ortho_n_meta = []
            flag_c_sub_o_para = []
            others = []
            for frag in Chem.GetMolFrags(fragments, asMols=True):
                # display(frag)
                # prompt2_sub_level2_check = "[O;D1]c1ccccc(N)1"
                # template_level2_check = Chem.MolFromSmarts(prompt2_sub_level2_check)
                if frag.HasSubstructMatch(template_level2):
                    for atom in frag.GetAtoms():
                        atom.SetIsotope(0)
                        atom.SetAtomMapNum(0)
                    flag_core.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                else:
                    number_of_dummies = sum(1 for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0)
                    if number_of_dummies == 2:
                        for atom in frag.GetAtoms():
                            atom.SetIsotope(0)
                            atom.SetAtomMapNum(0)
                        flag_n_bridge.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                    elif number_of_dummies == 1:
                        # for frag_atom in frag.GetAtoms():
                        #     print("ID", frag_atom.GetIdx(), "Symbol", frag_atom.GetSymbol(), "Isotope", frag_atom.GetIsotope())
                        if any(atom.GetIsotope() in n_sub for atom in frag.GetAtoms()):
                            for atom in frag.GetAtoms():
                                atom.SetIsotope(0)
                                atom.SetAtomMapNum(0)
                            flag_n_sub.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                        elif any(atom.GetIsotope() in c_sub for atom in frag.GetAtoms()):
                            if any(atom.GetIsotope() in o_ortho_n_meta for atom in frag.GetAtoms()):
                                for atom in frag.GetAtoms():
                                    atom.SetIsotope(0)
                                    atom.SetAtomMapNum(0)
                                flag_c_sub_o_ortho_n_meta.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                            elif any(atom.GetIsotope() in o_para for atom in frag.GetAtoms()):
                                for atom in frag.GetAtoms():
                                    atom.SetIsotope(0)
                                    atom.SetAtomMapNum(0)
                                flag_c_sub_o_para.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                            else:
                                others.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                

            # print("flag_core:", flag_core)
            # print("flag_n_sub:", flag_n_sub)
            # print("flag_n_bridge:", flag_n_bridge)
            # print("flag_c_sub_o_ortho_n_meta:", flag_c_sub_o_ortho_n_meta)
            # print("flag_c_sub_o_para:", flag_c_sub_o_para)
            # print("others:", others)
            if len(flag_core)%2 == 0 and len(flag_n_bridge) == 1 \
                and len(flag_n_sub)%2 == 0 \
                    and len(flag_c_sub_o_ortho_n_meta)%2 == 0 \
                        and len(flag_c_sub_o_para)%2 == 0:
                if len(others) == 0:
                    flag = True
            if flag:
                final_flag = True
                break
    else:
        flag = False
        final_flag = False
    
    return final_flag

# prompt 2
def check_prompt_2_homo(smiles):
    if smiles == "P" or smiles == "P[H]":
        return True
    substructure_smarts = "*P(*)(*)"
    mol = Chem.MolFromSmiles(smiles)
    # display(mol)
    substructure = Chem.MolFromSmarts(substructure_smarts)
    if mol is None or substructure is None:
        return False
    matches = mol.GetSubstructMatches(substructure)
    # print("All matches:", matches)
    selected_match = list()
    selected_match_p_index = list()
    for m in matches:
        for idx in m:
            atom_m = mol.GetAtomWithIdx(idx)
            if atom_m.GetSymbol() == "P":
                if atom_m.GetDegree() == 3:
                    selected_match_p_index.append(idx)
                    selected_match.append(m)

    # print("P index:", selected_match_p_index)
    # print("All matches:", selected_match)
    if len(selected_match) == 0:
        selected_match = [matches[0]] if matches else None

    # print("All matches:", selected_match)
    if selected_match is None:
        return False
    
    final_flag = False
    for m, p in zip(selected_match, selected_match_p_index):
        # print("Selected match:", m)

        cut_bonds = []
        if matches:
            for bond in mol.GetBonds():
                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                if begin_atom_idx == p or end_atom_idx == p:
                    # print(f"Cutting bond between atoms {begin_atom_idx} and {end_atom_idx}")
                    cut_bonds.append(bond.GetIdx()) 
        
        # print("Cut bonds:", cut_bonds)
        substituent_smiles = list()
        if cut_bonds:
            fragments = Chem.FragmentOnBonds(mol, cut_bonds)
            # display(fragments)
            for frag in Chem.GetMolFrags(fragments, asMols=True):
                # display(frag)
                if len([atom for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0]) == 1:
                    for atom in frag.GetAtoms():
                        # print(f"Atom {atom.GetIdx()}: {atom.GetSymbol()}, {atom.GetAtomicNum()}, {atom.GetAtomMapNum()}, {atom.GetIsotope()}")
                        atom.SetIsotope(0)
                        atom.SetAtomMapNum(0)
                    substituent_smiles.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
            
        # print("Substituent SMILES:", substituent_smiles)
        if len(substituent_smiles) != 3:
            final_flag = False
        
        flag = True  
        for sub in substituent_smiles:
            # print("Comparing:", sub, substituent_smiles[0])
            # print("Comparing:", sub.replace("*", "C"), substituent_smiles[0].replace("*", "C"))
            if sub != substituent_smiles[0]:
                inchi1 = Chem.MolToInchi(Chem.MolFromSmiles(substituent_smiles[0].replace("*", "C")))
                inchi2 = Chem.MolToInchi(Chem.MolFromSmiles(sub.replace("*", "C")))
                if inchi1 != inchi2:
                    flag = False
                    break
        if len(substituent_smiles) != 3:
            flag = False
        if flag:
            return True

    return final_flag

# prompt 2
def check_prompt_2_hetero(smiles):
    if smiles == "P" or smiles == "P[H]":
        return False
    substructure_smarts = "*P(*)(*)"
    mol = Chem.MolFromSmiles(smiles)
    # display(mol)
    if any(atom.GetSymbol() == "P" and atom.GetDegree() == 1 for atom in mol.GetAtoms()):
        return True

    substructure = Chem.MolFromSmarts(substructure_smarts)
    if mol is None or substructure is None:
        return False
    matches = mol.GetSubstructMatches(substructure)
    # print("All matches:", matches)
    if len(matches) > 0:
        selected_match = list()
        selected_match_p_index = list()
        for m in matches:
            for idx in m:
                atom_m = mol.GetAtomWithIdx(idx)
                if atom_m.GetSymbol() == "P":
                    if atom_m.GetDegree() == 3:
                        selected_match_p_index.append(idx)
                        selected_match.append(m)
    else:
        substructure_smarts = "*P*"
        substructure = Chem.MolFromSmarts(substructure_smarts)
        matches = mol.GetSubstructMatches(substructure)
        selected_match = list()
        selected_match_p_index = list()
        for m in matches:
            for idx in m:
                atom_m = mol.GetAtomWithIdx(idx)
                if atom_m.GetSymbol() == "P":
                    if atom_m.GetDegree() == 2:
                        selected_match_p_index.append(idx)
                        selected_match.append(m)

    # print("P index:", selected_match_p_index)
    # print("All matches:", selected_match)
    if len(selected_match) == 0:
        selected_match = [matches[0]] if matches else None

    # print("All matches:", selected_match)
    if selected_match is None:
        return False
    
    final_flag = False
    for m, p in zip(selected_match, selected_match_p_index):
        # print("Selected match:", m)

        cut_bonds = []
        if matches:
            for bond in mol.GetBonds():
                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                if begin_atom_idx == p or end_atom_idx == p:
                    # print(f"Cutting bond between atoms {begin_atom_idx} and {end_atom_idx}")
                    cut_bonds.append(bond.GetIdx()) 
        
        # print("Cut bonds:", cut_bonds)
        substituent_smiles = list()
        other_substituents = []
        core = []
        if cut_bonds:
            fragments = Chem.FragmentOnBonds(mol, cut_bonds)
            # display(fragments)
            for frag in Chem.GetMolFrags(fragments, asMols=True):
                # display(frag)
                if len([atom for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0]) == 1:
                    for atom in frag.GetAtoms():
                        atom.SetIsotope(0)
                        atom.SetAtomMapNum(0)
                    substituent_smiles.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                elif frag.HasSubstructMatch(Chem.MolFromSmarts("*P(*)(*)")):
                    for atom in frag.GetAtoms():
                        atom.SetIsotope(0)
                        atom.SetAtomMapNum(0)
                    core.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                elif frag.HasSubstructMatch(Chem.MolFromSmarts("*P*")):
                    for atom in frag.GetAtoms():
                        atom.SetIsotope(0)
                        atom.SetAtomMapNum(0)
                    core.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                else:
                    for atom in frag.GetAtoms():
                        atom.SetIsotope(0)
                        atom.SetAtomMapNum(0)
                    other_substituents.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
            
        # print("Substituent SMILES:", substituent_smiles)
        # print("Core SMILES:", core)
        # print("Other substituents SMILES:", other_substituents)
        if len(other_substituents) > 0:
            return False

        if len(substituent_smiles) != 3:
            # final_flag = False
            if len(substituent_smiles) == 1:
                return True
            elif len(substituent_smiles) == 2:
                if substituent_smiles[0] == substituent_smiles[1]:
                    return True
                else:
                    inchi1 = Chem.MolToInchi(Chem.MolFromSmiles(substituent_smiles[0].replace("*", "C")))
                    inchi2 = Chem.MolToInchi(Chem.MolFromSmiles(substituent_smiles[1].replace("*", "C")))
                    if inchi1 == inchi2:
                        return True
                    else:
                        return False
            else:
                return False
        
        flag = False  
        # for sub in substituent_smiles:
        #     print("Comparing:", sub, substituent_smiles[0])
        #     if sub != substituent_smiles[0]:
        #         inchi1 = Chem.MolToInchi(Chem.MolFromSmiles(substituent_smiles[0].replace("*", "")))
        #         inchi2 = Chem.MolToInchi(Chem.MolFromSmiles(sub.replace("*", "")))
        #         if inchi1 != inchi2:
        #             flag = False
        #             break
        # have exactly 2 is the same and the other one is different
        # count
        substituent_smiles_inchi = [Chem.MolToInchi(Chem.MolFromSmiles(sub.replace("*", "C"))) for sub in substituent_smiles]
        count_dict_inchi = {}
        for sub in substituent_smiles_inchi:
            if sub in count_dict_inchi:
                count_dict_inchi[sub] += 1
            else:
                count_dict_inchi[sub] = 1
        if len(count_dict_inchi) == 2 and 2 in count_dict_inchi.values() and 1 in count_dict_inchi.values():
            flag = True 

        # if not flag:
        #     count_dict = {}
        #     for sub in substituent_smiles:
        #         if sub in count_dict:
        #             count_dict[sub] += 1
        #         else:
        #             count_dict[sub] = 1
        #     if len(count_dict) == 2 and 2 in count_dict.values() and 1 in count_dict.values():
        #         flag = True
                
        if len(substituent_smiles) != 3:
            flag = False
        if flag:
            return True

    return final_flag

# prompt 3
def check_prompt_3(smiles):
    flag = False
    mol = Chem.MolFromSmiles(smiles)
    # display(mol)
    # template = Chem.MolFromSmarts(prompt3)
    # display(template)
    prompt3_sub_level2 = "c1ccc(N=c2c3cccc4c3c(ccc4)c2=Nc2ccccc2)cc1"
    template_level2 = Chem.MolFromSmarts(Chem.MolToSmarts(Chem.MolFromSmiles(prompt3_sub_level2)))
    # display(template_level2)
    if mol is None or template_level2 is None:
        return False

    matches_level2 = mol.GetSubstructMatches(template_level2)
    if len(matches_level2) == 0:
        template_level2 = Chem.MolFromSmiles(prompt3_sub_level2)
        matches_level2 = mol.GetSubstructMatches(template_level2)
        if len(matches_level2) == 0:
            return False
    # print(matches_level2)

    final_flag = False
    if len(matches_level2) >= 1:
        for match in matches_level2:
            n_para = []
            n_ortho = {}
            c_sub = []
            bonds_to_break = set()
            for a in match:
                atom = mol.GetAtomWithIdx(a)
                for bond in atom.GetBonds():
                    if atom.GetSymbol() == "N" and bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                        atom_to_n = bond.GetBeginAtomIdx() if bond.GetBeginAtomIdx() != a else bond.GetEndAtomIdx() 
                        if mol.GetAtomWithIdx(atom_to_n).GetSymbol() == "C" and mol.GetAtomWithIdx(atom_to_n).IsInRing():
                            pos = ring_relative_positions(mol, atom_to_n)
                            if 'para' in pos: n_para.extend(pos['para'])
                            if 'ortho' in pos:
                                n_ortho.setdefault(atom_to_n, []).extend(pos['ortho'])
                    elif bond.GetBeginAtomIdx() == a and atom.GetSymbol() == "C" and atom.IsInRing() and bond.GetEndAtomIdx() not in match:
                        bonds_to_break.add(bond.GetIdx())
                        c_sub.append(a)
                    elif bond.GetEndAtomIdx() == a and atom.GetSymbol() == "C" and atom.IsInRing() and bond.GetBeginAtomIdx() not in match:
                        bonds_to_break.add(bond.GetIdx())
                        c_sub.append(a)

            # print("n_para:", n_para)
            # print("n_ortho:", n_ortho)
            # print("c_sub:", c_sub)
            bonds_to_break = list(set(bonds_to_break))
            # print(bonds_to_break)
            if len(bonds_to_break) == 0:
                return True

            fragments = Chem.FragmentOnBonds(mol, bonds_to_break, addDummies=True)
            # display(fragments)
            flag_core = []
            flag_c_sub_n_para = []
            flag_c_sub_n_ortho = {}
            others = []
            for frag in Chem.GetMolFrags(fragments, asMols=True):
                # display(frag)
                if frag.HasSubstructMatch(template_level2):
                    for atom in frag.GetAtoms():
                        atom.SetIsotope(0)
                        atom.SetAtomMapNum(0)
                    flag_core.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                else:
                    number_of_dummies = sum(1 for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0)
                    if number_of_dummies == 1:
                        # for frag_atom in frag.GetAtoms():
                        #     print("ID", frag_atom.GetIdx(), "Symbol", frag_atom.GetSymbol(), "Isotope", frag_atom.GetIsotope())
                        if any(atom.GetIsotope() in c_sub for atom in frag.GetAtoms()):
                            if any(atom.GetIsotope() in n_para for atom in frag.GetAtoms()):
                                for atom in frag.GetAtoms():
                                    atom.SetIsotope(0)
                                    atom.SetAtomMapNum(0)
                                flag_c_sub_n_para.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                            else:
                                other = True
                                for n_ortho_atoms in n_ortho:
                                    # print("n_ortho_atoms:", n_ortho_atoms, "n_ortho:", n_ortho[n_ortho_atoms])
                                    if any(atom.GetIsotope() in n_ortho[n_ortho_atoms] for atom in frag.GetAtoms()):
                                        for atom in frag.GetAtoms():
                                            atom.SetIsotope(0)
                                            atom.SetAtomMapNum(0)
                                        flag_c_sub_n_ortho.setdefault(n_ortho_atoms, []).append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                                        other = False
                                if other:
                                    others.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                    

            # print("flag_core:", flag_core)
            # print("flag_c_sub_n_para:", flag_c_sub_n_para)
            # print("flag_c_sub_n_ortho:", flag_c_sub_n_ortho)
            # print("others:", others)
            if len(flag_core) >= 1 \
                and len(flag_c_sub_n_para) <= 2 \
                    and len(others) == 0:
                if len(flag_c_sub_n_ortho) == 0:
                    flag = True
                elif len(flag_c_sub_n_ortho) == 1:
                    flag = True
                else:   
                    key1 = list(flag_c_sub_n_ortho.keys())[0]
                    key2 = list(flag_c_sub_n_ortho.keys())[1]
                    if (len(set(flag_c_sub_n_ortho[key1])) == 1 and len(flag_c_sub_n_ortho[key1]) == 2 and len(flag_c_sub_n_ortho[key2]) <= 2) \
                        or (len(flag_c_sub_n_ortho[key1]) <= 2 and len(set(flag_c_sub_n_ortho[key2])) == 1 and len(flag_c_sub_n_ortho[key2]) == 2):
                        flag = True
            if flag:
                final_flag = True
                break
            
    else:
        flag = False
        final_flag = False
    
    return final_flag

# prompt 4
def check_prompt_4(smiles):
    flag = False
    mol = Chem.MolFromSmiles(smiles)
    # display(mol)
    if mol is None:
        return False
    # template = Chem.MolFromSmarts(prompt4)
    # display(template)

    # get rings from mol
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()
    large_rings = [ring for ring in rings if len(ring) >= 8]
    # print(large_rings)

    final_flag = False
    for match in large_rings:
        bonds_to_break = set()
        for a in match:
            atom = mol.GetAtomWithIdx(a)
            for bond in atom.GetBonds():
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    bonds_to_break.add(bond.GetIdx())

        bonds_to_break = list(set(bonds_to_break))
        # print("bonds_to_break:", bonds_to_break)
        if len(bonds_to_break) == 0:
            return False

        fragments = Chem.FragmentOnBonds(mol, bonds_to_break, addDummies=True)
        # display(fragments)
        flag_in_ring = []
        for frag in Chem.GetMolFrags(fragments, asMols=True):
            for atom in frag.GetAtoms():
                atom.SetIsotope(0)
                atom.SetAtomMapNum(0)
            flag_in_ring.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))

        # count the occurrences of each fragment
        from collections import Counter
        fragment_counts = Counter(flag_in_ring)
        # print("Fragment counts:", fragment_counts)

        # all are even
        even_check = all(count % 2 == 0 for count in fragment_counts.values())
        # greater than or equal to 4 at least 1 item
        ge_4_al_1_check = any(count >= 4 for count in fragment_counts.values())

        if even_check and ge_4_al_1_check:
            flag = True
        
        if flag:
            final_flag = True
            break
    
    return final_flag

# prompt 5
def check_prompt_5(smiles):
    flag = False
    mol = Chem.MolFromSmiles(smiles)
    # display(mol)
    # template = Chem.MolFromSmarts(prompt5)
    # display(template)
    prompt5_sub_level2 = "OP1(=O)Oc2ccc3ccccc3c2-c4c(O1)ccc5ccccc45"
    template_level2 = Chem.MolFromSmarts(Chem.MolToSmarts(Chem.MolFromSmiles(prompt5_sub_level2)))
    # display(template_level2)
    if mol is None or template_level2 is None:
        return False

    matches_level2 = mol.GetSubstructMatches(template_level2)
    if len(matches_level2) == 0:
        template_level2 = Chem.MolFromSmiles(prompt5_sub_level2)
        matches_level2 = mol.GetSubstructMatches(template_level2)
        if len(matches_level2) == 0:
            return False
    # print(matches_level2)

    final_flag = False

    if len(matches_level2) >= 1:
        for match in matches_level2:
            flag = False
            c_sub = []
            bonds_to_break = set()
            for a in match:
                atom = mol.GetAtomWithIdx(a)
                for bond in atom.GetBonds():
                    if bond.GetBeginAtomIdx() == a and atom.GetSymbol() == "C" and atom.IsInRing() and bond.GetEndAtomIdx() not in match:
                        bonds_to_break.add(bond.GetIdx())
                        c_sub.append(a)
                    elif bond.GetEndAtomIdx() == a and atom.GetSymbol() == "C" and atom.IsInRing() and bond.GetBeginAtomIdx() not in match:
                        bonds_to_break.add(bond.GetIdx())
                        c_sub.append(a)

            # print("c_sub:", c_sub)
            bonds_to_break = list(set(bonds_to_break))
            # print(bonds_to_break)
            if len(bonds_to_break) == 0:
                return True

            fragments = Chem.FragmentOnBonds(mol, bonds_to_break, addDummies=True)
            # display(fragments)
            flag_core = []
            flag_c_sub = []
            for frag in Chem.GetMolFrags(fragments, asMols=True):
                # display(frag)
                if frag.HasSubstructMatch(template_level2):
                    for atom in frag.GetAtoms():
                        atom.SetIsotope(0)
                        atom.SetAtomMapNum(0)
                    flag_core.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                else:
                    number_of_dummies = sum(1 for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0)
                    if number_of_dummies == 1:
                        # for frag_atom in frag.GetAtoms():
                        #     print("ID", frag_atom.GetIdx(), "Symbol", frag_atom.GetSymbol(), "Isotope", frag_atom.GetIsotope())
                        if any(atom.GetIsotope() in c_sub for atom in frag.GetAtoms()):
                            for atom in frag.GetAtoms():
                                atom.SetIsotope(0)
                                atom.SetAtomMapNum(0)
                            flag_c_sub.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))

            # print("flag_core:", flag_core)
            # print("flag_c_sub:", flag_c_sub)
            if len(flag_core) >= 1 \
                and len(flag_c_sub)%2 == 0 \
                    and len(set(flag_c_sub)) == 1:
                flag = True
            if flag:
                final_flag = True
                break
    else:
        flag = False
        final_flag = False
    
    return final_flag

# prompt 6
def check_prompt_6(smiles):
    if smiles == "NCN":
        return True
    mol = Chem.MolFromSmiles(smiles)
    # display(mol)
    # template = Chem.MolFromSmarts(prompt6)
    # display(template)
    prompt6_sub_level2 = "NCN"
    template_level2 = Chem.MolFromSmarts(Chem.MolToSmarts(Chem.MolFromSmiles(prompt6_sub_level2)))
    # display(template_level2)
    if mol is None or template_level2 is None:
        return False

    matches_level2 = mol.GetSubstructMatches(template_level2)
    if len(matches_level2) == 0:
        template_level2 = Chem.MolFromSmiles(prompt6_sub_level2)
        matches_level2 = mol.GetSubstructMatches(template_level2)
        if len(matches_level2) == 0:
            return False
    # print(matches_level2)

    final_flag = False

    if len(matches_level2) >= 1:
        for match in matches_level2:
            flag = False
            n_sub = []
            c_sub = []
            bonds_to_break = set()
            for a in match:
                atom = mol.GetAtomWithIdx(a)
                for bond in atom.GetBonds():
                    if bond.GetBeginAtomIdx() == a and atom.GetSymbol() == "C" and atom.GetDegree() >= 2 and bond.GetEndAtomIdx() not in match:
                        bonds_to_break.add(bond.GetIdx())
                        c_sub.append(a)
                    elif bond.GetEndAtomIdx() == a and atom.GetSymbol() == "C" and atom.GetDegree() >= 2 and bond.GetBeginAtomIdx() not in match:
                        bonds_to_break.add(bond.GetIdx())
                        c_sub.append(a)
                    elif bond.GetBeginAtomIdx() == a and atom.GetSymbol() == "N" and bond.GetEndAtomIdx() not in match:
                        bonds_to_break.add(bond.GetIdx())
                        n_sub.append(a)
                    elif bond.GetEndAtomIdx() == a and atom.GetSymbol() == "N" and bond.GetBeginAtomIdx() not in match:
                        bonds_to_break.add(bond.GetIdx())
                        n_sub.append(a)

            # print("c_sub:", c_sub)
            # print("n_sub:", n_sub)
            bonds_to_break = list(set(bonds_to_break))
            # print(bonds_to_break)
            if len(bonds_to_break) == 0:
                return True

            fragments = Chem.FragmentOnBonds(mol, bonds_to_break, addDummies=True)
            # display(fragments)
            flag_core = []
            flag_c_sub = []
            flag_n_sub = []
            for frag in Chem.GetMolFrags(fragments, asMols=True):
                # display(frag)
                if frag.HasSubstructMatch(template_level2):
                    frag_temp = Chem.MolFromSmiles(Chem.MolToSmiles(frag))
                    for atom in frag_temp.GetAtoms():
                        atom.SetIsotope(0)
                        atom.SetAtomMapNum(0)
                    flag_core.append(Chem.CanonSmiles(Chem.MolToSmiles(frag_temp)))
                number_of_dummies = sum(1 for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0)
                if number_of_dummies == 1:
                    # for frag_atom in frag.GetAtoms():
                    #     print("ID", frag_atom.GetIdx(), "Symbol", frag_atom.GetSymbol(), "Isotope", frag_atom.GetIsotope())
                    if any(atom.GetIsotope() in c_sub for atom in frag.GetAtoms()):
                        for atom in frag.GetAtoms():
                            atom.SetIsotope(0)
                            atom.SetAtomMapNum(0)
                        flag_c_sub.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))
                    elif any(atom.GetIsotope() in n_sub for atom in frag.GetAtoms()):
                        for atom in frag.GetAtoms():
                            atom.SetIsotope(0)
                            atom.SetAtomMapNum(0)
                        flag_n_sub.append(Chem.CanonSmiles(Chem.MolToSmiles(frag)))

            # print("flag_core:", flag_core)
            # print("flag_c_sub:", flag_c_sub)
            # print("flag_n_sub:", flag_n_sub)
            from collections import Counter
            if len(flag_n_sub) == 0 and len(flag_c_sub) == 0 and len(c_sub) == 0 and len(n_sub) == 0:
                flag = False
            elif len(flag_core) >= 1 \
                and len(flag_c_sub) == 1 and len(flag_n_sub) == 0 and len(n_sub) == 0:
                flag = True
            elif len(flag_core) >= 1 \
                and len(flag_c_sub) == 0 \
                    and len(flag_n_sub)%2 == 0 \
                        and len(set(flag_n_sub)) == 1 and len(c_sub) == 0:
                flag = True
            elif len(flag_core) >= 1 \
                and len(flag_n_sub)%2 == 0 \
                    and len(flag_c_sub) == 1 \
                        and len(set(flag_n_sub)) == 1:
                flag = True
            if flag:
                final_flag = True
                break 
    else:
        final_flag = False
    
    return final_flag


def sample(args):
    print("Training with args:", args)
    device = device_init(args)
    
    print("### Loading data and splits...")
    if os.path.exists(os.path.join(args.data_dir, args.data_source, f"split_{args.seed}.csv")):
        print("Loading existing train/val/test split from CSV...")
        df_all = pd.read_csv(os.path.join(args.data_dir, args.data_source, f"split_{args.seed}.csv"))
    else:
        print("No existing split found. Please run 02_train_mol.py first to create train/val/test splits and save them as CSV files.")
        return

    df_train = df_all[df_all['s']=='train']
    df_val = df_all[df_all['s']=='val']
    df_test = df_all[df_all['s']=='test']
    # train_idx = df_train['id'].to_numpy()
    # val_idx = df_val['id'].to_numpy()
    # test_idx = df_test['id'].to_numpy()
    train_mols = df_train['smiles'].to_numpy()
    val_mols = df_val['smiles'].to_numpy()
    test_mols = df_test['smiles'].to_numpy()

    print("[AFTER] Train size: {}, Val size: {}, Test size: {}".format(len(train_mols), len(val_mols), len(test_mols)))
    print("#############################")
    print()

    ### Test model
    print("### Loading model...")
    ckpt_path = args.checkpoint
    vae = TransVAE(args, mode="inference", load_fn=ckpt_path)
    print("Loaded best model from:", ckpt_path)
    print("Best val loss:", vae.best_loss)
    print("Best epoch:", vae.best_epoch)
    vae.model.eval()
    print("#############################")
    print()

    # decode_method_choices = ['greedy']
    # sample_mode_choices = ['rand', 'k_high_entropy', 'rand_training', 'rand_target']
    # # entropy_cutoff_choices = [1, 2, 5]
    # k_entropy_choices = [10, 50, 100, 200] # Only for k_high_entropy
    # temperature_choices = [0.5, 0.8, 1.0, 1.5, 2.0] # Only for do_sample=True
    # top_k_choices = [3, 5, 10] # Only for do_sample=True
    # do_sample_choices = [True, False]
    # dummy_attaches_enabled_choices = [True, False]

    combination = []
    combination = [
        ('greedy', 'rand', 10, 1.0, -1, False, False),
        # ('greedy', 'k_high_entropy', 50, 1.0, -1, False, False),
        # ('greedy', 'k_high_entropy', 100, 1.0, -1, False, False),
        # ('greedy', 'rand_training', 99, 1.0, -1, False, False),
        # ('greedy', 'rand_target', 99, 1.0, -1, False, False),
        # ('greedy', 'rand', 99, 1.0, 10, True, False),
        # ('greedy', 'k_high_entropy', 50, 1.0, 10, True, False),
        # ('greedy', 'k_high_entropy', 100, 1.0, 10, True, False),
        # ('greedy', 'rand_training', 99, 1.0, 10, True, False),
        # ('greedy', 'rand_target', 99, 1.0, 10, True, False),
        # ('greedy', 'rand', 99, 0.7, 10, True, False),
        # ('greedy', 'k_high_entropy', 50, 0.7, 10, True, False),
        # ('greedy', 'k_high_entropy', 100, 0.7, 10, True, False),
        # ('greedy', 'rand_training', 99, 0.7, 10, True, False),
        # ('greedy', 'rand_target', 99, 0.7, 10, True, False),
        # ('greedy', 'rand', 99, 1.5, 10, True, False),
        # ('greedy', 'k_high_entropy', 50, 1.5, 10, True, False),
        # ('greedy', 'k_high_entropy', 100, 1.5, 10, True, False),
        # ('greedy', 'rand_training', 99, 1.5, 10, True, False),
        # ('greedy', 'rand_target', 99, 1.5, 10, True, False),
    ]

    
    for c in tqdm(combination, desc="Testing combinations of sampling settings"):
        decode_method_selected, sample_mode_selected, k_entropy_selected, temperature_selected, top_k_selected, do_sample_selected, dummy_attaches_enabled_selected = c
        print(f"### Testing combination: \
                decode_method={decode_method_selected}, \
                sample_mode={sample_mode_selected}, \
                k_entropy={k_entropy_selected}, \
                temperature={temperature_selected}, \
                top_k={top_k_selected}, \
                do_sample={do_sample_selected}, \
                dummy_attaches_enabled={dummy_attaches_enabled_selected}...")


        ### Make directory
        save_name = f"{args.seed}_{args.save_name}"
        os.makedirs(os.path.join(args.data_dir, 
                                args.data_source, 
                                'generated', 
                                f'{vae.name}_{save_name}_test_sample'), 
                                exist_ok=True)

        ### Sampling after training
        print("### Testing sampling...")
        train_mols_sample = train_mols

        sample_mode = sample_mode_selected
        decode_method = decode_method_selected
        entropy_cutoff = args.entropy_cutoff
        k_entropy = k_entropy_selected
        temperature = temperature_selected
        top_k = top_k_selected
        do_sample = do_sample_selected
        n_samples = args.n_samples
        n_samples_per_batch = args.n_samples_per_batch
        # total samples = n_samples * n_samples_per_batch
        dummy_attaches_enabled = dummy_attaches_enabled_selected

        prompt_dict = {
                       "P1": 'Oc1c([*:1])cc([*:2])cc1CN([*:3])[*:4]N([*:3])Cc2cc([*:2])cc([*:1])c2O',
                       "P2": '[*:1]P([*:2])([*:2])',
                       "P3": 'c1cc2c3c(cccc3c1)C(=Nc1c([*:3])cc([*:5])cc1[*:4])C2=Nc1c([*:1])cc([*:2])cc1[*:1]',
                       "P4": '[*:1]1[*:2][*:1][*:3][*:1][*:3][*:1][*:2]1',
                       "P5": 'OP1(=O)Oc2c([*:1])cc3ccccc3c2-c4c(O1)c([*:1])cc5ccccc45',
                       "P6": '[*:1]NC([*:2])N[*:1]'
                       }

        for p in prompt_dict:
            # prompt = 'none'
            # if 'none' not in str(args.prompt):
            #     prompt = args.prompt.split(',')
            prompt = [prompt_dict[p]]

            save_path = os.path.join(args.data_dir, 
                                    args.data_source, 
                                    'generated', 
                                    f'{vae.name}_{save_name}_test_sample',
                                    f'{p}_{str(c)}.csv')
            start_time = perf_counter()
            samples, metrics_sampling = sampling(vae, 
                                        sample_mode=sample_mode,
                                        decode_method=decode_method,
                                        n_samples=n_samples, 
                                        n_samples_per_batch=n_samples_per_batch,
                                        prompt=prompt, 
                                        k_entropy=k_entropy, 
                                        entropy_cutoff=entropy_cutoff,
                                        temperature=temperature,
                                        top_k=top_k,
                                        do_sample=do_sample,
                                        dummy_attaches_enabled=dummy_attaches_enabled,
                                        ref_mols=train_mols_sample,
                                        metrics=True, 
                                        save_path=save_path,
                                        seed=args.seed)

            stop_time = perf_counter()
            run_time = round(stop_time - start_time, 5)
            
            # check valid task
            if p == "P1":
                valid_check_fn = check_prompt_1
            elif p == "P2":
                valid_check_fn = check_prompt_2_hetero
                valid_check_fn_2 = check_prompt_2_homo
            elif p == "P3":
                valid_check_fn = check_prompt_3
            elif p == "P4":
                valid_check_fn = check_prompt_4
            elif p == "P5":
                valid_check_fn = check_prompt_5
            elif p == "P6":
                valid_check_fn = check_prompt_6
            else:
                valid_check_fn = None

            valid_task = []
            valid_task_homo = []
            valid_task_hetero = []
            valid_task_invalid = []
            if valid_check_fn is not None:
                for s in tqdm(samples, desc=f"Valid Task {p}"):
                    if s is not None:
                        try: 
                            checked = valid_check_fn(s)
                            if p == "P2":
                                checked_hetero = valid_check_fn(s)
                                checked_homo = valid_check_fn_2(s)
                                checked = checked_homo or checked_hetero
                        except:
                            checked = False
                    else:
                        checked = False
                    if checked:
                        valid_task.append(s)
                        with open(os.path.join(args.data_dir, 
                                                args.data_source, 
                                                'generated', 
                                                f'{vae.name}_{save_name}_test_sample',
                                                f'{p}_valid_task.csv'), 'a') as f:
                            f.write(f"{s}\n")
                    else:
                        valid_task_invalid.append(s)
                        with open(os.path.join(args.data_dir, 
                                                args.data_source, 
                                                'generated', 
                                                f'{vae.name}_{save_name}_test_sample',
                                                f'{p}_invalid_task.csv'), 'a') as f:
                            f.write(f"{s}\n")
                    if p == "P2":
                        try:
                            if checked_homo:
                                valid_task_homo.append(s)
                            if checked_hetero:
                                valid_task_hetero.append(s)
                        except:
                            pass
            # visualize
            invalid_samples_30 = list(set(valid_task_invalid))[:30] if len(valid_task_invalid) > 30 else list(set(valid_task_invalid))
            mols_invalid = []
            for smi in invalid_samples_30:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    mols_invalid.append(mol)
                else:
                    mol = Chem.MolFromSmiles(smi, sanitize=False)
                    if mol is not None:
                        mols_invalid.append(mol)
            if len(mols_invalid) > 0:
                display_molecule(mols_invalid)
                plt.savefig(os.path.join(args.data_dir, 
                                        args.data_source, 
                                        'generated', 
                                        f'{vae.name}_{save_name}_test_sample',
                                        f'{p}_invalid_task.png'))
                
            metrics_analysis = sampling_analysis(samples, save_path, metrics=True)

            valid_task_percentage = len(valid_task)/len(samples)*100 if len(samples) > 0 else 0
            print(f"Valid task percentage for prompt {p}: {len(valid_task)}/{len(samples)} ({valid_task_percentage:.4f}%)")
            
            # save comparison metrics
            save_path = os.path.join(args.data_dir, 
                                    args.data_source, 
                                    'generated', 
                                    f'{vae.name}_{save_name}_test_sample',
                                    f'_metrics_comparison_{p}.csv')
            if not os.path.exists(save_path):
                with open(save_path, 'w') as f:
                    f.write("model,setting,prompt,sample_mode,decode_method,"
                            "k_entropy,entropy_cutoff,temperature,top_k,do_sample,dummy_attaches_enabled,n_samples,n_samples_per_batch,"
                            "valid,valid_percentage,"
                            "unique_total,unique_total_percentage,unique_valid,unique_valid_percentage,"
                            "novel_total,novel_total_percentage,novel_validunique,novel_validunique_percentage,"
                            "intdiv,intdiv_std,snn,snn_std,valid_task,valid_task_percentage,run_time_sample,run_time_metrics,"
                            "mol_weight_avg,mol_weight_std,num_atoms_avg,num_atoms_std,sascore_avg,sascore_std\n")
            with open(save_path, 'a') as f:
                f.write(f"{vae.name},{save_name},{prompt},{sample_mode},{decode_method},"
                        f"{k_entropy},{entropy_cutoff},{temperature},{top_k},{do_sample},{dummy_attaches_enabled},{n_samples},{n_samples_per_batch},"
                        f"{metrics_sampling['Valid'][0]},{metrics_sampling['Valid'][1]},"
                        f"{metrics_sampling['Unique_Total'][0]},{metrics_sampling['Unique_Total'][1]},{metrics_sampling['Unique_Valid'][0]},{metrics_sampling['Unique_Valid'][1]},"
                        f"{metrics_sampling['Novel_Total'][0]},{metrics_sampling['Novel_Total'][1]},{metrics_sampling['Novel_ValidUnique'][0]},{metrics_sampling['Novel_ValidUnique'][1]},"
                        f"{metrics_sampling['IntDiv'][0]},{metrics_sampling['IntDiv'][1]},{metrics_sampling['SNN'][0]},{metrics_sampling['SNN'][1]},"
                        f"{len(valid_task)},{valid_task_percentage:.2f},{metrics_sampling['RunTime']},{run_time},"
                        f"{metrics_analysis['MW'][0]},{metrics_analysis['MW'][1]},{metrics_analysis['NumAtoms'][0]},{metrics_analysis['NumAtoms'][1]},{metrics_analysis['SAScore'][0]},{metrics_analysis['SAScore'][1]}\n")
            # P2
            save_path2 = os.path.join(args.data_dir, 
                                    args.data_source, 
                                    'generated', 
                                    f'{vae.name}_{save_name}_test_sample',
                                    f'_metrics_comparison_{p}_2.csv')
            if p == "P2":
                valid_task_percentage_homo = len(valid_task_homo) / len(samples) * 100 if len(samples) > 0 else 0
                valid_task_percentage_hetero = len(valid_task_hetero) / len(samples) * 100 if len(samples) > 0 else 0
                if not os.path.exists(save_path2):
                    with open(save_path2, 'w') as f:
                        f.write("model,setting,prompt,sample_mode,decode_method,"
                                "k_entropy,entropy_cutoff,temperature,top_k,do_sample,dummy_attaches_enabled,n_samples,n_samples_per_batch,"
                                "valid_task_homo,valid_task_percentage_homo,valid_task_hetero,valid_task_percentage_hetero\n")
                with open(save_path2, 'a') as f:
                    f.write(f"{vae.name},{save_name},{prompt},{sample_mode},{decode_method},"
                            f"{k_entropy},{entropy_cutoff},{temperature},{top_k},{do_sample},{dummy_attaches_enabled},{n_samples},{n_samples_per_batch},"
                            f"{len(valid_task_homo)},{valid_task_percentage_homo:.2f},{len(valid_task_hetero)},{valid_task_percentage_hetero:.2f}\n")
            
            # save comparison metrics simple
            save_path = os.path.join(args.data_dir, 
                                    args.data_source, 
                                    'generated', 
                                    f'{vae.name}_{save_name}_test_sample',
                                    f'_metrics_comparison_simple.csv')
            if not os.path.exists(save_path):
                with open(save_path, 'w') as f:
                    f.write("model,setting,prompt,sample_mode,decode_method,"
                            "k_entropy,entropy_cutoff,temperature,top_k,do_sample,dummy_attaches_enabled,n_samples,n_samples_per_batch,"
                            "valid,valid_percentage,"
                            "unique_valid,unique_valid_percentage,"
                            "novel_validunique,novel_validunique_percentage,"
                            "intdiv,intdiv_std,valid_task,valid_task_percentage,snn,snn_std,run_time_sample,run_time_metrics,"
                            "mol_weight_avg,mol_weight_std,num_atoms_avg,num_atoms_std,sascore_avg,sascore_std\n")
            with open(save_path, 'a') as f:
                f.write(f"{vae.name},{save_name},{prompt},{sample_mode},{decode_method},"
                        f"{k_entropy},{entropy_cutoff},{temperature},{top_k},{do_sample},{dummy_attaches_enabled},{n_samples},{n_samples_per_batch},"
                        f"{metrics_sampling['Valid'][0]},{metrics_sampling['Valid'][1]},"
                        f"{metrics_sampling['Unique_Valid'][0]},{metrics_sampling['Unique_Valid'][1]},"
                        f"{metrics_sampling['Novel_ValidUnique'][0]},{metrics_sampling['Novel_ValidUnique'][1]},"
                        f"{metrics_sampling['IntDiv'][0]},{metrics_sampling['IntDiv'][1]},"
                        f"{len(valid_task)},{valid_task_percentage:.2f},"
                        f"{metrics_sampling['SNN'][0]},{metrics_sampling['SNN'][1]},"
                        f"{metrics_sampling['RunTime']},{run_time},"
                        f"{metrics_analysis['MW'][0]},{metrics_analysis['MW'][1]},{metrics_analysis['NumAtoms'][0]},{metrics_analysis['NumAtoms'][1]},{metrics_analysis['SAScore'][0]},{metrics_analysis['SAScore'][1]}\n")

            # print("#############################")
            # print()

            # print("### Testing sampling analysis...")
            # save_path = os.path.join(args.data_dir, 
            #                         args.data_source, 
            #                         'generated', 
            #                         f'{vae.name}_{save_name}_test_sample',
            #                         f'{p}_analysis.csv')
            # sampling_analysis(samples, save_path)

            print("##############################")
            print()

            print(f"### Testing prompt {p} valid task visualization...")
            # visualize 20 samples ordered by SA score
            valid_task = list(set(valid_task))
            sascores = [get_sa_score(smi) for smi in valid_task]
            valid_task_sascores = list(zip(valid_task, sascores))
            valid_task_sascores_sorted = sorted(valid_task_sascores, key=lambda x: x[1])
            valid_task_sorted = [x[0] for x in valid_task_sascores_sorted]
            valid_task_sorted_20 = valid_task_sorted[:20] if len(valid_task) > 20 else valid_task_sorted
            mols_valid = [Chem.MolFromSmiles(smi) for smi in valid_task_sorted_20]
            display_molecule(mols_valid, col=5, title=None, texts=None)
            save_path = os.path.join(args.data_dir, 
                        args.data_source, 
                        'generated', 
                        f'{vae.name}_{save_name}_test_sample',
                        f'{p}_valid_task.png')
            plt.savefig(save_path.replace('.csv', '.png'))

        
    print("#############################")
    print()


if __name__ == '__main__':
    parser = sample_parser_mol()
    args = parser.parse_args()
    sample(args)
