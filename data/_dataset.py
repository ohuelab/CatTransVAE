dataset_args = dict()

dataset_args['catalyst'] = {
    'file': 'catalyst',
    'smiles': 'smiles'
}

dataset_args['rxncat_procedure'] = {
    'file': 'rxncat_procedure',
    'smiles': {'reactant': 'Reactant_SMILES_new', 
               'reagent': 'Reagent_SMILES_new', #without catalyst
               'product': 'Product_SMILES_new', 
               'catalyst': 'Catalyst_SMILES_new'},
    'task': 'Yield_Clipped',
    'ids': 'index',
    'splitting': None,
    'predictiontask': 'yield',
    'predictiontype': 'regression',
    'procedure': 'procedure',
}

# rxn_id,catalyst_smiles_new,reagentall_smiles_new,startingmat_smiles_new,
# product_smiles_new,time_h,procedure,source_id,rxn_type,catalyst_smiles,
# solvent_smiles,reagent_smiles,startingmat_smiles,product_smiles,yield,
# reagentall_smiles,rxn_smiles,Catalyst_SMILES_mw,Catalyst_SMILES_no

# index,Catalyst_SMILES_new,Reagent_SMILES_new,Reactant_SMILES_new,Product_SMILES_new,time_h,
# procedure,source_id,rxn_type,catalyst_smiles,solvent_smiles,reagent_smiles,startingmat_smiles,
# product_smiles,yield,reagentall_smiles,Reaction_SMILES,Catalyst_SMILES_mw,Catalyst_SMILES_no,Yield_Clipped


dataset_args['ps_ddG_avg_804'] = {
    'file': 'ps_ddG_avg_804',
    'smiles': {'reactant': 'Sub_SMILES_new', 
               'reagent': 'Reagent_SMILES_new',
               'product': 'Product_SMILES_new', 
               'catalyst': 'Cat_SMILES_new'},
    'task': 'DeltaDeltaG',
    'ids': 'index',
    'splitting': None,
    'predictiontask': 'others',
    'predictiontype': 'regression',
    'procedure': 'Procedure',
}


dataset_args['um_cui_hte_split'] = {
    'file': 'um_cui_hte_split',
    'smiles': {'reactant': 'Reactant', 
               'reagent': 'Reagent',
               'product': 'PRODUCT_STRUCTURE', 
               'catalyst': 'Catalyst_Ligand'},
    'task': 'Product_Yield_PCT_Area_UV',
    'ids': 'index',
    'splitting': None,
    'predictiontask': 'yield',
    'predictiontype': 'regression',
    'procedure': 'Procedure',
}

dataset_args['niwa_lewis_246_ligand_2'] = {
    'file': 'niwa_lewis_246_ligand_2',
    'smiles': {'reactant': 'Input1_Input2', 
               'reagent': 'PdCatalyst_Additive', #with catalyst
               'product': 'Product', 
               'catalyst': 'Ligand'},
    'task': 'Output',
    'ids': 'index',
    'splitting': 'splitting',
    'predictiontask': 'yield',
    'predictiontype': 'regression',
    'procedure': 'Procedure',
}

dataset_args['greencat_yield'] = {
    'file': 'greencat_yield',
    'smiles': {'reactant': 'Reactant_final', 
               'reagent': 'Reagent_final', #with catalyst
               'product': 'Product_final', 
               'catalyst': 'Ligand_only_final'},
    'task': 'Yield (%)',
    'ids': 'index',
    'splitting': None,
    'predictiontask': 'yield',
    'predictiontype': 'regression',
    'procedure': 'Procedure',
}

dataset_args['greencat_yield_Split_Year'] = {
    'file': 'greencat_yield_Split_Year',
    'smiles': {'reactant': 'Reactant_final', 
               'reagent': 'Reagent_final', #with catalyst
               'product': 'Product_final', 
               'catalyst': 'Ligand_only_final'},
    'task': 'Yield (%)',
    'ids': 'index',
    'splitting': 'Split_Year',
    'predictiontask': 'yield',
    'predictiontype': 'regression',
    'procedure': 'Procedure',
}

dataset_args['greencat_yield_Split_Year_Add'] = {
    'file': 'greencat_yield_Split_Year_Add',
    'smiles': {'reactant': 'Reactant_final', 
               'reagent': 'Reagent_final', #with catalyst
               'product': 'Product_final', 
               'catalyst': 'Ligand_only_final'},
    'task': 'Yield (%)',
    'ids': 'index',
    'splitting': 'Split_Year',
    'predictiontask': 'yield',
    'predictiontype': 'regression',
    'procedure': 'Procedure',
}

dataset_args['Borylation'] = {
    'file': 'Borylation',
    'smiles': {'reactant': 'Reactant_final', 
               'reagent': 'Reagent_final', #with catalyst
               'product': 'Product_final', 
               'catalyst': 'Ligand_only_final'},
    'task': 'Yield (%)',
    'ids': 'index',
    'splitting': None,
    'predictiontask': 'yield',
    'predictiontype': 'regression',
    'procedure': 'Procedure',
}


