import re
import pandas as pd
from rdkit import Chem
from transvae.tvae_util import tokenizer
import random

# REGEX
SQUARE_BRACKET = re.compile(r"(\[[^\]]*\])")
# SQUARE_BRACKET without following with *
SQUARE_BRACKET_NEW = re.compile(r"(\[(?:(?!\*)[^\]]*)\])") # update square bracket
SQUARE_BRACKET_noH = re.compile(r"(\[(?:(?!H)[^\]]*)\])")
# square bracket with no H and no index, e.g., [H] or [*:1]
SQUARE_BRACKET_noH_noINDEX = re.compile(r"(\[(?:(?!H)(?!\*:[0-9]+)[^\]]*)\])")
BRCL = re.compile(r"(Br|Cl)")
ATOM = re.compile(r"([a-zA-Z])")
# ---- RING_ATOM needs some explaining ----
# An atom followed by ring connections (0-9) or double ring connections (%0-9)
# The atom may also have an explicit bond to the ring connection -/=/#
# The atom may also be wrapped in square brackets []
# The atom should not be *within* square brackets, e.g., "[NH2+]2" is a ring atom but not the "H2" inside it
RING_ATOM = re.compile(
    r"([a-zA-Z][%0-9]+(?![^[]*\])|[a-zA-Z][-=#:][%0-9]+(?![^[]*\])|\[[^\]]*\][%0-9]+(?![^[]*\])|\[[^\]]*\][-=#:][%0-9]+(?![^[]*\]))"
)
# The following regex identifies ring numbers seperated by an explicit bond e.g., C2-3, whereas above only recognizes C-2
RING_ATOM_2 = re.compile(
    r"([a-zA-Z][%0-9][-=#][%0-9]+(?![^[]*\])|[a-zA-Z][%0-9]+(?![^[]*\])|[a-zA-Z][-=#][%0-9]+(?![^[]*\])|\[[^\]]*\][%0-9][-=#][%0-9]+(?![^[]*\])|\[[^\]]*\][%0-9]+(?![^[]*\])|\[[^\]]*\][-=#][%0-9]+(?![^[]*\]))"
)
SINGLE_RING =  re.compile(r"((?<!:)[0-9]{1})") # re.compile(r"([0-9]{1})") 
DOUBLE_RING = re.compile(r"(%[0-9]{2})")
BR_ATTCH = re.compile(r"(\(\*\))")
ATTCH = re.compile(r"(\*(?![^[]*\]))") # Not * in square brackets
SUB_ATTCH = re.compile(r"(\[99\*\])")
BR_OPEN = re.compile(r"(\()")
BR_CLOSE = re.compile(r"(\))")
BR = re.compile(r"(\(|\))")
# Attachment point with index and square bracket, e.g., [*:1]
ATTCH_INDEX = re.compile(r"(\[\*:[0-9]+\][0-9]*)")
BR_ATTCH_INDEX = re.compile(r"(\(\[\*:[0-9]+\][0-9]*\))")  

def split_by_regex(data: str, regexes: list):
    if not regexes:
        return list(data)
    # print("before regexp", data)
    regexp = regexes[0]
    # print("regexp", regexp)
    splitted = regexp.split(data)
    # print("after regexp", splitted)
    tokens = []
    for i, split in enumerate(splitted):
        if i % 2 == 0:
            tokens += split_by_regex(split, regexes[1:])
        else:
            tokens.append(split)
        # print("tokens", tokens)
    return tokens

def xsmiles2smiles(smiles):
    smiles = smiles.replace("X", "[99*]")
    smiles = re.sub(r"\[\*:([0-9]+)\]([0-9]*)", r"[*:\1]\2", smiles)
    # NOTE: Remove explicit aromatic bonds, likely not in many vocabularies
    aromatic_bond = r"(\:(?![^[]*\]))"
    smiles = re.sub(aromatic_bond, "", smiles)
    return smiles

def smiles2xsmiles(smiles):
    smiles = smiles.replace("[99*]", "X")
    smiles = re.sub(r"\[([0-9]+)\*\]([0-9]*)", r"[*:\1]\2", smiles)
    return smiles

def _check_ring_numbers(smiles, debug=False, v=False):
    """Check and re-index ring numbers sequentially if needed"""
    smiles = xsmiles2smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    ringinfo = mol.GetRingInfo()
    N_rings = ringinfo.NumRings()
    L_rings = [
        int(t.strip("%"))
        for t in DOUBLE_RING.findall(smiles) + SINGLE_RING.findall(smiles)
    ]
    if L_rings:
        L_rings = max(L_rings)

    # ---- Check rings
    if (not L_rings) or (L_rings == N_rings):
        # Assume they're labelled correctly
        return smiles

    # ---- Otherwise let's re-index
    rings = list(ringinfo.AtomRings())
    if v:
        print(rings)
    tokens = split_by_regex(smiles, [SQUARE_BRACKET, DOUBLE_RING, BRCL])

    ring_count = 0
    atom_count = -1  # Counting what the last atom was
    ring_map = {}  # {old_ring -> new_ring}
    new_tokens = []
    for i, t in enumerate(tokens):
        # If it's a ring token -> update
        if any([regex.fullmatch(t) for regex in [DOUBLE_RING, SINGLE_RING]]):
            if debug:
                import pdb

                pdb.set_trace()
            # Check to see if it's already mapped
            if t in ring_map.keys():
                new_tokens.append(ring_map[t])
                # if v:
                #     print(f"Relabelled {t} -> {ring_map[t]} at atom {atom_count}")
                ring_map.pop(t)
            else:
                # Add it
                ring_count += 1
                new_ring_id = int2ring_number(ring_count)
                ring_map[t] = new_ring_id
                new_tokens.append(new_ring_id)
                # if v:
                #     print(f"Relabelled {t} -> {new_ring_id} at atom {atom_count}")
        # If it's an atom -> count
        elif any(
            [
                regex.fullmatch(t)
                for regex in [SQUARE_BRACKET_noH, BRCL, ATOM, BR_ATTCH, ATTCH]
            ]
        ):
            atom_count += 1
            new_tokens.append(t)
        else:
            new_tokens.append(t)
    if v:
        print(ring_map)
    new_smiles = "".join(new_tokens)
    new_smiles = smiles2xsmiles(new_smiles)
    # logger.debug(f"Re-indexed SMILES rings from {smiles} -> {new_smiles}")
    return new_smiles

def reverse_smiles(smiles, renumber_rings=True, v=False):
    """
    Reverse a SMILES string
    """
    smiles_checked = _check_ring_numbers(smiles)
    # print("Checked rings:", smiles_checked)
    # Tokenize
    tokens = split_by_regex(smiles_checked, [RING_ATOM_2, SQUARE_BRACKET, BRCL, ATTCH, ATOM, ATTCH_INDEX])
    # print(f"Tokenized:\n\t{tokens}")

    # Find parenthesis
    branching_idxs = _seek_parenthesis(tokens)
    # print(f"Branches identified:\n\t{branching_idxs}")

    # Merge branches with source atom
    new_tokens = []
    i = 0
    while i < len(tokens):
        # Check if we need to combine branches into one token
        if i + 1 in branching_idxs:
            t = "".join(tokens[i : branching_idxs[i + 1] + 1])
            new_tokens.append(t)
            i = branching_idxs[i + 1] + 1
        else:
            new_tokens.append(tokens[i])
            i += 1
    # print(f"Tokens corrected by branch:\n\t{new_tokens}")

    # Reverse
    rev_tokens = list(reversed(new_tokens))
    rsmiles = "".join(rev_tokens)
    # print(f"Tokens reversed:\n\t{rev_tokens}")
    # print(f"SMILES reversed: {rsmiles}")

    # Re-number rings
    if renumber_rings:
        rsmiles = _reverse_ring_numbers(rsmiles)
        # print(f"Rings reindexed:\n\t {rsmiles}")

    return rsmiles

def _seek_parenthesis(smiles_or_tokens):
    """
    Return the indices of top level parenthesis only as a dict map
    """
    br_open = 0
    multiple = False
    open_close_idxs = {}  # {idx1: idx2}
    for i, t in enumerate(smiles_or_tokens):
        # Open
        if t == "(":
            # If first open, save index
            if (br_open == 0) and not multiple:
                br_open_idx = i
            br_open += 1
            multiple = False
        # Close
        if t == ")":
            br_open -= 1
            # If last close, save index
            if br_open == 0:
                # Check to see we aren't immediately branching afterwards
                if (i < len(smiles_or_tokens) - 1) and smiles_or_tokens[i + 1] == "(":
                    multiple = True
                else:
                    br_close_idx = i
                    open_close_idxs[br_open_idx] = br_close_idx
    return open_close_idxs

def _reverse_ring_numbers(smi: str) -> str:
    """Given ring numbers in smi, reindex the rings in smi from left to right"""
    ring_map = {}
    ring_count = 1
    new_smiles = []
    for c in split_by_regex(smi, [SQUARE_BRACKET, BRCL, DOUBLE_RING, SINGLE_RING]):
    #     print(f"Processing token: {c}")
        if SINGLE_RING.fullmatch(c) or DOUBLE_RING.fullmatch(c):
    # for c in tokenizer(smi):
        # if c.isdigit() or (c.startswith("%") and c[1:].isdigit()):
            # Add new ring to map
            if c not in ring_map.keys():
                ring_map[c] = int2ring_number(ring_count)
                ring_count += 1
            # Update ring close
            c = ring_map[c]
        # Add token
        # print(ring_map)
        new_smiles.append(c)
    smiles = "".join(new_smiles)
    return smiles

def int2ring_number(x):
    if x < 10:
        return str(x)
    else:
        return "%" + str(x)

def clean_add_module(start_smiles, add_module, ending_attach="", limit_gen_size=300):
    flag = False
    if add_module != "":
        if add_module[0] == ")":
            add_module = add_module[1:]
        elif add_module == "[H]":
            add_module = add_module
        elif add_module[0] == "]":
            add_module = add_module[1:]
        elif add_module[0] in ["=", "#", "-", "(", "[", "~"]:
            add_module = "C" + add_module
        if "." in add_module:
            add_module = add_module.replace(".", "")
    else:
        add_module = "[H]"

    if add_module != "[H]":
        while not flag:
            combined = start_smiles + add_module + ending_attach
            mol = Chem.MolFromSmiles(combined)
            mol2 = Chem.MolFromSmiles(combined, sanitize=False)
            if mol and mol2 and len(add_module) <= limit_gen_size:
                flag = True
            else:
                add_module = add_module[:-1]
            if add_module == "":
                break
    if add_module == "":
        add_module = "[H]"
    if add_module == "[H]":
        mol = Chem.MolFromSmiles(start_smiles + add_module + ending_attach)
        mol2 = Chem.MolFromSmiles(start_smiles + add_module + ending_attach, sanitize=False)
        if not mol and mol2:
            add_module = "C"
        mol = Chem.MolFromSmiles(start_smiles + add_module + ending_attach)
        mol2 = Chem.MolFromSmiles(start_smiles + add_module + ending_attach, sanitize=False)
        if not mol and mol2:
            add_module = "([H])" 

    return add_module

def xprompt_sampler(model_sampler, start_prompt, n_samples=1, limit_gen_size=300, dummy_attaches_enabled=False, seed=0):

    # set seed
    random.seed(seed)
    
    template_smiles = start_prompt
    template_smiles = template_smiles.replace('"', '').replace('"', '')  # Clean up quotes if any
    template_mol = Chem.MolFromSmiles(template_smiles)
    template_smiles_normalized = Chem.MolToSmiles(template_mol)
    # print("Normalized:", template_smiles_normalized)
    # display(template_mol)

    results = []
    for n in range(n_samples):
        # print(f"\n=== Sample {n+1} ===")

        prompt_record = {}
        prompt_count = 0
        for token in tokenizer(template_smiles_normalized):
            if "*" in token:
                prompt_record[token] = None
                prompt_count += 1
        prompt_record

        results_smiles = template_smiles_normalized

        for _ in range(prompt_count):
            
            if results_smiles is None:
                break
            try:
                template_mol_normalized = Chem.MolFromSmiles(results_smiles)
                if template_mol_normalized is None:
                    template_mol_normalized = Chem.MolFromSmiles(results_smiles, sanitize=False)
                    if template_mol_normalized is None:
                        print("A. Invalid SMILES generated during normalization:", results_smiles, template_smiles_normalized)
                        results_smiles = None
                        break
            except:
                print("B. Invalid SMILES generated during normalization:", results_smiles, template_smiles_normalized)
                results_smiles = None
                break
            
            try:
                if "." not in results_smiles:
                    atom_order = []
                    atom_list = []
                    for atom in template_mol_normalized.GetAtoms():
                        # print(f"Atom {atom.GetIdx()}: {atom.GetSymbol()}, {atom.GetAtomMapNum()}")
                        atom_list.append({
                            "atom_idx": atom.GetIdx(),
                            "symbol": atom.GetSymbol(),
                            "atom_map_num": atom.GetAtomMapNum()
                        })
                        atom_order.append(atom.GetIdx())
                    template_atom_df = pd.DataFrame(atom_list)
                    # display(template_atom_df)
                    template_atom_order = template_atom_df[template_atom_df["atom_map_num"] > 0].sort_values(["atom_map_num", "atom_idx"])
                    # display(template_atom_order)
                else:
                    atom_list = []
                    atom_order = []
                    frag_dict = dict()
                    for frag_idx, frag in enumerate(Chem.GetMolFrags(template_mol_normalized, asMols=True)):
                        frag = Chem.MolFromSmiles(Chem.MolToSmiles(frag))
                        frag_dict[frag_idx] = Chem.MolToSmiles(frag)
                        for atom in frag.GetAtoms():
                            # print(f"Atom {atom.GetIdx()}: {atom.GetSymbol()}, {atom.GetAtomMapNum()}")
                            atom_list.append({
                                "frag_idx": frag_idx,
                                "atom_idx": atom.GetIdx(),
                                "symbol": atom.GetSymbol(),
                                "atom_map_num": atom.GetAtomMapNum(),
                            })
                            atom_order.append(atom.GetIdx())
                    template_atom_df = pd.DataFrame(atom_list)
                    # display(template_atom_df)
                    template_atom_order = template_atom_df[template_atom_df["atom_map_num"] > 0].sort_values(["frag_idx", "atom_map_num", "atom_idx"])
                    # display(template_atom_order)
            except:
                print("C. Invalid SMILES generated during atom mapping:", results_smiles, template_smiles_normalized)
                results_smiles = None
                break

            for idx, row in template_atom_order.iterrows():

                if "." not in results_smiles:
                    atom_idx = row["atom_idx"]
                    # atom_order.append(atom_order.pop(atom_order.index(atom_idx)))
                    # print(atom_order)
                    random_rooted_smiles = Chem.MolToSmiles(
                        template_mol_normalized,
                        canonical=True, #False,
                        doRandom=False, #True,
                        isomericSmiles=True, #False,
                        rootedAtAtom=atom_idx,
                    )
                    # print("Random rooted SMILES:", random_rooted_smiles)
                else:
                    frag_idx = row["frag_idx"]
                    atom_idx = row["atom_idx"]
                    frag_current = frag_dict[frag_idx]
                    smiles_frag_rooted = Chem.MolToSmiles(
                        Chem.MolFromSmiles(frag_current),
                        canonical=True, #False,
                        doRandom=False, #True,
                        isomericSmiles=True, #False,
                        rootedAtAtom=atom_idx,
                    )
                    smiles_frag_rooted = ".".join([smiles_frag_rooted] + [frag_dict[i] for i in range(len(frag_dict)) if i != frag_idx])
                    random_rooted_smiles = smiles_frag_rooted
                    # print("Random rooted SMILES:", random_rooted_smiles)
                
                # MY OWN: USING RENUMBER
                try:
                    random_rooted_mol = Chem.MolFromSmiles(random_rooted_smiles)
                    if random_rooted_mol is None:
                        random_rooted_mol = Chem.MolFromSmiles(random_rooted_smiles, sanitize=False)
                        if random_rooted_mol is None:
                            print("A. Invalid SMILES generated during random rooting:", random_rooted_smiles, results_smiles)
                            next_prompt = None
                            break
                except:
                    print("B. Invalid SMILES generated during random rooting:", random_rooted_smiles, results_smiles)
                    next_prompt = None
                    break  
                # for atom in random_rooted_mol.GetAtoms():
                #     print(f"Atom {atom.GetIdx()}: {atom.GetSymbol()}, {atom.GetAtomMapNum()}")
                # display(random_rooted_mol)
                atom_order = list(reversed([atom.GetIdx() for atom in random_rooted_mol.GetAtoms()]))
                random_rooted_mol_reversed = Chem.RenumberAtoms(random_rooted_mol, newOrder=atom_order)
                # for atom in random_rooted_mol_reversed.GetAtoms():
                #     print(f"Reversed Atom {atom.GetIdx()}: {atom.GetSymbol()}, {atom.GetAtomMapNum()}")
                # display(random_rooted_mol_reversed)
                current_prompt = Chem.MolToSmiles(
                    random_rooted_mol_reversed, 
                    canonical=False, 
                    doRandom=False,
                    isomericSmiles=True,
                )

                # FROM PROMPTSMILES 
                # current_prompt = reverse_smiles(random_rooted_smiles) # From PromptSMILES paper
                
                # print("Current prompt:", current_prompt)
                # display(Chem.MolFromSmiles(current_prompt))

                tokens = tokenizer(current_prompt)
                # print("Tokens:", tokens)

                ending_attach = ""
                for _ in range(len(tokens)-1, -1, -1):
                    # print("Checking token:", tokens[_])
                    if "*" in tokens[_]:
                        tokens_start = tokens[:_]
                        if prompt_record[tokens[_]] is not None:
                            # print("Using recorded prompt for", tokens[_])
                            next_prompt = "".join(tokens_start) + prompt_record[tokens[_]] + ending_attach
                            # print("Next prompt:", "".join(tokens_start), "+", prompt_record[tokens[_]], "+", ending_attach)
                            if Chem.MolFromSmiles(next_prompt) is None:
                                if "".join(tokens_start)[-1] in ["=", "#", "-", "~"]:
                                    next_prompt = "".join(tokens_start)[:-1] + prompt_record[tokens[_]] + ending_attach
                            # display(Chem.MolFromSmiles(next_prompt))
                            break
                        if dummy_attaches_enabled:
                            # dummy_attach = ["C"]
                            dummy_attach = ["C", "[H]", "C"*random.randint(2, 5), "N", "Cl", "Br", "C(c1ccccc1)", "C(C)(C)C"]
                        else:
                            dummy_attach = ["C"]
                        start_prompt = [t if "*" not in t else random.choice(dummy_attach) for t in tokens_start]
                        start_prompt = "".join(start_prompt)
                        # print("New prompt:", start_prompt)
                        if dummy_attaches_enabled:
                            if len(start_prompt) < 20:
                                dump_prompt = "C"*random.randint(5, 10)
                            elif len(start_prompt) < 50:
                                dump_prompt = "C"*random.randint(1, 5)
                            # dump_prompt = "CCCCCCCC"
                        else:
                            dump_prompt = ""
                        if "." in start_prompt:
                            before_last_dot, after_last_dot = start_prompt.rsplit(".", 1)
                            final_prompt = before_last_dot + "." + dump_prompt + after_last_dot
                        else:
                            final_prompt = dump_prompt + start_prompt
                        # final_prompt = start_prompt
                        # print("Final prompt:", final_prompt)
                        if ":" in tokenizer(final_prompt):
                            print(final_prompt)
                            next_prompt = None
                            break
                        tokens_gen = model_sampler(prompt=final_prompt)
                        # print("Generated tokens:", tokens_gen)
                        for gen in tokens_gen:
                            # print("Generated SMILES:", gen)
                            # display(Chem.MolFromSmiles(gen))
                            new_gen = gen.replace(final_prompt, "")
                            # print("Raw generated module:", new_gen)
                            new_gen = clean_add_module(start_prompt, new_gen, ending_attach, limit_gen_size=limit_gen_size)
                            # print("New gen:", new_gen)
                            # final_gen = start_prompt + new_gen
                            # print("Cleaned Generated SMILES:", final_gen)
                            # display(Chem.MolFromSmiles(final_gen))
                            next_prompt = "".join(tokens_start) + new_gen + ending_attach
                            # print("Next prompt:", "".join(tokens_start), "+", new_gen, "+", ending_attach)
                            # display(Chem.MolFromSmiles(next_prompt))
                            prompt_record[tokens[_]] = new_gen
                            # print("Recorded prompt", prompt_record)
                        break
                    else:
                        ending_attach = tokens[_] + ending_attach
                break

            results_smiles = next_prompt
            # print("results_smiles", results_smiles)

        if results_smiles is not None:
            results.append(results_smiles)
        else:
            results.append("")

    return results