from importlib.resources import path
from alphafold.common import residue_constants
import numpy as np
import os

restype_3to1 = {
    'ALA': 'A' ,
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
}


"""
dict_keys(['template_aatype', 'template_all_atom_masks', 'template_all_atom_positions', 'template_domain_names', 'template_sequence', 'template_sum_probs'])
"""
def parse_pdb(input_pdb):
    # input: pdb_name
    # output:
    # seq_list : {A:[], B:[], C:[]}
    # index_list : [A,B,C]
    # atom_list : {A:{index1:{atom1:[], atom2:[]...}, index2:{}....},...}
    # pdb line: 
    # ATOM    994  CE2 PHE A  64      11.485   1.004  -2.799  1.00 97.61           C
    # ATOM   1624 HE21 GLN A 105      -4.248  13.875  -3.152  1.00 97.99           H 
    
    seq_list = []
    atom_list = []
    last_index = ''
    last_chain = ''
    for line in open(input_pdb, 'r'):
        if len(line)<4:
            continue
        if line[0:4] != 'ATOM':
            continue

        atom_name = line[11:17].replace(' ', '')
        res3 = line[17:20].replace(' ', '')
        chain = line[20:22].replace(' ', '')
        index = line[22:27].replace(' ', '')
        coor = line[27:55]
        coor_list = coor.split(' ')
        coor_list_float = []
        for item in coor_list:
            if item!='':
                try:
                    coor_list_float.append(float(item))
                except:
                    raise ValueError(f'Found an unknown coordinate:{coor} in {line} of {input_pdb}!')
        
        if len(coor_list_float)!=3:
            raise ValueError(f'Found an unknown coordinate:{coor} in {line} of {input_pdb}!')

        if chain != last_chain :
            if last_chain != '':
                break
            last_chain = chain

        if res3 not in restype_3to1.keys():
            raise ValueError(f'Found an unknown residual: {res3}')
        res1 = restype_3to1[res3]
        if last_index != index:
            seq_list.append(res1)
            last_index = index
            atom_list.append({})
        
        atom_list[-1][atom_name] = coor_list_float
    
    assert(len(seq_list)== len(atom_list))
    return seq_list, atom_list

    

def list2atom37(atom_list, templates_all_atom_positions, templates_all_atom_masks):
    # input: 
    # [[atoms in res1], [atoms in res2]]
    # templates_all_atom_positions : list of all atoms. shape: [res, 37, 3]
    # templates_all_atom_masks: list of all atoms. shape: [res, 37]
    index = 0
    for item in atom_list:
        # item is a dict
        for atom, coordinate in item.items():
            if atom in residue_constants.atom_order.keys():
                templates_all_atom_positions[index][residue_constants.atom_order[atom]][0] = coordinate[0]
                templates_all_atom_positions[index][residue_constants.atom_order[atom]][1] = coordinate[1]
                templates_all_atom_positions[index][residue_constants.atom_order[atom]][2] = coordinate[2]
                templates_all_atom_masks[index][residue_constants.atom_order[atom]] = 1.0
            
        index += 1

def list2atom37_ala(atom_list, templates_all_atom_positions, templates_all_atom_masks):
    # input: 
    # [[atoms in res1], [atoms in res2]]
    # templates_all_atom_positions : list of all atoms. shape: [res, 37, 3]
    # templates_all_atom_masks: list of all atoms. shape: [res, 37]
    ala_atom_list = ['C', 'CA', 'CB', 'N', 'O']
    index = 0
    for item in atom_list:
        # item is a dict
        for atom, coordinate in item.items():
            if atom not in ala_atom_list:
                continue
            if atom in residue_constants.atom_order.keys():
                templates_all_atom_positions[index][residue_constants.atom_order[atom]][0] = coordinate[0]
                templates_all_atom_positions[index][residue_constants.atom_order[atom]][1] = coordinate[1]
                templates_all_atom_positions[index][residue_constants.atom_order[atom]][2] = coordinate[2]
                templates_all_atom_masks[index][residue_constants.atom_order[atom]] = 1.0
        
        index += 1

def align(input_seq, template_seq):
    l1 = len(input_seq)
    l2 = len(template_seq)
    dp = []
    for i in range(l1+1):
        dp.append([0]*(l2+1))
    
    for i in range(l1):
        dp[i+1][0] = i+1
    for i in range(l2):
        dp[0][i+1] = i+1
    
    for i in range(l1):
        for j in range(l2):
            if input_seq[i] == template_seq[j]:
                dp[i+1][j+1] = dp[i][j]
            else:
                dp[i+1][j+1] = min(dp[i][j], dp[i+1][j], dp[i][j+1])+1
    
    ret_list = []
    
    min_dis = max(l1+1, l2+1)
    last_line = dp[-1]
    for i in range(l2+1):
        if last_line[l2-i] < min_dis:
            j = l2-i
            min_dis = last_line[l2-i]
    i = l1
    while(i>=0 and j >=0):
        if i == 0 and j == 0:
            break
        
        pre_list = [max(l1+1,l2+1)]*3
        if i>0 and j>0:
            pre_list[0] = dp[i-1][j-1]
        if i>0:
            pre_list[1] = dp[i-1][j]
        if j>0:
            pre_list[2] = dp[i][j-1]
        
        if min(pre_list) == pre_list[0]:
            ret_list.append(j-1)
            i = i-1
            j = j-1
            continue
        elif min(pre_list) == pre_list[1]:
            ret_list.append(-1)
            i = i-1
            continue
        else:
            j = j - 1
            continue
    
    ret_list.reverse()
    assert(len(ret_list) == len(input_seq))
    return ret_list

def make_empty_features(num_res):
    empty_template_features = {
        'template_aatype': np.zeros((1, num_res, len(residue_constants.restypes_with_x_and_gap)),np.float32),
        'template_all_atom_masks': np.zeros((1, num_res, residue_constants.atom_type_num), np.float32),
        'template_all_atom_positions': np.zeros((1, num_res, residue_constants.atom_type_num, 3), np.float32),
        'template_domain_names': np.array([''.encode()], dtype=np.object),
        'template_sequence': np.array([''.encode()], dtype=np.object),
        'template_sum_probs': np.array([0], dtype=np.float32)
        }
    return empty_template_features
    

def make_template_features(input_seq, input_pdb,use_ala_template):
    num_res  =len(input_seq)
    empty_template = make_empty_features(num_res)
    if input_pdb is None:
        print(f'Use empty template features')
        return empty_template

    template_all_atom_positions = []
    template_all_atom_masks = []
    template_sequence = []
    template_aatype = []
    template_domain_names = []
    template_sum_probs = []


    for pdb_path in input_pdb:
        if not os.path.isfile(pdb_path):
            print(f'Cant find pdb file: {pdb_path}. Use empty template features')
            template_aatype.append(np.zeros((num_res, len(residue_constants.restypes_with_x_and_gap)),np.float32))
            template_all_atom_masks.append(np.zeros((num_res, residue_constants.atom_type_num), np.float32))
            template_all_atom_positions.append(np.zeros((num_res, residue_constants.atom_type_num, 3), np.float32))
            template_domain_names.append(''.encode())
            template_sequence.append(np.array(''.encode(), dtype=np.object))
            template_sum_probs.append(0.0)
            continue

        print(f'Use pdb template: {pdb_path}')
        template_seq_list, template_atom_list = parse_pdb(pdb_path)
        template_seq = ''.join(template_seq_list)
        align_list = align(input_seq, template_seq)
        insert_count = 0
        for item in align_list:
            if item == -1:
                insert_count += 1
        
        if insert_count/len(input_seq) > 0.5:
            raise ValueError(f'The template is too short, the align ratio is {1-insert_count/len(input_seq)}')

        templates_all_atom_positions = []
        templates_all_atom_masks = []

        for _ in input_seq:
        # Residues in the query_sequence that are not in the template_sequence:
            templates_all_atom_positions.append(
                np.zeros((residue_constants.atom_type_num, 3)))
            templates_all_atom_masks.append(np.zeros(residue_constants.atom_type_num))

        atom_list = []
        template_align_seq = ''
        for index in align_list:
            if index == -1:
                atom_list.append({})
                template_align_seq = template_align_seq+'-'
            else:
                atom_list.append(template_atom_list[index])
                template_align_seq = template_align_seq + template_seq_list[index]

        if use_ala_template:
            list2atom37_ala(atom_list, templates_all_atom_positions, templates_all_atom_masks)
            output_templates_sequence = ''
            for item in template_align_seq:
                if item == '-':
                    output_templates_sequence = output_templates_sequence + '-'
                else:
                    output_templates_sequence = output_templates_sequence + 'A'

            
        else:
            list2atom37(atom_list, templates_all_atom_positions, templates_all_atom_masks)
            output_templates_sequence = template_align_seq

        templates_aatype = residue_constants.sequence_to_onehot(output_templates_sequence, residue_constants.HHBLITS_AA_TO_ID)
        template_aatype.append(templates_aatype)
        template_all_atom_masks.append(templates_all_atom_masks)
        template_all_atom_positions.append(templates_all_atom_positions)
        template_domain_names.append(f'custom template_{pdb_path}'.encode())
        template_sequence.append(np.array(output_templates_sequence.encode(), dtype=np.object))
        template_sum_probs.append(0.0)

    return {
            'template_all_atom_positions': np.array(template_all_atom_positions, np.float32),
            'template_all_atom_masks': np.array(template_all_atom_masks, np.float32),
            'template_sequence': np.array(template_sequence,dtype=np.object),
            'template_aatype': np.array(template_aatype, np.float32),
            'template_domain_names': np.array(template_domain_names, dtype=np.object),
            'template_sum_probs':np.array(template_sum_probs, dtype=np.float32)
        }

if __name__ == '__main__':
    t = make_template_features('GMTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEK',['test_pdb.pdb'])