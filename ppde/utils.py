import torch
from collections import defaultdict


def mut_distance(x, wt):
    """
    edit distance from wt
    
    x is [n_chains,seq_len,vocab_size]
    wt is [1,seq_len,vocab_size]
    """
    wt = wt.repeat(x.size(0),1,1)
    edits = ((x != wt).float().sum(-1) > 0).float().sum(-1)
    return edits


def mutation_mask(x, wt):
    """
    Allow mutations wherever mask is False
    
    For every pos where x and wt differ, the mask is set to False.
    Everywhere else set to True.
    """
    mask = torch.ones_like(x).to(x.device)
    wt = wt.repeat(x.size(0),1,1)
    positions = (x != wt) & (wt == 1)
    mask[positions] = 0
    return mask.bool()


def load_MSA(filename):
    """
    Loads the MSA in filename and returns it
    as a one-hot numpy array of 
        shape [num_sequences,num_alignment_columns,alphabet]

    Code stripped out from https://github.com/debbiemarkslab/DeepSequence/blob/master/DeepSequence/helper.py
    """
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    aa_dict = {}
    for i,aa in enumerate(alphabet):
        aa_dict[aa] = i
    # Read alignment
    seq_name_to_sequence = defaultdict(str)
    seq_names = []

    name = ""
    INPUT = open(filename, "r")
    for i, line in enumerate(INPUT):
        line = line.rstrip()
        if line.startswith(">"):
            name = line
            seq_names.append(name)
        else:
            seq_name_to_sequence[name] += line
    INPUT.close()

    # If we don"t have a focus sequence, pick the one that
    #   we used to generate the alignment
    focus_seq_name = seq_names[0]

    # Select focus columns
    #  These columns are the uppercase residues of the .a2m file
    focus_seq = seq_name_to_sequence[focus_seq_name]
    focus_cols = [ix for ix, s in enumerate(focus_seq) if s == s.upper()]
    
    # Get only the focus columns
    for seq_name,sequence in seq_name_to_sequence.items():
        # Replace periods with dashes (the uppercase equivalent)
        sequence = sequence.replace(".","-")

        #then get only the focus columns
        seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in focus_cols]

    # Remove sequences that have bad characters
    alphabet_set = set(list(alphabet))
    seq_names_to_remove = []
    for seq_name,sequence in seq_name_to_sequence.items():
        for letter in sequence:
            if letter not in alphabet_set and letter != "-":
                seq_names_to_remove.append(seq_name)

    seq_names_to_remove = list(set(seq_names_to_remove))
    for seq_name in seq_names_to_remove:
        del seq_name_to_sequence[seq_name]

    # Encode the sequences
    #msa = np.zeros((len(seq_name_to_sequence.keys()),len(focus_cols),len(alphabet)))
    #import pdb; pdb.set_trace()
    # x_train_name_list = []
    # for i,seq_name in enumerate(seq_name_to_sequence.keys()):
    #     sequence = seq_name_to_sequence[seq_name]
    #     x_train_name_list.append(seq_name)
    #     for j,letter in enumerate(sequence):
    #         if letter in aa_dict:
    #             k = aa_dict[letter]
    #             msa[i,j,k] = 1.0

    #return msa 
    msa = []
    for i,seq_name in enumerate(seq_name_to_sequence.keys()):
        sequence = seq_name_to_sequence[seq_name]
        msa += [(seq_name, ''.join([letter for letter in sequence]))]
    return msa

# TODO: Delete
# if __name__ == '__main__':

#     msa = load_MSA('/gpfs/alpine/bie108/proj-shared/pppo/alignments/PABP_YEAST.a2m')
#     print(len(msa[0][1]))
#     print(len(msa[10][1]))
