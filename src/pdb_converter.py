import numpy as np
from sklearn.manifold import MDS
from scipy.ndimage import gaussian_filter1d
from Bio.PDB import PDBIO, Atom, Residue, Chain, Model, Structure
from Bio.SeqUtils import seq3

# === Parameters for idealized bond geometry (in Ångströms) ===
BOND_LENGTH_N_CA = 1.46
BOND_LENGTH_CA_C = 1.52
BOND_LENGTH_C_O = 1.23
BACKBONE_SPACING = 3.8  # average CA-CA distance

# === Generate backbone atoms around CA with approximate geometry ===
def generate_backbone_atoms(ca_coord):
    """
    Place N, CA, C, O atoms near CA based on idealized bond lengths.
    All atoms placed linearly for simplicity but more spread than original.
    """
    x, y, z = ca_coord
    return {
        "N":  np.array([x - BOND_LENGTH_N_CA, y, z]),
        "CA": np.array([x, y, z]),
        "C":  np.array([x + BOND_LENGTH_CA_C, y, z]),
        "O":  np.array([x + BOND_LENGTH_CA_C + BOND_LENGTH_C_O, y, z]),
    }

# === Reconstruct 3D coordinates from distance map using MDS ===
def distmap_to_coords(distmap, random_state=42, clip_value=20.0, smooth_sigma=1.0):
    """
    Applies distance clipping and MDS to reconstruct 3D coords from predicted map.
    Optionally smooths the resulting coordinates.
    """
    distmap = np.clip(distmap, 0, clip_value)
    distmap = 0.5 * (distmap + distmap.T)
    np.fill_diagonal(distmap, 0)

    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=random_state)
    coords = mds.fit_transform(distmap)

    # Smooth noisy coordinate jumps
    coords = gaussian_filter1d(coords, sigma=smooth_sigma, axis=0)

    return coords

# === Write atoms into a .pdb file from coords and sequence ===
def coords_to_pdb(coords, sequence, pdb_path="predicted_structure.pdb"):
    structure = Structure.Structure("X")
    model = Model.Model(0)
    chain = Chain.Chain("A")

    for i, coord in enumerate(coords):
        aa = sequence[i].upper()
        try:
            resname = seq3(aa)
        except KeyError:
            resname = "GLY"  # fallback for unknown residues

        res_id = (" ", i + 1, " ")
        residue = Residue.Residue(res_id, resname, " ")

        atoms = generate_backbone_atoms(coord)
        for atom_name, atom_coord in atoms.items():
            atom = Atom.Atom(atom_name, atom_coord, 1.0, 1.0, " ", f" {atom_name:<2}", i + 1, "C")
            residue.add(atom)

        chain.add(residue)

    model.add(chain)
    structure.add(model)

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)
