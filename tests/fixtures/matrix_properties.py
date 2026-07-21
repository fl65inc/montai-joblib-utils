"""Matrix property names for usage tests.

Source: platform_cdk/stacks/matrix_cdk_stack.py job_definition_name values (38)
plus related Step Functions / property names to reach 50.
"""

from __future__ import annotations

# All Batch job definitions declared in matrix_cdk_stack.py
MATRIX_JOB_DEFINITIONS: list[str] = [
    "acn_predictions",
    "balaban_j",
    "bertz_ct",
    "camptothecin_substructure",
    "chloroformate_functional_group",
    "contains_acid_chloride_group",
    "contains_extra_atoms",
    "contains_glycoside_substructure",
    "count_unique_atoms",
    "covalent_warhead",
    "etoposide_substructure",
    "fcfp2_fp",
    "highly_halogenated",
    "kmeans_diversity_labels",
    "largest_ring_size",
    "lipinski_heavy_atom_count",
    "lipinski_num_h_acceptors",
    "lipinski_num_h_donors",
    "log_p",
    "longest_non_carbon_chain",
    "maccs_fp",
    "molecular_descriptors",
    "montility_v1_processed_smiles",
    "morgan1_fp",
    "morgan2_fp",
    "morgan_fingerprints_2019_09_02",
    "murcko_scaffolds",
    "np_classifier_predict",
    "np_likeness_score",
    "oxygen_sulfur_nitrogen",
    "pmi",
    "proportion_halogenated",
    "qed",
    "rdkit_fp",
    "reaction_sites_count",
    "sulfonyl_chloride_functional_group",
    "tanimoto_to_anthromolecule",
    "undefined_stereochemistry",
]

# Extra matrix-related state machines / properties (pad to 50)
MATRIX_EXTRA_PROPERTIES: list[str] = [
    "CarbonCount",
    "ChiralCenters",
    "CompoundProperties",
    "ExactMolecularWeight",
    "SA_Score",
    "TransitionMetals",
    "rd_filters_PAINS",
    "rd_filters_Glaxo",
    "rd_filters_BMS",
    "rd_filters_SureChEMBL",
    "substructure_flags",
    "identify_oxygen_sulfur_nitrogen",
]

MATRIX_PROPERTIES_50: list[str] = (MATRIX_JOB_DEFINITIONS + MATRIX_EXTRA_PROPERTIES)[:50]

assert len(MATRIX_PROPERTIES_50) == 50, len(MATRIX_PROPERTIES_50)
