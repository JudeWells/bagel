"""
Standalone SolubleMPNN sequence scoring script.

Runs in an isolated conda env (e.g. ``proteinmpnn``) via subprocess from
``SolMPNNPerplexityEnergy``.  Reads a PDB file and emits a JSON result.

The script is designed to be called with a **full complex** PDB when used
for binder design: the MPNN encoder sees all chains in the structure as
binding context, but only residues on ``--chains_to_score`` contribute to
the autoregressive perplexity loss.  This is critical — monomer-only
scoring penalises legitimate interface sequences; see
``test_mpnn_context_significance.py`` for validation on a real heterodimer
(1YCR / p53-MDM2), where the p53 peptide's perplexity drops from 16.57 in
isolation to 4.47 in MDM2 context (Welch t ≈ -188, n=10 repeats).

CLI args:
    --pdb PATH                  Input PDB file (typically the full complex).
    --chains_to_score STR       Comma-separated chain IDs the perplexity is
                                computed on (e.g. "A" or "A,C").  Other
                                chains are visible to the encoder only.
    --proteinmpnn_path DIR      Path to cloned ProteinMPNN repo.
    --checkpoint PATH           Path to model .pt weights.  Defaults to
                                soluble_model_weights/v_48_020.pt.
    --backbone_noise FLOAT      ``augment_eps`` — std of Gaussian noise
                                applied to backbone coordinates each
                                forward pass.  0.0 for deterministic
                                featurisation, > 0 to get an ensemble of
                                independently-noised backbones.
    --ensemble_n INT            Number of forward passes.  Each pass uses
                                an independent backbone-noise draw AND an
                                independent decoding order.
    --decoding_order STR        ``"random"`` (default; fresh randn per pass)
                                or ``"fixed:<seed>"`` for deterministic
                                ordering.
    --output_json PATH          Where to write the result JSON.

Output JSON:
    {
        "perplexity": float,           # exp(mean NLL) over scored residues
        "mean_nll": float,
        "std_nll": float,              # std across ensemble passes
        "global_perplexity": float,    # perplexity over all residues (reference)
        "per_pass_nll": [float, ...],  # NLL from each pass
        "backbone_noise": float,       # echoed hyperparameters
        "ensemble_n": int,
        "decoding_order": str,
    }
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb", type=str, required=True)
    p.add_argument("--chains_to_score", type=str, default="A")
    p.add_argument("--proteinmpnn_path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to model .pt; defaults to soluble_model_weights/v_48_020.pt")
    p.add_argument("--backbone_noise", type=float, default=0.0,
                   help="Gaussian noise stddev applied to backbone coords (augment_eps).")
    p.add_argument("--ensemble_n", type=int, default=10,
                   help="Number of forward passes (each with independent noise+order).")
    p.add_argument("--decoding_order", type=str, default="random",
                   help="'random' or 'fixed:<seed>'.")
    p.add_argument("--output_json", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    sys.path.insert(0, args.proteinmpnn_path)
    from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize, _scores

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load SolubleMPNN model ---
    ckpt_path = args.checkpoint or os.path.join(
        args.proteinmpnn_path, "soluble_model_weights", "v_48_020.pt"
    )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    hidden_dim = 128
    num_layers = 3
    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=args.backbone_noise,  # key: controls per-pass backbone noise
        k_neighbors=checkpoint["num_edges"],
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Parse PDB ---
    pdb_dict_list = parse_PDB(args.pdb, ca_only=False)
    if not pdb_dict_list:
        raise ValueError(f"parse_PDB returned empty list for {args.pdb}")
    pdb_dict = pdb_dict_list[0]

    designed = [c.strip() for c in args.chains_to_score.split(",") if c.strip()]
    all_chains = sorted(
        k.split("_")[-1] for k in pdb_dict if k.startswith("seq_chain_")
    )
    fixed = [c for c in all_chains if c not in designed]
    chain_id_dict = {pdb_dict["name"]: (designed, fixed)}

    # --- Decoding order generator ---
    if args.decoding_order.startswith("fixed"):
        # "fixed:<seed>" — deterministic
        seed_part = args.decoding_order.split(":", 1)
        base_seed = int(seed_part[1]) if len(seed_part) > 1 else 0
        def get_randn(shape, i):
            g = torch.Generator(device=device)
            g.manual_seed(base_seed + i)
            return torch.randn(shape, generator=g, device=device)
    else:
        def get_randn(shape, i):
            return torch.randn(shape, device=device)

    # --- Score over N independent passes ---
    # Each pass re-featurises (with fresh backbone noise if augment_eps > 0)
    # and uses an independent decoding order.
    pass_scores = []
    pass_global_scores = []

    with torch.no_grad():
        for i in range(args.ensemble_n):
            batch = [copy.deepcopy(pdb_dict)]
            (
                X, S, mask, lengths, chain_M, chain_encoding_all,
                chain_list_list, visible_list_list, masked_list_list,
                masked_chain_length_list_list, chain_M_pos, omit_AA_mask,
                residue_idx, dihedral_mask, tied_pos_list_of_lists_list,
                pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all,
                tied_beta,
            ) = tied_featurize(batch, device, chain_id_dict, ca_only=False)

            randn = get_randn(chain_M.shape, i)
            log_probs = model(
                X, S, mask,
                chain_M * chain_M_pos,
                residue_idx, chain_encoding_all,
                randn,
            )
            mask_for_loss = mask * chain_M * chain_M_pos
            scores = _scores(S, log_probs, mask_for_loss)
            global_scores = _scores(S, log_probs, mask)
            pass_scores.append(float(scores.mean().cpu()))
            pass_global_scores.append(float(global_scores.mean().cpu()))

    mean_nll = float(np.mean(pass_scores))
    std_nll = float(np.std(pass_scores))
    mean_global = float(np.mean(pass_global_scores))

    result = {
        "perplexity": float(np.exp(mean_nll)),
        "mean_nll": mean_nll,
        "std_nll": std_nll,
        "global_perplexity": float(np.exp(mean_global)),
        "per_pass_nll": pass_scores,
        "backbone_noise": args.backbone_noise,
        "ensemble_n": args.ensemble_n,
        "decoding_order": args.decoding_order,
    }

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
