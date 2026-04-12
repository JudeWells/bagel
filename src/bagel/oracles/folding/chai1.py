"""
Chai-1 folding oracle.

Runs Chai-1 prediction locally via the chai_lab Python API.
Returns results compatible with BoltzResult fields.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Type

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, get_structure

from ...chain import Chain
from .base import FoldingOracle, FoldingResult

logger = logging.getLogger(__name__)


class Chai1Result(FoldingResult):
    """Result from Chai-1 prediction."""

    input_chains: list[Chain]
    structure: AtomArray
    local_plddt: np.ndarray  # [1, n_residues]
    pae: np.ndarray  # [1, n_residues, n_residues]
    ptm: np.ndarray  # [1,]
    chain_pair_iptm: np.ndarray  # [n_chains, n_chains]

    class Config:
        arbitrary_types_allowed = True

    def save_attributes(self, filepath: Path) -> None:
        np.savetxt(filepath.with_suffix('.plddt'), self.local_plddt[0], fmt='%.6f', header='plddt')
        np.savetxt(filepath.with_suffix('.pae'), self.pae[0], fmt='%.6f', header='pae')
        np.savetxt(filepath.with_suffix('.iptm'), self.chain_pair_iptm, fmt='%.6f', header='chain_pair_iptm')


class Chai1(FoldingOracle):
    """
    Chai-1 structure prediction via the chai_lab Python API.

    Parameters
    ----------
    num_trunk_recycles : int
        Number of recycling iterations (default 3).
    num_diffn_timesteps : int
        Number of diffusion timesteps (default 200).
    num_diffn_samples : int
        Number of diffusion samples to generate. Best is returned.
    seed : int or None
        Random seed for reproducibility. None = random.
    use_esm_embeddings : bool
        Whether to use ESM embeddings (default True).
    use_msa_server : bool
        Whether to query MSA server (default False for speed).
    device : str or None
        CUDA device. None = auto-detect.
    """

    result_class: Type[Chai1Result] = Chai1Result

    def __init__(
        self,
        num_trunk_recycles: int = 3,
        num_diffn_timesteps: int = 200,
        num_diffn_samples: int = 5,
        seed: int | None = None,
        use_esm_embeddings: bool = True,
        use_msa_server: bool = False,
        device: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.num_trunk_recycles = num_trunk_recycles
        self.num_diffn_timesteps = num_diffn_timesteps
        self.num_diffn_samples = num_diffn_samples
        self.seed = seed
        self.use_esm_embeddings = use_esm_embeddings
        self.use_msa_server = use_msa_server
        self.device = device
        self.config = config or {}

    def _write_fasta(self, chains: list[Chain], fasta_path: Path) -> None:
        """Write chains as a multi-sequence FASTA for Chai-1 input."""
        with open(fasta_path, "w") as f:
            for chain in chains:
                f.write(f">protein|name={chain.chain_ID}\n")
                f.write(f"{chain.sequence}\n")

    def fold(self, chains: list[Chain]) -> Chai1Result:
        """Fold chains using Chai-1."""
        import torch
        from chai_lab.chai1 import run_inference

        with tempfile.TemporaryDirectory(prefix="chai1_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            fasta_path = tmpdir_path / "input.fasta"
            output_dir = tmpdir_path / "output"
            output_dir.mkdir()

            self._write_fasta(chains, fasta_path)

            logger.info(f"Running Chai-1: {len(chains)} chains, "
                        f"samples={self.num_diffn_samples}, "
                        f"seed={self.seed}")

            candidates = run_inference(
                fasta_file=fasta_path,
                output_dir=output_dir,
                num_trunk_recycles=self.num_trunk_recycles,
                num_diffn_timesteps=self.num_diffn_timesteps,
                num_diffn_samples=self.num_diffn_samples,
                seed=self.seed,
                use_esm_embeddings=self.use_esm_embeddings,
                use_msa_server=self.use_msa_server,
                device=self.device,
            )

            # Sort by aggregate score and take the best
            sorted_candidates = candidates.sorted()
            best_idx = 0  # sorted() puts best first

            # Load structure from CIF
            best_cif = sorted_candidates.cif_paths[best_idx]
            cif = CIFFile.read(str(best_cif))
            atoms = get_structure(cif, model=1)

            # Extract metrics from the best candidate
            plddt = sorted_candidates.plddt[best_idx].cpu().numpy()  # (n_tokens,)
            pae = sorted_candidates.pae[best_idx].cpu().numpy()  # (n_tokens, n_tokens)

            # Extract PTM scores from ranking data
            ranking = sorted_candidates.ranking_data[best_idx]
            ptm_scores = ranking.ptm_scores
            ptm_val = float(ptm_scores.complex_ptm.cpu())
            iptm_val = float(ptm_scores.interface_ptm.cpu())

            # Build chain_pair_iptm matrix
            n_chains = len(chains)
            pair_iptm = ptm_scores.per_chain_pair_iptm.cpu().numpy()
            if pair_iptm.ndim > 2:
                pair_iptm = pair_iptm.squeeze()
            # Ensure correct shape
            if pair_iptm.shape == (n_chains, n_chains):
                chain_pair_iptm = pair_iptm.astype(np.float64)
            else:
                # Fallback: fill with interface_ptm for off-diagonal
                chain_pair_iptm = np.zeros((n_chains, n_chains), dtype=np.float64)
                for i in range(n_chains):
                    for j in range(n_chains):
                        if i != j:
                            chain_pair_iptm[i, j] = iptm_val

            # Reshape to BAGEL conventions
            plddt = plddt[np.newaxis, :].astype(np.float64)  # [1, n_residues]
            pae = pae[np.newaxis, :, :].astype(np.float64)  # [1, n_residues, n_residues]
            ptm = np.array([ptm_val], dtype=np.float64)

            from .utils import reindex_chains
            atoms = reindex_chains([atoms], [c.chain_ID for c in chains])

            return Chai1Result(
                input_chains=chains,
                structure=atoms,
                local_plddt=plddt,
                pae=pae,
                ptm=ptm,
                chain_pair_iptm=chain_pair_iptm,
            )
