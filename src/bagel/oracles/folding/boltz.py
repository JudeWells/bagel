"""
Boltz folding oracle — Boltz-1/Boltz-2 structure prediction via local CLI.

Boltz (https://github.com/jwohlwend/boltz) is a fast open-source protein
structure predictor.  This oracle invokes ``boltz predict`` as a subprocess
and parses the output CIF structures, NPZ confidence data (PAE + pLDDT),
and chain-pair iPTM scores.

The implementation follows the nipah_ipsae_pipeline reference
(https://github.com/adaptyvbio/nipah_ipsae_pipeline) for input/output
handling and token-level PAE masking.

MIT License
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Type

import numpy as np
import numpy.typing as npt
from pydantic import field_validator

from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, get_structure

from ...chain import Chain
from .base import FoldingOracle, FoldingResult

logger = logging.getLogger(__name__)

# Standard amino acid and nucleic acid 3-letter codes that get one token each
# in Boltz tokenisation.  Non-standard residues get one token per heavy atom.
_STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "DA", "DC", "DT", "DG", "A", "C", "U", "G",
}


# ---------------------------------------------------------------------------
# Result class
# ---------------------------------------------------------------------------

class BoltzResult(FoldingResult):
    """FoldingResult with Boltz confidence metrics.

    Fields mirror ESMFoldResult (``local_plddt``, ``pae``, ``ptm``) so that
    existing energy terms (ipSAEEnergy, LISEnergy, PAEEnergy, PTMEnergy, etc.)
    work out of the box.  ``chain_pair_iptm`` enables iPTMEnergy scoring.
    """

    input_chains: list[Chain]
    structure: AtomArray                      # from CIF output
    local_plddt: npt.NDArray[np.float64]      # [1, n_residues] — 0-1 scale
    pae: npt.NDArray[np.float64]              # [1, n_residues, n_residues]
    ptm: npt.NDArray[np.float64]              # [1] — overall pTM (0-1)
    chain_pair_iptm: npt.NDArray[np.float64]  # [n_chains, n_chains]

    class Config:
        arbitrary_types_allowed = True

    @field_validator('local_plddt')
    def validate_local_plddt(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if not isinstance(v, np.ndarray):
            raise ValueError('local_plddt must be a numpy array')
        if not np.all((v >= 0) & (v <= 1)):
            raise ValueError('All values in local_plddt must be between 0 and 1')
        return v

    @field_validator('ptm')
    def validate_ptm(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if not isinstance(v, np.ndarray):
            raise ValueError('ptm must be a numpy array')
        if not np.all((v >= 0) & (v <= 1)):
            raise ValueError('All values in ptm must be between 0 and 1')
        return v

    def save_attributes(self, filepath: Path) -> None:
        np.savetxt(filepath.with_suffix('.plddt'), self.local_plddt[0], fmt='%.6f', header='plddt')
        np.savetxt(filepath.with_suffix('.pae'), self.pae[0], fmt='%.6f', header='pae')
        np.savetxt(filepath.with_suffix('.iptm'), self.chain_pair_iptm, fmt='%.6f', header='chain_pair_iptm')


# ---------------------------------------------------------------------------
# Oracle class
# ---------------------------------------------------------------------------

class Boltz(FoldingOracle):
    """
    Boltz structure prediction via local CLI subprocess.

    Invokes ``boltz predict`` on YAML input and parses the resulting CIF
    structures, NPZ PAE/pLDDT files, and confidence JSON.

    Parameters
    ----------
    model_seeds : list[int]
        Seeds to use for prediction. Default: ``[1]``.
    recycling_steps : int or None
        Number of recycling steps.  Passed as ``--recycling_steps`` if set.
    sampling_steps : int or None
        Number of diffusion sampling steps.  Passed as ``--sampling_steps``.
    boltz_command : str
        CLI command to invoke boltz.  Default: ``"boltz"``.
    extra_args : list[str]
        Additional CLI arguments passed verbatim to ``boltz predict``.
    msa_directory : str or None
        Path to directory containing pre-computed MSA files (.a3m).
        If None, uses ``msa: "empty"`` for all chains.
    """

    result_class: Type[BoltzResult] = BoltzResult

    def __init__(
        self,
        model_seeds: list[int] | None = None,
        recycling_steps: int | None = None,
        sampling_steps: int | None = None,
        boltz_command: str = "boltz",
        extra_args: list[str] | None = None,
        msa_directory: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.model_seeds = model_seeds or [1]
        self.recycling_steps = recycling_steps
        self.sampling_steps = sampling_steps
        self.boltz_command = boltz_command
        self.extra_args = extra_args or []
        self.msa_directory = msa_directory
        self.config = config or {}

    # --------------------------------------------------------------------- #
    # Boltz YAML input
    # --------------------------------------------------------------------- #

    def _chains_to_boltz_yaml(self, chains: list[Chain], yaml_path: Path) -> None:
        """
        Write a Boltz input YAML from BAGEL Chain objects.

        Format (matching nipah_ipsae_pipeline)::

            version: 1
            sequences:
            - protein:
                id: A
                sequence: MKLL...
                msa: empty
            - protein:
                id: B
                sequence: GFED...
                msa: empty
        """
        import yaml

        data: dict[str, Any] = {"version": 1, "sequences": []}
        for chain in chains:
            entry: dict[str, Any] = {
                "protein": {
                    "id": chain.chain_ID,
                    "sequence": chain.sequence,
                }
            }
            # Check for pre-computed MSA
            if self.msa_directory:
                msa_path = Path(self.msa_directory) / f"{chain.chain_ID}.a3m"
                if msa_path.exists():
                    entry["protein"]["msa"] = str(msa_path)
                else:
                    entry["protein"]["msa"] = "empty"
            else:
                entry["protein"]["msa"] = "empty"

            data["sequences"].append(entry)

        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    # --------------------------------------------------------------------- #
    # Token mask (CA-only filtering for Boltz PAE/pLDDT)
    # --------------------------------------------------------------------- #

    @staticmethod
    def _build_token_mask_from_cif(cif_path: Path) -> npt.NDArray[np.bool_]:
        """
        Build a boolean token mask from a Boltz CIF file.

        Boltz outputs PAE and pLDDT at the *token* level.  For standard
        amino acids each residue has one token (the CA atom).  Non-standard
        residues (ligands, PTMs) contribute one token per heavy atom.

        This mask is True for CA tokens and False for non-CA tokens, allowing
        us to extract residue-level PAE and pLDDT from the raw token-level
        arrays.

        Follows the reference implementation from DunbrackLab/IPSAE.
        """
        token_mask: list[int] = []
        atomsitefield_dict: dict[str, int] = {}
        atomsitefield_num = 0

        with open(cif_path) as f:
            for line in f:
                # Parse mmCIF header to get field positions
                if line.startswith("_atom_site."):
                    fieldname = line.strip().split(".")[1]
                    atomsitefield_dict[fieldname] = atomsitefield_num
                    atomsitefield_num += 1
                    continue

                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue

                fields = line.split()
                if len(fields) < max(atomsitefield_dict.values(), default=0) + 1:
                    continue

                # Extract atom name and residue name
                atom_name = fields[atomsitefield_dict.get("label_atom_id", 3)]
                res_name = fields[atomsitefield_dict.get("label_comp_id", 5)]

                # Standard residue atom handling
                if res_name in _STANDARD_RESIDUES:
                    if atom_name == "CA" or "C1" in atom_name:
                        token_mask.append(1)
                    # Non-CA atoms in standard residues are NOT tokens — skip
                else:
                    # Non-standard residue: each atom is a separate token
                    if atom_name == "CA" or "C1" in atom_name:
                        token_mask.append(1)
                    else:
                        token_mask.append(0)

        return np.array(token_mask, dtype=bool)

    # --------------------------------------------------------------------- #
    # Output parsing
    # --------------------------------------------------------------------- #

    def _find_output_files(self, out_dir: Path) -> dict[str, Path]:
        """
        Locate Boltz output files in the output directory.

        Boltz writes to: ``{out_dir}/boltz_results_{hash}/predictions/{hash}/``
        with files like ``{hash}_model_0.cif``, ``pae_{hash}_model_0.npz``, etc.
        """
        files: dict[str, Path] = {}

        # Find CIF file
        cif_files = list(out_dir.rglob("*_model_*.cif"))
        # Exclude pae/plddt/confidence prefixed files
        cif_files = [f for f in cif_files if not any(
            f.name.startswith(p) for p in ("pae_", "plddt_", "confidence_")
        )]
        if cif_files:
            files["cif"] = cif_files[0]

        # Find PAE NPZ
        pae_files = list(out_dir.rglob("pae_*_model_*.npz"))
        if pae_files:
            files["pae"] = pae_files[0]

        # Find pLDDT NPZ
        plddt_files = list(out_dir.rglob("plddt_*_model_*.npz"))
        if plddt_files:
            files["plddt"] = plddt_files[0]

        # Find confidence JSON
        conf_files = list(out_dir.rglob("confidence_*_model_*.json"))
        if conf_files:
            files["confidence"] = conf_files[0]

        return files

    def _parse_output(
        self,
        output_files: dict[str, Path],
        chains: list[Chain],
    ) -> BoltzResult:
        """Parse Boltz output files into a BoltzResult."""

        # --- CIF → structure (AtomArray) ---
        cif_path = output_files.get("cif")
        if cif_path is None:
            raise FileNotFoundError("No CIF file found in Boltz output")
        cif = CIFFile.read(str(cif_path))
        structure = get_structure(cif, model=1)

        # Reindex chain IDs to match input chain ordering
        import pandas as pd
        original_chain_ids = pd.unique(structure.chain_id)
        chain_id_map = {
            old: chain.chain_ID
            for old, chain in zip(original_chain_ids, chains)
        }
        new_chain_ids = structure.chain_id.copy()
        for i in range(len(new_chain_ids)):
            if new_chain_ids[i] in chain_id_map:
                new_chain_ids[i] = chain_id_map[new_chain_ids[i]]
        structure.chain_id = new_chain_ids

        # --- Token mask for CA-only extraction ---
        token_mask = self._build_token_mask_from_cif(cif_path)
        ca_mask = token_mask  # True = CA token

        # --- pLDDT (NPZ, key 'plddt', 0-1 scale) ---
        plddt_path = output_files.get("plddt")
        if plddt_path is not None:
            raw_plddt = np.load(str(plddt_path))["plddt"]
            # Apply token mask to get CA-only pLDDT
            if len(raw_plddt) == len(ca_mask):
                plddt = raw_plddt[ca_mask]
            else:
                # Boltz pLDDT may already be residue-level
                plddt = raw_plddt
            # Boltz pLDDT is 0-1 scale; keep as-is (BAGEL convention)
            plddt = np.array(plddt, dtype=np.float64)
        else:
            n_res = sum(c.length for c in chains)
            plddt = np.zeros(n_res, dtype=np.float64)
            logger.warning("No pLDDT NPZ found; using zeros")

        local_plddt = plddt[np.newaxis, :]  # [1, n_residues]

        # --- PAE (NPZ, key 'pae', token-level) ---
        pae_path = output_files.get("pae")
        if pae_path is not None:
            raw_pae = np.array(np.load(str(pae_path))["pae"], dtype=np.float64)
            # Apply token mask to get CA-only PAE
            if raw_pae.shape[0] == len(ca_mask):
                pae = raw_pae[np.ix_(ca_mask, ca_mask)]
            else:
                # Already residue-level
                pae = raw_pae
        else:
            n_res = sum(c.length for c in chains)
            pae = np.zeros((n_res, n_res), dtype=np.float64)
            logger.warning("No PAE NPZ found; using zeros")

        pae = pae[np.newaxis, :, :]  # [1, n_residues, n_residues]

        # --- Confidence JSON (pTM, chain_pair_iptm) ---
        conf_path = output_files.get("confidence")
        n_chains = len(chains)
        ptm_val = 0.0
        chain_pair_iptm = np.zeros((n_chains, n_chains), dtype=np.float64)

        if conf_path is not None:
            with open(conf_path) as f:
                conf = json.load(f)

            ptm_val = float(conf.get("ptm", 0.0))

            # Parse chain_pair_iptm — Boltz uses string keys: {"0": {"1": 0.8}, ...}
            pair_iptm_raw = conf.get("pair_chains_iptm", {})
            if pair_iptm_raw:
                for i in range(n_chains):
                    for j in range(n_chains):
                        if i == j:
                            continue
                        val = pair_iptm_raw.get(str(i), {}).get(str(j), 0.0)
                        chain_pair_iptm[i, j] = float(val)
        else:
            logger.warning("No confidence JSON found; pTM and iPTM will be 0")

        ptm = np.array([ptm_val], dtype=np.float64)

        return BoltzResult(
            input_chains=chains,
            structure=structure,
            local_plddt=local_plddt,
            pae=pae,
            ptm=ptm,
            chain_pair_iptm=chain_pair_iptm,
        )

    # --------------------------------------------------------------------- #
    # Main fold method
    # --------------------------------------------------------------------- #

    def fold(self, chains: list[Chain]) -> BoltzResult:
        """
        Fold chains using Boltz via local CLI.

        Creates a temporary YAML input, runs ``boltz predict``, parses the
        output, and returns a BoltzResult.
        """
        with tempfile.TemporaryDirectory(prefix="boltz_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write Boltz input YAML
            yaml_path = tmpdir_path / "input.yaml"
            self._chains_to_boltz_yaml(chains, yaml_path)

            # Build command
            out_dir = tmpdir_path / "output"
            out_dir.mkdir()
            cmd = [
                self.boltz_command, "predict",
                str(yaml_path),
                "--out_dir", str(out_dir),
                "--write_full_pae",
            ]
            if self.recycling_steps is not None:
                cmd.extend(["--recycling_steps", str(self.recycling_steps)])
            if self.sampling_steps is not None:
                cmd.extend(["--sampling_steps", str(self.sampling_steps)])
            cmd.extend(self.extra_args)

            logger.info(f"Running Boltz: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout
            )
            if result.returncode != 0:
                logger.error(f"Boltz stderr:\n{result.stderr}")
                raise RuntimeError(
                    f"Boltz predict failed (exit code {result.returncode}):\n"
                    f"{result.stderr[-2000:]}"
                )
            if result.stdout:
                logger.debug(f"Boltz stdout:\n{result.stdout[-1000:]}")

            # Find and parse output files
            output_files = self._find_output_files(out_dir)
            if "cif" not in output_files:
                raise FileNotFoundError(
                    f"No CIF output found in {out_dir}. "
                    f"Boltz may have failed silently. Check stderr above."
                )

            return self._parse_output(output_files, chains)
