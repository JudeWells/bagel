"""
AlphaFold2 folding oracle using the BindCraft/ColabDesign pipeline.

Runs AF2 prediction via ``conda run -n BindCraft`` subprocess to avoid
JAX/PyTorch dependency conflicts. Requires the BindCraft conda env to be
set up with ColabDesign + AF2 params.

Returns a result compatible with BoltzResult fields so all energy terms work.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Type

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

from ...chain import Chain
from .base import FoldingOracle, FoldingResult

logger = logging.getLogger(__name__)


class AF2BindCraftResult(FoldingResult):
    """Result from AF2 BindCraft prediction."""

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


class AF2BindCraft(FoldingOracle):
    """
    AF2 structure prediction using BindCraft/ColabDesign.

    Runs in the BindCraft conda environment via subprocess.
    Requires a target PDB file for binder design protocol.

    Parameters
    ----------
    target_pdb : str
        Path to the target PDB file.
    target_chain : str
        Chain ID in the target PDB to use (default "A").
    conda_env : str
        Name of the conda environment with ColabDesign installed.
    af_params_dir : str
        Path to AF2 parameter files.
    num_recycles : int
        Number of AF2 recycling iterations.
    prediction_models : list[int]
        AF2 model numbers to average over (0-indexed).
    """

    result_class: Type[AF2BindCraftResult] = AF2BindCraftResult

    def __init__(
        self,
        target_pdb: str = "",
        target_chain: str = "A",
        conda_env: str = "BindCraft",
        af_params_dir: str = "/mnt/disk2/BindCraft/params",
        num_recycles: int = 3,
        prediction_models: list[int] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.target_pdb = target_pdb
        self.target_chain = target_chain
        self.conda_env = conda_env
        self.af_params_dir = af_params_dir
        self.num_recycles = num_recycles
        self.prediction_models = prediction_models or [0, 1]
        self.config = config or {}

    def fold(self, chains: list[Chain]) -> AF2BindCraftResult:
        """Fold chains using AF2 via BindCraft subprocess."""

        # Extract binder sequence (first chain) and target info
        binder_chain = chains[0]
        binder_seq = binder_chain.sequence
        binder_len = binder_chain.length

        with tempfile.TemporaryDirectory(prefix="af2_bc_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_json = tmpdir_path / "result.json"
            output_pdb = tmpdir_path / "prediction.pdb"
            output_npz = tmpdir_path / "arrays.npz"

            # Write a helper script that runs in the BindCraft env
            script = tmpdir_path / "predict.py"
            script.write_text(textwrap.dedent(f"""\
                import json, sys, warnings, os
                import numpy as np
                warnings.filterwarnings("ignore")
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

                from colabdesign.af import mk_afdesign_model
                from colabdesign.shared.utils import clear_mem

                target_pdb = "{self.target_pdb}"
                target_chain = "{self.target_chain}"
                binder_seq = "{binder_seq}"
                binder_len = {binder_len}
                af_params_dir = "{self.af_params_dir}"
                num_recycles = {self.num_recycles}
                models = {self.prediction_models}
                output_json = "{output_json}"
                output_pdb = "{output_pdb}"
                output_npz = "{output_npz}"

                clear_mem()
                model = mk_afdesign_model(
                    protocol="binder",
                    num_recycles=num_recycles,
                    data_dir=af_params_dir,
                    use_multimer=False,
                )
                model.prep_inputs(
                    pdb_filename=target_pdb,
                    chain=target_chain,
                    binder_len=binder_len,
                    rm_target_seq=False,
                    rm_target_sc=False,
                )
                target_len = model._target_len

                per_model = []
                all_plddt = []
                all_pae = []
                for model_num in models:
                    model.predict(seq=binder_seq, models=[model_num],
                                  num_recycles=num_recycles, verbose=False)
                    aux = model.aux
                    log = aux["log"]
                    plddt = np.asarray(aux["plddt"])
                    pae = np.asarray(aux["pae"])
                    all_plddt.append(plddt)
                    all_pae.append(pae)
                    per_model.append({{
                        "ptm": float(log.get("ptm", float("nan"))),
                        "iptm": float(log.get("i_ptm", float("nan"))),
                    }})

                # Save best model PDB (last one)
                model.save_pdb(output_pdb)

                # Average metrics across models
                avg_ptm = np.mean([m["ptm"] for m in per_model])
                avg_iptm = np.mean([m["iptm"] for m in per_model])
                avg_plddt = np.mean(all_plddt, axis=0)
                avg_pae = np.mean(all_pae, axis=0)

                np.savez_compressed(output_npz,
                    plddt=avg_plddt, pae=avg_pae,
                    target_len=target_len, binder_len=binder_len)

                result = {{
                    "ptm": float(avg_ptm),
                    "iptm": float(avg_iptm),
                    "target_len": int(target_len),
                    "binder_len": int(binder_len),
                    "per_model": per_model,
                }}
                with open(output_json, "w") as f:
                    json.dump(result, f)
            """))

            # Run in BindCraft conda env
            cmd = ["conda", "run", "-n", self.conda_env, "python", str(script)]
            logger.info(f"Running AF2 BindCraft: {' '.join(cmd)}")

            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=600,
            )
            if proc.returncode != 0:
                logger.error(f"AF2 BindCraft stderr: {proc.stderr[-500:]}")
                raise RuntimeError(
                    f"AF2 BindCraft prediction failed (exit {proc.returncode}): "
                    f"{proc.stderr[-200:]}"
                )

            # Parse results
            with open(output_json) as f:
                result_data = json.load(f)

            arrays = np.load(output_npz)
            plddt_raw = arrays["plddt"]  # (total_residues,) — target first, binder second
            pae_raw = arrays["pae"]  # (total_residues, total_residues)
            target_len = int(result_data["target_len"])
            total_len = target_len + binder_len

            # AF2 outputs [target, binder] order. Reorder to [binder, target]
            # to match BAGEL convention where GEN chain comes first.
            reorder = list(range(target_len, total_len)) + list(range(target_len))
            plddt = plddt_raw[reorder]
            pae = pae_raw[np.ix_(reorder, reorder)]

            # Load structure from PDB and reorder chains
            pdb_file = PDBFile.read(str(output_pdb))
            atoms_raw = pdb_file.get_structure(model=1)

            # Split atoms by chain and reorder: binder (B) first, target (A) second
            import pandas as pd
            chain_ids_in_pdb = pd.unique(atoms_raw.chain_id)
            binder_mask = atoms_raw.chain_id == chain_ids_in_pdb[-1]  # binder is last chain
            target_mask = ~binder_mask
            from biotite.structure import AtomArray
            # Concatenate binder atoms first, then target atoms
            atoms = atoms_raw[binder_mask] + atoms_raw[target_mask]

            # Assign chain IDs from input chains list
            n_chains = len(chains)
            binder_chain_id = chains[0].chain_ID if n_chains > 0 else "GEN"
            target_chain_id = chains[1].chain_ID if n_chains > 1 else "B"
            for i in range(len(atoms)):
                if i < sum(binder_mask):
                    atoms.chain_id[i] = binder_chain_id
                else:
                    atoms.chain_id[i] = target_chain_id

            # Build chain_pair_iptm matrix
            chain_pair_iptm = np.zeros((n_chains, n_chains), dtype=np.float64)
            iptm_val = float(result_data["iptm"])
            for i in range(n_chains):
                for j in range(n_chains):
                    if i != j:
                        chain_pair_iptm[i, j] = iptm_val

            # Reshape arrays to match BAGEL conventions
            plddt = plddt[np.newaxis, :]  # [1, n_residues]
            pae = pae[np.newaxis, :, :]  # [1, n_residues, n_residues]
            ptm = np.array([float(result_data["ptm"])], dtype=np.float64)

            return AF2BindCraftResult(
                input_chains=chains,
                structure=atoms,
                local_plddt=plddt,
                pae=pae,
                ptm=ptm,
                chain_pair_iptm=chain_pair_iptm,
            )
