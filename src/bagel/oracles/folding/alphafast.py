"""
AlphaFast folding oracle — GPU-accelerated AlphaFold3 via Modal serverless.

AlphaFast (https://github.com/RomeroLab/alphafast) wraps AlphaFold3 with
GPU-accelerated MMseqs2 MSA search, running ~23x faster end-to-end.  This
oracle sends prediction jobs to a deployed AlphaFast Modal app and parses
the returned CIF structures and confidence metrics.

MIT License
"""

import json
import logging
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


# ---------------------------------------------------------------------------
# Result class
# ---------------------------------------------------------------------------

class AlphaFastResult(FoldingResult):
    """FoldingResult with AlphaFold3 confidence metrics."""

    input_chains: list[Chain]
    structure: AtomArray              # parsed from CIF output
    local_plddt: npt.NDArray[np.float64]   # [1, n_residues] — normalised 0-1
    pae: npt.NDArray[np.float64]           # [1, n_residues, n_residues]
    ptm: npt.NDArray[np.float64]           # [1] — overall pTM
    chain_pair_iptm: npt.NDArray[np.float64]  # [n_chains, n_chains]
    ranking_score: float

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

class AlphaFast(FoldingOracle):
    """
    AlphaFold3 structure prediction via AlphaFast on Modal.

    AlphaFast is always called remotely (Modal serverless); there is no local
    execution path because it requires Docker + ~800 GB databases.

    Parameters
    ----------
    model_seeds : list[int]
        Model seeds to run.  When multiple seeds are provided the result
        with the best ``ranking_score`` is returned.
    modal_app_name : str
        Name of the deployed Modal app that exposes the AlphaFast predict
        function.  Default: ``"alphafast-predict"``.
    modal_function_name : str
        Name of the Modal function within the app.
        Default: ``"predict_structure"``.
    config : dict
        Additional config forwarded to the Modal function (e.g. GPU type,
        MSA server URL).
    """

    result_class: Type[AlphaFastResult] = AlphaFastResult

    def __init__(
        self,
        model_seeds: list[int] | None = None,
        modal_app_name: str = "alphafold3-inference",
        modal_function_name: str = "predict_structure",
        config: dict[str, Any] | None = None,
        skip_msa: bool = True,
    ) -> None:
        self.model_seeds = model_seeds or [1]
        self.modal_app_name = modal_app_name
        self.modal_function_name = modal_function_name
        self.skip_msa = skip_msa
        config = config or {}
        if self.skip_msa:
            config.setdefault("run_msa", False)
        self.config = config

    # --------------------------------------------------------------------- #
    # AF3 JSON input format
    # --------------------------------------------------------------------- #

    @staticmethod
    def _chains_to_af3_json(
        chains: list[Chain],
        model_seeds: list[int],
        skip_msa: bool = False,
    ) -> dict:
        """
        Convert BAGEL Chain objects to the AF3 JSON input format.

        When *skip_msa* is ``True``, each protein entry includes inline dummy
        ``unpairedMsa`` / ``pairedMsa`` fields (query-only) and empty
        ``templates`` so that the AF3 data pipeline can be skipped entirely.
        """
        sequences = []
        for chain in chains:
            protein: dict = {
                "id": [chain.chain_ID],
                "sequence": chain.sequence,
            }
            if skip_msa:
                dummy_msa = f">query\n{chain.sequence}\n"
                protein["unpairedMsa"] = dummy_msa
                protein["pairedMsa"] = dummy_msa
                protein["templates"] = []
            sequences.append({"protein": protein})
        return {
            "name": "bagel_prediction",
            "modelSeeds": model_seeds,
            "sequences": sequences,
            "dialect": "alphafold3",
            "version": 1,
        }

    # --------------------------------------------------------------------- #
    # Output parsing
    # --------------------------------------------------------------------- #

    @staticmethod
    def _parse_cif(cif_content: str) -> AtomArray:
        """Parse a CIF string into a biotite AtomArray."""
        with tempfile.NamedTemporaryFile(suffix=".cif", mode="w", delete=False) as f:
            f.write(cif_content)
            f.flush()
            cif = CIFFile.read(f.name)
        return get_structure(cif, model=1)

    @staticmethod
    def _per_residue_plddt(atom_plddt: npt.NDArray[np.float64], structure: AtomArray) -> npt.NDArray[np.float64]:
        """
        Reduce per-atom pLDDT to per-residue by taking the CA atom value
        (matching ESMFold convention).  Falls back to mean over atoms in each
        residue if CA is missing.
        """
        # Build unique (chain_id, res_id) pairs preserving order
        seen: set[tuple[str, int]] = set()
        unique_res: list[tuple[str, int]] = []
        for cid, rid in zip(structure.chain_id, structure.res_id):
            key = (str(cid), int(rid))
            if key not in seen:
                seen.add(key)
                unique_res.append(key)

        per_res = []
        for chain_id, res_id in unique_res:
            mask = (structure.chain_id == chain_id) & (structure.res_id == res_id)
            ca_mask = mask & (structure.atom_name == "CA")
            if np.any(ca_mask):
                per_res.append(float(atom_plddt[ca_mask][0]))
            else:
                per_res.append(float(np.mean(atom_plddt[mask])))
        return np.array(per_res, dtype=np.float64)

    def _parse_output(
        self,
        output: dict[str, Any],
        chains: list[Chain],
    ) -> AlphaFastResult:
        """
        Parse the output dictionary returned by the AlphaFast Modal function.

        Expected keys:
        - ``cif``: CIF file content as a string
        - ``summary_confidences``: dict with ``ptm``, ``iptm``, ``ranking_score``,
          ``chain_pair_iptm``, ``atom_plddts``
        - ``full_data`` (optional): dict with ``pae`` matrix

        If the Modal function returns raw file paths instead, this method
        handles that case too.
        """
        # Parse structure — handle both key conventions:
        # Modal predict_structure returns "structure", BAGEL convention uses "cif"
        cif_content = output.get("cif", "") or output.get("structure", "")
        if not cif_content:
            raise ValueError("AlphaFast output missing 'cif'/'structure' key")
        structure = self._parse_cif(cif_content)

        # Remap chain IDs: AF3 assigns its own chain IDs (A, B, C, ...) which
        # may not match the input BAGEL chain IDs (e.g. "GEN", "B").  Match
        # output chains to input chains by sequence length, then rename.
        output_chain_ids = list(dict.fromkeys(structure.chain_id))  # unique, order-preserving
        input_chain_ids = [c.chain_ID for c in chains]
        if set(output_chain_ids) != set(input_chain_ids):
            # Build mapping: output_chain_id -> input_chain_id by matching lengths
            out_lengths = {}
            for oc in output_chain_ids:
                out_lengths[oc] = int(np.sum(structure.atom_name[structure.chain_id == oc] == "CA"))
            in_lengths = {c.chain_ID: len(c.sequence) for c in chains}
            chain_map: dict[str, str] = {}
            used_input: set[str] = set()
            for oc, olen in out_lengths.items():
                for ic, ilen in in_lengths.items():
                    if ic not in used_input and olen == ilen:
                        chain_map[oc] = ic
                        used_input.add(ic)
                        break
            if len(chain_map) == len(output_chain_ids):
                new_chain_ids = np.array([chain_map.get(c, c) for c in structure.chain_id])
                structure.chain_id = new_chain_ids
                logger.info(f"Remapped AF3 chain IDs: {chain_map}")
            else:
                logger.warning(
                    f"Could not remap all chain IDs (output={output_chain_ids}, "
                    f"input={input_chain_ids}); ipSAE may be incorrect"
                )

        # Parse confidence metrics — handle both key conventions:
        # Modal returns "summary" + "confidence", BAGEL uses "summary_confidences" + "full_data"
        summary = output.get("summary_confidences", {}) or output.get("summary", {})
        full_data = output.get("full_data", {}) or output.get("confidence", {})

        # AF3 Modal output stores per-atom pLDDT and PAE in per-sample files
        # (e.g. "seed-1_sample-0/..._confidences.json") rather than in the
        # top-level confidence dict.  Extract from the best sample (sample-0).
        if not full_data.get("atom_plddts") and not full_data.get("pae"):
            files = output.get("files", {})
            for fname, fdata in files.items():
                if "confidences.json" in fname and "summary" not in fname and "sample-0" in fname:
                    if isinstance(fdata, dict):
                        full_data = fdata
                    break

        # The confidence dict from Modal contains summary-level metrics too
        conf = output.get("confidence", {})

        # pTM — check summary first, then confidence dict
        ptm_val = float(summary.get("ptm", 0.0) or conf.get("ptm", 0.0))
        ptm = np.array([ptm_val], dtype=np.float64)

        # Ranking score
        ranking_score = float(summary.get("ranking_score", 0.0) or conf.get("ranking_score", 0.0))

        # chain_pair_iptm: [n_chains, n_chains]
        chain_pair_iptm_raw = summary.get("chain_pair_iptm", None) or conf.get("chain_pair_iptm", None)
        if chain_pair_iptm_raw is not None:
            chain_pair_iptm = np.array(chain_pair_iptm_raw, dtype=np.float64)
        else:
            # Fallback: use scalar iptm on off-diagonal
            iptm_val = float(summary.get("iptm", 0.0) or conf.get("iptm", 0.0))
            n_chains = len(chains)
            chain_pair_iptm = np.eye(n_chains, dtype=np.float64)
            for i in range(n_chains):
                for j in range(n_chains):
                    if i != j:
                        chain_pair_iptm[i, j] = iptm_val

        # Per-atom pLDDT → per-residue, normalise from 0-100 to 0-1
        # atom_plddts may be in summary (BAGEL convention) or full_data/confidence (AF3 convention)
        atom_plddts = summary.get("atom_plddts", None) or full_data.get("atom_plddts", None)
        if atom_plddts is not None:
            atom_plddt_arr = np.array(atom_plddts, dtype=np.float64)
            # AF3 pLDDT is on 0-100 scale; normalise to 0-1
            if np.max(atom_plddt_arr) > 1.0:
                atom_plddt_arr = atom_plddt_arr / 100.0
            per_res_plddt = self._per_residue_plddt(atom_plddt_arr, structure)
        else:
            n_res = sum(c.length for c in chains)
            per_res_plddt = np.zeros(n_res, dtype=np.float64)
        local_plddt = per_res_plddt[np.newaxis, :]  # [1, n_residues]

        # PAE matrix — shape [1, n_residues, n_residues]
        pae_raw = full_data.get("pae", None)
        if pae_raw is not None:
            pae = np.array(pae_raw, dtype=np.float64)
            if pae.ndim == 2:
                pae = pae[np.newaxis, :]  # add batch dim
        else:
            n_res = sum(c.length for c in chains)
            pae = np.zeros((1, n_res, n_res), dtype=np.float64)
            logger.warning("AlphaFast output missing PAE matrix; using zeros")

        return AlphaFastResult(
            input_chains=chains,
            structure=structure,
            local_plddt=local_plddt,
            pae=pae,
            ptm=ptm,
            chain_pair_iptm=chain_pair_iptm,
            ranking_score=ranking_score,
        )

    # --------------------------------------------------------------------- #
    # Remote prediction
    # --------------------------------------------------------------------- #

    def _call_modal(self, af3_input: dict) -> dict[str, Any]:
        """
        Call the AlphaFast Modal function and return parsed output.

        Tries ``modal.Function.from_name()`` first (for persistent deployments).
        Falls back to ``modal run`` subprocess as a last resort.
        """
        try:
            import modal
        except ImportError:
            logger.warning("modal not installed; falling back to subprocess")
            return self._call_modal_subprocess(af3_input)

        try:
            fn = modal.Function.from_name(self.modal_app_name, self.modal_function_name)
            result = fn.remote(af3_input, **self.config)
            # Check for error status from the Modal function
            if isinstance(result, dict) and result.get("status") == "error":
                error_msg = result.get("error", "unknown")
                raise RuntimeError(
                    f"AlphaFast Modal function returned error: {error_msg}"
                )
            return result
        except modal.exception.NotFoundError as e:
            logger.warning(
                f"modal.Function.from_name failed ({e}); "
                f"falling back to subprocess 'modal run'"
            )

        return self._call_modal_subprocess(af3_input)

    def _call_modal_subprocess(self, af3_input: dict[str, Any]) -> dict[str, Any]:
        """Fallback: write input JSON to temp file and call via subprocess."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, prefix="af3_input_"
        ) as f:
            json.dump(af3_input, f)
            input_path = f.name

        with tempfile.TemporaryDirectory(prefix="af3_output_") as output_dir:
            cmd = [
                "modal", "run",
                f"{self.modal_app_name}",
                "--input-json", input_path,
                "--output-dir", output_dir,
            ]
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, timeout=600)

            # Parse output files from the output directory
            return self._parse_output_dir(Path(output_dir))

    @staticmethod
    def _parse_output_dir(output_dir: Path) -> dict[str, Any]:
        """Parse AlphaFast output files from a directory."""
        result: dict[str, Any] = {}

        # Find CIF file
        cif_files = list(output_dir.rglob("*.cif"))
        if cif_files:
            result["cif"] = cif_files[0].read_text()

        # Find summary confidences JSON
        summary_files = list(output_dir.rglob("summary_confidences*.json"))
        if summary_files:
            with open(summary_files[0]) as f:
                result["summary_confidences"] = json.load(f)

        # Find full data JSON (contains PAE)
        full_data_files = list(output_dir.rglob("full_data*.json"))
        if full_data_files:
            with open(full_data_files[0]) as f:
                result["full_data"] = json.load(f)

        return result

    # --------------------------------------------------------------------- #
    # Main fold method
    # --------------------------------------------------------------------- #

    def fold(self, chains: list[Chain]) -> AlphaFastResult:
        """
        Fold a list of chains using AlphaFast (AlphaFold3) via Modal.

        When multiple ``model_seeds`` are configured, runs all seeds and
        returns the result with the best ``ranking_score``.
        """
        af3_input = self._chains_to_af3_json(chains, self.model_seeds, skip_msa=self.skip_msa)

        logger.info(
            f"Calling AlphaFast via Modal with {len(chains)} chain(s), "
            f"{len(self.model_seeds)} seed(s)"
        )

        output = self._call_modal(af3_input)

        # If multiple seeds returned as a list, pick the best
        if isinstance(output, list):
            best = None
            best_score = -float("inf")
            for out in output:
                parsed = self._parse_output(out, chains)
                if parsed.ranking_score > best_score:
                    best = parsed
                    best_score = parsed.ranking_score
            if best is None:
                raise RuntimeError("AlphaFast returned no results")
            return best

        return self._parse_output(output, chains)
