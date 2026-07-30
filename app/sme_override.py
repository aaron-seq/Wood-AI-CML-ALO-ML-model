"""SME (Subject Matter Expert) Override System."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


class SMEOverrideManager:
    """Manage SME manual overrides for CML elimination decisions."""

    def __init__(self, override_file: Path = Path("data/sme_overrides.json")):
        self.override_file = override_file
        self.override_file.parent.mkdir(exist_ok=True, parents=True)

        if not self.override_file.exists():
            self._save_overrides([])

    def _load_overrides(self) -> list[dict]:
        """Load all SME overrides from file."""
        try:
            with open(self.override_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_overrides(self, overrides: list[dict]):
        """Save overrides to file."""
        with open(self.override_file, "w") as f:
            json.dump(overrides, f, indent=2, default=str)

    def add_override(
        self,
        id_number: str,
        sme_decision: str,
        reason: str,
        sme_name: str,
        original_prediction: str | None = None,
        original_probability: float | None = None,
    ) -> dict:
        """Add a new SME override decision."""
        if sme_decision not in ["KEEP", "ELIMINATE"]:
            raise ValueError("sme_decision must be 'KEEP' or 'ELIMINATE'")

        override = {
            "id_number": id_number,
            "sme_decision": sme_decision,
            "reason": reason,
            "sme_name": sme_name,
            "override_date": datetime.now().isoformat(),
            "original_prediction": original_prediction,
            "original_probability": original_probability,
        }

        overrides = self._load_overrides()

        existing_idx = None
        for idx, ov in enumerate(overrides):
            if ov["id_number"] == id_number:
                existing_idx = idx
                break

        if existing_idx is not None:
            overrides[existing_idx] = override
        else:
            overrides.append(override)

        self._save_overrides(overrides)

        return override

    def get_override(self, id_number: str) -> dict | None:
        """Get SME override for a specific CML."""
        overrides = self._load_overrides()

        for override in overrides:
            if override["id_number"] == id_number:
                return override

        return None

    def get_all_overrides(self) -> list[dict]:
        """Get all SME overrides."""
        return self._load_overrides()

    def remove_override(self, id_number: str) -> bool:
        """Remove SME override for a specific CML."""
        overrides = self._load_overrides()

        original_length = len(overrides)
        overrides = [ov for ov in overrides if ov["id_number"] != id_number]

        if len(overrides) < original_length:
            self._save_overrides(overrides)
            return True

        return False

    def apply_overrides_to_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Apply SME overrides to prediction results."""
        if "id_number" not in predictions_df.columns:
            return predictions_df

        overrides = self._load_overrides()

        if not overrides:
            return predictions_df

        override_dict = {ov["id_number"]: ov for ov in overrides}

        predictions_df["has_sme_override"] = False
        predictions_df["sme_decision"] = None
        predictions_df["sme_reason"] = None
        predictions_df["sme_name"] = None
        predictions_df["sme_override_date"] = None

        for idx, row in predictions_df.iterrows():
            cml_id = row["id_number"]

            if cml_id in override_dict:
                override = override_dict[cml_id]

                predictions_df.at[idx, "has_sme_override"] = True
                predictions_df.at[idx, "sme_decision"] = override["sme_decision"]
                predictions_df.at[idx, "sme_reason"] = override["reason"]
                predictions_df.at[idx, "sme_name"] = override["sme_name"]
                predictions_df.at[idx, "sme_override_date"] = override["override_date"]

                if "recommendation" in predictions_df.columns:
                    predictions_df.at[idx, "final_decision"] = override["sme_decision"]
                else:
                    predictions_df.at[idx, "recommendation"] = override["sme_decision"]

        return predictions_df

    @staticmethod
    def _recent_overrides(df: pd.DataFrame, limit: int = 10) -> list[dict]:
        """Return the most recently recorded overrides, newest first.

        ``override_date`` is persisted as an ISO-8601 *string*, and
        ``DataFrame.nlargest`` rejects non-numeric dtypes with a
        TypeError, so the column is parsed to timestamps and sorted.
        Unparseable dates sort last rather than aborting the statistics.
        """
        if "override_date" not in df.columns:
            return []

        columns = [
            column
            for column in ("id_number", "sme_decision", "sme_name", "override_date")
            if column in df.columns
        ]
        ordered = df.assign(
            _sort_key=pd.to_datetime(df["override_date"], errors="coerce", format="mixed")
        ).sort_values("_sort_key", ascending=False, na_position="last")

        return ordered.head(limit)[columns].to_dict("records")

    def get_override_statistics(self) -> dict:
        """Get statistics about SME overrides."""
        overrides = self._load_overrides()

        if not overrides:
            return {"total_overrides": 0, "keep_overrides": 0, "eliminate_overrides": 0}

        df = pd.DataFrame(overrides)

        stats = {
            "total_overrides": len(overrides),
            "keep_overrides": len(df[df["sme_decision"] == "KEEP"]),
            "eliminate_overrides": len(df[df["sme_decision"] == "ELIMINATE"]),
            "sme_distribution": df["sme_name"].value_counts().to_dict()
            if "sme_name" in df.columns
            else {},
            "recent_overrides": self._recent_overrides(df),
        }

        if "original_prediction" in df.columns:
            disagreements = df[df["sme_decision"] != df["original_prediction"]]
            stats["disagreements_with_ml"] = len(disagreements)
            stats["agreement_rate"] = (
                round((len(df) - len(disagreements)) / len(df) * 100, 1) if len(df) > 0 else 0
            )

        return stats


def create_override_manager(override_file: Path = None) -> SMEOverrideManager:
    """Factory function to create SME override manager."""
    if override_file is None:
        override_file = Path("data/sme_overrides.json")

    return SMEOverrideManager(override_file)
