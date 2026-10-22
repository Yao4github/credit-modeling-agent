"""Data-Pointer Registry & Synthetic Credit Data Warehouse Generator.

Implements the Data-Pointer Pattern: Large DataFrames (thousands/millions of credit records)
are never serialized into LLM prompt contexts. Instead, tools exchange lightweight URI pointers
(e.g., `ds://credit_a_card_train`, `ds://credit_a_card_oot`) and return only compact statistical
summaries to the Agent state machine.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from app.core.paths import get_data_dir

logger = logging.getLogger(__name__)
_store_lock = threading.Lock()


@dataclass
class DatasetMetadata:
    """Metadata describing a registered credit dataset pointer."""

    uri: str
    description: str
    obs_start_date: Optional[str]
    obs_end_date: Optional[str]
    perf_window_months: int
    target_col: str
    row_count: int
    bad_rate: float
    columns: List[str]
    tags: Dict[str, Any] = field(default_factory=dict)


class DataPointerRegistry:
    """Thread-safe registry mapping dataset URI pointers to in-memory/parquet DataFrames."""

    _instance: Optional["DataPointerRegistry"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._meta: Dict[str, DatasetMetadata] = {}
        self._models: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_instance(cls) -> "DataPointerRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                cls._instance.bootstrap_synthetic_credit_warehouse()
                cls._instance._load_persisted()
            return cls._instance

    # ---- persistence of file-mounted registrations (parameters only, never the data) ----------------
    @staticmethod
    def _store_path():
        return get_data_dir() / "dataset_registry.json"

    @classmethod
    def _read_records(cls) -> List[Dict[str, Any]]:
        path = cls._store_path()
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and all(isinstance(r, dict) and "uri" in r for r in data):
                return data
        except (OSError, ValueError):
            pass
        return []

    @classmethod
    def _write_records(cls, records: List[Dict[str, Any]]) -> None:
        path = cls._store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                os.replace(path, path.with_suffix(".json.corrupt"))  # keep the broken file, never overwrite silently
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)

    @classmethod
    def _persist_record(cls, record: Dict[str, Any]) -> None:
        with _store_lock:
            records = [r for r in cls._read_records() if r["uri"] != record["uri"]]
            cls._write_records(records + [record])

    @classmethod
    def _forget_records(cls, uris: List[str]) -> None:
        with _store_lock:
            records = cls._read_records()
            kept = [r for r in records if r["uri"] not in uris]
            if len(kept) != len(records):
                cls._write_records(kept)

    def _load_persisted(self) -> None:
        """Re-mounts every stored registration; a record that fails (file gone, bad labels...) is skipped and kept."""
        for record in self._read_records():
            try:
                self.register_from_file(**record, _persist=False)
            except Exception as exc:  # noqa: BLE001 - one bad record must not block service start-up
                logger.warning("skipped persisted dataset %s: %s", record.get("uri"), exc)

    def register_dataset(
        self,
        uri: str,
        df: pd.DataFrame,
        description: str,
        obs_start_date: Optional[str],
        obs_end_date: Optional[str],
        perf_window_months: int = 3,
        target_col: str = "target_mob3_dpd30",
        tags: Optional[Dict[str, Any]] = None,
    ) -> DatasetMetadata:
        bad_rate = float(df[target_col].mean()) if target_col in df.columns else 0.0
        meta = DatasetMetadata(
            uri=uri,
            description=description,
            obs_start_date=obs_start_date,
            obs_end_date=obs_end_date,
            perf_window_months=perf_window_months,
            target_col=target_col,
            row_count=len(df),
            bad_rate=round(bad_rate, 4),
            columns=list(df.columns),
            tags=tags or {},
        )
        self._datasets[uri] = df.copy()
        self._meta[uri] = meta
        return meta

    def register_from_file(
        self,
        uri: str,
        path: str,
        target_col: str,
        obs_start_date: Optional[str] = None,
        obs_end_date: Optional[str] = None,
        perf_window_months: int = 3,
        description: str = "",
        date_col: Optional[str] = None,
        _persist: bool = True,
    ) -> DatasetMetadata:
        """Mounts a CSV/Parquet file located inside the data directory as a dataset pointer.

        Observation window: explicit dates win; otherwise it is inferred from `date_col` (min / max); with neither
        it stays None and the chronological check is skipped (reported as not verifiable).
        """
        if not uri.startswith("ds://"):
            raise ValueError("uri must start with 'ds://'")
        explicit_start, explicit_end = obs_start_date, obs_end_date
        data_dir = get_data_dir()
        file_path = (data_dir / path).resolve()
        if not file_path.is_relative_to(data_dir):
            raise ValueError(f"path '{path}' is outside the data directory")
        if not file_path.is_file():
            raise FileNotFoundError(f"data file '{path}' not found in data directory")

        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError("only .csv and .parquet files are supported")

        if target_col not in df.columns:
            raise ValueError(f"target column '{target_col}' not found in file")
        if not df[target_col].isin([0, 1]).all():
            raise ValueError(f"target column '{target_col}' must contain only 0/1 labels")

        tags: Dict[str, Any] = {"source_file": path}
        if date_col:
            if date_col not in df.columns:
                raise ValueError(f"date column '{date_col}' not found in file")
            dates = pd.to_datetime(df[date_col], errors="coerce")
            if dates.isna().any():
                raise ValueError(f"date column '{date_col}' contains values that cannot be parsed as dates")
            obs_start_date = obs_start_date or dates.min().strftime("%Y-%m-%d")
            obs_end_date = obs_end_date or dates.max().strftime("%Y-%m-%d")
            tags["date_col"] = date_col

        meta = self.register_dataset(
            uri=uri,
            df=df,
            description=description or f"External dataset mounted from {path}",
            obs_start_date=obs_start_date,
            obs_end_date=obs_end_date,
            perf_window_months=perf_window_months,
            target_col=target_col,
            tags=tags,
        )
        if _persist:
            # store the arguments as given (not the inferred window) so a changed file is re-inferred on restart
            self._persist_record(
                {
                    "uri": uri,
                    "path": path,
                    "target_col": target_col,
                    "obs_start_date": explicit_start,
                    "obs_end_date": explicit_end,
                    "perf_window_months": perf_window_months,
                    "description": description,
                    "date_col": date_col,
                }
            )
        return meta

    def get_df(self, uri: str) -> pd.DataFrame:
        if uri not in self._datasets:
            raise KeyError(f"Dataset URI '{uri}' not found in DataPointerRegistry. Available: {list(self._datasets.keys())}")
        return self._datasets[uri]

    def get_meta(self, uri: str) -> DatasetMetadata:
        if uri not in self._meta:
            raise KeyError(f"Metadata for URI '{uri}' not found.")
        return self._meta[uri]

    def unregister(self, uri: str) -> None:
        """Drops a file-mounted pointer (the file stays). KeyError if unknown, ValueError for built-in datasets.

        Row-split pointers derived from `uri` are dropped with it.
        """
        if uri not in self._meta:
            raise KeyError(f"dataset '{uri}' not found")
        if "source_file" not in self._meta[uri].tags:
            raise ValueError(f"dataset '{uri}' is built-in and cannot be unregistered")
        for derived in [u for u, m in self._meta.items() if m.tags.get("row_split_of") == uri]:
            del self._datasets[derived]
            del self._meta[derived]
        del self._datasets[uri]
        del self._meta[uri]
        self._forget_records([uri])

    def split_by_rows(self, uri: str, dev_ratio: float = 0.7) -> tuple[str, str]:
        """Registers (and returns) `<uri>#dev` = first `dev_ratio` of the rows and `<uri>#oot` = the rest.

        Row order is taken as given, so this is NOT a guaranteed out-of-time split. Raises ValueError when a part
        would be too small or contain a single class only.
        """
        df, meta = self.get_df(uri), self.get_meta(uri)
        cut = int(len(df) * dev_ratio)
        parts = {"dev": df.iloc[:cut], "oot": df.iloc[cut:]}
        for name, part in parts.items():
            if len(part) < 30 or part[meta.target_col].nunique() < 2:
                raise ValueError(
                    f"row split of '{uri}' failed: {name} part has {len(part)} rows or a single label class; "
                    "provide a separate out-of-time dataset instead"
                )
        date_col = meta.tags.get("date_col")
        pct = int(dev_ratio * 100)
        uris = {}
        for name, part in parts.items():
            start = end = None
            if date_col:
                dates = pd.to_datetime(part[date_col])
                start, end = dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")
            uris[name] = f"{uri}#{name}"
            self.register_dataset(
                uri=uris[name],
                df=part,
                description=f"{uri} 按行切分{'前' if name == 'dev' else '后'}{pct if name == 'dev' else 100 - pct}%（非严格跨期）",
                obs_start_date=start,
                obs_end_date=end,
                perf_window_months=meta.perf_window_months,
                target_col=meta.target_col,
                tags={**meta.tags, "row_split_of": uri, "split_part": name},
            )
        return uris["dev"], uris["oot"]

    def resolve_modeling_uris(self, train_uri: Optional[str], oot_uri: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """Train only -> row split into dev/OOT; OOT only -> ValueError; otherwise unchanged."""
        if oot_uri and not train_uri:
            raise ValueError("oot_uri given without train_uri; select the development dataset as well")
        if train_uri and not oot_uri:
            return self.split_by_rows(train_uri)
        return train_uri, oot_uri

    def uris_for_file(self, filename: str) -> List[str]:
        """URIs of pointers mounted from `filename` (a path relative to the data directory)."""
        wanted = os.path.normpath(filename)
        return [u for u, m in self._meta.items() if "source_file" in m.tags and os.path.normpath(m.tags["source_file"]) == wanted]

    def list_datasets(self) -> List[Dict[str, Any]]:
        return [
            {
                "uri": m.uri,
                "description": m.description,
                "obs_window": f"{m.obs_start_date} ~ {m.obs_end_date}" if m.obs_start_date and m.obs_end_date else "未提供",
                "perf_window_months": m.perf_window_months,
                "row_count": m.row_count,
                "bad_rate": m.bad_rate,
                "column_count": len(m.columns),
                "removable": "source_file" in m.tags,
            }
            for m in self._meta.values()
        ]

    def register_model(self, model_id: str, payload: Dict[str, Any]) -> None:
        self._models[model_id] = payload

    def get_model(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self._models:
            raise KeyError(f"Model artifact '{model_id}' not found in registry.")
        return self._models[model_id]

    def bootstrap_synthetic_credit_warehouse(self, seed: int = 42) -> None:
        """Generates realistic A-Card (Application Scorecard) Train & OOT datasets with intentional domain traps."""
        rng = np.random.default_rng(seed)

        def _generate_cohort(n_rows: int, is_oot: bool = False) -> pd.DataFrame:
            # Latent true credit risk factor z
            z_risk = rng.normal(0.0, 1.0, size=n_rows)

            # 1. Stable high-IV features
            credit_inq_cnt_3m = np.clip(np.round(3.5 + 2.2 * z_risk + rng.normal(0, 1.2, n_rows)), 0, 25)
            multi_platform_loan_cnt_6m = np.clip(np.round(2.8 + 1.9 * z_risk + rng.normal(0, 1.0, n_rows)), 0, 20)
            hist_max_dpd_days_12m = np.clip(np.round(np.exp(1.2 + 0.95 * z_risk + rng.normal(0, 0.5, n_rows)) - 1), 0, 90)
            debt_to_income_ratio = np.clip(0.38 + 0.16 * z_risk + rng.normal(0, 0.08, n_rows), 0.05, 0.98)
            credit_card_utilization = np.clip(0.45 + 0.18 * z_risk + rng.normal(0, 0.10, n_rows), 0.0, 1.20)
            verified_income_monthly = np.clip(np.round(11500 - 2600 * z_risk + rng.normal(0, 1800, n_rows)), 2500, 50000)

            # 2. Trap Feature A: Severe OOT PSI drift (`night_apply_ratio_7d`)
            # In Train, correlates strongly with risk; in OOT, channel shift causes massive distribution shift!
            if not is_oot:
                night_apply_ratio_7d = np.clip(0.22 + 0.15 * z_risk + rng.normal(0, 0.07, n_rows), 0.0, 1.0)
            else:
                # OOT distribution shifted upward by +0.38 due to night-time ad campaign
                night_apply_ratio_7d = np.clip(0.60 + 0.04 * z_risk + rng.normal(0, 0.12, n_rows), 0.0, 1.0)

            # 3. Trap Feature B: Extreme Multicollinearity (`multi_loan_composite_idx`, VIF > 20 with multi_platform_loan_cnt_6m)
            multi_loan_composite_idx = 2.0 * multi_platform_loan_cnt_6m + rng.normal(0, 0.15, n_rows)

            # 4. Trap Feature C: Data Leakage / Post-loan feature (`post_loan_repay_amt_m1`)
            post_loan_repay_amt_m1 = np.clip(2000 - 1200 * z_risk + rng.normal(0, 200, n_rows), 0, 5000)

            # 5. Trap Feature D: Prohibited discriminatory feature (`applicant_gender_code`)
            applicant_gender_code = rng.integers(0, 2, size=n_rows)

            # 6. Low-IV noise feature (`app_device_screen_dpi`)
            app_device_screen_dpi = rng.normal(400, 40, size=n_rows)

            # Generate binary default label `target_mob3_dpd30` (~8.5% - 10% bad rate)
            logit = (
                -2.65
                + 0.24 * (credit_inq_cnt_3m - 3.5)
                + 0.28 * (multi_platform_loan_cnt_6m - 2.8)
                + 0.045 * (hist_max_dpd_days_12m - 5.0)
                + 2.10 * (debt_to_income_ratio - 0.38)
                + 1.65 * (credit_card_utilization - 0.45)
                - 0.00012 * (verified_income_monthly - 11500)
                + (0.95 * (night_apply_ratio_7d - 0.22) if not is_oot else 0.05 * (night_apply_ratio_7d - 0.60))
            )
            prob = 1.0 / (1.0 + np.exp(-logit))
            target = (rng.uniform(0, 1, size=n_rows) < prob).astype(int)

            df = pd.DataFrame(
                {
                    "apply_id": [f"{'OOT' if is_oot else 'TRN'}_{i:05d}" for i in range(n_rows)],
                    "credit_inq_cnt_3m": np.round(credit_inq_cnt_3m, 2),
                    "multi_platform_loan_cnt_6m": np.round(multi_platform_loan_cnt_6m, 2),
                    "hist_max_dpd_days_12m": np.round(hist_max_dpd_days_12m, 2),
                    "debt_to_income_ratio": np.round(debt_to_income_ratio, 4),
                    "credit_card_utilization": np.round(credit_card_utilization, 4),
                    "verified_income_monthly": np.round(verified_income_monthly, 2),
                    "night_apply_ratio_7d": np.round(night_apply_ratio_7d, 4),
                    "multi_loan_composite_idx": np.round(multi_loan_composite_idx, 4),
                    "post_loan_repay_amt_m1": np.round(post_loan_repay_amt_m1, 2),
                    "applicant_gender_code": applicant_gender_code,
                    "app_device_screen_dpi": np.round(app_device_screen_dpi, 1),
                    "target_mob3_dpd30": target,
                }
            )
            return df

        train_df = _generate_cohort(4000, is_oot=False)
        oot_df = _generate_cohort(1600, is_oot=True)

        self.register_dataset(
            uri="ds://credit_a_card_train",
            df=train_df,
            description="首贷客群A卡开发样本集 (Train/Dev Cohort)",
            obs_start_date="2025-01-01",
            obs_end_date="2025-06-30",
            perf_window_months=3,
            target_col="target_mob3_dpd30",
        )
        self.register_dataset(
            uri="ds://credit_a_card_oot",
            df=oot_df,
            description="首贷客群A卡跨期验证样本集 (Out-of-Time Cohort)",
            obs_start_date="2025-07-01",
            obs_end_date="2025-09-30",
            perf_window_months=3,
            target_col="target_mob3_dpd30",
        )
        self._bootstrap_b_card(seed + 1)
        self._bootstrap_c_card(seed + 2)

    def _bootstrap_b_card(self, seed: int) -> None:
        """B-Card (behavior scorecard) cohorts for existing customers; target=1 means DPD30+ within 3 months."""
        rng = np.random.default_rng(seed)

        def _cohort(n: int, is_oot: bool) -> pd.DataFrame:
            z = rng.normal(0.0, 1.0, size=n)
            avg_util_3m = np.clip(0.45 + 0.17 * z + rng.normal(0, 0.10, n), 0.0, 1.2)
            max_dpd_cnt_6m = np.clip(np.round(1.2 + 1.1 * z + rng.normal(0, 0.9, n)), 0, 12)
            repay_ratio_3m = np.clip(0.80 - 0.18 * z + rng.normal(0, 0.10, n), 0.0, 1.5)
            cash_advance_cnt_3m = np.clip(np.round(1.5 + 1.2 * z + rng.normal(0, 0.9, n)), 0, 15)
            txn_amt_trend_3m = np.clip(-0.05 - 0.20 * z + rng.normal(0, 0.15, n), -1.0, 1.0)

            # Trap A: severe OOT drift (app redesign changed login behaviour)
            if not is_oot:
                login_freq_30d = np.clip(14.0 - 3.0 * z + rng.normal(0, 2.0, n), 0, 60)
            else:
                login_freq_30d = np.clip(24.0 - 0.5 * z + rng.normal(0, 3.0, n), 0, 60)
            # Trap B: collinearity with avg_util_3m
            avg_util_6m = 0.95 * avg_util_3m + rng.normal(0, 0.01, n)
            # Trap C: future information (observed after the observation point)
            fut_m1_overdue_amt = np.clip(300 + 400 * z + rng.normal(0, 60, n), 0, 3000)
            # Trap D: prohibited discriminatory attribute; noise feature
            applicant_gender_code = rng.integers(0, 2, size=n)
            app_device_screen_dpi = rng.normal(400, 40, size=n)

            logit = (
                -2.7
                + 2.2 * (avg_util_3m - 0.45)
                + 0.30 * (max_dpd_cnt_6m - 1.2)
                - 2.0 * (repay_ratio_3m - 0.80)
                + 0.22 * (cash_advance_cnt_3m - 1.5)
                - 1.2 * (txn_amt_trend_3m + 0.05)
                + (-0.09 * (login_freq_30d - 14.0) if not is_oot else -0.005 * (login_freq_30d - 24.0))
            )
            target = (rng.uniform(0, 1, size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
            return pd.DataFrame(
                {
                    "apply_id": [f"{'BOOT' if is_oot else 'BTRN'}_{i:05d}" for i in range(n)],
                    "avg_util_3m": np.round(avg_util_3m, 4),
                    "max_dpd_cnt_6m": max_dpd_cnt_6m,
                    "repay_ratio_3m": np.round(repay_ratio_3m, 4),
                    "cash_advance_cnt_3m": cash_advance_cnt_3m,
                    "txn_amt_trend_3m": np.round(txn_amt_trend_3m, 4),
                    "login_freq_30d": np.round(login_freq_30d, 2),
                    "avg_util_6m": np.round(avg_util_6m, 4),
                    "fut_m1_overdue_amt": np.round(fut_m1_overdue_amt, 2),
                    "applicant_gender_code": applicant_gender_code,
                    "app_device_screen_dpi": np.round(app_device_screen_dpi, 1),
                    "target_fut3m_dpd30": target,
                }
            )

        self.register_dataset(
            uri="ds://credit_b_card_train",
            df=_cohort(4000, False),
            description="存量客群B卡开发样本集 (Train/Dev Cohort)",
            obs_start_date="2025-01-01",
            obs_end_date="2025-06-30",
            perf_window_months=3,
            target_col="target_fut3m_dpd30",
        )
        self.register_dataset(
            uri="ds://credit_b_card_oot",
            df=_cohort(1600, True),
            description="存量客群B卡跨期验证样本集 (Out-of-Time Cohort)",
            obs_start_date="2025-07-01",
            obs_end_date="2025-09-30",
            perf_window_months=3,
            target_col="target_fut3m_dpd30",
        )

    def _bootstrap_c_card(self, seed: int) -> None:
        """C-Card (collection scorecard) cohorts of overdue accounts; target=1 means NOT recovered within 30 days."""
        rng = np.random.default_rng(seed)

        def _cohort(n: int, is_oot: bool) -> pd.DataFrame:
            z = rng.normal(0.0, 1.0, size=n)
            overdue_days = np.clip(np.round(np.exp(2.4 + 0.5 * z + rng.normal(0, 0.4, n))), 1, 180)
            overdue_amount = np.clip(np.round(np.exp(8.0 + 0.5 * z + rng.normal(0, 0.5, n))), 200, 200000)
            prior_collection_cnt = np.clip(np.round(1.0 + 1.0 * z + rng.normal(0, 0.9, n)), 0, 10)
            promise_to_pay_cnt = np.clip(np.round(2.0 - 1.0 * z + rng.normal(0, 1.0, n)), 0, 10)
            debt_to_income_ratio = np.clip(0.42 + 0.15 * z + rng.normal(0, 0.08, n), 0.05, 0.98)

            # Trap A: severe OOT drift (outbound-call vendor switch shifts the connect rate upward).
            # Only the distribution drifts: the label relationship stays the same relative to each cohort's baseline.
            connect_base = 0.72 if is_oot else 0.45
            call_connect_rate_7d = np.clip(connect_base - 0.12 * z + rng.normal(0, 0.07, n), 0.0, 1.0)
            # Trap B: collinearity with overdue_amount
            overdue_amt_bucket = overdue_amount / 10000.0 + rng.normal(0, 0.01, n)
            # Trap C: future information (repayment observed after the collection decision point)
            post_collection_repay_amt_d7 = np.clip(800 - 500 * z + rng.normal(0, 100, n), 0, 3000)
            # Trap D: prohibited discriminatory attribute; noise feature
            applicant_gender_code = rng.integers(0, 2, size=n)
            app_device_screen_dpi = rng.normal(400, 40, size=n)

            logit = (
                -0.5
                + 0.045 * (overdue_days - 12.0)
                + 0.00018 * (overdue_amount - 3800.0)
                + 0.28 * (prior_collection_cnt - 1.0)
                - 0.30 * (promise_to_pay_cnt - 2.0)
                + 2.0 * (debt_to_income_ratio - 0.42)
                - 3.5 * (call_connect_rate_7d - connect_base)
            )
            target = (rng.uniform(0, 1, size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
            return pd.DataFrame(
                {
                    "apply_id": [f"{'COOT' if is_oot else 'CTRN'}_{i:05d}" for i in range(n)],
                    "overdue_days": overdue_days,
                    "overdue_amount": overdue_amount,
                    "prior_collection_cnt": prior_collection_cnt,
                    "promise_to_pay_cnt": promise_to_pay_cnt,
                    "debt_to_income_ratio": np.round(debt_to_income_ratio, 4),
                    "call_connect_rate_7d": np.round(call_connect_rate_7d, 4),
                    "overdue_amt_bucket": np.round(overdue_amt_bucket, 4),
                    "post_collection_repay_amt_d7": np.round(post_collection_repay_amt_d7, 2),
                    "applicant_gender_code": applicant_gender_code,
                    "app_device_screen_dpi": np.round(app_device_screen_dpi, 1),
                    "target_unrecovered_d30": target,
                }
            )

        self.register_dataset(
            uri="ds://credit_c_card_train",
            df=_cohort(4000, False),
            description="逾期客群C卡开发样本集 (Train/Dev Cohort)",
            obs_start_date="2025-01-01",
            obs_end_date="2025-06-30",
            perf_window_months=1,
            target_col="target_unrecovered_d30",
        )
        self.register_dataset(
            uri="ds://credit_c_card_oot",
            df=_cohort(1600, True),
            description="逾期客群C卡跨期验证样本集 (Out-of-Time Cohort)",
            obs_start_date="2025-07-01",
            obs_end_date="2025-09-30",
            perf_window_months=1,
            target_col="target_unrecovered_d30",
        )
