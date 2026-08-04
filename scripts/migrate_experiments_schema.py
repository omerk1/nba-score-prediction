"""
One-off: seeds outputs/experiments_v2.csv (the new CV-protocol leaderboard
schema -- adds val_score_mean, val_score_per_fold, test_score_mean, protocol,
session_id, and fixes the stale `rolling_window` header, which the code has
written as "rolling_windows" for a while) from outputs/experiments.csv's
existing rows. See CLAUDE.md's "Project Rules (ML experimentation)" section.

outputs/experiments.csv itself is NEVER modified -- it stays exactly as it
was, a frozen historical snapshot at the old 16-column schema.
outputs/experiments_v2.csv is the new file train_model.py writes to going
forward (its --experiments-csv default).

Every existing row gets protocol="single_split" (all of them came from the
pre-CV harness's one-fixed-split pipeline); session_id and the 3 new score
columns are left EMPTY -- they can't be retroactively computed, since each
row's own naive-baseline values (needed to normalize the composite score)
were never recorded and can't be reconstructed after the fact without
re-running every historical experiment.

Usage: venv/bin/python3 scripts/migrate_experiments_schema.py [--dry-run]
"""

import argparse
import csv
from pathlib import Path

NEW_COLUMNS = ["val_score_mean", "val_score_per_fold", "test_score_mean", "protocol", "session_id"]


def migrate(source_path: Path, dest_path: Path, dry_run: bool) -> None:
    with open(source_path, newline="") as f:
        reader = csv.DictReader(f)
        old_fieldnames = reader.fieldnames
        rows = list(reader)

    if old_fieldnames is None:
        raise ValueError(f"{source_path} has no header row -- nothing to seed from")

    if dest_path.exists():
        print(f"{dest_path} already exists -- not overwriting. Delete it first if you want to reseed.")
        return

    # Fix the stale header (values were always correct; only the label was wrong).
    new_fieldnames = ["rolling_windows" if c == "rolling_window" else c for c in old_fieldnames]
    # Insert the 5 new columns right before "notes" (matches train_model.py's
    # _save_experiment row-dict ordering going forward).
    notes_idx = new_fieldnames.index("notes")
    new_fieldnames = new_fieldnames[:notes_idx] + NEW_COLUMNS + new_fieldnames[notes_idx:]

    new_rows = []
    for row in rows:
        new_row = {k: row.get("rolling_window" if k == "rolling_windows" else k, "") for k in new_fieldnames}
        new_row["protocol"] = "single_split"
        new_row["val_score_mean"] = ""
        new_row["val_score_per_fold"] = ""
        new_row["test_score_mean"] = ""
        new_row["session_id"] = ""
        new_rows.append(new_row)

    print(f"Seeding {dest_path} from {len(rows)} rows in {source_path} ({source_path} left untouched).")
    print(f"Old header: {old_fieldnames}")
    print(f"New header: {new_fieldnames}")

    if dry_run:
        print("--dry-run set: not writing anything.")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"Wrote {dest_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", default="outputs/experiments.csv")
    parser.add_argument("--dest-csv", default="outputs/experiments_v2.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written, write nothing.")
    args = parser.parse_args()
    migrate(Path(args.source_csv), Path(args.dest_csv), args.dry_run)
