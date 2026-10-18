"""Stop one explicitly bound new-account Brev host when its cutoff is due.

Run in WSL after logging in with the isolated /home/wezeso/brev-new CLI home.
Dry-run is the default; --execute permits a stop only at the effective cutoff.
This is a one-shot check: an operator must arrange repeated/deadline invocation.
Stopping compute can leave billable storage; the estimate is not account credit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
import os
import re
import subprocess
import time


CLI = "/home/wezeso/.local/bin/brev"
CLI_HOME = "/home/wezeso/brev-new"
LATEST_CUTOFF = datetime.fromisoformat("2026-10-18T13:00:00+00:00")
SESSION_BUDGET_USD = Decimal("50")
PROTECTED_IDS = {"wd88a7vvt", "rrbfqp3zd"}
KNOWN_STATUSES = {
    "RUNNING", "STOPPED", "STOPPING", "DELETED", "STARTING", "PENDING",
    "CREATING", "PROVISIONING", "DEPLOYING", "FAILED", "ERROR", "RESTARTING", "DELETING",
}


def utc_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def nonnegative_decimal(value: str) -> Decimal:
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("expected a finite nonnegative number") from error
    if not number.is_finite() or number < 0:
        raise argparse.ArgumentTypeError("expected a finite nonnegative number")
    return number


@dataclass(frozen=True)
class GuardConfig:
    instance_name: str
    instance_id: str
    organization: str
    started_at: datetime
    hourly_usd: Decimal
    budget_usd: Decimal
    spent_before_start_usd: Decimal
    reserve_usd: Decimal
    cutoff_utc: datetime

    def effective_cutoff(self) -> datetime:
        deadline = min(LATEST_CUTOFF, self.cutoff_utc)
        allowance = max(Decimal(0), self.budget_usd - self.spent_before_start_usd - self.reserve_usd)
        runtime_seconds = allowance * Decimal(3600) / self.hourly_usd
        seconds_until_deadline = Decimal(str(max(0.0, (deadline - self.started_at).total_seconds())))
        if runtime_seconds >= seconds_until_deadline:
            return deadline
        # Round toward an earlier stop; do not extend the deadline by rounding.
        budget_cutoff = self.started_at + timedelta(seconds=int(runtime_seconds))
        return min(deadline, budget_cutoff)

    def estimated_spend(self, now: datetime) -> Decimal:
        elapsed_seconds = Decimal(str(max(0.0, (now - self.started_at).total_seconds())))
        return self.spent_before_start_usd + self.hourly_usd * elapsed_seconds / Decimal(3600)


def run_cli(config: GuardConfig, *arguments: str, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    cli_environment = os.environ.copy()
    # Only the child process receives this HOME; the default CLI profile is untouched.
    cli_environment["HOME"] = CLI_HOME
    return subprocess.run(
        [CLI, "--org", config.organization, *arguments],
        env=cli_environment, capture_output=True, text=True, timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


def instance_status(config: GuardConfig) -> str:
    result = run_cli(config, "ls", "--json")
    if result.returncode:
        raise RuntimeError("provider_inventory_unavailable")
    payload = json.loads(result.stdout)
    workspaces = payload.get("workspaces") if isinstance(payload, dict) else None
    if not isinstance(workspaces, list):
        raise RuntimeError("provider_inventory_schema_mismatch")
    # Validate both selectors independently so a duplicate name cannot redirect stop.
    id_matches = [item for item in workspaces if isinstance(item, dict) and item.get("id") == config.instance_id]
    name_matches = [item for item in workspaces if isinstance(item, dict) and item.get("name") == config.instance_name]
    if len(id_matches) != 1 or len(name_matches) != 1 or id_matches[0] != name_matches[0]:
        raise RuntimeError("authorized_instance_identity_mismatch")
    status = id_matches[0].get("status")
    if not isinstance(status, str) or status not in KNOWN_STATUSES:
        raise RuntimeError("provider_status_unrecognized")
    return status


def parse_args() -> tuple[GuardConfig, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-name", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--organization", required=True)
    parser.add_argument("--started-at", required=True, type=utc_timestamp,
                        help="conservative start of billable host time, with timezone")
    parser.add_argument("--hourly-usd", required=True, type=nonnegative_decimal,
                        help="estimated all-in hourly rate for this host")
    parser.add_argument("--budget-usd", default=SESSION_BUDGET_USD, type=nonnegative_decimal,
                        help="session spend ceiling, at most USD 50")
    parser.add_argument("--spent-before-start-usd", default=Decimal("0"), type=nonnegative_decimal)
    parser.add_argument("--reserve-usd", default=Decimal("0"), type=nonnegative_decimal)
    parser.add_argument("--cutoff-utc", default=LATEST_CUTOFF, type=utc_timestamp,
                        help="optional earlier stop; never later than 18 Oct 18:00 Asia/Qyzylorda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"forte-new-[a-z0-9][a-z0-9-]{0,62}", args.instance_name):
        parser.error("instance name must be an explicitly bound forte-new-* test host")
    if not re.fullmatch(r"[a-zA-Z0-9]{4,64}", args.instance_id) or args.instance_id in PROTECTED_IDS:
        parser.error("instance ID is invalid or belongs to a protected old-account host")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}", args.organization):
        parser.error("organization selector is invalid")
    if args.hourly_usd <= 0:
        parser.error("hourly rate must be greater than zero")
    if args.budget_usd > SESSION_BUDGET_USD:
        parser.error("budget cannot exceed the authorized USD 50 session ceiling")
    if args.started_at > LATEST_CUTOFF:
        parser.error("billable start cannot be after the fixed session deadline")
    config = GuardConfig(
        instance_name=args.instance_name, instance_id=args.instance_id,
        organization=args.organization, started_at=args.started_at,
        hourly_usd=args.hourly_usd, budget_usd=args.budget_usd,
        spent_before_start_usd=args.spent_before_start_usd, reserve_usd=args.reserve_usd,
        cutoff_utc=args.cutoff_utc,
    )
    return config, args.execute


def main() -> int:
    config, execute = parse_args()
    now = datetime.now(timezone.utc)
    try:
        cutoff = config.effective_cutoff()
        status = instance_status(config)
        common = {
            "instance_name": config.instance_name, "instance_id": config.instance_id,
            "status": status, "cli_home": CLI_HOME, "cutoff_utc": cutoff.isoformat(),
            "latest_cutoff_utc": LATEST_CUTOFF.isoformat(), "due": now >= cutoff,
            "estimated_spend_usd": str(config.estimated_spend(now).quantize(Decimal("0.01"))),
            "budget_usd": str(config.budget_usd), "reserve_usd": str(config.reserve_usd),
            "remaining_seconds": max(0, int((cutoff - now).total_seconds())),
        }
        if not execute or now < cutoff:
            print(json.dumps({"guard": "ready", "mode": "execute_if_due" if execute else "dry_run", **common}))
            return 0
        if status in {"STOPPED", "DELETED", "STOPPING"}:
            print(json.dumps({"guard": "already_stopped_or_stopping", **common, "storage_still_bills": True}))
            return 0
        # The validated exact name is the supported selector in the existing operator workflow.
        result = run_cli(config, "stop", config.instance_name, timeout=180)
        if result.returncode:
            raise RuntimeError("provider_stop_failed")
        for attempt in range(12):
            status = instance_status(config)
            if status in {"STOPPED", "STOPPING"}:
                print(json.dumps({"guard": "stop_verified", **common, "status": status,
                                  "storage_still_bills": True, "compute_stop_complete": status == "STOPPED"}))
                return 0
            if attempt < 11:
                time.sleep(5)
        raise RuntimeError("provider_stop_not_verified")
    except Exception as error:
        # Never print raw provider output, exception text, credentials, or inventory.
        print(json.dumps({"guard": "failed", "instance_id": config.instance_id,
                          "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
