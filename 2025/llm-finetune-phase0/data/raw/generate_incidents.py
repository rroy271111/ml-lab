# Generate incidents

import json
import random
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "incidents.jsonl"

SERVICES = [
    "payments",
    "auth",
    "orders",
    "inventory",
    "notifications",
]

ROOT_CAUSES = [
    "deployment regression",
    "database connection exhaustion",
    "misconfiguration environment variable",
    "downstream service timeout",
    "unexpected traffic spike",
]

IMPACTS = [
    "checkout failures",
    "user login failures",
    "order processing delays",
    "missing notifications",
    "inventory desync",
]

SEVERITY_MAP = {
    "checkout failures": "HIGH",
    "user login failures": "HIGH",
    "order processing delays": "MEDIUM",
    "missing notifications": "LOW",
    "inventory desync": "MEDIUM",
}


def generate_incident():
    service = random.choice(SERVICES)
    root_cause = random.choice(ROOT_CAUSES)
    impact = random.choice(IMPACTS)
    severity = SEVERITY_MAP[impact]
    action_required = severity in {"HIGH", "CRITICAL"}

    text = (
        f"Following a recent deployment, the {services} services started "
        f"experiencing issues. Engineers observed {impact} caused by "
        f"{root_cause}. Immediate investigation was initiated."
    )

    return {
        "text": text,
        "label": {
            "service": service,
            "severity": severity,
            "root_cause": root_cause,
            "impact": impact,
            "action_required": action_required,
        },
    }


def main(n_samples: int = 300):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        for _ in range(n_samples):
            record = generate_incident()
            f.write(json.dumps(record) + "\n")
    print(f"Generated {n_samples} incidents at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
