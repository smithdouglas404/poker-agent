#!/usr/bin/env python3
"""
verify_contract.py — Enforces AGENT_CONTRACT.md on the codebase.
Run this BEFORE committing any change to model.js or server.py.

Exits 0 = clean, 1 = contract violation found.
"""
import sys, re, os

REPO = '.'
EXT = 'extension'

# Forbidden patterns — hardcoded poker theory that violates the contract
FORBIDDEN_PATTERNS = [
    (r"DECK_PROMOTE\s*=\s*\[", "Hardcoded promote list (frozen snapshot — banned)"),
    (r"DECK_DOWNGRADE\s*=\s*\[", "Hardcoded downgrade list (frozen snapshot — banned)"),
    (r"TOXIC_RANKS\s*=\s*\[", "Hardcoded toxic ranks (frozen — banned)"),
    (r"GOLDEN_RANK\s*=\s*['\"]", "Hardcoded golden rank (frozen — banned)"),
    (r"PAIRED[\s_-]*BOARD.{0,40}(FOLD|fold)", "Paired-board fold rule (textbook poker — banned)"),
    (r"STACK[\s_-]*COMMIT.{0,80}(FOLD|fold)", "Pot-commitment rule (textbook poker — banned)"),
    (r"bottom[\s_-]*range[\s_-]*(veto|trash)", "Bottom-range veto (Sklansky — banned)"),
    (r"facing[34]bet", "3-bet/4-bet classification (textbook — banned)"),
    (r"isTrash\s*=\s*", "Trash-hand classifier (textbook — banned)"),
    (r"empirical\s*(profile|tier|deck)", "'Empirical' frozen-snapshot logic (banned)"),
]

REQUIRED_PATTERNS = [
    (r"posCarry|carryHot|carryRate", "Card-bleed tracking (Pattern 1)"),
    (r"position(WinRate|_win_rate)|recentWinnerPositions", "Position win-rate (Pattern 2/3)"),
    (r"build_model_from_db|buildModel", "Live model rebuild (no frozen constants)"),
]

def check_file(path, file_label):
    if not os.path.exists(path):
        return [f"MISSING FILE: {path}"]
    with open(path) as f:
        code = f.read()
    violations = []
    for pat, desc in FORBIDDEN_PATTERNS:
        for m in re.finditer(pat, code, re.IGNORECASE):
            line = code[:m.start()].count('\n') + 1
            violations.append(f"  {file_label}:{line}  {desc}")
    return violations

def check_required(files):
    all_code = ""
    for f in files:
        if os.path.exists(f):
            with open(f) as fh:
                all_code += fh.read() + "\n"
    missing = []
    for pat, desc in REQUIRED_PATTERNS:
        if not re.search(pat, all_code, re.IGNORECASE):
            missing.append(f"  REQUIRED MISSING: {desc} (regex: {pat})")
    return missing

def main():
    print("=" * 70)
    print("AGENT_CONTRACT.md VERIFICATION")
    print("=" * 70)

    files = [
        (os.path.join(EXT, 'content.js'), 'extension/content.js'),
        (os.path.join(EXT, 'lib/model.js'), 'extension/lib/model.js'),
        (os.path.join(EXT, 'model.js'), 'extension/model.js'),
    ]

    violations = []
    for path, label in files:
        if os.path.exists(path):
            violations += check_file(path, label)

    print("\n[1/2] CHECK: forbidden patterns (textbook poker contamination)")
    if violations:
        print(f"  X FOUND {len(violations)} VIOLATIONS:")
        for v in violations:
            print(v)
    else:
        print("  OK No forbidden patterns found")

    print("\n[2/2] CHECK: required patterns (live-data infrastructure intact)")
    file_paths = [p for p, _ in files if os.path.exists(p)]
    file_paths.append('server.py')
    missing = check_required(file_paths)
    if missing:
        print(f"  X MISSING REQUIRED:")
        for m in missing:
            print(m)
    else:
        print("  OK All required pattern infrastructure present")

    print("\n" + "=" * 70)
    if violations or missing:
        print("VERDICT: X CONTRACT VIOLATIONS FOUND. Do not deploy.")
        print("Read AGENT_CONTRACT.md before fixing.")
        sys.exit(1)
    else:
        print("VERDICT: OK Contract satisfied. Safe to commit.")
        sys.exit(0)

if __name__ == '__main__':
    main()
