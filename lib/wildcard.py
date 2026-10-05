"""Wildcard expansion for MiniMax H3 Director prompts.

Supports {option1|option2|option3} syntax in segment prompts.
Each wildcard is expanded independently, so multiple wildcards in one
prompt produce all combinations.
"""

from __future__ import annotations

import random
import re

WILDCARD_RE = re.compile(r"\{([^}]+)\}")


def expand_wildcards(prompt: str, *, seed: int | None = None) -> str:
    """Expand all {a|b|c} wildcards in the prompt.

    If seed is provided, choices are deterministic for that seed.
    Multiple wildcards are expanded independently (Cartesian product).
    """
    if not prompt:
        return prompt

    def replace_match(match):
        options = [o.strip() for o in match.group(1).split("|") if o.strip()]
        if not options:
            return match.group(0)
        if seed is not None:
            # Deterministic choice based on seed and position
            r = random.Random(seed + hash(match.group(0)))
        else:
            r = random
        return r.choice(options)

    result = WILDCARD_RE.sub(replace_match, prompt)
    return result


def expand_wildcards_all_combinations(prompt: str) -> list[str]:
    """Expand all {a|b|c} wildcards into all possible combinations.

    Returns a list of all possible prompt variations.
    For multiple wildcards, this is the Cartesian product.
    """
    if not prompt:
        return [prompt]

    matches = list(WILDCARD_RE.finditer(prompt))
    if not matches:
        return [prompt]

    # Extract all wildcard option lists
    wildcard_options = []
    for match in matches:
        options = [o.strip() for o in match.group(1).split("|") if o.strip()]
        if options:
            wildcard_options.append(options)

    # Generate all combinations (Cartesian product)
    from itertools import product
    combinations = list(product(*wildcard_options))

    results = []
    for combo in combinations:
        result = prompt
        for match, choice in zip(matches, combo):
            result = result[:match.start()] + choice + result[match.end():]
        results.append(result)

    return results


def has_wildcards(prompt: str) -> bool:
    """Check if a prompt contains any wildcards."""
    return WILDCARD_RE.search(prompt) is not None
