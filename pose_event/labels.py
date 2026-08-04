from __future__ import annotations

ACTION_CLASSES = [
    "fall_down",
    "lying_down",
    "sit_down",
    "sitting",
    "stand_up",
    "standing",
    "walking",
    "unstable_gait",
    "other",
]

SAFETY_CLASSES = [
    "normal",
    "unstable_motion",
    "fall_transition",
    "ground_state",
]

ACTION_TO_SAFETY = {
    "fall_down": "fall_transition",
    "lying_down": "ground_state",
    "sit_down": "normal",
    "sitting": "normal",
    "stand_up": "normal",
    "standing": "normal",
    "walking": "normal",
    "unstable_gait": "unstable_motion",
    "other": "normal",
}

ACTION_ALIASES = {
    "fall": "fall_down",
    "falling": "fall_down",
    "fallen": "lying_down",
    "lying": "lying_down",
    "lying down": "lying_down",
    "lie down": "lying_down",
    "sit down": "sit_down",
    "sitting down": "sit_down",
    "sit": "sitting",
    "stand up": "stand_up",
    "standing up": "stand_up",
    "stand": "standing",
    "walk": "walking",
    "normal walk": "walking",
    "staggering": "unstable_gait",
    "limping": "unstable_gait",
    "shuffle": "unstable_gait",
    "shuffling": "unstable_gait",
    "abnormal gait": "unstable_gait",
    "kneeling": "other",
    "hopping": "other",
    "pick up": "other",
    "pickup": "other",
    "drop": "other",
    "jump": "other",
}


def normalize_action(label: str) -> str:
    cleaned = " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())
    value = ACTION_ALIASES.get(cleaned, cleaned.replace(" ", "_"))
    if value not in ACTION_CLASSES:
        return "other"
    return value


def safety_for_action(action: str) -> str:
    return ACTION_TO_SAFETY.get(normalize_action(action), "normal")
