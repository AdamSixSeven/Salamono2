from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass
class EventDecision:
    event_type: str | None
    severity: str
    status: str
    confirmed: bool
    score: float
    reason: str


class LearnedEventController:
    """
    Alert stabilizer driven by learned probabilities and data quality.

    Fall/ground can work with a torso-dominant pose. Unstable gait needs enough
    lower-body evidence; otherwise the system returns insufficient_pose instead
    of guessing.

    There are three independent safety paths:

    * transition -> persistent ground state: confirmed fall,
    * persistent, very strong fall evidence in both model heads: confirmed fall,
    * persistent ground state without a visible transition: confirmed
      ``person_on_ground`` warning (it does not claim how the person got there).
    """

    def __init__(
        self,
        *,
        unstable_threshold: float = 0.72,
        unstable_confirm_seconds: float = 1.6,
        fall_threshold: float = 0.68,
        ground_threshold: float = 0.72,
        direct_fall_threshold: float = 0.88,
        direct_fall_action_threshold: float = 0.72,
        direct_fall_confirm_seconds: float = 0.8,
        fall_followup_seconds: float = 4.0,
        ground_confirm_seconds: float = 0.8,
        cooldown_seconds: float = 15.0,
        min_torso_quality_for_fall: float = 0.45,
        min_lower_body_quality_for_unstable: float = 0.35,
    ):
        self.unstable_threshold = unstable_threshold
        self.unstable_confirm_seconds = unstable_confirm_seconds
        self.fall_threshold = fall_threshold
        self.ground_threshold = ground_threshold
        self.direct_fall_threshold = direct_fall_threshold
        self.direct_fall_action_threshold = direct_fall_action_threshold
        self.direct_fall_confirm_seconds = direct_fall_confirm_seconds
        self.fall_followup_seconds = fall_followup_seconds
        self.ground_confirm_seconds = ground_confirm_seconds
        self.cooldown_seconds = cooldown_seconds
        self.min_torso_quality_for_fall = min_torso_quality_for_fall
        self.min_lower_body_quality_for_unstable = min_lower_body_quality_for_unstable
        self._history = defaultdict(lambda: deque(maxlen=80))
        self._fall_started: dict[Any, float] = {}
        self._last_alert: dict[tuple[Any, str], float] = {}

    def reset(self, track_id=None):
        if track_id is None:
            self._history.clear()
            self._fall_started.clear()
            self._last_alert.clear()
        else:
            self._history.pop(track_id, None)
            self._fall_started.pop(track_id, None)
            for key in list(self._last_alert):
                if key[0] == track_id:
                    self._last_alert.pop(key, None)

    @staticmethod
    def _duration_above(history, label: str, threshold: float, now: float) -> float:
        start = now
        found = False
        for timestamp, probs in reversed(history):
            if float(probs.get(label, 0.0)) < threshold:
                break
            start = timestamp
            found = True
        return max(0.0, now - start) if found else 0.0

    def _cooldown_ok(self, track_id, event_key: str, now: float) -> bool:
        return (
            now - self._last_alert.get((track_id, event_key), float("-inf"))
            >= self.cooldown_seconds
        )

    def update(self, track_id, timestamp: float, prediction) -> EventDecision:
        now = float(timestamp)
        probs = dict(prediction.safety_probabilities)
        action_probs = dict(
            getattr(
                prediction,
                "probabilities",
                getattr(prediction, "action_probabilities", {}),
            )
            or {}
        )
        history = self._history[track_id]
        history.append((now, probs))

        torso_quality = float(getattr(prediction, "torso_quality", prediction.valid_ratio))
        lower_body_quality = float(getattr(prediction, "lower_body_quality", prediction.valid_ratio))

        fall_p = float(probs.get("fall_transition", 0.0))
        ground_p = float(probs.get("ground_state", 0.0))
        unstable_p = float(probs.get("unstable_motion", 0.0))
        action_fall_p = float(action_probs.get("fall_down", 0.0))

        fall_evidence_ok = torso_quality >= self.min_torso_quality_for_fall
        unstable_evidence_ok = lower_body_quality >= self.min_lower_body_quality_for_unstable

        if fall_evidence_ok and fall_p >= self.fall_threshold:
            self._fall_started.setdefault(track_id, now)

        started = self._fall_started.get(track_id)
        if started is not None and now - started > self.fall_followup_seconds:
            self._fall_started.pop(track_id, None)
            started = None

        ground_duration = (
            self._duration_above(history, "ground_state", self.ground_threshold, now)
            if fall_evidence_ok
            else 0.0
        )

        # Preferred route: a visible fall transition followed by a stable
        # ground state. It has the clearest semantics and therefore wins over
        # the direct high-confidence fallback below.
        if started is not None and ground_duration >= self.ground_confirm_seconds:
            cooldown_ok = self._cooldown_ok(track_id, "fall", now)
            if cooldown_ok:
                self._last_alert[(track_id, "fall")] = now
                self._fall_started.pop(track_id, None)
                return EventDecision(
                    "fall_detected",
                    "DANGER",
                    "confirmed",
                    True,
                    max(fall_p, ground_p),
                    "learned fall transition followed by persistent ground state",
                )
            return EventDecision(
                "fall_detected",
                "DANGER",
                "cooldown",
                False,
                max(fall_p, ground_p),
                "fall sequence detected during cooldown",
            )

        # The baseline sometimes keeps the four-second window classified as a
        # transition while the person is already down. Confirm this only when
        # both heads independently agree and the signal persists long enough.
        direct_fall_duration = (
            self._duration_above(
                history,
                "fall_transition",
                self.direct_fall_threshold,
                now,
            )
            if fall_evidence_ok
            else 0.0
        )
        direct_fall_ready = (
            direct_fall_duration >= self.direct_fall_confirm_seconds
            and action_fall_p >= self.direct_fall_action_threshold
        )
        if direct_fall_ready:
            cooldown_ok = self._cooldown_ok(track_id, "fall", now)
            if cooldown_ok:
                self._last_alert[(track_id, "fall")] = now
                self._fall_started.pop(track_id, None)
                return EventDecision(
                    "fall_detected",
                    "DANGER",
                    "confirmed_direct_fall",
                    True,
                    max(fall_p, action_fall_p),
                    "persistent high-confidence fall in action and safety heads",
                )
            return EventDecision(
                "fall_detected",
                "DANGER",
                "cooldown",
                False,
                max(fall_p, action_fall_p),
                "direct fall detected during cooldown",
            )

        if started is not None:
            return EventDecision(
                "fall_suspected",
                "WARNING",
                "awaiting_ground_confirmation",
                False,
                fall_p,
                "learned fall transition; waiting for ground state or persistent dual-head confirmation",
            )

        # A camera can start after the fall or miss the transition. Persistent
        # ground state is still a real safety event, but it is labelled as such
        # instead of claiming a fall that was not observed.
        if ground_duration >= self.ground_confirm_seconds:
            cooldown_ok = self._cooldown_ok(track_id, "ground", now)
            if cooldown_ok:
                self._last_alert[(track_id, "ground")] = now
                return EventDecision(
                    "person_on_ground",
                    "DANGER",
                    "verification_required",
                    True,
                    ground_p,
                    "persistent learned ground state without observed transition",
                )
            return EventDecision(
                "person_on_ground",
                "DANGER",
                "cooldown",
                False,
                ground_p,
                "ground-state event detected during cooldown",
            )

        unstable_duration = (
            self._duration_above(history, "unstable_motion", self.unstable_threshold, now)
            if unstable_evidence_ok
            else 0.0
        )

        if unstable_duration >= self.unstable_confirm_seconds:
            cooldown_ok = self._cooldown_ok(track_id, "unstable", now)
            if cooldown_ok:
                self._last_alert[(track_id, "unstable")] = now
                return EventDecision(
                    "unstable_movement",
                    "WARNING",
                    "verification_required",
                    True,
                    unstable_p,
                    "persistent learned unstable motion; cause is not inferred",
                )
            return EventDecision(
                "unstable_movement",
                "WARNING",
                "cooldown",
                False,
                unstable_p,
                "unstable movement detected during cooldown",
            )

        if unstable_p >= self.unstable_threshold and not unstable_evidence_ok:
            return EventDecision(
                None,
                "INFO",
                "insufficient_lower_body_pose",
                False,
                unstable_p,
                "unstable gait suppressed because legs are not sufficiently visible",
            )

        if (
            max(fall_p, ground_p) >= min(self.fall_threshold, self.ground_threshold)
            and not fall_evidence_ok
        ):
            return EventDecision(
                None,
                "INFO",
                "insufficient_torso_pose",
                False,
                max(fall_p, ground_p),
                "fall/ground alert suppressed because torso evidence is insufficient",
            )

        return EventDecision(
            None,
            "INFO",
            "normal",
            False,
            max(fall_p, ground_p, unstable_p),
            "",
        )
