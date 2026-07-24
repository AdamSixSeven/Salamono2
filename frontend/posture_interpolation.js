/* Smooth, bounded MediaPipe landmark interpolation for the live overlay. */
(function (root, factory) {
    "use strict";

    const api = factory();
    if (typeof module === "object" && module.exports) {
        module.exports = api;
    }
    if (root) {
        root.PerimetrPostureInterpolation = api;
    }
})(typeof window !== "undefined" ? window : globalThis, function () {
    "use strict";

    const DEFAULT_DURATION_MS = 180;
    const MAX_LANDMARKS = 33;

    function cloneLandmarks(landmarks) {
        if (!Array.isArray(landmarks)) return [];
        return landmarks.slice(0, MAX_LANDMARKS).map(landmark => {
            if (!Array.isArray(landmark)) return [];
            return landmark.slice(0, 4).map(value => Number(value));
        });
    }

    function sameLandmarks(left, right) {
        if (!Array.isArray(left) || !Array.isArray(right) ||
            left.length !== right.length) return false;
        for (let index = 0; index < left.length; index += 1) {
            const a = left[index];
            const b = right[index];
            if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
                return false;
            }
            for (let component = 0; component < a.length; component += 1) {
                const av = Number(a[component]);
                const bv = Number(b[component]);
                if (Number.isFinite(av) && Number.isFinite(bv)) {
                    if (Math.abs(av - bv) > 1e-7) return false;
                } else if (!(Number.isNaN(av) && Number.isNaN(bv))) {
                    return false;
                }
            }
        }
        return true;
    }

    function smoothStep(progress) {
        const t = Math.max(0, Math.min(1, Number(progress) || 0));
        return t * t * (3 - 2 * t);
    }

    function interpolateLandmarks(previous, current, progress) {
        const eased = smoothStep(progress);
        const target = cloneLandmarks(current);
        const source = cloneLandmarks(previous);
        return target.map((landmark, index) => {
            const before = source[index];
            if (!Array.isArray(before) || before.length !== landmark.length) {
                return landmark.slice();
            }
            return landmark.map((value, component) => {
                const from = Number(before[component]);
                const to = Number(value);
                if (!Number.isFinite(from) || !Number.isFinite(to)) return to;
                return from + (to - from) * eased;
            });
        });
    }

    function assessmentToken(assessment, fallbackToken) {
        if (assessment && assessment.timestamp !== undefined &&
            assessment.timestamp !== null) {
            return "assessment:" + String(assessment.timestamp);
        }
        if (fallbackToken !== undefined && fallbackToken !== null) {
            return "frame:" + String(fallbackToken);
        }
        return null;
    }

    class TrackInterpolator {
        constructor(options) {
            const requested = options && Number(options.durationMs);
            this.durationMs = Number.isFinite(requested) && requested > 0
                ? requested : DEFAULT_DURATION_MS;
            this.tracks = new Map();
        }

        update(assessments, nowMs, frameToken) {
            const now = Number(nowMs) || 0;
            const seen = new Set();
            (Array.isArray(assessments) ? assessments : []).forEach(assessment => {
                if (!assessment || assessment.track_id === undefined ||
                    assessment.track_id === null) return;
                const target = cloneLandmarks(assessment.pose_landmarks);
                if (!target.length) return;

                const trackId = String(assessment.track_id);
                const token = assessmentToken(assessment, frameToken);
                const existing = this.tracks.get(trackId);
                seen.add(trackId);

                // app.js emits the same payload before and after image decode.
                // Do not reset a running transition for that duplicate event.
                if (existing && token !== null && existing.token === token &&
                    sameLandmarks(existing.current, target)) {
                    return;
                }
                if (existing && sameLandmarks(existing.current, target)) {
                    existing.token = token;
                    return;
                }

                const previous = existing
                    ? this._sampleState(existing, now)
                    : cloneLandmarks(target);
                this.tracks.set(trackId, {
                    previous: previous,
                    current: cloneLandmarks(target),
                    startedAt: existing ? now : now - this.durationMs,
                    token: token,
                });
            });

            Array.from(this.tracks.keys()).forEach(trackId => {
                if (!seen.has(trackId)) this.tracks.delete(trackId);
            });
            return this.isAnimating(now);
        }

        sample(trackId, nowMs) {
            const state = this.tracks.get(String(trackId));
            if (!state) return null;
            return this._sampleState(state, Number(nowMs) || 0);
        }

        isAnimating(nowMs) {
            const now = Number(nowMs) || 0;
            for (const state of this.tracks.values()) {
                if (!sameLandmarks(state.previous, state.current) &&
                    now - state.startedAt < this.durationMs) {
                    return true;
                }
            }
            return false;
        }

        clear() {
            this.tracks.clear();
        }

        get size() {
            return this.tracks.size;
        }

        _sampleState(state, now) {
            const progress = (now - state.startedAt) / this.durationMs;
            if (progress >= 1) return cloneLandmarks(state.current);
            if (progress <= 0) return cloneLandmarks(state.previous);
            return interpolateLandmarks(state.previous, state.current, progress);
        }
    }

    return {
        DEFAULT_DURATION_MS,
        TrackInterpolator,
        cloneLandmarks,
        interpolateLandmarks,
        smoothStep,
    };
});
