(function () {
    "use strict";


    const State = {
        wsStatus: "connecting",
        stats: {
            fps: 0,
            detectionMs: 0,
            frameCount: 0,
            overlayMs: 0,
            overlayAverageMs: 0,
            overlaySamples: 0,
        },
        mode: "site",           // "site" | "checkpoint"
        layers: loadLayers(),   // { boxes, posture, zones, markers, distances }
        activeAlarm: null,      // { title, meta, ts, source }
        alarmMuteUntil: 0,
        alertsCount: 0,
        alertsFilter: "",       // "" | "DANGER" | "WARNING"
        lastMarkerCount: 0,
        lastCalibrationActive: false,
        lastCameraSize: "—",
        cameraId: loadCameraId(),
    };

    function loadCameraId() {
        const fromUrl = new URLSearchParams(location.search).get("camera_id");
        if (fromUrl) return fromUrl;
        try { return localStorage.getItem("perimetr:camera_id") || "cam_default"; }
        catch (_) { return "cam_default"; }
    }

    function loadLayers() {
        try {
            const raw = localStorage.getItem("perimetr:layers");
            if (raw) {
                const parsed = JSON.parse(raw);
                return {
                    boxes: parsed.boxes !== false,
                    posture: parsed.posture !== false,
                    zones: parsed.zones !== false,
                    markers: parsed.markers !== false,
                    distances: parsed.distances !== false,
                    worker_id: parsed.worker_id !== false,
                };
            }
        } catch (_) { /* ignore */ }
        return {
            boxes: true,
            posture: true,
            zones: true,
            markers: true,
            distances: true,
            worker_id: true,
        };
    }
    function saveLayers() {
        try { localStorage.setItem("perimetr:layers", JSON.stringify(State.layers)); } catch (_) { }
    }


    const canvas = document.getElementById("liveCanvas");
    const ctx = canvas.getContext("2d");
    const noSignal = document.getElementById("noSignal");
    const wsStatusEl = document.getElementById("wsStatus");
    const connText = document.getElementById("connectionText");
    const connWrap = document.getElementById("connectionStatus");
    const frameCountEl = document.getElementById("frameCount");
    const processingTimeEl = document.getElementById("processingTime");
    const fpsLine = document.getElementById("fpsLine");
    const camMeta = document.getElementById("camMeta");
    const liveText = document.getElementById("liveText");
    const liveChip = document.getElementById("liveChip");
    const markerChip = document.getElementById("markerChip");
    const videoEl = document.getElementById("canvasStack");
    const videoStatus = document.getElementById("videoStatus");
    const alertList = document.getElementById("alertList");
    const alertsEmpty = document.getElementById("alertsEmpty");
    const alertTotal = document.getElementById("alertTotal");
    const alarmsFilters = document.getElementById("alarmsFilters");
    const alarmBanner = document.getElementById("alarmBanner");
    const alarmTitle = document.getElementById("alarmTitle");
    const alarmMeta = document.getElementById("alarmMeta");
    const alarmAckBtn = document.getElementById("alarmAckBtn");
    const alarmMuteBtn = document.getElementById("alarmMuteBtn");
    const headerAlarmTag = document.getElementById("headerAlarmTag");
    const snapshotBtn = document.getElementById("snapshotBtn");
    const pairPhoneBtn = document.getElementById("pairPhoneBtn");
    const addCameraBtn = document.getElementById("addCameraBtn");
    const demoVideoBtn = document.getElementById("demoVideoBtn");
    const demoVideoStopBtn = document.getElementById("demoVideoStopBtn");
    const demoVideoChangeBtn = document.getElementById("demoVideoChangeBtn");
    const demoVideoInput = document.getElementById("demoVideoInput");
    const demoVideoStatusPanel = document.getElementById("demoVideoStatusPanel");
    const demoVideoStatusText = document.getElementById("demoVideoStatusText");
    const demoVideoStatusMeta = document.getElementById("demoVideoStatusMeta");
    const demoVideoProgress = document.getElementById("demoVideoProgress");
    const modeSegmented = document.getElementById("modeSegmented");
    const aiModeBtn = document.getElementById("aiModeBtn");
    const ppeBadge = document.getElementById("ppeBadge");
    const ppeHeadline = document.getElementById("ppeHeadline");
    const ppeHardhat = document.getElementById("ppeHardhat");
    const ppeVest = document.getElementById("ppeVest");
    const postureBadge = document.getElementById("postureBadge");
    const postureScore = document.getElementById("postureScore");
    const postureSignals = document.getElementById("postureSignals");
    const postureStatus = document.getElementById("postureStatus");
    const workerBadge = document.getElementById("workerBadge");
    const workerIds = document.getElementById("workerIds");
    const workerMeta = document.getElementById("workerMeta");
    const workerStatus = document.getElementById("workerStatus");
    const workerQrInput = document.getElementById("workerQrInput");
    const workerQrPreview = document.getElementById("workerQrPreview");
    const workerQrPreviewImg = document.getElementById("workerQrPreviewImg");
    const workerQrPreviewTitle = document.getElementById("workerQrPreviewTitle");
    const workerQrPreviewPayload = document.getElementById("workerQrPreviewPayload");
    const workerQrDownloadBtn = document.getElementById("workerQrDownloadBtn");
    const workerQrOpenBtn = document.getElementById("workerQrOpenBtn");
    const workerRegistryForm = document.getElementById("workerRegistryForm");
    const workerRegistrySelect = document.getElementById("workerRegistrySelect");
    const workerRegistryCount = document.getElementById("workerRegistryCount");
    const workerRegistryStatus = document.getElementById("workerRegistryStatus");
    const workerFirstName = document.getElementById("workerFirstName");
    const workerLastName = document.getElementById("workerLastName");
    const workerPosition = document.getElementById("workerPosition");
    const workerDepartment = document.getElementById("workerDepartment");
    const workerSaveBtn = document.getElementById("workerSaveBtn");
    const workerDeleteBtn = document.getElementById("workerDeleteBtn");
    const workerNewBtn = document.getElementById("workerNewBtn");
    const workerToolsCard = document.getElementById("workerToolsCard");
    const workerToolsToggle = document.getElementById("workerToolsToggle");
    const workerToolsBody = document.getElementById("workerToolsBody");
    const layerTags = document.querySelectorAll("[data-layer]");
    const cameraSelect = document.getElementById("cameraSelect");

    window.Perimetr = window.Perimetr || {};
    window.Perimetr.getCameraId = () => State.cameraId;
    window.Perimetr.getLayers = () => ({ ...State.layers });

    let runtimeSyncTimer = null;
    let runtimeSyncDesiredRevision = 0;
    let runtimeSyncReadyRevision = 0;
    let runtimeSyncCompletedRevision = 0;
    let runtimeSyncRunning = false;
    const runtimeLayerKeys = [
        "boxes",
        "posture",
        "zones",
        "markers",
        "distances",
        "worker_id",
    ];
    const runtimeSyncDebug = {
        status: "idle",
        requested_revision: 0,
        completed_revision: 0,
        camera_id: State.cameraId,
        mode: State.mode,
        detector_required: null,
        last_error: null,
        recovered: false,
    };
    let overlayDebug = {
        overlay_ms: 0,
        source: null,
        camera_id: State.cameraId,
        job_id: null,
        run_id: null,
        frame_index: null,
        frame_id: null,
        timestamp: null,
    };

    window.Perimetr.getDebugState = () => ({
        cameraId: State.cameraId,
        mode: State.mode,
        wsStatus: State.wsStatus,
        stats: { ...State.stats },
        overlay: { ...overlayDebug },
        runtimeSync: { ...runtimeSyncDebug },
    });
    window.Perimetr.reportOverlayMetric = metric => {
        const overlayMs = Number(metric && metric.overlay_ms);
        if (!Number.isFinite(overlayMs) || overlayMs < 0) return;
        const rounded = Math.round(overlayMs * 1000) / 1000;
        State.stats.overlayMs = rounded;
        State.stats.overlaySamples += 1;
        const alpha = 0.15;
        State.stats.overlayAverageMs = State.stats.overlaySamples === 1
            ? rounded
            : State.stats.overlayAverageMs * (1 - alpha) + rounded * alpha;
        overlayDebug = {
            ...overlayDebug,
            ...(metric || {}),
            overlay_ms: rounded,
        };
        fpsLine.dataset.overlayMs = rounded.toFixed(3);
        fpsLine.title = "Overlay ostatniej klatki: " + rounded.toFixed(2) + " ms";
    };

    function runtimeLayerPayload() {
        return {
            boxes: !!State.layers.boxes,
            posture: !!State.layers.posture,
            zones: !!State.layers.zones,
            markers: !!State.layers.markers,
            distances: !!State.layers.distances,
            worker_id: !!State.layers.worker_id,
        };
    }

    function runtimeContext() {
        return {
            cameraId: State.cameraId,
            mode: State.mode,
        };
    }

    function runtimeContextMatches(context) {
        return context.cameraId === State.cameraId && context.mode === State.mode;
    }

    function runtimeUrl(context) {
        return "/api/runtime/" + encodeURIComponent(context.cameraId) +
            "?mode=" + encodeURIComponent(context.mode);
    }

    function runtimeErrorMessage(error) {
        if (error && error.message) return String(error.message);
        return String(error || "Nieznany błąd synchronizacji");
    }

    function setRuntimeSyncStatus(status, details) {
        const info = details || {};
        const detectorRequired = typeof info.detectorRequired === "boolean"
            ? info.detectorRequired
            : runtimeSyncDebug.detector_required;
        const errorMessage = info.error ? runtimeErrorMessage(info.error) : null;
        Object.assign(runtimeSyncDebug, {
            status: status,
            requested_revision: runtimeSyncDesiredRevision,
            completed_revision: runtimeSyncCompletedRevision,
            camera_id: State.cameraId,
            mode: State.mode,
            detector_required: detectorRequired,
            last_error: errorMessage || info.lastError || null,
            recovered: !!info.recovered,
        });

        let suffix = "Ustawienia lokalne";
        if (status === "pending") suffix = "Oczekiwanie na zapis ustawień…";
        else if (status === "syncing") suffix = "Zapisywanie ustawień…";
        else if (status === "error") suffix = "Błąd synchronizacji: " + errorMessage;
        else if (typeof detectorRequired === "boolean") {
            suffix = detectorRequired ? "YOLO aktywne" : "YOLO zatrzymane";
            if (info.recovered) suffix += " · stan odczytany ponownie z serwera";
        }

        layerTags.forEach(tag => {
            if (tag.dataset.baseTitle === undefined) {
                tag.dataset.baseTitle = tag.title || "";
            }
            tag.dataset.runtimeSync = status;
            tag.setAttribute(
                "aria-busy",
                status === "pending" || status === "syncing" ? "true" : "false",
            );
            if (status === "error") {
                tag.setAttribute("aria-invalid", "true");
                tag.style.boxShadow = "0 0 0 2px var(--msbp-red)";
            } else {
                tag.removeAttribute("aria-invalid");
                tag.style.boxShadow = "";
            }
            tag.title = (tag.dataset.baseTitle ? tag.dataset.baseTitle + " · " : "") + suffix;
        });
    }

    function applyRuntimeOptions(options) {
        if (!options || typeof options !== "object") {
            throw new Error("Serwer nie zwrócił ustawień modułów");
        }
        const next = { ...State.layers };
        runtimeLayerKeys.forEach(key => {
            if (typeof options[key] !== "boolean") {
                throw new Error("Niepełna odpowiedź ustawień modułów: " + key);
            }
            next[key] = options[key];
        });
        const changed = runtimeLayerKeys.some(key => next[key] !== State.layers[key]);
        if (!changed) return;
        State.layers = next;
        saveLayers();
        syncLayerButtons();
        document.dispatchEvent(new CustomEvent("perimetr-layers", { detail: State.layers }));
    }

    async function requestRuntimeState(context, method, options) {
        const requestOptions = {
            method: method,
            credentials: "same-origin",
        };
        if (method === "PATCH") {
            requestOptions.headers = { "Content-Type": "application/json" };
            requestOptions.body = JSON.stringify(options);
        }
        const response = await fetch(runtimeUrl(context), requestOptions);
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
    }

    async function reconcileRuntimeState(context, revision, writeError) {
        try {
            const payload = await requestRuntimeState(context, "GET");
            if (
                revision === runtimeSyncDesiredRevision &&
                runtimeContextMatches(context)
            ) {
                applyRuntimeOptions(payload.options);
                setRuntimeSyncStatus("synced", {
                    detectorRequired: !!payload.detector_required,
                    lastError: runtimeErrorMessage(writeError),
                    recovered: true,
                });
            }
        } catch (readError) {
            if (
                revision === runtimeSyncDesiredRevision &&
                runtimeContextMatches(context)
            ) {
                setRuntimeSyncStatus("error", {
                    error: new Error(
                        runtimeErrorMessage(writeError) +
                        "; odczyt stanu serwera również się nie udał: " +
                        runtimeErrorMessage(readError),
                    ),
                });
            }
        }
    }

    async function flushRuntimeProcessing() {
        if (runtimeSyncRunning) return;
        runtimeSyncRunning = true;
        try {
            while (runtimeSyncCompletedRevision < runtimeSyncReadyRevision) {
                const revision = runtimeSyncReadyRevision;
                const context = runtimeContext();
                const requestedOptions = runtimeLayerPayload();
                if (
                    revision === runtimeSyncDesiredRevision &&
                    runtimeContextMatches(context)
                ) {
                    setRuntimeSyncStatus("syncing");
                }
                try {
                    const payload = await requestRuntimeState(
                        context,
                        "PATCH",
                        requestedOptions,
                    );
                    runtimeSyncCompletedRevision = Math.max(
                        runtimeSyncCompletedRevision,
                        revision,
                    );
                    if (
                        revision === runtimeSyncDesiredRevision &&
                        runtimeContextMatches(context)
                    ) {
                        applyRuntimeOptions(payload.options);
                        setRuntimeSyncStatus("synced", {
                            detectorRequired: !!payload.detector_required,
                        });
                    }
                } catch (error) {
                    runtimeSyncCompletedRevision = Math.max(
                        runtimeSyncCompletedRevision,
                        revision,
                    );
                    console.error("Nie udało się zapisać ustawień modułów.", error);
                    if (
                        revision === runtimeSyncDesiredRevision &&
                        runtimeContextMatches(context)
                    ) {
                        await reconcileRuntimeState(context, revision, error);
                    }
                }
            }
        } finally {
            runtimeSyncRunning = false;
            if (runtimeSyncCompletedRevision < runtimeSyncReadyRevision) {
                void flushRuntimeProcessing();
            }
        }
    }

    function syncRuntimeProcessing(delayMs) {
        clearTimeout(runtimeSyncTimer);
        const revision = ++runtimeSyncDesiredRevision;
        setRuntimeSyncStatus("pending");
        runtimeSyncTimer = setTimeout(() => {
            runtimeSyncTimer = null;
            runtimeSyncReadyRevision = Math.max(runtimeSyncReadyRevision, revision);
            void flushRuntimeProcessing();
        }, Math.max(0, Number(delayMs) || 0));
    }


    function syncLayerButtons() {
        layerTags.forEach(el => {
            const on = !!State.layers[el.dataset.layer];
            el.classList.toggle("msbp-tag--solid", on);
            el.classList.toggle("msbp-tag--outline", !on);
            if (!on) {
                el.style.color = "var(--ink-3)";
                el.style.borderColor = "var(--border-2)";
            } else {
                el.style.color = "";
                el.style.borderColor = "";
            }
        });
    }
    layerTags.forEach(el => {
        el.addEventListener("click", () => {
            const key = el.dataset.layer;
            State.layers[key] = !State.layers[key];
            saveLayers();
            syncLayerButtons();
            document.dispatchEvent(new CustomEvent("perimetr-layers", { detail: State.layers }));
            syncRuntimeProcessing(60);
        });
    });
    syncLayerButtons();


    modeSegmented.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => {
            State.mode = btn.dataset.mode;
            modeSegmented.querySelectorAll("button").forEach(b => {
                b.classList.toggle("active", b === btn);
            });
            if (State.mode !== "checkpoint") ppeBadge.classList.add("hidden");
            document.dispatchEvent(new CustomEvent("perimetr-mode", { detail: State.mode }));
            syncRuntimeProcessing(60);
        });
    });


    alarmsFilters.querySelectorAll("[data-filter]").forEach(el => {
        el.addEventListener("click", () => {
            State.alertsFilter = el.dataset.filter;
            alarmsFilters.querySelectorAll("[data-filter]").forEach(x => {
                const on = x === el;
                x.classList.toggle("msbp-tag--solid", on);
                x.classList.toggle("msbp-tag--outline", !on);
                x.classList.toggle("active", on);
                if (!on) {
                    x.style.color = "var(--ink-3)";
                    x.style.borderColor = "var(--border-2)";
                } else {
                    x.style.color = "";
                    x.style.borderColor = "";
                }
            });
            filterAlertsUI();
        });
    });

    function filterAlertsUI() {
        const rows = alertList.querySelectorAll(".px-alert-row");
        let visible = 0;
        rows.forEach(row => {
            const sev = row.dataset.severity;
            const show = !State.alertsFilter || sev === State.alertsFilter;
            row.style.display = show ? "" : "none";
            if (show) visible++;
        });
        alertsEmpty.style.display = visible === 0 ? "" : "none";
    }


    alarmAckBtn.addEventListener("click", async () => {
        const recordId = State.activeAlarm && State.activeAlarm.recordId;
        if (recordId) {
            try {
                await fetch("/api/alerts/" + encodeURIComponent(recordId) + "/review", {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    credentials: "same-origin",
                    body: JSON.stringify({
                        status: "acknowledged",
                        reviewed_by: "panel-live",
                        note: "Potwierdzono odbiór alarmu w panelu LIVE",
                    }),
                });
            } catch (_) {
            }
        }
        clearAlarm();
    });
    alarmMuteBtn.addEventListener("click", () => {
        State.alarmMuteUntil = Date.now() + 5 * 60 * 1000;
        clearAlarm();
    });

    function raiseAlarm(kind, description, meta, recordId) {
        if (Date.now() < State.alarmMuteUntil) return;
        State.activeAlarm = { kind, description, meta, recordId: recordId || null, ts: Date.now() };
        alarmTitle.textContent = description || "STOP — Naruszenie";
        alarmMeta.textContent = meta;
        alarmBanner.classList.remove("hidden");
        headerAlarmTag.classList.remove("hidden");
        videoEl.classList.add("alarm");
    }
    function clearAlarm() {
        State.activeAlarm = null;
        alarmBanner.classList.add("hidden");
        headerAlarmTag.classList.add("hidden");
        videoEl.classList.remove("alarm");
    }


    snapshotBtn.addEventListener("click", () => {
        if (!canvas.width || !canvas.height) return;
        const url = canvas.toDataURL("image/png");
        const a = document.createElement("a");
        a.href = url;
        const ts = new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");
        a.download = `perimetr-${ts}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });


    let demoState = "idle";
    let demoJob = null;
    let demoJobId = null;
    let demoCameraId = null;
    let demoRunId = null;
    let demoLastFrameIndex = -1;
    let demoGeneration = 0;
    let demoUploadXhr = null;
    let demoPollTimer = null;
    let demoStatusController = null;
    let demoControlController = null;
    let demoLoadingTimer = null;
    let demoActionPending = false;
    let demoPollFailures = 0;
    let demoReturnCameraId = null;
    let demoDownloadedOutputUrl = null;
    let demoPageDisposed = false;

    // Kolejka analizy filmów uruchamiana zielonym przyciskiem.
    let demoPickerPurpose = "single";
    let demoBatchActive = false;
    let demoBatchCancelled = false;
    let demoBatchFiles = [];
    let demoBatchIndex = -1;
    let demoBatchDone = 0;
    let demoBatchFailed = [];
    let demoBatchFinalMessage = "";

    const DEMO_CAMERA_ID = "demo_upload";
    const DEMO_SESSION_STORAGE_KEY = "perimetr:demo-video-session";
    const DEMO_STATUS_POLL_MS = 600;
    const DEMO_PREPARE_TIMEOUT_MS = 45_000;
    const DEMO_UPLOAD_TIMEOUT_MS = 30 * 60 * 1000;
    const DEMO_BATCH_PREPARE_TIMEOUT_MS = 3 * 60 * 1000;
    const DEMO_BATCH_EXPORT_TIMEOUT_MS = 6 * 60 * 60 * 1000;
    const DEMO_BATCH_DOWNLOAD_GRACE_MS = 2500;
    const DEMO_BACKEND_STATES = new Set([
        "uploaded", "loading", "ready", "playing", "paused", "finished",
        "stopped", "failed", "deleted",
    ]);

    function loadPersistedDemoSession() {
        try {
            const raw = localStorage.getItem(DEMO_SESSION_STORAGE_KEY);
            if (!raw) return null;
            const parsed = JSON.parse(raw);
            const jobId = String(parsed && parsed.job_id || "").trim();
            if (!jobId) {
                localStorage.removeItem(DEMO_SESSION_STORAGE_KEY);
                return null;
            }
            return {
                jobId: jobId,
                cameraId: String(parsed.camera_id || DEMO_CAMERA_ID),
                returnCameraId: parsed.return_camera_id
                    ? String(parsed.return_camera_id)
                    : null,
            };
        } catch (_) {
            return null;
        }
    }

    function persistDemoSession() {
        if (!demoJobId) return;
        try {
            localStorage.setItem(DEMO_SESSION_STORAGE_KEY, JSON.stringify({
                job_id: demoJobId,
                camera_id: demoCameraId || DEMO_CAMERA_ID,
                return_camera_id: demoReturnCameraId,
            }));
        } catch (_) { /* localStorage can be unavailable in privacy mode */ }
    }

    function clearPersistedDemoSession(jobId) {
        try {
            const persisted = loadPersistedDemoSession();
            if (jobId && persisted && persisted.jobId !== String(jobId)) return;
            localStorage.removeItem(DEMO_SESSION_STORAGE_KEY);
        } catch (_) { /* localStorage can be unavailable in privacy mode */ }
    }

    function demoControlPresentation(state) {
        const primaryLabels = {
            idle: "Wybierz film",
            uploading: "Wysyłanie…",
            uploaded: "Przygotowywanie…",
            loading: "Przygotowywanie…",
            ready: "Odtwórz",
            playing: "Pauza",
            paused: "Wznów",
            finished: "Odtwórz ponownie",
            stopped: "Odtwórz",
            failed: "Wybierz film",
            deleted: "Wybierz film",
        };
        return {
            primaryLabel: primaryLabels[state] || "Wybierz film",
            primaryDisabled: ["uploading", "uploaded", "loading"].includes(state),
            showStop: !!demoJobId && ["loading", "ready", "playing", "paused"].includes(state),
            showChange: state !== "idle" && state !== "deleted",
        };
    }

    function formatDemoBytes(value) {
        const bytes = Number(value);
        if (!Number.isFinite(bytes) || bytes < 0) return "";
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function formatDemoTime(value) {
        const seconds = Number(value);
        if (!Number.isFinite(seconds) || seconds < 0) return "";
        const whole = Math.floor(seconds);
        const minutes = Math.floor(whole / 60);
        const rest = String(whole % 60).padStart(2, "0");
        return minutes + ":" + rest;
    }

    function demoStatusMessage(state, job) {
        if (demoBatchFinalMessage && !demoBatchActive) {
            return demoBatchFinalMessage;
        }
        const batchPrefix = demoBatchActive && demoBatchFiles.length
            ? "AI " + (demoBatchIndex + 1) + "/" + demoBatchFiles.length + " · "
            : "";
        if (job && job.export_status === "processing") {
            return batchPrefix + "Pełna analiza AI i zapis filmu…";
        }
        if (job && job.export_status === "ready") {
            return "Film z analizą AI zapisany";
        }
        const messages = {
            idle: "Wybierz plik MP4, AVI, MOV lub MKV",
            uploading: "Wysyłanie: " + Math.round(Number(job.upload_percent) || 0) + "%",
            uploaded: "Zapisywanie pliku…",
            loading: "Odczytywanie informacji o filmie…",
            ready: "Gotowy",
            playing: "Odtwarzanie",
            paused: "Pauza",
            finished: "Zakończono",
            stopped: "Zatrzymano",
            failed: "Błąd: " + (job.error || "Nie udało się przygotować filmu"),
            deleted: "Usunięto",
        };
        return batchPrefix + (messages[state] || messages.failed);
    }

    function demoStatusMeta(job) {
        const parts = [];
        if (job.filename) parts.push(job.filename);
        const size = formatDemoBytes(job.size_bytes);
        if (size) parts.push(size);
        const current = formatDemoTime(job.current_time_sec ?? job.source_time_sec);
        const duration = formatDemoTime(job.duration_sec);
        if (current || duration) parts.push((current || "0:00") + (duration ? " / " + duration : ""));
        const processingFps = Number(job.processing_fps);
        if (Number.isFinite(processingFps) && processingFps > 0) {
            parts.push(processingFps.toFixed(1) + " kl/s");
        }
        return parts.join(" · ");
    }

    function updateDemoControls(state, job) {
        demoState = state;
        demoJob = job || demoJob || {};
        const presentation = demoControlPresentation(state);

        demoVideoBtn.textContent = presentation.primaryLabel;
        demoVideoBtn.disabled = presentation.primaryDisabled || demoActionPending;
        demoVideoBtn.classList.toggle(
            "px-btn--solid",
            ["ready", "playing", "paused", "finished", "stopped"].includes(state),
        );
        demoVideoStopBtn.classList.toggle("hidden", !presentation.showStop);
        demoVideoStopBtn.disabled = demoActionPending;
        demoVideoChangeBtn.classList.toggle("hidden", !presentation.showChange);
        demoVideoChangeBtn.disabled = false;
        const exportRunning = job && job.export_status === "processing";
        if (demoBatchActive) {
            aiModeBtn.disabled = false;
            aiModeBtn.classList.add("is-processing");
            aiModeBtn.title = "Kolejka AI " + (demoBatchIndex + 1) + "/" +
                demoBatchFiles.length + " — kliknij, aby zatrzymać po bieżącym filmie";
        } else {
            aiModeBtn.disabled = demoActionPending || exportRunning;
            aiModeBtn.classList.toggle("is-processing", !!exportRunning);
            aiModeBtn.title = exportRunning
                ? "Trwa pełna analiza AI i zapis filmu"
                : "Wybierz jeden lub wiele filmów do pełnej analizy AI";
        }

        demoVideoStatusPanel.classList.toggle("hidden", state === "idle");
        demoVideoStatusPanel.classList.toggle("is-error", state === "failed");
        demoVideoStatusText.textContent = demoStatusMessage(state, demoJob);
        demoVideoStatusMeta.textContent = demoStatusMeta(demoJob);
        demoVideoProgress.classList.toggle("hidden", state !== "uploading");
        demoVideoProgress.value = Math.max(
            0,
            Math.min(100, Number(demoJob.upload_percent) || 0),
        );
        demoVideoProgress.textContent = Math.round(demoVideoProgress.value) + "%";
    }

    function clearDemoPreparationWatchdog() {
        if (demoLoadingTimer !== null) {
            clearTimeout(demoLoadingTimer);
            demoLoadingTimer = null;
        }
    }

    function clearDemoPollTimer() {
        if (demoPollTimer !== null) {
            clearTimeout(demoPollTimer);
            demoPollTimer = null;
        }
    }

    function abortDemoController(controller) {
        if (!controller) return;
        try { controller.abort(); } catch (_) {}
    }

    function cancelDemoRequests() {
        clearDemoPollTimer();
        clearDemoPreparationWatchdog();
        abortDemoController(demoStatusController);
        abortDemoController(demoControlController);
        demoStatusController = null;
        demoControlController = null;
        if (demoUploadXhr) {
            try { demoUploadXhr.abort(); } catch (_) {}
            demoUploadXhr = null;
        }
        demoActionPending = false;
    }

    function resetDemoFrameAcceptance() {
        demoRunId = null;
        demoLastFrameIndex = -1;
        lastRenderedTs = 0;
        frameRenderSequence += 1;
        pendingBinaryFrames.length = 0;
        document.dispatchEvent(new CustomEvent("perimetr-demo-run-changed", {
            detail: { job_id: demoJobId, run_id: null },
        }));
    }

    function activateDemoRun(runId, reconnect, force) {
        const nextRunId = runId === null || runId === undefined ? null : String(runId);
        if (!force && nextRunId === demoRunId) return false;
        demoRunId = nextRunId;
        demoLastFrameIndex = -1;
        lastRenderedTs = 0;
        frameRenderSequence += 1;
        pendingBinaryFrames.length = 0;
        document.dispatchEvent(new CustomEvent("perimetr-demo-run-changed", {
            detail: { job_id: demoJobId, run_id: demoRunId },
        }));
        if (reconnect && demoCameraId && State.cameraId === demoCameraId) {
            connectWebSocket();
        }
        return true;
    }

    function markDemoFailed(message, generation) {
        if (generation !== undefined && generation !== demoGeneration) return;
        clearDemoPollTimer();
        clearDemoPreparationWatchdog();
        demoActionPending = false;
        demoJob = Object.assign({}, demoJob || {}, {
            job_id: demoJobId,
            status: "failed",
            error: String(message || "Nieznany błąd"),
        });
        updateDemoControls("failed", demoJob);
    }

    function startDemoPreparationWatchdog(generation) {
        if (demoPageDisposed || demoLoadingTimer !== null) return;
        demoLoadingTimer = setTimeout(() => {
            demoLoadingTimer = null;
            if (generation !== demoGeneration) return;
            if (!["uploaded", "loading"].includes(demoState)) return;
            markDemoFailed(
                "Backend nie przygotował filmu w ciągu " +
                    Math.round(DEMO_PREPARE_TIMEOUT_MS / 1000) + " s.",
                generation,
            );
        }, DEMO_PREPARE_TIMEOUT_MS);
    }

    function normalizeDemoBackendState(value) {
        const state = String(value || "").toLowerCase();
        return DEMO_BACKEND_STATES.has(state) ? state : "failed";
    }

    function shouldPollDemoStatus(state) {
        // Eksport może kończyć się już po zakończeniu odtwarzania źródła.
        return ["uploaded", "loading", "playing"].includes(state) ||
            (!!demoJob && demoJob.export_status === "processing");
    }

    function applyDemoSnapshot(snapshot, generation, options) {
        if (generation !== demoGeneration || !snapshot) return false;
        if (snapshot.job_id && demoJobId && String(snapshot.job_id) !== demoJobId) return false;

        const nextState = normalizeDemoBackendState(snapshot.status);
        demoJob = Object.assign({}, demoJob || {}, snapshot, { status: nextState });
        if (snapshot.camera_id) demoCameraId = String(snapshot.camera_id);

        const reconnectOnRunChange = !(options && options.reconnectOnRunChange === false);
        if (snapshot.run_id !== undefined && snapshot.run_id !== null) {
            activateDemoRun(snapshot.run_id, reconnectOnRunChange);
        }

        if (["uploaded", "loading"].includes(nextState)) {
            startDemoPreparationWatchdog(generation);
        } else {
            clearDemoPreparationWatchdog();
        }
        updateDemoControls(nextState, demoJob);
        if (
            snapshot.export_status === "ready" && snapshot.output_url &&
            demoDownloadedOutputUrl !== snapshot.output_url
        ) {
            demoDownloadedOutputUrl = snapshot.output_url;
            const download = document.createElement("a");
            download.href = snapshot.output_url;
            download.download = snapshot.output_filename || "perimetr_ai.mp4";
            document.body.appendChild(download);
            download.click();
            download.remove();
        }
        return true;
    }

    async function requestDemoExport() {
        if (!demoJobId) {
            if (!demoBatchActive) openDemoAiBatchPicker();
            return;
        }
        if (!["ready", "finished", "stopped"].includes(demoState)) return;
        demoActionPending = true;
        demoDownloadedOutputUrl = null;
        updateDemoControls(demoState, demoJob);
        try {
            const response = await fetch(
                "/api/demo-videos/" + encodeURIComponent(demoJobId) + "/export",
                { method: "POST", credentials: "same-origin" },
            );
            if (!response.ok) {
                const raw = await response.text();
                throw new Error(demoResponseError(raw, "Nie udało się uruchomić eksportu AI"));
            }
            const snapshot = await response.json();
            applyDemoSnapshot(snapshot, demoGeneration);
            scheduleDemoStatusPoll(0, demoGeneration);
        } catch (err) {
            markDemoFailed("Eksport AI: " + (err.message || err), demoGeneration);
        } finally {
            demoActionPending = false;
            updateDemoControls(demoState, demoJob);
        }
    }

    function demoResponseError(raw, fallback) {
        try {
            const parsed = JSON.parse(raw || "{}");
            return parsed.detail || parsed.error || fallback;
        } catch (_) {
            return fallback;
        }
    }

    function scheduleDemoStatusPoll(delayMs, generation) {
        clearDemoPollTimer();
        if (demoPageDisposed || !demoJobId || generation !== demoGeneration) return;
        demoPollTimer = setTimeout(() => {
            demoPollTimer = null;
            void pollDemoStatus(generation);
        }, Math.max(0, Number(delayMs) || 0));
    }

    async function pollDemoStatus(generation) {
        if (demoPageDisposed || !demoJobId || generation !== demoGeneration) return;
        const jobId = demoJobId;
        const controller = new AbortController();
        demoStatusController = controller;
        try {
            const response = await fetch(
                "/api/demo-videos/" + encodeURIComponent(jobId),
                { credentials: "same-origin", signal: controller.signal },
            );
            if (!response.ok) {
                const raw = await response.text();
                throw new Error(demoResponseError(raw, "HTTP " + response.status));
            }
            const snapshot = await response.json();
            if (!applyDemoSnapshot(snapshot, generation)) return;
            demoPollFailures = 0;
        } catch (err) {
            if (err && err.name === "AbortError") return;
            if (generation !== demoGeneration) return;
            demoPollFailures += 1;
            if (!["uploaded", "loading"].includes(demoState) && demoPollFailures >= 3) {
                markDemoFailed("Nie można odczytać stanu filmu: " + (err.message || err), generation);
            }
        } finally {
            if (demoStatusController === controller) demoStatusController = null;
            if (
                !demoPageDisposed &&
                generation === demoGeneration &&
                demoJobId === jobId &&
                shouldPollDemoStatus(demoState)
            ) {
                scheduleDemoStatusPoll(DEMO_STATUS_POLL_MS, generation);
            }
        }
    }

    async function restorePersistedDemoJob() {
        if (demoPageDisposed || demoUploadXhr || demoJobId) return false;
        const persisted = loadPersistedDemoSession();
        if (!persisted) return false;

        const generation = ++demoGeneration;
        demoJobId = persisted.jobId;
        demoCameraId = persisted.cameraId || DEMO_CAMERA_ID;
        demoReturnCameraId = persisted.returnCameraId;
        demoPollFailures = 0;
        demoJob = { job_id: demoJobId, status: "loading" };
        updateDemoControls("loading", demoJob);

        const controller = new AbortController();
        demoStatusController = controller;
        try {
            const response = await fetch(
                "/api/demo-videos/" + encodeURIComponent(demoJobId),
                { credentials: "same-origin", signal: controller.signal },
            );
            if (generation !== demoGeneration) return false;
            if (response.status === 404 || response.status === 410) {
                clearPersistedDemoSession(demoJobId);
                resetDemoClient({ returnToCamera: true });
                return false;
            }
            if (!response.ok) {
                const raw = await response.text();
                throw new Error(demoResponseError(raw, "HTTP " + response.status));
            }

            const snapshot = await response.json();
            if (!snapshot.job_id || String(snapshot.job_id) !== demoJobId) {
                throw new Error("Backend zwrócił stan innego filmu demonstracyjnego.");
            }
            if (!applyDemoSnapshot(snapshot, generation)) return false;
            persistDemoSession();
            if (demoCameraId && State.cameraId !== demoCameraId) {
                selectCamera(demoCameraId);
                cameraSelect.value = demoCameraId;
            }
            refreshCameras();
            scheduleDemoStatusPoll(0, generation);
            return true;
        } catch (err) {
            if (err && err.name === "AbortError") return false;
            if (generation === demoGeneration) {
                markDemoFailed(
                    "Nie można odtworzyć stanu filmu: " + (err.message || err),
                    generation,
                );
            }
            return false;
        } finally {
            if (demoStatusController === controller) demoStatusController = null;
        }
    }

    function deleteDemoJobBestEffort(jobId) {
        if (!jobId) return;
        clearPersistedDemoSession(jobId);
        void fetch("/api/demo-videos/" + encodeURIComponent(jobId), {
            method: "DELETE",
            credentials: "same-origin",
            keepalive: true,
        }).catch(() => {});
    }

    function returnFromDemoCamera(cameraId, returnCameraId) {
        if (!cameraId || State.cameraId !== cameraId) return;
        const target = returnCameraId && returnCameraId !== cameraId
            ? returnCameraId
            : "cam_default";
        selectCamera(target);
        cameraSelect.value = target;
    }

    function resetDemoClient(options) {
        const oldCameraId = demoCameraId;
        const returnCameraId = demoReturnCameraId;
        demoGeneration += 1;
        cancelDemoRequests();
        resetDemoFrameAcceptance();
        demoJob = null;
        demoJobId = null;
        demoCameraId = null;
        demoPollFailures = 0;
        demoReturnCameraId = null;
        updateDemoControls("idle", {});
        if (options && options.returnToCamera) {
            returnFromDemoCamera(oldCameraId, returnCameraId);
        }
    }

    function openDemoFilePicker(purpose) {
        demoPickerPurpose = purpose === "ai-batch" ? "ai-batch" : "single";
        demoVideoInput.value = "";
        demoVideoInput.click();
    }

    function openDemoAiBatchPicker() {
        if (demoBatchActive) {
            demoBatchCancelled = true;
            demoBatchFinalMessage = "Zatrzymywanie kolejki AI po bieżącym filmie…";
            updateDemoControls(demoState, demoJob);
            return;
        }
        demoBatchFinalMessage = "";
        openDemoFilePicker("ai-batch");
    }

    function changeDemoVideo() {
        // Stary job jest zastępowany dopiero po wybraniu nowego pliku.
        openDemoFilePicker();
    }

    function startDemoUpload(file) {
        if (!file) return;
        if (!demoBatchActive) demoBatchFinalMessage = "";

        const oldJobId = demoJobId;
        const oldCameraId = demoCameraId;
        const oldReturnCameraId = demoReturnCameraId;
        demoGeneration += 1;
        const generation = demoGeneration;
        cancelDemoRequests();
        resetDemoFrameAcceptance();
        deleteDemoJobBestEffort(oldJobId);
        returnFromDemoCamera(oldCameraId, oldReturnCameraId);

        demoJobId = null;
        demoCameraId = DEMO_CAMERA_ID;
        demoReturnCameraId = State.cameraId === DEMO_CAMERA_ID ? "cam_default" : State.cameraId;
        demoPollFailures = 0;
        demoJob = {
            status: "uploading",
            filename: file.name,
            size_bytes: file.size,
            upload_percent: 0,
            error: null,
        };
        updateDemoControls("uploading", demoJob);

        const form = new FormData();
        form.append("video", file, file.name);
        form.append("camera_id", DEMO_CAMERA_ID);
        form.append("mode", State.mode);
        form.append("playback_mode", "realtime");

        const xhr = new XMLHttpRequest();
        demoUploadXhr = xhr;
        xhr.open("POST", "/api/demo-videos", true);
        xhr.withCredentials = true;
        xhr.timeout = DEMO_UPLOAD_TIMEOUT_MS;
        xhr.upload.onprogress = event => {
            if (generation !== demoGeneration || demoUploadXhr !== xhr) return;
            if (!event.lengthComputable || event.total <= 0) return;
            demoJob.upload_percent = Math.min(100, (event.loaded / event.total) * 100);
            updateDemoControls("uploading", demoJob);
        };
        xhr.onload = () => {
            if (generation !== demoGeneration || demoUploadXhr !== xhr) return;
            demoUploadXhr = null;
            if (xhr.status < 200 || xhr.status >= 300) {
                markDemoFailed(
                    demoResponseError(xhr.responseText, "Upload nie powiódł się: HTTP " + xhr.status),
                    generation,
                );
                return;
            }

            let snapshot;
            try {
                snapshot = JSON.parse(xhr.responseText || "{}");
            } catch (_) {
                markDemoFailed("Backend zwrócił nieprawidłową odpowiedź po uploadzie.", generation);
                return;
            }
            if (!snapshot.job_id) {
                markDemoFailed("Backend nie zwrócił identyfikatora zadania.", generation);
                return;
            }

            demoJobId = String(snapshot.job_id);
            demoCameraId = String(snapshot.camera_id || DEMO_CAMERA_ID);
            demoJob = Object.assign({}, demoJob, snapshot, { upload_percent: 100 });
            persistDemoSession();
            selectCamera(demoCameraId);
            refreshCameras();
            applyDemoSnapshot(snapshot, generation, { reconnectOnRunChange: false });
            scheduleDemoStatusPoll(0, generation);
        };
        xhr.onerror = () => {
            if (generation !== demoGeneration || demoUploadXhr !== xhr) return;
            demoUploadXhr = null;
            markDemoFailed("Nie udało się wysłać filmu do backendu.", generation);
        };
        xhr.ontimeout = () => {
            if (generation !== demoGeneration || demoUploadXhr !== xhr) return;
            demoUploadXhr = null;
            markDemoFailed("Przekroczono limit czasu wysyłania filmu.", generation);
        };
        xhr.onabort = () => {
            if (generation !== demoGeneration || demoUploadXhr !== xhr) return;
            demoUploadXhr = null;
            markDemoFailed("Wysyłanie filmu zostało anulowane.", generation);
        };
        xhr.send(form);
    }

    async function requestDemoAction(action) {
        if (!demoJobId || demoActionPending) return;
        const generation = demoGeneration;
        const jobId = demoJobId;
        const previousRunId = demoRunId;
        const expectsNewRun = action === "restart" ||
            (action === "play" && demoState === "stopped");
        if (expectsNewRun) {
            // Keep the already-connected demo socket able to accept frame 0.
            // The backend starts a fresh run before its HTTP response reaches
            // us, so retaining the previous run_id here would discard those
            // first, otherwise perfectly synchronized frames.
            activateDemoRun(null, false, true);
        }
        demoActionPending = true;
        updateDemoControls(demoState, demoJob);
        const controller = new AbortController();
        demoControlController = controller;
        try {
            const response = await fetch(
                "/api/demo-videos/" + encodeURIComponent(jobId) + "/" + action,
                {
                    method: "POST",
                    credentials: "same-origin",
                    signal: controller.signal,
                },
            );
            if (!response.ok) {
                const raw = await response.text();
                throw new Error(demoResponseError(raw, "HTTP " + response.status));
            }
            const snapshot = await response.json();
            if (generation !== demoGeneration || jobId !== demoJobId) return;
            if (
                action === "restart" &&
                (snapshot.run_id === null || snapshot.run_id === undefined ||
                    String(snapshot.run_id) === previousRunId)
            ) {
                activateDemoRun(snapshot.run_id ?? null, true, true);
            }
            applyDemoSnapshot(snapshot, generation);
            if (shouldPollDemoStatus(demoState)) {
                scheduleDemoStatusPoll(0, generation);
            } else {
                clearDemoPollTimer();
            }
        } catch (err) {
            if (err && err.name === "AbortError") return;
            if (generation === demoGeneration) {
                markDemoFailed(
                    "Operacja „" + action + "” nie powiodła się: " + (err.message || err),
                    generation,
                );
            }
        } finally {
            if (demoControlController === controller) demoControlController = null;
            if (generation === demoGeneration) {
                demoActionPending = false;
                updateDemoControls(demoState, demoJob);
            }
        }
    }

    function sleepDemoBatch(ms) {
        return new Promise(resolve => setTimeout(resolve, Math.max(0, Number(ms) || 0)));
    }

    function waitForDemoBatch(predicate, timeoutMs, label) {
        const started = Date.now();
        return new Promise((resolve, reject) => {
            const tick = () => {
                if (demoBatchCancelled) {
                    reject(new Error("Kolejka AI została zatrzymana"));
                    return;
                }
                if (demoState === "failed") {
                    reject(new Error((demoJob && demoJob.error) || "Przetwarzanie filmu nie powiodło się"));
                    return;
                }
                if (demoJob && demoJob.export_status === "failed") {
                    reject(new Error(demoJob.export_error || "Eksport AI nie powiódł się"));
                    return;
                }
                try {
                    if (predicate()) {
                        resolve();
                        return;
                    }
                } catch (err) {
                    reject(err);
                    return;
                }
                if (Date.now() - started >= timeoutMs) {
                    reject(new Error("Przekroczono limit czasu: " + label));
                    return;
                }
                setTimeout(tick, 250);
            };
            tick();
        });
    }

    async function runDemoAiBatch(files) {
        const queue = Array.from(files || []).filter(Boolean);
        if (!queue.length || demoBatchActive) return;

        demoBatchActive = true;
        demoBatchCancelled = false;
        demoBatchFiles = queue;
        demoBatchIndex = -1;
        demoBatchDone = 0;
        demoBatchFailed = [];
        demoBatchFinalMessage = "";
        updateDemoControls(demoState, demoJob);

        for (let index = 0; index < queue.length; index += 1) {
            if (demoBatchCancelled) break;
            const file = queue[index];
            demoBatchIndex = index;
            updateDemoControls(demoState, demoJob);

            try {
                startDemoUpload(file);
                await waitForDemoBatch(
                    () => !!demoJobId && ["ready", "finished", "stopped"].includes(demoState),
                    DEMO_BATCH_PREPARE_TIMEOUT_MS,
                    "przygotowanie " + file.name,
                );

                if (demoBatchCancelled) break;
                await requestDemoExport();
                await waitForDemoBatch(
                    () => !!demoJob && demoJob.export_status === "ready" && !!demoJob.output_url,
                    DEMO_BATCH_EXPORT_TIMEOUT_MS,
                    "analiza AI " + file.name,
                );

                demoBatchDone += 1;
                // Daj przeglądarce chwilę na rozpoczęcie pobierania.
                await sleepDemoBatch(DEMO_BATCH_DOWNLOAD_GRACE_MS);
            } catch (err) {
                if (demoBatchCancelled) break;
                demoBatchFailed.push({
                    name: file.name,
                    error: String((err && err.message) || err || "Nieznany błąd"),
                });
                await sleepDemoBatch(600);
            }
        }

        const total = queue.length;
        const failedCount = demoBatchFailed.length;
        const cancelled = demoBatchCancelled;
        demoBatchActive = false;
        demoBatchFiles = [];
        demoBatchIndex = -1;
        demoBatchCancelled = false;

        if (cancelled) {
            demoBatchFinalMessage = "Kolejka AI zatrzymana · ukończono " + demoBatchDone + "/" + total;
        } else if (failedCount) {
            demoBatchFinalMessage = "Kolejka AI zakończona · gotowe " + demoBatchDone +
                "/" + total + " · błędy " + failedCount;
            console.warn("Perimetr AI batch failures:", demoBatchFailed);
        } else {
            demoBatchFinalMessage = "Kolejka AI zakończona · przetworzono " + demoBatchDone + "/" + total;
        }
        updateDemoControls(demoState, demoJob);
    }

    aiModeBtn.addEventListener("click", () => {
        openDemoAiBatchPicker();
    });

    demoVideoBtn.addEventListener("click", () => {
        if (["idle", "deleted"].includes(demoState)) {
            openDemoFilePicker();
        } else if (demoState === "failed") {
            changeDemoVideo();
        } else if (["ready", "stopped"].includes(demoState)) {
            void requestDemoAction("play");
        } else if (demoState === "playing") {
            void requestDemoAction("pause");
        } else if (demoState === "paused") {
            void requestDemoAction("resume");
        } else if (demoState === "finished") {
            void requestDemoAction("restart");
        }
    });

    demoVideoStopBtn.addEventListener("click", () => {
        void requestDemoAction("stop");
    });

    demoVideoChangeBtn.addEventListener("click", changeDemoVideo);

    demoVideoInput.addEventListener("change", () => {
        const files = Array.from(demoVideoInput.files || []);
        if (!files.length) return;
        if (demoPickerPurpose === "ai-batch") {
            void runDemoAiBatch(files);
            return;
        }
        startDemoUpload(files[0]);
    });

    function disposeDemoPageClient() {
        if (demoPageDisposed) return;
        demoPageDisposed = true;
        clearDemoPollTimer();
        clearDemoPreparationWatchdog();
        abortDemoController(demoStatusController);
        abortDemoController(demoControlController);
        demoStatusController = null;
        demoControlController = null;
        demoActionPending = false;
        disconnectWebSocketForPageLifecycle();
        // Backendowe odtwarzanie pozostaje aktywne po zmianie podstrony.
    }

    function resumeDemoPageClient() {
        if (!demoPageDisposed) return;
        demoPageDisposed = false;
        if (demoJobId) {
            if (demoCameraId && State.cameraId !== demoCameraId) {
                selectCamera(demoCameraId);
                cameraSelect.value = demoCameraId;
            } else {
                connectWebSocket();
            }
            scheduleDemoStatusPoll(0, demoGeneration);
            return;
        }
        connectWebSocket();
        if (!demoUploadXhr) void restorePersistedDemoJob();
    }

    window.addEventListener("pagehide", disposeDemoPageClient);
    window.addEventListener("pageshow", resumeDemoPageClient);

    const pairModal = document.getElementById("pairModal");
    const pairModalClose = document.getElementById("pairModalClose");
    const pairQrImg = document.getElementById("pairQrImg");
    const pairUrlInput = document.getElementById("pairUrl");
    const pairCopyBtn = document.getElementById("pairCopyBtn");

    function captureUrl() {
        const u = new URL("/phone/capture.html", location.origin);
        u.searchParams.set("server", location.origin);
        u.searchParams.set("camera_id", State.cameraId);
        return u.toString();
    }

    function openPairModal() {
        const url = captureUrl();
        pairUrlInput.value = url;
        pairQrImg.src = "/api/pair-qr?target=" + encodeURIComponent(url);
        pairModal.classList.remove("hidden");
    }
    function closePairModal() {
        pairModal.classList.add("hidden");
    }

    pairPhoneBtn.addEventListener("click", openPairModal);
    addCameraBtn.addEventListener("click", () => {
        const suggested = "cam_" + String(Date.now()).slice(-6);
        const cameraId = prompt("Podaj identyfikator nowej kamery", suggested);
        if (!cameraId || !cameraId.trim()) return;
        selectCamera(cameraId.trim().replace(/[^a-zA-Z0-9_.-]/g, "_"));
        refreshCameras();
        openPairModal();
    });
    pairModalClose.addEventListener("click", closePairModal);
    pairModal.addEventListener("click", (e) => {
        if (e.target === pairModal) closePairModal();  // click backdrop
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !pairModal.classList.contains("hidden")) closePairModal();
    });
    pairCopyBtn.addEventListener("click", () => {
        const done = () => {
            const prev = pairCopyBtn.textContent;
            pairCopyBtn.textContent = "Skopiowano";
            setTimeout(() => { pairCopyBtn.textContent = prev; }, 1500);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(pairUrlInput.value).then(done, () => {
                pairUrlInput.select(); document.execCommand("copy"); done();
            });
        } else {
            pairUrlInput.select(); document.execCommand("copy"); done();
        }
    });


    let selectedWorkerId = null;
    const WORKER_TOOLS_OPEN_KEY = "perimetr:worker_tools_expanded";

    function loadWorkerToolsExpanded() {
        try {
            return localStorage.getItem(WORKER_TOOLS_OPEN_KEY) === "true";
        } catch (_) {
            return false;
        }
    }

    function saveWorkerToolsExpanded(expanded) {
        try {
            localStorage.setItem(WORKER_TOOLS_OPEN_KEY, expanded ? "true" : "false");
        } catch (_) { /* localStorage can be unavailable in privacy mode */ }
    }

    function setWorkerToolsExpanded(expanded, options) {
        if (!workerToolsCard || !workerToolsToggle || !workerToolsBody) return;

        const animate = !(options && options.animate === false);
        const persist = !(options && options.persist === false);
        const currentlyExpanded = workerToolsCard.classList.contains("px-worker-tools--expanded");

        workerToolsToggle.setAttribute("aria-expanded", expanded ? "true" : "false");
        workerToolsBody.setAttribute("aria-hidden", expanded ? "false" : "true");
        workerToolsCard.classList.toggle("px-worker-tools--expanded", expanded);
        workerToolsCard.classList.toggle("px-worker-tools--collapsed", !expanded);

        if (persist) saveWorkerToolsExpanded(expanded);

        if (!animate) {
            const previousTransition = workerToolsBody.style.transition;
            workerToolsBody.style.transition = "none";
            workerToolsBody.style.height = expanded ? "auto" : "0px";
            workerToolsBody.style.opacity = expanded ? "1" : "0";
            void workerToolsBody.offsetHeight;
            workerToolsBody.style.transition = previousTransition;
            return;
        }

        const currentHeight = workerToolsBody.getBoundingClientRect().height;
        workerToolsBody.style.height = currentHeight + "px";
        workerToolsBody.style.opacity = currentlyExpanded ? "1" : "0";
        void workerToolsBody.offsetHeight;

        requestAnimationFrame(() => {
            workerToolsBody.style.height = expanded
                ? workerToolsBody.scrollHeight + "px"
                : "0px";
            workerToolsBody.style.opacity = expanded ? "1" : "0";
        });
    }

    if (workerToolsBody) {
        workerToolsBody.addEventListener("transitionend", event => {
            if (event.propertyName !== "height") return;
            const expanded = workerToolsCard && workerToolsCard.classList.contains("px-worker-tools--expanded");
            if (expanded) workerToolsBody.style.height = "auto";
        });
    }

    if (workerToolsToggle) {
        workerToolsToggle.addEventListener("click", () => {
            const expanded = workerToolsToggle.getAttribute("aria-expanded") === "true";
            setWorkerToolsExpanded(!expanded);
        });
    }

    setWorkerToolsExpanded(loadWorkerToolsExpanded(), { animate: false, persist: false });

    let workerRegistryBusy = false;
    const workersById = new Map();
    const workerRequiredFields = [
        [workerQrInput, "ID znacznika"],
        [workerFirstName, "imię"],
        [workerLastName, "nazwisko"],
        [workerPosition, "stanowisko"],
        [workerDepartment, "dział"],
    ];

    function workerCountLabel(count) {
        if (count === 1) return "1 wpis";
        const lastTwo = count % 100;
        const last = count % 10;
        return count + ((last >= 2 && last <= 4 && !(lastTwo >= 12 && lastTwo <= 14))
            ? " wpisy"
            : " wpisów");
    }

    function setWorkerRegistryStatus(message, tone) {
        if (!workerRegistryStatus) return;
        workerRegistryStatus.textContent = message;
        workerRegistryStatus.classList.toggle("success", tone === "success");
        workerRegistryStatus.classList.toggle("error", tone === "error");
    }

    function setWorkerRegistryBusy(busy) {
        workerRegistryBusy = busy;
        workerSaveBtn.disabled = busy;
        workerNewBtn.disabled = busy;
        workerRegistrySelect.disabled = busy;
        workerDeleteBtn.disabled = busy;
        const label = workerSaveBtn.querySelector("span");
        if (label) label.textContent = busy
            ? "Zapisywanie…"
            : (selectedWorkerId ? "Aktualizuj" : "Zapisz");
    }

    function validateWorkerRegistryForm() {
        const missing = workerRequiredFields.filter(([field]) => {
            const isMissing = !(field.value || "").trim();
            if (isMissing) field.setAttribute("aria-invalid", "true");
            else field.removeAttribute("aria-invalid");
            return isMissing;
        });
        if (!missing.length) return true;

        setWorkerRegistryStatus(
            "Uzupełnij wymagane pola: " +
                missing.map(([, label]) => label).join(", ") + ".",
            "error"
        );
        missing[0][0].focus();
        return false;
    }

    function workerOptionLabel(worker) {
        const name = worker.full_name ||
            [worker.first_name, worker.last_name].filter(Boolean).join(" ");
        return worker.worker_id + (name ? " — " + name : "");
    }

    function renderWorkerRegistryOptions(preferredId) {
        const currentId = preferredId === undefined
            ? (workerRegistrySelect.value || selectedWorkerId || "")
            : (preferredId || "");
        workerRegistrySelect.innerHTML = "";
        const newOption = document.createElement("option");
        newOption.value = "";
        newOption.textContent = "Nowy pracownik…";
        workerRegistrySelect.appendChild(newOption);

        Array.from(workersById.values())
            .sort((a, b) => workerOptionLabel(a).localeCompare(workerOptionLabel(b), "pl"))
            .forEach(worker => {
                const option = document.createElement("option");
                option.value = worker.worker_id;
                option.textContent = workerOptionLabel(worker);
                workerRegistrySelect.appendChild(option);
            });

        workerRegistryCount.textContent = workerCountLabel(workersById.size);
        workerRegistrySelect.value = workersById.has(currentId) ? currentId : "";
    }

function workerTagPngUrl(workerId, download) {
    const params = new URLSearchParams({
        worker_id: workerId,
    });

    if (download) {
        params.set("download", "true");
    }

    return "/api/worker-tag.png?" + params.toString();
}

function workerTagFilename(workerId) {
    const safe = String(workerId || "worker")
        .replace(/[^A-Za-z0-9_.-]+/g, "_")
        .replace(/^[._]+|[._]+$/g, "") || "worker";

    return "worker-" + safe + ".png";
}

function hideWorkerQrPreview() {
    if (!workerQrPreview) {
        return;
    }

    workerQrPreview.classList.add("hidden");

    if (workerQrPreviewImg) {
        workerQrPreviewImg.removeAttribute("src");
        workerQrPreviewImg.alt = "";
    }

    if (workerQrDownloadBtn) {
        workerQrDownloadBtn.setAttribute("href", "#");
        workerQrDownloadBtn.removeAttribute("download");
    }

    if (workerQrPreviewTitle) {
        workerQrPreviewTitle.textContent = "Znacznik pracownika";
    }

    if (workerQrPreviewPayload) {
        workerQrPreviewPayload.textContent = "tag:—";
    }
}

function showWorkerQrPreview(workerId) {
    if (!workerQrPreview || !workerId) {
        return;
    }

    const previewUrl = workerTagPngUrl(workerId, false);
    const downloadUrl = workerTagPngUrl(workerId, true);
    const filename = workerTagFilename(workerId);

    if (workerQrPreviewImg) {
        workerQrPreviewImg.src = previewUrl;
        workerQrPreviewImg.alt = "Znacznik pracownika " + workerId;
    }

    if (workerQrPreviewTitle) {
        workerQrPreviewTitle.textContent = "Tag · " + workerId;
    }

    if (workerQrPreviewPayload) {
        workerQrPreviewPayload.textContent = "worker tag → " + workerId;
    }

    if (workerQrDownloadBtn) {
        workerQrDownloadBtn.href = downloadUrl;
        workerQrDownloadBtn.download = filename;
    }

    workerQrPreview.classList.remove("hidden");
}


    function editWorker(worker, options) {
        const focusId = options && options.focusId;
        if (!worker) {
            selectedWorkerId = null;
            workerRegistryForm.reset();
            workerRegistrySelect.value = "";
            workerQrInput.readOnly = false;
            workerQrInput.removeAttribute("aria-readonly");
            workerRequiredFields.forEach(([field]) => field.removeAttribute("aria-invalid"));
            workerDeleteBtn.classList.add("hidden");
            const saveLabel = workerSaveBtn.querySelector("span");
            if (saveLabel) saveLabel.textContent = "Zapisz";
            hideWorkerQrPreview();
            setWorkerRegistryStatus("");
            if (focusId) workerQrInput.focus();
            return;
        }

        selectedWorkerId = worker.worker_id;
        workerRegistrySelect.value = worker.worker_id;
        workerQrInput.value = worker.worker_id || "";
        workerQrInput.readOnly = true;
        workerQrInput.setAttribute("aria-readonly", "true");
        workerFirstName.value = worker.first_name || "";
        workerLastName.value = worker.last_name || "";
        workerPosition.value = worker.position || "";
        workerDepartment.value = worker.department || "";
        workerDeleteBtn.classList.remove("hidden");
        const saveLabel = workerSaveBtn.querySelector("span");
        if (saveLabel) saveLabel.textContent = "Aktualizuj";
        showWorkerQrPreview(worker.worker_id);
        setWorkerRegistryStatus("Edytujesz zapisany profil " + worker.worker_id + ".");
    }

    async function apiErrorMessage(response) {
        let body = null;
        try { body = await response.json(); } catch (_) { /* no JSON body */ }
        if (body && Array.isArray(body.detail)) {
            const messages = body.detail.map(item => item && item.msg).filter(Boolean);
            if (messages.length) return messages.join(", ");
        }
        if (body && typeof body.detail === "string") return body.detail;
        return "HTTP " + response.status;
    }

    async function loadWorkerRegistry(preferredId) {
        setWorkerRegistryStatus("Wczytywanie rejestru…");
        try {
            const response = await fetch("/api/workers", { credentials: "same-origin" });
            if (!response.ok) throw new Error(await apiErrorMessage(response));
            const workers = await response.json();
            workersById.clear();
            (Array.isArray(workers) ? workers : []).forEach(worker => {
                if (worker && worker.worker_id) workersById.set(worker.worker_id, worker);
            });
            renderWorkerRegistryOptions(preferredId);
            const selected = workersById.get(preferredId || "");
            if (selected) editWorker(selected);
            else editWorker(null);
        } catch (error) {
            setWorkerRegistryStatus(
                "Nie udało się wczytać rejestru: " + (error.message || error),
                "error"
            );
            throw error;
        }
    }

    function workerFormPayload() {
        return {
            worker_id: (workerQrInput.value || "").trim(),
            first_name: (workerFirstName.value || "").trim(),
            last_name: (workerLastName.value || "").trim(),
            position: (workerPosition.value || "").trim(),
            department: (workerDepartment.value || "").trim(),
        };
    }

    async function saveWorker(event) {
        event.preventDefault();
        if (workerRegistryBusy || !validateWorkerRegistryForm()) return;

        const payload = workerFormPayload();
        setWorkerRegistryBusy(true);
        setWorkerRegistryStatus("Sprawdzanie identyfikatora " + payload.worker_id + "…");

        try {
            const itemUrl = "/api/workers/" + encodeURIComponent(payload.worker_id);
            const lookup = await fetch(itemUrl, { credentials: "same-origin" });
            if (!lookup.ok && lookup.status !== 404) {
                throw new Error(await apiErrorMessage(lookup));
            }

            const exists = lookup.ok;
            const response = await fetch(exists ? itemUrl : "/api/workers", {
                method: exists ? "PUT" : "POST",
                credentials: "same-origin",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(exists ? {
                    first_name: payload.first_name,
                    last_name: payload.last_name,
                    position: payload.position,
                    department: payload.department,
                } : payload),
            });
            if (!response.ok) {
                if (response.status === 409) {
                    throw new Error("To ID jest już zapisane. Odśwież rejestr i wybierz pracownika z listy.");
                }
                if (response.status === 404 && exists) {
                    throw new Error("Profil został w międzyczasie usunięty. Odśwież rejestr i spróbuj ponownie.");
                }
                throw new Error(await apiErrorMessage(response));
            }

            const saved = await response.json();
            workersById.set(saved.worker_id, saved);
            renderWorkerRegistryOptions(saved.worker_id);
            editWorker(saved);
            setWorkerRegistryStatus(
                exists ? "Zaktualizowano profil pracownika." : "Dodano pracownika do rejestru.",
                "success"
            );
        } catch (error) {
            setWorkerRegistryStatus("Nie udało się zapisać: " + (error.message || error), "error");
        } finally {
            setWorkerRegistryBusy(false);
        }
    }

    async function deleteWorker() {
        if (workerRegistryBusy || !selectedWorkerId) return;
        const worker = workersById.get(selectedWorkerId);
        const label = worker ? workerOptionLabel(worker) : selectedWorkerId;
        if (!confirm("Usunąć pracownika „" + label + "” z rejestru?")) return;

        const deletedId = selectedWorkerId;
        setWorkerRegistryBusy(true);
        setWorkerRegistryStatus("Usuwanie profilu " + deletedId + "…");
        try {
            const response = await fetch("/api/workers/" + encodeURIComponent(deletedId), {
                method: "DELETE",
                credentials: "same-origin",
            });
            if (!response.ok && response.status !== 404) {
                throw new Error(await apiErrorMessage(response));
            }
            workersById.delete(deletedId);
            renderWorkerRegistryOptions("");
            editWorker(null);
            setWorkerRegistryStatus(
                response.status === 404
                    ? "Profil nie istniał już w rejestrze."
                    : "Usunięto pracownika " + deletedId + ".",
                "success"
            );
        } catch (error) {
            setWorkerRegistryStatus("Nie udało się usunąć: " + (error.message || error), "error");
        } finally {
            setWorkerRegistryBusy(false);
        }
    }

    workerRegistrySelect.addEventListener("change", () => {
        const worker = workersById.get(workerRegistrySelect.value);
        editWorker(worker || null, { focusId: !worker });
    });
    workerRegistryForm.addEventListener("input", event => {
        if (event.target && (event.target.value || "").trim()) {
            event.target.removeAttribute("aria-invalid");
        }
        if (event.target === workerQrInput && !workerQrInput.readOnly) {
            hideWorkerQrPreview();
        }
    });
    workerNewBtn.addEventListener("click", () => editWorker(null, { focusId: true }));
    workerRegistryForm.addEventListener("submit", saveWorker);
    workerDeleteBtn.addEventListener("click", deleteWorker);

    function generateWorkerTagPng() {
        const workerId = (workerQrInput.value || "").trim();
        if (!workerId) {
            workerQrInput.focus();
            setWorkerRegistryStatus("Najpierw wpisz ID pracownika.", "error");
            return;
        }

        showWorkerQrPreview(workerId);

        const link = document.createElement("a");
        link.href = workerTagPngUrl(workerId, true);
        link.download = workerTagFilename(workerId);
        link.style.display = "none";
        document.body.appendChild(link);
        link.click();
        link.remove();

        setWorkerRegistryStatus(
            "Wygenerowano i pobrano PNG dla identyfikatora " + workerId + ".",
            "success"
        );
    }
    workerQrOpenBtn.addEventListener("click", generateWorkerTagPng);
    loadWorkerRegistry().catch(() => {});


    let ws = null;
    let lastRenderedTs = 0;
    let wsGeneration = 0;
    let wsReconnectTimer = null;
    const pendingBinaryFrames = [];

    function setWsStatus(state) {
        State.wsStatus = state;
        const labels = { connecting: "łączenie", connected: "połączono", error: "błąd" };
        wsStatusEl.textContent = labels[state] || state;
        connText.textContent = labels[state] || state;
        connWrap.classList.toggle("err", state !== "connected");
        if (state !== "connected") {
            liveText.textContent = "OFFLINE";
        }
    }

    function connectWebSocket() {
        const generation = ++wsGeneration;
        if (wsReconnectTimer) {
            clearTimeout(wsReconnectTimer);
            wsReconnectTimer = null;
        }
        if (ws) {
            ws.onopen = null;
            ws.onmessage = null;
            ws.onerror = null;
            ws.onclose = null;
            try { ws.close(); } catch (_) {}
        }
        pendingBinaryFrames.length = 0;
        setWsStatus("connecting");
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        const query = "?camera_id=" + encodeURIComponent(State.cameraId) + "&binary=true";
        const socket = new WebSocket(proto + "//" + location.host + "/ws/live" + query);
        socket.binaryType = "blob";
        ws = socket;
        socket.onopen = () => {
            if (generation === wsGeneration) setWsStatus("connected");
        };
        socket.onerror = () => {
            if (generation !== wsGeneration) return;
            setWsStatus("error");
            try { socket.close(); } catch (_) {}
        };
        socket.onclose = () => {
            if (generation !== wsGeneration) return;
            setWsStatus("error");
            wsReconnectTimer = setTimeout(connectWebSocket, 2000);
        };
        socket.onmessage = event => {
            if (generation === wsGeneration) onWsMessage(event);
        };
    }

    function disconnectWebSocketForPageLifecycle() {
        wsGeneration += 1;
        if (wsReconnectTimer) {
            clearTimeout(wsReconnectTimer);
            wsReconnectTimer = null;
        }
        const socket = ws;
        ws = null;
        pendingBinaryFrames.length = 0;
        if (!socket) return;
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        try { socket.close(); } catch (_) {}
    }

    function queueBinaryMetadata(metadata) {
        pendingBinaryFrames.push(metadata);
        if (pendingBinaryFrames.length > 4) pendingBinaryFrames.shift();
    }

    function rejectWsMetadata(data) {
        if (data && data.frame_transport === "binary-jpeg") {
            // Preserve the JSON/binary message pairing even when this frame is
            // intentionally rejected. The following blob consumes this slot.
            queueBinaryMetadata(null);
        }
    }

    function acceptDemoFrame(data) {
        if (!demoJobId || String(data.job_id || "") !== demoJobId) return false;

        const incomingRunId = data.run_id === null || data.run_id === undefined
            ? null
            : String(data.run_id);
        if (demoRunId !== null && incomingRunId !== demoRunId) return false;
        if (demoRunId === null && incomingRunId !== null) {
            activateDemoRun(incomingRunId, false);
        }

        const frameIndex = Number(data.frame_index);
        if (!Number.isInteger(frameIndex) || frameIndex < 0) return false;
        if (frameIndex <= demoLastFrameIndex) return false;
        demoLastFrameIndex = frameIndex;

        demoJob = Object.assign({}, demoJob || {}, {
            status: data.status || demoState,
            current_frame: frameIndex,
            current_time_sec: data.source_time_sec,
            source_time_sec: data.source_time_sec,
            processing_fps: data.processing_fps,
        });
        const frameState = String(data.status || "").toLowerCase();
        if (DEMO_BACKEND_STATES.has(frameState)) {
            const nextState = frameState;
            if (!["uploaded", "loading"].includes(nextState)) {
                clearDemoPreparationWatchdog();
            }
            updateDemoControls(nextState, demoJob);
            if (!shouldPollDemoStatus(nextState)) clearDemoPollTimer();
        }
        return true;
    }

    function onWsMessage(evt) {
        if (typeof evt.data !== "string") {
            const metadata = pendingBinaryFrames.shift();
            if (metadata) renderBinaryFrame(evt.data, metadata);
            return;
        }
        let data; try { data = JSON.parse(evt.data); } catch (_) { return; }
        if ((data.camera_id || "cam_default") !== State.cameraId) {
            rejectWsMetadata(data);
            return;
        }

        const isDemoFrame = data.type === "frame" && data.job_id !== undefined;
        if (isDemoFrame && !acceptDemoFrame(data)) {
            rejectWsMetadata(data);
            return;
        }
        const ts = data.timestamp || 0;
        if (!isDemoFrame && ts && ts < lastRenderedTs) {
            rejectWsMetadata(data);
            return;
        }
        lastRenderedTs = ts;

        if (data.frame_transport === "binary-jpeg") {
            pendingBinaryFrames.push(data);
            if (pendingBinaryFrames.length > 4) pendingBinaryFrames.shift();
        } else {
            renderFrame(data);
        }
        updateStats(data);
        updateStatusChips(data);
        updateAlerts(data);
        updatePPE(data);
        updatePosture(data);
        updateWorkers(data);
        maybeRaiseAlarm(data);

        document.dispatchEvent(new CustomEvent("perimetr-frame", { detail: data }));
    }


    let frameRenderSequence = 0;

    function applyDecodedFrame(image, width, height, data, sequence) {
        if (sequence !== frameRenderSequence) {
            if (image && typeof image.close === "function") image.close();
            return;
        }
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        const aspectRatio = `${width} / ${height}`;
        if (videoEl.style.aspectRatio !== aspectRatio) {
            videoEl.style.aspectRatio = aspectRatio;
        }
        ctx.drawImage(image, 0, 0);
        if (image && typeof image.close === "function") image.close();
        State.lastCameraSize = width + "×" + height;
        document.dispatchEvent(new CustomEvent("perimetr-frame-rendered", {
            detail: data,
        }));
    }

    function renderFrame(data) {
        if (!data.frame_jpeg_b64) return;
        noSignal.classList.add("hidden");
        const sequence = ++frameRenderSequence;
        const img = new Image();
        img.onload = () => {
            applyDecodedFrame(img, img.width, img.height, data, sequence);
        };
        img.src = "data:image/jpeg;base64," + data.frame_jpeg_b64;
    }

    async function renderBinaryFrame(blob, data) {
        noSignal.classList.add("hidden");
        const sequence = ++frameRenderSequence;
        try {
            const bitmap = await createImageBitmap(blob);
            applyDecodedFrame(
                bitmap,
                bitmap.width,
                bitmap.height,
                data,
                sequence,
            );
        } catch (_) {
            if (sequence === frameRenderSequence) setWsStatus("error");
        }
    }


    let fpsFrames = 0, fpsLastTime = performance.now();
    function updateStats(data) {
        State.stats.frameCount = data.frame_id || (State.stats.frameCount + 1);
        State.stats.detectionMs = data.processing_ms || 0;
        frameCountEl.textContent = State.stats.frameCount;
        processingTimeEl.textContent = Math.round(State.stats.detectionMs);
        fpsFrames++;
        const now = performance.now();
        const elapsed = now - fpsLastTime;
        if (elapsed >= 1000) {
            const fps = (fpsFrames / elapsed) * 1000;
            State.stats.fps = fps;
            fpsLine.textContent = fps.toFixed(1) + " kl/s";
            camMeta.textContent = State.lastCameraSize + " · " + fps.toFixed(1) + " kl/s";
            const shortText = fps.toFixed(1) + " kl/s · " + Math.round(State.stats.detectionMs) + " ms";
            connText.textContent = "Połączono · " + shortText;
            fpsFrames = 0;
            fpsLastTime = now;
        }
    }

    function updateStatusChips(data) {
        const time = data.timestamp ?
            new Date(data.timestamp * 1000).toLocaleTimeString("pl-PL") : "—";
        const activeAlarm = !!State.activeAlarm;
        liveText.textContent = (activeAlarm ? "ALARM" : "LIVE") + " · " + State.cameraId + " · " + time;

        const markers = (data.markers || []).length;
        State.lastMarkerCount = markers;
        State.lastCalibrationActive = !!data.calibration_active;
        const calibrationValid = data.calibration_valid === undefined
            ? !!data.calibration_active
            : !!data.calibration_valid;
        const homo = !data.calibration_active
            ? "brak homografii"
            : calibrationValid
                ? "homografia OK"
                : "homografia niezgodna";
        markerChip.textContent = "ArUco " + markers + " · " + homo;
        markerChip.title = data.calibration_warning || "";
    }


    const MAX_ALERTS_IN_UI = 100;

    function formatHazardDistance(alert) {
        if (alert.distance_m !== null && alert.distance_m !== undefined) {
            return " · " + Number(alert.distance_m).toFixed(2) + " m";
        }
        if (alert.distance_px !== null && alert.distance_px !== undefined) {
            return " · " + Math.round(alert.distance_px) + " px";
        }
        return "";
    }

    function workerSuffix(item) {
        return item && item.worker_id ? " · ID " + item.worker_id : "";
    }

    function updateAlerts(data) {
        const items = [];
        (data.confirmed_zone_breaches || []).forEach(b => items.push({
            severity: b.severity,
            time: b.timestamp,
            desc: (b.rule_name === "zone_approach" ? "Zbliżanie do strefy: " : "Wejście w strefę: ") +
                (b.zone_name || "?") + formatHazardDistance(b),
            kind: b.rule_name === "zone_approach" ? "zone_approach" : "zone_breach",
            thumb: b.frame_thumbnail_url,
        }));
        (data.confirmed_alerts || []).forEach(a => {
            const descByRule = {
                person_vehicle_overlap: "Osoba w obrysie maszyny",
                person_vehicle_danger_zone: "Krytyczna odległość od maszyny",
                person_near_vehicle: "Ostrzegawcza odległość od maszyny",
            };
            items.push({
                severity: a.severity,
                time: a.timestamp,
                desc: (descByRule[a.rule_name] || "Osoba przy maszynie") + formatHazardDistance(a),
                kind: "site_hazard",
                thumb: a.frame_thumbnail_url,
            });
        });
        (data.confirmed_posture_alerts || []).forEach(p => {
            const details = (p.signals || []).slice(0, 2).map(x => POSTURE_SIGNAL_LABELS[x] || x);
            const signals = p.signals || [];
            const fall = signals.includes("fall_detected") ||
                signals.includes("fall_suspected") ||
                signals.includes("possible_fall") || signals.includes("ml_fall_down");
            const lying = signals.includes("person_on_ground") || signals.includes("ml_lying_down");
            const unstable = signals.includes("unstable_movement");
            const coordination = unstable || signals.some(signal => [
                "repeated_body_sway", "unstable_trajectory", "irregular_step_pattern",
                "upper_body_instability", "sudden_balance_loss",
            ].includes(signal));
            const smoking = signals.includes("smoking_detected") ||
                (signals.includes("hand_to_mouth_pattern") && !coordination);
            const title = fall ? "Wykryto upadek"
                        : lying ? "Wykryto osobę na ziemi"
                        : unstable ? "Niestabilny ruch — weryfikacja"
                        : smoking ? "WYKRYTO PALENIE"
                        : "Nietypowa koordynacja";
            items.push({
                severity: p.severity,
                time: p.timestamp,
                desc: smoking
                    ? title
                    : title + (details.length ? ": " + details.join(", ") : ""),
                kind: fall ? "fall_detected" : (smoking ? "smoking_gesture" : "posture_anomaly"),
                thumb: p.frame_thumbnail_url,
            });
        });
        (data.ppe_checks || []).forEach(c => {
            if (c.severity !== "DANGER" || !c.missing || !c.missing.length) return;
            const parts = c.missing.map(m => m === "hardhat" ? "kaska" : (m === "vest" ? "kamizelki" : m));
            items.push({
                severity: "DANGER",
                time: c.timestamp,
                desc: "Brak PPE: " + parts.join(" + "),
                kind: "ppe_missing",
                thumb: c.frame_thumbnail_url,
            });
        });
        (data.unidentified_workers || []).forEach(u => items.push({
            severity: u.severity || "WARNING",
            time: u.timestamp,
            desc: "Osoba bez znacznika identyfikacyjnego",
            kind: "unidentified_worker",
            thumb: u.frame_thumbnail_url,
        }));
        items.forEach(prependAlert);
    }

    function prependAlert(a) {
        const row = document.createElement("div");
        row.className = "px-alert-row " + (a.severity === "WARNING" ? "warning" : "danger");
        row.dataset.severity = a.severity;

        const thumb = document.createElement("div");
        thumb.className = "px-alert-thumb";
        if (a.thumb) thumb.style.backgroundImage = "url(" + a.thumb + ")";
        else         thumb.textContent = State.cameraId;

        const body = document.createElement("div");
        body.className = "px-alert-body";
        const line = document.createElement("div");
        line.className = "px-alert-line";
        const tag = document.createElement("span");
        tag.className = "msbp-tag " + (a.severity === "WARNING" ? "msbp-tag--warning" : "msbp-tag--red");
        tag.textContent = a.severity === "WARNING" ? "Warning" : "Danger";
        const desc = document.createElement("span");
        desc.className = "px-alert-desc";
        desc.textContent = a.desc;
        line.appendChild(tag);
        line.appendChild(desc);
        const meta = document.createElement("span");
        meta.className = "px-alert-meta";
        const t = a.time ? new Date(a.time * 1000).toLocaleTimeString("pl-PL") : "—";
        meta.textContent = t + " · " + a.kind;
        body.appendChild(line);
        body.appendChild(meta);

        row.appendChild(thumb);
        row.appendChild(body);
        alertsEmpty.style.display = "none";
        alertList.insertBefore(row, alertList.firstChild);
        const rows = alertList.querySelectorAll(".px-alert-row");
        if (rows.length > MAX_ALERTS_IN_UI) rows[rows.length - 1].remove();
        State.alertsCount++;
        alertTotal.textContent = State.alertsCount;
        filterAlertsUI();
    }


    let ppeHideTimeout = null;
    function updatePPE(data) {
        if (State.mode !== "checkpoint") {
            ppeBadge.classList.add("hidden");
            return;
        }
        const checks = data.ppe_checks || [];
        if (!checks.length) return;
        const c = checks[0];
        const ok = c.severity === "OK";
        ppeBadge.classList.remove("hidden", "ok", "fail");
        ppeBadge.classList.add(ok ? "ok" : "fail");
        ppeHeadline.textContent = ok
            ? "PPE OK"
            : "Brak: " + c.missing.map(x => x.toUpperCase()).join(" + ");
        setPpeItem(ppeHardhat, c.has_hardhat);
        setPpeItem(ppeVest, c.has_vest);
        if (ppeHideTimeout) clearTimeout(ppeHideTimeout);
        ppeHideTimeout = setTimeout(() => { ppeBadge.classList.add("hidden"); }, 6000);
    }
    function setPpeItem(el, ok) {
        el.classList.remove("ok", "fail");
        el.classList.add(ok ? "ok" : "fail");
        el.querySelector(".px-ppe-mark").textContent = ok ? "✓" : "✗";
    }


    const POSTURE_SIGNAL_LABELS = {
        repeated_body_sway: "kołysanie tułowia",
        unstable_trajectory: "niestabilny tor ruchu",
        irregular_step_pattern: "nieregularny krok",
        upper_body_instability: "niestabilna postawa",
        sudden_balance_loss: "utrata równowagi",
        possible_fall: "możliwy upadek / osunięcie",
        hand_to_mouth_pattern: "powtarzalny gest ręka–usta",
        smoking_detected: "WYKRYTO PALENIE",
        ml_fall_down: "TCN+GRU: upadek",
        ml_lying_down: "TCN+GRU: pozycja leżąca",
        fall_suspected: "podejrzenie upadku",
        fall_detected: "potwierdzony upadek",
        person_on_ground: "osoba na ziemi",
        unstable_movement: "niestabilny ruch",
    };

    const BEHAVIOR_LABELS = {
        fall_down: "upadek",
        lying_down: "leżenie",
        sit_down: "siadanie",
        sitting: "siedzenie",
        stand_up: "wstawanie",
        standing: "stanie",
        walking: "chodzenie",
        unstable_gait: "niestabilny chód",
        other: "inna czynność",
    };

    function updatePosture(data) {
        if (!data.posture_available) {
            postureStatus.textContent = "postura wyłączona";
            postureBadge.classList.add("hidden");
            return;
        }
        postureStatus.textContent = data.behavior_classifier_available
            ? "postura + TCN/GRU aktywne" : "postura aktywna";
        const assessments = (data.posture_assessments || []).slice();
        if (!assessments.length) {
            postureBadge.classList.add("hidden");
            return;
        }
        assessments.sort((a, b) => (b.risk_score || 0) - (a.risk_score || 0));
        const p = assessments[0];
        if (p.status === "normal" || p.status === "collecting_history" ||
            String(p.status || "").startsWith("insufficient_")) {
            postureBadge.classList.add("hidden");
            return;
        }
        postureBadge.classList.remove("hidden", "observe", "warning", "danger");
        const cls = p.severity === "DANGER" ? "danger"
                  : p.severity === "WARNING" ? "warning" : "observe";
        postureBadge.classList.add(cls);
        const statusLabels = {
            observation: "OBSERWACJA",
            verification_required: "WYMAGA WERYFIKACJI",
            high_risk: "WYSOKIE RYZYKO",
            awaiting_ground_confirmation: "PODEJRZENIE UPADKU",
            confirmed: "ZDARZENIE POTWIERDZONE",
            cooldown: "ZDARZENIE W COOLDOWN",
        };
        const smoking = (p.signals || []).includes("smoking_detected");
        postureScore.textContent = smoking
            ? "WYKRYTO PALENIE"
            : (statusLabels[p.status] || p.status);
        const labels = (p.signals || []).slice(0, 3).map(x => POSTURE_SIGNAL_LABELS[x] || x);
        if (p.behavior_label) {
            const behaviorName = BEHAVIOR_LABELS[p.behavior_label] || p.behavior_label;
            labels.unshift("TCN: " + behaviorName);
        }
        postureSignals.textContent = labels.length ? labels.slice(0, 4).join(" · ") : "nietypowy wzorzec ruchu";
    }



    const workerUiTracks = new Map();
    // The backend refreshes the cached identity against the current detector
    // box on every frame. Keep only a short UI grace period so a deleted track
    // cannot leave a worker profile visible for several seconds.
    const WORKER_UI_HOLD_MS = 900;
    const WORKER_UI_LIVE_GRACE_MS = 500;

    function workerUiTrackKey(identity) {
        if (identity && identity.track_id !== null &&
            identity.track_id !== undefined) {
            return "track:" + String(identity.track_id);
        }
        return "worker:" + String(identity && identity.worker_id || "");
    }

    function stableWorkerIdentities(rawIdentities) {
        const now = performance.now();
        (rawIdentities || []).forEach(identity => {
            if (!identity || !identity.worker_id) return;
            const key = workerUiTrackKey(identity);
            const previous = workerUiTracks.get(key);
            const sameWorker = previous &&
                previous.identity.worker_id === identity.worker_id;
            const directRead = identity.cached !== true;
            workerUiTracks.set(key, {
                identity: Object.assign(
                    {},
                    sameWorker ? previous.identity : {},
                    identity,
                ),
                lastSeenAt: now,
                lastLiveAt: directRead
                    ? now
                    : (sameWorker ? previous.lastLiveAt : now),
            });
        });
        const stable = [];
        workerUiTracks.forEach((track, key) => {
            if (now - track.lastSeenAt > WORKER_UI_HOLD_MS) {
                workerUiTracks.delete(key);
                return;
            }
            const liveAgeMs = Math.max(0, now - track.lastLiveAt);
            stable.push(Object.assign({}, track.identity, {
                ui_live_age_ms: liveAgeMs,
                ui_held_only: liveAgeMs > WORKER_UI_LIVE_GRACE_MS,
            }));
        });
        return stable;
    }

    function setTextIfChanged(element, value) {
        if (!element) return;
        const next = String(value == null ? "" : value);
        if (element.textContent !== next) element.textContent = next;
    }

    function setHiddenState(element, hidden) {
        if (!element) return;
        if (element.classList.contains("hidden") !== !!hidden) {
            element.classList.toggle("hidden", !!hidden);
        }
    }

    function workerIdentityName(identity) {
        if (!identity) return "";
        return (identity.full_name ||
            [identity.first_name, identity.last_name].filter(Boolean).join(" ")).trim();
    }

    function updateWorkers(data) {
        const identities = stableWorkerIdentities(data.worker_identifications || []);
        const unidentified = data.unidentified_workers || [];
        if (!data.worker_identification_available && identities.length === 0) {
            setTextIfChanged(workerStatus, "znaczniki wyłączone");
            setHiddenState(workerBadge, true);
            return;
        }
        setTextIfChanged(
            workerStatus,
            identities.length
                ? "znaczniki " + identities.length
                : (unidentified.length ? "brak znacznika " + unidentified.length : "znaczniki aktywne")
        );
        if (!identities.length && !unidentified.length) {
            setHiddenState(workerBadge, true);
            return;
        }
        const unique = new Map();
        identities.forEach(identity => {
            if (!identity.worker_id) return;
            const previous = unique.get(identity.worker_id);
            if (!previous || (!workerIdentityName(previous) && workerIdentityName(identity))) {
                unique.set(identity.worker_id, identity);
            }
        });
        const identified = Array.from(unique.values());
        setTextIfChanged(
            workerIds,
            identified.length
                ? identified.map(identity => {
                    const name = workerIdentityName(identity);
                    return name || "ID " + identity.worker_id;
                }).join(" · ")
                : "BRAK ID"
        );

        const profileMeta = identified
            .filter(identity => workerIdentityName(identity))
            .map(identity => [
                "ID " + identity.worker_id,
                identity.position,
                identity.department,
            ].filter(Boolean).join(" · "))
            .join(" | ");

        const heldOnly = identified.length > 0 && identified.every(identity => identity.ui_held_only);
        const trackingMeta = unidentified.length
            ? "osoby bez widocznego znacznika: " + unidentified.length
            : heldOnly
            ? "identyfikacja podtrzymana"
            : "znacznik pracownika aktywny";
        setTextIfChanged(workerMeta, [profileMeta, trackingMeta].filter(Boolean).join(" • "));
        setHiddenState(workerBadge, false);
    }


    function maybeRaiseAlarm(data) {
        if (State.activeAlarm) return;

        const zoneBreach = (data.confirmed_zone_breaches || []).find(
            b => b.severity === "DANGER" && b.rule_name !== "zone_approach"
        );
        if (zoneBreach) {
            const t = zoneBreach.timestamp ?
                new Date(zoneBreach.timestamp * 1000).toLocaleTimeString("pl-PL") : "—";
            raiseAlarm(
                "zone_breach",
                "STOP — Osoba w strefie: " + (zoneBreach.zone_name || "?"),
                t + " · " + State.cameraId + " · reguła zone_breach",
                zoneBreach.id
            );
            return;
        }

        const siteHazard = (data.confirmed_alerts || []).find(a => a.severity === "DANGER");
        if (siteHazard) {
            const t = siteHazard.timestamp ?
                new Date(siteHazard.timestamp * 1000).toLocaleTimeString("pl-PL") : "—";
            const distance = formatHazardDistance(siteHazard);
            const metricMode = siteHazard.calibrated ? "pomiar metryczny" : "tryb obrazu";
            raiseAlarm(
                "site_hazard",
                "STOP — Krytyczna strefa maszyny" + distance,
                t + " · " + State.cameraId + " · " + metricMode + " · reguła " + siteHazard.rule_name,
                siteHazard.id
            );
            return;
        }

        const postureDanger = (data.confirmed_posture_alerts || []).find(p => p.severity === "DANGER");
        if (postureDanger) {
            const t = postureDanger.timestamp ?
                new Date(postureDanger.timestamp * 1000).toLocaleTimeString("pl-PL") : "—";
            const postureSignals = postureDanger.signals || [];
            const fall = postureSignals.includes("fall_detected") ||
                postureSignals.includes("fall_suspected") ||
                postureSignals.includes("possible_fall") || postureSignals.includes("ml_fall_down");
            const ground = postureSignals.includes("person_on_ground") ||
                postureSignals.includes("ml_lying_down");
            raiseAlarm(
                fall ? "fall_detected" : (ground ? "person_on_ground" : "posture_anomaly"),
                fall ? "STOP — Wykryto upadek pracownika"
                     : ground ? "STOP — Wykryto osobę na ziemi"
                     : "STOP — Wykryto niestabilny ruch",
                t + " · " + State.cameraId + " · wymagana weryfikacja człowieka",
                postureDanger.id
            );
            return;
        }

        const ppeFail = (data.ppe_checks || []).find(c => c.severity === "DANGER" && c.missing && c.missing.length);
        if (ppeFail && State.mode === "checkpoint") {
            const missing = ppeFail.missing.map(m => m === "hardhat" ? "kaska" : (m === "vest" ? "kamizelki" : m));
            const t = ppeFail.timestamp ?
                new Date(ppeFail.timestamp * 1000).toLocaleTimeString("pl-PL") : "—";
            raiseAlarm(
                "ppe_missing",
                "BRAK " + missing.join(" + ").toUpperCase() + " — WEJŚCIE WSTRZYMANE",
                t + " · " + State.cameraId + " · bramka główna · komunikat głosowy odtworzony",
                ppeFail.id
            );
        }
    }


    function selectCamera(cameraId) {
        State.cameraId = cameraId || "cam_default";
        workerUiTracks.clear();
        lastRenderedTs = 0;
        frameRenderSequence++;
        pendingBinaryFrames.length = 0;
        noSignal.classList.remove("hidden");
        try { localStorage.setItem("perimetr:camera_id", State.cameraId); } catch (_) {}
        const u = new URL(location.href);
        u.searchParams.set("camera_id", State.cameraId);
        history.replaceState(null, "", u);
        document.dispatchEvent(new CustomEvent("perimetr-camera-changed", { detail: State.cameraId }));
        syncRuntimeProcessing(0);
        connectWebSocket();
    }

    function refreshCameras() {
        fetch("/api/cameras", { credentials: "same-origin" })
            .then(r => r.ok ? r.json() : { cameras: [] })
            .then(data => {
                const rows = data.cameras || [];
                const ids = rows.map(row => row.camera_id);
                if (!ids.includes(State.cameraId)) ids.unshift(State.cameraId);
                cameraSelect.innerHTML = "";
                ids.forEach(id => {
                    const row = rows.find(item => item.camera_id === id);
                    const option = document.createElement("option");
                    option.value = id;
                    option.textContent = id + (row && row.online ? " · online" : "");
                    option.selected = id === State.cameraId;
                    cameraSelect.appendChild(option);
                });
            }).catch(() => {});
    }

    cameraSelect.addEventListener("change", () => selectCamera(cameraSelect.value));
    refreshCameras();
    setInterval(refreshCameras, 5000);

    filterAlertsUI();
    syncRuntimeProcessing(0);
    connectWebSocket();
    void restorePersistedDemoJob();

})();
