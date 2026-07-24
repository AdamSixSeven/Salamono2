/* ============================================================
   Perimetr — Panel kierownika, app runtime
   ============================================================ */
(function () {
    "use strict";

    // ---------- State store (prosty pub/sub) -----------------------

    const State = {
        wsStatus: "connecting",
        stats: { fps: 0, detectionMs: 0, frameCount: 0 },
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
                    // Older saved preferences do not contain this key, so
                    // MediaPipe remains visible until POS is explicitly off.
                    posture: parsed.posture !== false,
                    zones: parsed.zones !== false,
                    markers: parsed.markers !== false,
                    distances: parsed.distances !== false,
                };
            }
        } catch (_) { /* ignore */ }
        return {
            boxes: true,
            posture: true,
            zones: true,
            markers: true,
            distances: true,
        };
    }
    function saveLayers() {
        try { localStorage.setItem("perimetr:layers", JSON.stringify(State.layers)); } catch (_) { }
    }

    // ---------- DOM refs ----------------------------------------

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
    const demoVideoInput = document.getElementById("demoVideoInput");
    const modeSegmented = document.getElementById("modeSegmented");
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
    const layerTags = document.querySelectorAll("[data-layer]");
    const cameraSelect = document.getElementById("cameraSelect");

    window.Perimetr = window.Perimetr || {};
    window.Perimetr.getCameraId = () => State.cameraId;
    window.Perimetr.getLayers = () => ({ ...State.layers });

    // ---------- Init: layer buttons ---------------------------

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
        });
    });
    syncLayerButtons();

    // ---------- Init: mode segmented ---------------------------

    modeSegmented.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => {
            State.mode = btn.dataset.mode;
            modeSegmented.querySelectorAll("button").forEach(b => {
                b.classList.toggle("active", b === btn);
            });
            // hide PPE badge when leaving checkpoint
            if (State.mode !== "checkpoint") ppeBadge.classList.add("hidden");
            document.dispatchEvent(new CustomEvent("perimetr-mode", { detail: State.mode }));
        });
    });

    // ---------- Init: alarms filter ---------------------------

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

    // ---------- Alarm banner controls ----------------------------

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
                // The local alarm may still be cleared if the network briefly
                // fails; the incident remains in history with status `new`.
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

    // ---------- Snapshot + phone pair ---------------------------

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


    // ---------- Browser video demo -------------------------------

    let demoVideoAbort = false;
    let demoVideoRunning = false;

    const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

    async function postDemoFrame(canvasEl, timestamp) {
        const blob = await new Promise(resolve => canvasEl.toBlob(resolve, "image/jpeg", 0.78));
        if (!blob) return;
        const form = new FormData();
        form.append("image", blob, "demo-frame.jpg");
        form.append("camera_id", State.cameraId);
        form.append("timestamp", String(timestamp));
        form.append("mode", State.mode);
        const response = await fetch("/api/frame", {
            method: "POST",
            body: form,
            credentials: "same-origin",
        });
        if (!response.ok) throw new Error("HTTP " + response.status);
    }

    async function runDemoVideo(file) {
        if (!file || demoVideoRunning) return;
        demoVideoRunning = true;
        demoVideoAbort = false;
        const oldLabel = demoVideoBtn.textContent;
        demoVideoBtn.textContent = "Zatrzymaj film";
        demoVideoBtn.classList.add("px-btn--solid");

        const demoCamera = "demo_upload";
        selectCamera(demoCamera);
        cameraSelect.value = demoCamera;

        const video = document.createElement("video");
        video.muted = true;
        video.playsInline = true;
        video.preload = "auto";
        const objectUrl = URL.createObjectURL(file);
        video.src = objectUrl;
        const capture = document.createElement("canvas");
        const cctx = capture.getContext("2d");

        try {
            await new Promise((resolve, reject) => {
                video.onloadedmetadata = resolve;
                video.onerror = () => reject(new Error("Nie można odczytać filmu"));
            });
            const scale = Math.min(1, 1280 / Math.max(1, video.videoWidth));
            capture.width = Math.max(2, Math.round(video.videoWidth * scale));
            capture.height = Math.max(2, Math.round(video.videoHeight * scale));
            await video.play();

            const sampleIntervalMs = 200; // 5 FPS: enough for posture and fast CPU demo
            while (!video.ended && !demoVideoAbort) {
                const started = performance.now();
                cctx.drawImage(video, 0, 0, capture.width, capture.height);
                await postDemoFrame(capture, Date.now() / 1000);
                const remaining = sampleIntervalMs - (performance.now() - started);
                if (remaining > 0) await sleep(remaining);
            }
        } catch (err) {
            console.error("Demo video failed", err);
            alert("Nie udało się przeanalizować filmu: " + (err.message || err));
        } finally {
            video.pause();
            URL.revokeObjectURL(objectUrl);
            demoVideoRunning = false;
            demoVideoAbort = false;
            demoVideoBtn.textContent = oldLabel || "Wczytaj film";
            demoVideoBtn.classList.remove("px-btn--solid");
            demoVideoInput.value = "";
        }
    }

    demoVideoBtn.addEventListener("click", () => {
        if (demoVideoRunning) {
            demoVideoAbort = true;
            return;
        }
        demoVideoInput.click();
    });
    demoVideoInput.addEventListener("change", () => runDemoVideo(demoVideoInput.files[0]));

    // ---------- Phone pair QR modal ----------------------------
    const pairModal = document.getElementById("pairModal");
    const pairModalClose = document.getElementById("pairModalClose");
    const pairQrImg = document.getElementById("pairQrImg");
    const pairUrlInput = document.getElementById("pairUrl");
    const pairCopyBtn = document.getElementById("pairCopyBtn");

    // The explicit server parameter keeps phone uploads pointed at this
    // backend when the QR is displayed inside a different-origin embed.
    function captureUrl() {
        const u = new URL("/phone/capture.html", location.origin);
        u.searchParams.set("server", location.origin);
        u.searchParams.set("camera_id", State.cameraId);
        return u.toString();
    }

    function openPairModal() {
        const url = captureUrl();
        pairUrlInput.value = url;
        // QR is rendered server-side, avoiding an external CDN dependency.
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

    // ---------- Worker directory + printable QR -----------------

    let selectedWorkerId = null;
    let workerRegistryBusy = false;
    const workersById = new Map();
    const workerRequiredFields = [
        [workerQrInput, "ID QR"],
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

function workerQrPngUrl(workerId, download) {
    const params = new URLSearchParams({
        worker_id: workerId,
    });

    if (download) {
        params.set("download", "true");
    }

    return "/api/worker-qr.png?" + params.toString();
}

function workerQrFilename(workerId) {
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
        workerQrPreviewTitle.textContent = "QR pracownika";
    }

    if (workerQrPreviewPayload) {
        workerQrPreviewPayload.textContent = "worker:—";
    }
}

function showWorkerQrPreview(workerId) {
    if (!workerQrPreview || !workerId) {
        return;
    }

    const previewUrl = workerQrPngUrl(workerId, false);
    const downloadUrl = workerQrPngUrl(workerId, true);
    const filename = workerQrFilename(workerId);

    if (workerQrPreviewImg) {
        workerQrPreviewImg.src = previewUrl;
        workerQrPreviewImg.alt = "Kod QR pracownika " + workerId;
    }

    if (workerQrPreviewTitle) {
        workerQrPreviewTitle.textContent = "QR · " + workerId;
    }

    if (workerQrPreviewPayload) {
        workerQrPreviewPayload.textContent = "worker:" + workerId;
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
            setWorkerRegistryStatus("Pola oznaczone * są wymagane.");
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

    function generateWorkerQrPng() {
        const workerId = (workerQrInput.value || "").trim();
        if (!workerId) {
            workerQrInput.focus();
            setWorkerRegistryStatus("Najpierw wpisz ID pracownika.", "error");
            return;
        }

        showWorkerQrPreview(workerId);

        const link = document.createElement("a");
        link.href = workerQrPngUrl(workerId, true);
        link.download = workerQrFilename(workerId);
        link.style.display = "none";
        document.body.appendChild(link);
        link.click();
        link.remove();

        setWorkerRegistryStatus(
            "Wygenerowano i pobrano PNG dla identyfikatora " + workerId + ".",
            "success"
        );
    }
    workerQrOpenBtn.addEventListener("click", generateWorkerQrPng);
    loadWorkerRegistry().catch(() => {});

    // ---------- WebSocket ----------------------------------

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

    function onWsMessage(evt) {
        if (typeof evt.data !== "string") {
            const metadata = pendingBinaryFrames.shift();
            if (metadata) renderBinaryFrame(evt.data, metadata);
            return;
        }
        let data; try { data = JSON.parse(evt.data); } catch (_) { return; }
        if ((data.camera_id || "cam_default") !== State.cameraId) return;
        const ts = data.timestamp || 0;
        if (ts && ts < lastRenderedTs) return;
        lastRenderedTs = ts;

        if (data.frame_transport === "binary-jpeg") {
            pendingBinaryFrames.push(data);
            if (pendingBinaryFrames.length > 2) pendingBinaryFrames.shift();
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

        // For zones.js — draws marker-zone polygons and other overlays
        document.dispatchEvent(new CustomEvent("perimetr-frame", { detail: data }));
    }

    // ---------- Frame render ---------------------------------

    let frameRenderSequence = 0;

    function applyDecodedFrame(image, width, height, data, sequence) {
        if (sequence !== frameRenderSequence) {
            if (image && typeof image.close === "function") image.close();
            return;
        }
        canvas.width = width;
        canvas.height = height;
        videoEl.style.aspectRatio = `${width} / ${height}`;
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
            // Image decoding is asynchronous.  Never let an older, slower
            // decode replace a newer frame and its matching client overlays.
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

    // ---------- Stats + chips ------------------------------

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

    // ---------- Alerts list --------------------------------

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
            const fall = signals.includes("possible_fall");
            const coordination = signals.some(signal => [
                "repeated_body_sway", "unstable_trajectory", "irregular_step_pattern",
                "upper_body_instability", "sudden_balance_loss",
            ].includes(signal));
            const smoking = signals.includes("hand_to_mouth_pattern") && !coordination;
            const title = fall ? "Możliwy upadek"
                        : smoking ? "Możliwy gest palenia"
                        : "Nietypowa koordynacja";
            items.push({
                severity: p.severity,
                time: p.timestamp,
                desc: title + (details.length ? ": " + details.join(", ") : ""),
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
            desc: "Osoba bez identyfikatora QR",
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
        // insert on top; alertsEmpty is kept at the end and hidden when there are rows
        alertList.insertBefore(row, alertList.firstChild);
        const rows = alertList.querySelectorAll(".px-alert-row");
        if (rows.length > MAX_ALERTS_IN_UI) rows[rows.length - 1].remove();
        State.alertsCount++;
        alertTotal.textContent = State.alertsCount;
        filterAlertsUI();
    }

    // ---------- PPE badge (state 1d) --------------------

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

    // ---------- Posture / coordination badge ----------------

    const POSTURE_SIGNAL_LABELS = {
        repeated_body_sway: "kołysanie tułowia",
        unstable_trajectory: "niestabilny tor ruchu",
        irregular_step_pattern: "nieregularny krok",
        upper_body_instability: "niestabilna postawa",
        sudden_balance_loss: "utrata równowagi",
        possible_fall: "możliwy upadek / osunięcie",
        hand_to_mouth_pattern: "powtarzalny gest ręka–usta",
    };

    function updatePosture(data) {
        if (!data.posture_available) {
            postureStatus.textContent = "postura wyłączona";
            postureBadge.classList.add("hidden");
            return;
        }
        postureStatus.textContent = "postura aktywna";
        const assessments = (data.posture_assessments || []).slice();
        if (!assessments.length) {
            postureBadge.classList.add("hidden");
            return;
        }
        assessments.sort((a, b) => (b.risk_score || 0) - (a.risk_score || 0));
        const p = assessments[0];
        if (p.status === "normal" || p.status === "collecting_history") {
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
        };
        postureScore.textContent = (statusLabels[p.status] || p.status) +
            " · " + Math.round((p.risk_score || 0) * 100) + "%";
        const labels = (p.signals || []).slice(0, 3).map(x => POSTURE_SIGNAL_LABELS[x] || x);
        postureSignals.textContent = labels.length ? labels.join(" · ") : "nietypowy wzorzec ruchu";
    }


    // ---------- Worker identification (visible QR tag) --------

    function workerIdentityName(identity) {
        if (!identity) return "";
        return (identity.full_name ||
            [identity.first_name, identity.last_name].filter(Boolean).join(" ")).trim();
    }

    function updateWorkers(data) {
        const identities = data.worker_identifications || [];
        const unidentified = data.unidentified_workers || [];
        if (!data.worker_identification_available && identities.length === 0) {
            workerStatus.textContent = "QR wyłączone";
            workerBadge.classList.add("hidden");
            return;
        }
        workerStatus.textContent = identities.length
            ? "QR " + identities.length
            : (unidentified.length ? "QR brak " + unidentified.length : "QR aktywne");
        if (!identities.length && !unidentified.length) {
            workerBadge.classList.add("hidden");
            return;
        }
        const unique = new Map();
        identities.forEach(identity => {
            if (!identity.worker_id) return;
            const previous = unique.get(identity.worker_id);
            // Prefer the enriched occurrence if the same ID appears more than once.
            if (!previous || (!workerIdentityName(previous) && workerIdentityName(identity))) {
                unique.set(identity.worker_id, identity);
            }
        });
        const identified = Array.from(unique.values());
        workerIds.textContent = identified.length
            ? identified.map(identity => {
                const name = workerIdentityName(identity);
                return name || "ID " + identity.worker_id;
            }).join(" · ")
            : "BRAK ID";

        const profileMeta = identified
            .filter(identity => workerIdentityName(identity))
            .map(identity => [
                "ID " + identity.worker_id,
                identity.position,
                identity.department,
            ].filter(Boolean).join(" · "))
            .join(" | ");
        const cached = identities.filter(identity => identity.cached).length;
        const trackingMeta = unidentified.length
            ? "osoby bez widocznego/cached QR: " + unidentified.length
            : cached
            ? "utrzymano po chwilowym zasłonięciu: " + cached
            : "odczytano widoczny znacznik QR";
        workerMeta.textContent = [profileMeta, trackingMeta].filter(Boolean).join(" • ");
        workerBadge.classList.remove("hidden");
    }

    // ---------- Alarm banner (state 1c) trigger ------------

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
            const fall = (postureDanger.signals || []).includes("possible_fall");
            raiseAlarm(
                fall ? "fall_detected" : "posture_anomaly",
                fall ? "STOP — Możliwy upadek pracownika" : "STOP — Możliwa utrata koordynacji ruchowej",
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

    // ---------- Bootstrap ---------------------------

    function selectCamera(cameraId) {
        State.cameraId = cameraId || "cam_default";
        lastRenderedTs = 0;
        frameRenderSequence++;
        pendingBinaryFrames.length = 0;
        noSignal.classList.remove("hidden");
        try { localStorage.setItem("perimetr:camera_id", State.cameraId); } catch (_) {}
        const u = new URL(location.href);
        u.searchParams.set("camera_id", State.cameraId);
        history.replaceState(null, "", u);
        document.dispatchEvent(new CustomEvent("perimetr-camera-changed", { detail: State.cameraId }));
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
    connectWebSocket();

})();
