(() => {
    "use strict";

    const $ = (id) => document.getElementById(id);
    const STORAGE_KEY = "perimetr.distanceVideoId";
    const live = {
        camera: $("distanceCamera"), max: $("distanceMax"), rate: $("distanceRate"),
        start: $("distanceStart"), once: $("distanceOnce"), stop: $("distanceStop"),
        status: $("distanceStatus"), image: $("distanceImage"), empty: $("distanceEmpty"),
        pairs: $("distancePairs"),
    };
    const clip = {
        liveTab: $("distanceLiveTab"), clipTab: $("distanceClipTab"),
        livePanel: $("distanceLivePanel"), clipPanel: $("distanceClipPanel"),
        form: $("clipUploadForm"), file: $("clipFile"), upload: $("clipUpload"),
        uploadStatus: $("clipUploadStatus"), metadata: $("clipMetadata"),
        metaName: $("clipMetaName"), metaResolution: $("clipMetaResolution"),
        metaFps: $("clipMetaFps"), metaFrames: $("clipMetaFrames"),
        metaDuration: $("clipMetaDuration"), workspace: $("clipWorkspace"),
        remove: $("clipRemove"), video: $("clipVideo"), playerMessage: $("clipPlayerMessage"),
        seek: $("clipSeek"), currentTime: $("clipCurrentTime"), duration: $("clipDuration"),
        previous: $("clipPreviousFrame"), next: $("clipNextFrame"),
        max: $("clipMaxDistance"), analyze: $("clipAnalyze"),
        analysisStatus: $("clipAnalysisStatus"), analysisMetadata: $("clipAnalysisMetadata"),
        requestedTime: $("clipRequestedTime"), actualTime: $("clipActualTime"),
        frameIndex: $("clipFrameIndex"), analysisState: $("clipAnalysisState"),
        analysisCounts: $("clipAnalysisCounts"), resultImage: $("clipResultImage"),
        resultEmpty: $("clipResultEmpty"), pairs: $("clipPairs"),
        detections: $("clipDetections"),
    };

    let liveTimer = null;
    let liveInFlight = false;
    let resumeLiveAfterClip = false;
    let clipId = null;
    let clipMeta = null;
    let uploading = false;
    let analyzing = false;
    let generation = 0;
    let clipAnalysisController = null;
    let clipSeekDragging = false;

    function numberOrNull(value) {
        if (value === null || value === undefined || value === "") return null;
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
    }

    function setStatus(node, text, kind = "") {
        node.textContent = text;
        node.className = "dist-status" + (kind ? ` ${kind}` : "");
    }

    function emptyList(node, text) {
        const child = document.createElement("div");
        child.className = "dist-empty";
        child.textContent = text;
        node.replaceChildren(child);
    }

    function confidence(value) {
        const parsed = numberOrNull(value);
        return parsed === null ? "—" : `${(parsed * 100).toFixed(0)}%`;
    }

    function formatTime(value) {
        const parsed = numberOrNull(value);
        const seconds = Math.max(0, parsed === null ? 0 : parsed);
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const whole = Math.floor(seconds % 60);
        const hundredths = Math.floor((seconds - Math.floor(seconds)) * 100 + 1e-6);
        const prefix = hours ? `${String(hours).padStart(2, "0")}:` : "";
        return `${prefix}${String(minutes).padStart(2, "0")}:${String(whole).padStart(2, "0")}.${String(hundredths).padStart(2, "0")}`;
    }

    function secondsLabel(value) {
        const parsed = numberOrNull(value);
        return parsed === null ? "—" : `${parsed.toFixed(3)} s`;
    }

    function normalizedStatus(value) {
        return String(value || "").trim().toLowerCase();
    }

    function statusLabel(value) {
        const state = normalizedStatus(value);
        return ({danger: "DANGER", warning: "WARNING", safe: "SAFE", no_pairs: "BRAK PAR",
            calibration_incompatible: "NIEZGODNA KALIBRACJA",
            calibration_unavailable: "BRAK KALIBRACJI"})[state] || (state ? state.toUpperCase() : "—");
    }

    function oneBased(value) {
        const parsed = numberOrNull(value);
        return parsed === null ? "?" : String(Math.max(0, Math.trunc(parsed)) + 1);
    }

    async function apiJson(url, options = {}) {
        const response = await fetch(url, options);
        if (response.status === 204) return null;
        let body = null;
        try { body = await response.json(); } catch (_) {
            try { body = await response.text(); } catch (_) { body = null; }
        }
        if (!response.ok) {
            const detail = body && typeof body === "object" ? body.detail : null;
            let message = `HTTP ${response.status}`;
            if (typeof detail === "string") message = detail;
            else if (Array.isArray(detail)) message = detail.map((item) => item.msg || String(item)).join("; ");
            else if (detail && typeof detail === "object") message = detail.message || detail.error || detail.code || message;
            else if (body && typeof body === "object") message = body.message || body.error || message;
            else if (typeof body === "string" && body.trim()) message = body.trim();
            const error = new Error(message);
            error.httpStatus = response.status;
            error.payload = body;
            error.code = normalizedStatus(
                (body && typeof body === "object" && (body.status || body.code)) ||
                (detail && typeof detail === "object" && (detail.status || detail.code))
            );
            throw error;
        }
        return body;
    }

    async function loadCameras() {
        try {
            const response = await fetch("/api/cameras", {
                cache: "no-store",
                credentials: "same-origin",
            });
            if (!response.ok) return;
            const rows = (await response.json()).cameras || [];
            const previous = live.camera.value;
            const ids = rows.map((row) => row.camera_id).filter(Boolean);
            if (!ids.includes("cam_default")) ids.push("cam_default");
            live.camera.replaceChildren();
            [...new Set(ids)].forEach((id) => {
                const option = document.createElement("option");
                const row = rows.find((item) => item.camera_id === id);
                option.value = id;
                option.textContent = row ? `${id}${row.online ? " · online" : " · offline"}` : id;
                live.camera.appendChild(option);
            });
            if ([...live.camera.options].some((option) => option.value === previous)) live.camera.value = previous;
        } catch (_) {}
    }

    function renderLivePairs(items) {
        if (!Array.isArray(items) || !items.length) {
            emptyList(live.pairs, "Brak pomiarów dla bieżącej klatki.");
            return;
        }
        live.pairs.replaceChildren();
        items.slice(0, 80).forEach((item) => {
            const node = document.createElement("div");
            node.className = "dist-item";
            const strong = document.createElement("strong");
            const distance = numberOrNull(item.distance_m);
            strong.textContent = distance === null ? "—" : `${distance.toFixed(2)} m`;
            const details = document.createElement("span");
            details.textContent = `${item.person_class || "person"} ${confidence(item.person_confidence)} ↔ ${item.hazard_class || "machine"} ${confidence(item.hazard_confidence)}`;
            node.append(strong, details);
            live.pairs.appendChild(node);
        });
    }

    async function refreshLive() {
        if (liveInFlight) return;
        liveInFlight = true;
        try {
            const params = new URLSearchParams({camera_id: live.camera.value, max_width: "1280"});
            const maxDistance = numberOrNull(live.max.value);
            if (maxDistance !== null && maxDistance > 0) {
                params.set("max_distance_m", String(maxDistance));
            }
            const data = await apiJson(`/api/distance/latest?${params.toString()}`, {
                cache: "no-store",
                credentials: "same-origin",
            });
            const calibration = data.calibration || {};
            $("metaFrame").textContent = data.frame_available ? `${data.frame_width}×${data.frame_height}` : "—";
            $("metaPersons").textContent = data.person_count ?? "—";
            $("metaHazards").textContent = data.hazard_count ?? "—";
            $("metaPairs").textContent = data.pair_count ?? "—";

            if (data.preview_jpeg_b64) {
                live.image.src = `data:image/jpeg;base64,${data.preview_jpeg_b64}`;
                live.image.hidden = false;
                live.empty.hidden = true;
            } else {
                live.image.hidden = true;
                live.image.removeAttribute("src");
                live.empty.hidden = false;
                live.empty.textContent = data.message || "Brak klatki do wyświetlenia.";
            }
            renderLivePairs(data.pairs);

            if (!data.frame_available) {
                setStatus(live.status, data.message || "Brak świeżej klatki dla tej kamery.", "warning");
            } else if (!calibration.available) {
                setStatus(live.status, data.message || calibration.error || "Brak kalibracji modułu dystansu.", "error");
            } else if (calibration.compatible === false) {
                setStatus(live.status, data.message || calibration.warning || "Niezgodna kalibracja.", "error");
            } else if (!data.detection_exact) {
                setStatus(live.status, data.message || "Oczekiwanie na detekcję tej samej klatki.", "warning");
            } else {
                const age = numberOrNull(data.frame_age_sec);
                setStatus(
                    live.status,
                    `Gotowe · ${data.pair_count || 0} par` + (age === null ? "" : ` · wiek klatki ${age.toFixed(2)} s`),
                    "ok",
                );
            }
        } catch (error) {
            setStatus(live.status, `Błąd pobierania dystansu: ${error.message || error}`, "error");
        } finally {
            liveInFlight = false;
        }
    }

    function stopLive() {
        if (liveTimer !== null) {
            clearInterval(liveTimer);
            liveTimer = null;
        }
        live.start.disabled = false;
        live.stop.disabled = true;
    }

    function startLive() {
        stopLive();
        live.start.disabled = true;
        live.stop.disabled = false;
        void refreshLive();
        liveTimer = setInterval(refreshLive, Math.max(250, Number(live.rate.value) || 1000));
    }

    function switchMode(mode) {
        const isClip = mode === "clip";
        if (isClip && liveTimer !== null) {
            resumeLiveAfterClip = true;
            stopLive();
        } else if (!isClip && resumeLiveAfterClip) {
            resumeLiveAfterClip = false;
            startLive();
        }
        clip.liveTab.classList.toggle("active", !isClip);
        clip.clipTab.classList.toggle("active", isClip);
        clip.liveTab.setAttribute("aria-selected", String(!isClip));
        clip.clipTab.setAttribute("aria-selected", String(isClip));
        clip.livePanel.hidden = isClip;
        clip.clipPanel.hidden = !isClip;
    }

    function setClipControlsDisabled(disabled) {
        [clip.previous, clip.next, clip.analyze, clip.remove, clip.seek, clip.max].forEach((node) => {
            node.disabled = Boolean(disabled);
        });
    }

    function clipDurationSeconds() {
        const backendDuration = numberOrNull(clipMeta && clipMeta.duration_sec);
        const playerDuration = numberOrNull(clip.video.duration);
        if (playerDuration !== null && playerDuration > 0) return playerDuration;
        return backendDuration !== null && backendDuration > 0 ? backendDuration : 0;
    }

    function clipPositionSeconds() {
        const slider = numberOrNull(clip.seek.value);
        const player = numberOrNull(clip.video.currentTime);
        const playerHasMedia = Number(clip.video.readyState || 0) > 0;
        const value = !clipSeekDragging && playerHasMedia && player !== null
            ? player
            : (slider === null ? (player || 0) : slider);
        return Math.min(Math.max(0, value), clipDurationSeconds() || Math.max(0, value));
    }

    function syncTimeline(value) {
        const duration = clipDurationSeconds();
        const position = Math.min(Math.max(0, numberOrNull(value) || 0), duration || Number.MAX_SAFE_INTEGER);
        clip.seek.max = String(duration || 0);
        clip.seek.value = String(position);
        clip.currentTime.textContent = formatTime(position);
        clip.duration.textContent = formatTime(duration);
    }

    function resetAnalysisResult() {
        clip.analysisMetadata.hidden = true;
        clip.resultImage.hidden = true;
        clip.resultImage.removeAttribute("src");
        clip.resultEmpty.hidden = false;
        clip.resultEmpty.textContent = "Nie analizowano jeszcze żadnej klatki.";
        emptyList(clip.pairs, "Brak pomiarów.");
        emptyList(clip.detections, "Brak detekcji.");
        setStatus(clip.analysisStatus, "Przewiń film do wybranego momentu i uruchom analizę jednej klatki.");
    }

    function clearClipUi(message = "Wybierz plik. Film zostanie zapisany bez analizy klatek.") {
        generation += 1;
        if (clipAnalysisController) clipAnalysisController.abort();
        clipAnalysisController = null;
        analyzing = false;
        clipId = null;
        clipMeta = null;
        clip.video.pause();
        clip.video.removeAttribute("src");
        clip.video.load();
        clip.metadata.hidden = true;
        clip.workspace.hidden = true;
        clip.playerMessage.hidden = true;
        clip.file.value = "";
        syncTimeline(0);
        resetAnalysisResult();
        setStatus(clip.uploadStatus, message);
        try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
    }

    function installClip(metadata, options = {}) {
        clipId = String(metadata.id || metadata.job_id || "");
        clipMeta = metadata;
        if (!clipId) throw new Error("Backend nie zwrócił identyfikatora klipu.");
        clip.metaName.textContent = metadata.filename || "—";
        clip.metaResolution.textContent = metadata.width && metadata.height
            ? `${metadata.width}×${metadata.height}` : "—";
        const fps = numberOrNull(metadata.fps ?? metadata.source_fps);
        clip.metaFps.textContent = fps === null ? "—" : fps.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
        clip.metaFrames.textContent = metadata.frame_count ?? metadata.total_frames ?? "—";
        clip.metaDuration.textContent = formatTime(metadata.duration_sec);
        clip.metadata.hidden = false;
        clip.workspace.hidden = false;
        clip.playerMessage.hidden = true;
        setClipControlsDisabled(false);
        const source = metadata.video_url || `/api/distance/videos/${encodeURIComponent(clipId)}/content`;
        clip.video.src = source;
        clip.video.load();
        syncTimeline(0);
        resetAnalysisResult();
        setStatus(
            clip.uploadStatus,
            options.restored ? `Przywrócono klip: ${metadata.filename || clipId}` : `Klip gotowy: ${metadata.filename || clipId}`,
            "ok",
        );
        try { localStorage.setItem(STORAGE_KEY, clipId); } catch (_) {}
    }

    function renderClipPairs(items) {
        if (!Array.isArray(items) || !items.length) {
            emptyList(clip.pairs, "Brak par osoba ↔ maszyna dla tej klatki.");
            return;
        }
        clip.pairs.replaceChildren();
        items.forEach((item) => {
            const state = normalizedStatus(item.status);
            const node = document.createElement("div");
            node.className = `dist-item ${state}`.trim();

            const header = document.createElement("div");
            header.className = "dist-item-header";
            const relation = document.createElement("strong");
            relation.textContent = `${item.person_class || "person"} #${oneBased(item.person_index)} → ` +
                `${item.hazard_class || "machine"} #${oneBased(item.hazard_index)}`;
            const badge = document.createElement("span");
            badge.className = `dist-state ${state}`.trim();
            badge.textContent = statusLabel(state);
            header.append(relation, badge);

            const distance = document.createElement("strong");
            const distanceValue = numberOrNull(item.distance_m);
            distance.textContent = distanceValue === null ? "—" : `${distanceValue.toFixed(2)} m`;
            const details = document.createElement("span");
            details.textContent = `person ${confidence(item.person_confidence)} · maszyna ${confidence(item.hazard_confidence)}`;
            node.append(header, distance, details);
            clip.pairs.appendChild(node);
        });
    }

    function renderClipDetections(items) {
        if (!Array.isArray(items) || !items.length) {
            emptyList(clip.detections, "Brak detekcji Perimetr dla tej klatki.");
            return;
        }
        clip.detections.replaceChildren();
        items.forEach((item) => {
            const node = document.createElement("div");
            node.className = "dist-item";
            const title = document.createElement("strong");
            const index = numberOrNull(item.index);
            title.textContent = `${item.class_name || item.category || "obiekt"} #${index === null ? "?" : index + 1}`;
            const details = document.createElement("span");
            const box = Array.isArray(item.box) ? item.box.map((value) => Math.round(Number(value) || 0)).join(", ") : "—";
            details.textContent = `${item.category || "unknown"} · confidence ${confidence(item.confidence)} · bbox [${box}]`;
            node.append(title, details);
            clip.detections.appendChild(node);
        });
    }

    function analysisStatusKind(state) {
        if (["danger", "calibration_incompatible", "calibration_unavailable"].includes(state)) return "error";
        if (state === "warning") return "warning";
        return "ok";
    }

    function renderAnalysis(data) {
        const state = normalizedStatus(data.status || data.analysis_status);
        clip.requestedTime.textContent = secondsLabel(data.requested_time_sec);
        clip.actualTime.textContent = secondsLabel(data.actual_time_sec);
        clip.frameIndex.textContent = data.frame_index ?? "—";
        clip.analysisState.textContent = statusLabel(state);
        clip.analysisCounts.textContent = `${data.person_count ?? 0} / ${data.hazard_count ?? 0}`;
        clip.analysisMetadata.hidden = false;

        if (data.preview_jpeg_b64) {
            clip.resultImage.src = `data:image/jpeg;base64,${data.preview_jpeg_b64}`;
            clip.resultImage.hidden = false;
            clip.resultEmpty.hidden = true;
        } else {
            clip.resultImage.hidden = true;
            clip.resultImage.removeAttribute("src");
            clip.resultEmpty.hidden = false;
            clip.resultEmpty.textContent = "Backend nie zwrócił obrazu wyniku.";
        }
        renderClipPairs(data.pairs);
        renderClipDetections(data.detections);

        const summary = data.message || (
            state === "no_pairs"
                ? "Klatka przeanalizowana poprawnie; brak par osoba ↔ maszyna."
                : `Klatka ${data.frame_index ?? "—"} · ${secondsLabel(data.actual_time_sec)} · ${statusLabel(state)}`
        );
        setStatus(clip.analysisStatus, summary, analysisStatusKind(state));
    }

    function requestPosition() {
        const time = clipPositionSeconds();
        // HTML currentTime is only a request. The backend seeks its own decoder
        // by timestamp and reports the frame/time it actually obtained. This
        // remains correct for variable-frame-rate clips where time*FPS is not.
        return {time_sec: time};
    }

    async function deleteClipAsset(videoId) {
        if (!videoId) return;
        await apiJson(`/api/distance/videos/${encodeURIComponent(videoId)}`, {
            method: "DELETE",
            credentials: "same-origin",
        });
    }

    async function uploadClip(event) {
        event.preventDefault();
        if (uploading || analyzing) return;
        const file = clip.file.files && clip.file.files[0];
        if (!file) {
            setStatus(clip.uploadStatus, "Wybierz plik MP4, MOV, AVI lub MKV.", "warning");
            return;
        }

        uploading = true;
        const requestGeneration = ++generation;
        const oldClipId = clipId;
        if (clipAnalysisController) clipAnalysisController.abort();
        clipAnalysisController = null;
        analyzing = false;
        clip.video.pause();
        setClipControlsDisabled(true);
        clip.upload.disabled = true;
        clip.file.disabled = true;
        setStatus(clip.uploadStatus, `Wysyłanie ${file.name}…`);
        try {
            const form = new FormData();
            form.append("video", file, file.name);
            const metadata = await apiJson("/api/distance/videos", {
                method: "POST",
                body: form,
                credentials: "same-origin",
            });
            if (requestGeneration !== generation) return;
            installClip(metadata);
            if (oldClipId && oldClipId !== clipId) {
                // Wczytanie nowego pliku jest świadomym zastąpieniem assetu.
                void deleteClipAsset(oldClipId).catch(() => {});
            }
        } catch (error) {
            if (requestGeneration !== generation) return;
            setStatus(clip.uploadStatus, `Nie udało się wczytać klipu: ${error.message || error}`, "error");
        } finally {
            if (requestGeneration === generation) {
                uploading = false;
                setClipControlsDisabled(!clipId);
                clip.upload.disabled = false;
                clip.file.disabled = false;
                clip.file.value = "";
            }
        }
    }

    async function analyzeClip() {
        if (!clipId || analyzing || uploading) return;
        analyzing = true;
        clip.video.pause();
        const playerTime = numberOrNull(clip.video.currentTime);
        syncTimeline(playerTime === null ? clipPositionSeconds() : playerTime);
        const requestGeneration = generation;
        if (clipAnalysisController) clipAnalysisController.abort();
        const controller = new AbortController();
        clipAnalysisController = controller;
        clip.analyze.disabled = true;
        clip.remove.disabled = true;
        clip.upload.disabled = true;
        clip.file.disabled = true;
        const position = requestPosition();
        setStatus(
            clip.analysisStatus,
            position.frame_index !== undefined
                ? `Analiza klatki ${position.frame_index} przez istniejący detektor Perimetr…`
                : `Analiza pozycji ${secondsLabel(position.time_sec)} przez istniejący detektor Perimetr…`,
        );
        try {
            const maxDistance = numberOrNull(clip.max.value);
            const payload = {...position, preview_max_width: 1280};
            if (maxDistance !== null && maxDistance > 0) payload.max_distance_m = maxDistance;
            const data = await apiJson(`/api/distance/videos/${encodeURIComponent(clipId)}/analyze`, {
                method: "POST",
                credentials: "same-origin",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload),
                signal: controller.signal,
            });
            if (requestGeneration !== generation || controller.signal.aborted) return;
            renderAnalysis(data);
        } catch (error) {
            if (error && error.name === "AbortError") return;
            if (requestGeneration === generation) {
                const state = normalizedStatus(error.code);
                const calibration = state === "calibration_incompatible"
                    ? "Kalibracja jest niezgodna z proporcjami obrazu. " : "";
                setStatus(clip.analysisStatus, `${calibration}Analiza nie powiodła się: ${error.message || error}`, "error");
            }
        } finally {
            if (clipAnalysisController === controller) clipAnalysisController = null;
            if (requestGeneration === generation) {
                analyzing = false;
                clip.analyze.disabled = !clipId;
                clip.remove.disabled = !clipId;
                clip.upload.disabled = false;
                clip.file.disabled = false;
            }
        }
    }

    async function removeClip() {
        if (!clipId || uploading) return;
        if (!window.confirm("Usunąć ten klip z serwera?")) return;
        const currentId = clipId;
        const previousTime = clipPositionSeconds();
        const previousSource = clip.video.currentSrc || clip.video.src;
        // Abort an outstanding native <video> Range request before DELETE.
        // The backend intentionally pins streamed files with a lease, so
        // leaving the media source attached could make an explicit removal
        // wait until the server-side delete timeout.
        clip.video.pause();
        clip.video.removeAttribute("src");
        clip.video.load();
        clip.remove.disabled = true;
        setStatus(clip.uploadStatus, "Usuwanie klipu…");
        try {
            await deleteClipAsset(currentId);
            clearClipUi("Klip został jawnie usunięty. Możesz wczytać kolejny plik.");
        } catch (error) {
            if (clipId === currentId && previousSource) {
                clip.video.src = previousSource;
                clip.video.addEventListener("loadedmetadata", () => {
                    try { clip.video.currentTime = previousTime; } catch (_) {}
                    syncTimeline(previousTime);
                }, {once: true});
                clip.video.load();
            }
            clip.remove.disabled = false;
            setStatus(clip.uploadStatus, `Nie udało się usunąć klipu: ${error.message || error}`, "error");
        }
    }

    function stepFrame(direction) {
        if (!clipId) return;
        const fps = numberOrNull(clipMeta && (clipMeta.fps ?? clipMeta.source_fps));
        const step = fps !== null && fps > 0 ? 1 / fps : 1 / 25;
        const duration = clipDurationSeconds();
        const next = Math.min(duration || Number.MAX_SAFE_INTEGER, Math.max(0, clipPositionSeconds() + direction * step));
        clip.video.pause();
        try { clip.video.currentTime = next; } catch (_) {}
        syncTimeline(next);
    }

    async function restoreClip() {
        const restoreGeneration = generation;
        let storedId = null;
        try { storedId = localStorage.getItem(STORAGE_KEY); } catch (_) {}
        if (!storedId) return;
        try {
            const metadata = await apiJson(`/api/distance/videos/${encodeURIComponent(storedId)}`, {
                cache: "no-store",
                credentials: "same-origin",
            });
            if (restoreGeneration !== generation || uploading || clipId) return;
            installClip(metadata, {restored: true});
        } catch (error) {
            if (restoreGeneration !== generation) return;
            if (error.httpStatus === 404 || error.httpStatus === 410) {
                try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
                return;
            }
            setStatus(clip.uploadStatus, `Nie udało się przywrócić poprzedniego klipu: ${error.message || error}`, "warning");
        }
    }

    live.start.addEventListener("click", startLive);
    live.once.addEventListener("click", () => void refreshLive());
    live.stop.addEventListener("click", stopLive);
    live.rate.addEventListener("change", () => {
        if (liveTimer !== null) startLive();
    });
    live.camera.addEventListener("change", () => {
        if (liveTimer !== null) void refreshLive();
    });

    clip.liveTab.addEventListener("click", () => switchMode("live"));
    clip.clipTab.addEventListener("click", () => switchMode("clip"));
    [clip.liveTab, clip.clipTab].forEach((tab) => {
        tab.addEventListener("keydown", (event) => {
            if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
            event.preventDefault();
            const target = tab === clip.liveTab ? clip.clipTab : clip.liveTab;
            target.focus();
            target.click();
        });
    });
    clip.form.addEventListener("submit", uploadClip);
    clip.analyze.addEventListener("click", () => void analyzeClip());
    clip.remove.addEventListener("click", () => void removeClip());
    clip.previous.addEventListener("click", () => stepFrame(-1));
    clip.next.addEventListener("click", () => stepFrame(1));

    clip.video.addEventListener("loadedmetadata", () => {
        clip.playerMessage.hidden = true;
        syncTimeline(clip.video.currentTime || 0);
    });
    clip.video.addEventListener("durationchange", () => syncTimeline(clip.video.currentTime || 0));
    clip.video.addEventListener("timeupdate", () => {
        if (!clipSeekDragging) syncTimeline(clip.video.currentTime || 0);
    });
    clip.video.addEventListener("error", () => {
        clip.playerMessage.hidden = false;
        clip.playerMessage.textContent = "Przeglądarka nie potrafi odtworzyć tego kontenera. Nadal możesz ustawić czas suwakiem i zlecić analizę klatki backendowi.";
    });
    clip.seek.addEventListener("pointerdown", () => { clipSeekDragging = true; });
    clip.seek.addEventListener("input", () => {
        clip.currentTime.textContent = formatTime(clip.seek.value);
    });
    const commitSeek = () => {
        clipSeekDragging = false;
        const next = numberOrNull(clip.seek.value) || 0;
        try { clip.video.currentTime = next; } catch (_) {}
        syncTimeline(next);
    };
    clip.seek.addEventListener("change", commitSeek);
    clip.seek.addEventListener("pointerup", commitSeek);

    switchMode("live");
    setClipControlsDisabled(true);
    syncTimeline(0);
    void loadCameras();
    void refreshLive();
    void restoreClip();
})();
