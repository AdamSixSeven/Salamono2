"use strict";

const $ = id => document.getElementById(id);
const cameraSelect = $("cameraSelect");
const profileSelect = $("profileSelect");
const rgbCanvas = $("rgbCanvas");
const depthCanvas = $("depthCanvas");
const rgbCtx = rgbCanvas.getContext("2d");
const depthCtx = depthCanvas.getContext("2d");
let running = false;
let inferBusy = false;
let timer = null;
let targetIntervalMs = 125;
let currentResult = null;
let lastCompletedAt = 0;
let displayFpsEma = 0;
let activeProfileId = null;
let lastAppliedProfileId = null;
let lastViewsKey = null;
let lastMetricPointsKey = null;
let pendingMetricPoint = false;

function checkerboardBody() {
    return {
        inner_corners_x: Number($("cornersX").value),
        inner_corners_y: Number($("cornersY").value),
        square_length_m: Number($("squareLengthMm").value) / 1000,
        lens_model: $("lensModel").value,
    };
}

async function api(url, options = {}) {
    const response = await fetch(url, options);
    let payload = null;
    try { payload = await response.json(); } catch (_) {}
    if (!response.ok) throw new Error(payload && payload.detail ? payload.detail : `HTTP ${response.status}`);
    return payload;
}

function setStatus(element, text, type = "") {
    element.textContent = text;
    element.className = `d3-status${type ? ` ${type}` : ""}`;
}

function postCheckerboard(path) {
    return api(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(checkerboardBody()),
    });
}

function cameraLabel(camera) {
    const online = camera.online ? "online" : "offline";
    const size = camera.width && camera.height ? `${camera.width}×${camera.height}` : "brak klatki";
    return `${camera.camera_id} · ${online} · ${size}`;
}

function profileLabel(profile) {
    const state = profile.calibrated
        ? `${Number(profile.calibration?.mean_reprojection_error_px || 0).toFixed(3)} px`
        : "bez wyniku";
    return `${profile.name} · ${profile.view_count} wid. · ${state}`;
}

function applyProfileSpec(profile) {
    if (!profile || profile.profile_id === lastAppliedProfileId) return;
    const spec = profile.checkerboard_spec || {};
    if (Number.isFinite(Number(spec.inner_corners_x))) $("cornersX").value = spec.inner_corners_x;
    if (Number.isFinite(Number(spec.inner_corners_y))) $("cornersY").value = spec.inner_corners_y;
    if (Number.isFinite(Number(spec.square_length_m))) $("squareLengthMm").value = Number(spec.square_length_m) * 1000;
    if (["pinhole", "fisheye"].includes(spec.lens_model)) $("lensModel").value = spec.lens_model;
    lastAppliedProfileId = profile.profile_id;
}

function renderProfiles(status) {
    const profiles = Array.isArray(status.profiles) ? status.profiles : [];
    activeProfileId = status.active_profile_id || null;
    profileSelect.innerHTML = "";
    for (const profile of profiles) {
        const option = new Option(profileLabel(profile), profile.profile_id);
        profileSelect.add(option);
    }
    if (!profiles.length) profileSelect.add(new Option("Brak profilu", ""));
    profileSelect.value = activeProfileId || "";
    applyProfileSpec(status.active_profile);
}

function correctionSummaryText(correction) {
    if (!correction) return "Brak korekcji. Dodaj co najmniej 3 punkty kontrolne.";
    return `Aktywna korekcja: ${correction.method} · a=${Number(correction.scale).toFixed(4)} · b=${Number(correction.offset).toFixed(4)} · punkty ${Number(correction.point_count || 0)} · RMSE ${Number(correction.rmse_m || 0).toFixed(3)} m · mediana |błędu| ${Number(correction.median_abs_error_m || 0).toFixed(3)} m`;
}

function metricPointCard(point) {
    const label = point.label ? `${point.label} · ` : "";
    const wrapper = document.createElement("div");
    wrapper.className = "d3-point-item";
    wrapper.innerHTML = `
        <div class="d3-point-meta">${label}piksel [${Number(point.x).toFixed(1)}, ${Number(point.y).toFixed(1)}]<br>model ${Number(point.predicted_depth_m).toFixed(3)} m → pomiar ${Number(point.measured_distance_m).toFixed(3)} m</div>
        <button class="px-btn" data-delete-metric-point="${point.point_id}">Usuń</button>`;
    return wrapper;
}

async function refreshMetricPoints() {
    const list = $("metricPointsList");
    if (!cameraSelect.value || !activeProfileId) {
        list.innerHTML = '<div class="d3-empty-views">Brak aktywnego profilu.</div>';
        $("metricCorrectionStatus").textContent = "Brak aktywnego profilu.";
        return;
    }
    const data = await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/profiles/${encodeURIComponent(activeProfileId)}/depth-points`);
    const points = Array.isArray(data.points) ? data.points : [];
    const correction = data.correction || null;
    const key = `${cameraSelect.value}:${activeProfileId}:` + points.map(item => `${item.point_id}:${item.measured_distance_m}:${item.predicted_depth_m}`).join('|') + `::${correction ? JSON.stringify(correction) : ''}`;
    if (key === lastMetricPointsKey) return;
    lastMetricPointsKey = key;
    list.innerHTML = "";
    if (!points.length) list.innerHTML = '<div class="d3-empty-views">Brak zapisanych punktów kontrolnych.</div>';
    for (const point of points) list.appendChild(metricPointCard(point));
    $("metricCorrectionStatus").textContent = correctionSummaryText(correction);
}

function viewCard(view) {
    const card = document.createElement("article");
    card.className = "d3-view-card";
    const error = view.reprojection_error_px == null ? "—" : `${Number(view.reprojection_error_px).toFixed(3)} px`;
    card.innerHTML = `
        <img src="${view.image_url}?t=${encodeURIComponent(view.created_at || Date.now())}" alt="Widok kalibracyjny">
        <div class="d3-view-card-body">
            <div class="d3-view-card-meta">${view.corner_count || 0} narożników · pokrycie ${(Number(view.coverage_ratio || 0) * 100).toFixed(1)}%<br>ostrość ${Number(view.blur_score || 0).toFixed(0)} · błąd ${error}</div>
            <div class="d3-view-card-actions"><button class="px-btn" data-delete-view="${view.view_id}">Usuń</button></div>
        </div>`;
    return card;
}

async function refreshViews() {
    const list = $("viewsList");
    if (!cameraSelect.value || !activeProfileId) {
        list.innerHTML = '<div class="d3-empty-views">Brak aktywnego profilu.</div>';
        $("viewsSummary").textContent = "0 widoków";
        return;
    }
    const data = await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/profiles/${encodeURIComponent(activeProfileId)}/views`);
    const views = Array.isArray(data.views) ? data.views : [];
    const viewsKey = `${cameraSelect.value}:${activeProfileId}:` + views.map(view => `${view.view_id}:${view.reprojection_error_px ?? ""}`).join("|");
    if (viewsKey === lastViewsKey) return;
    lastViewsKey = viewsKey;
    list.innerHTML = "";
    if (!views.length) list.innerHTML = '<div class="d3-empty-views">Brak zapisanych widoków w tym profilu.</div>';
    for (const view of views) list.appendChild(viewCard(view));
    $("viewsSummary").textContent = `${views.length} widoków`;
}

async function loadCameras() {
    const payload = await api("/api/cameras");
    const cameras = Array.isArray(payload) ? payload : (payload.cameras || []);
    const params = new URLSearchParams(location.search);
    const wanted = params.get("camera_id") || localStorage.getItem("perimetr:camera_id") || cameras.find(item => item.online)?.camera_id || "cam_default";
    cameraSelect.innerHTML = "";
    const ids = new Set();
    for (const camera of cameras) {
        ids.add(camera.camera_id);
        cameraSelect.add(new Option(cameraLabel(camera), camera.camera_id));
    }
    if (!ids.has(wanted)) cameraSelect.add(new Option(wanted, wanted));
    cameraSelect.value = wanted;
    await refreshStatus();
}

async function refreshStatus() {
    if (!cameraSelect.value) return;
    try {
        const status = await api(`/api/depth3d/status?camera_id=${encodeURIComponent(cameraSelect.value)}`);
        renderProfiles(status);
        const minimum = Number(status.minimum_captures || 8);
        const captures = Number(status.captures || 0);
        $("captureProgress").style.width = `${Math.min(100, captures / minimum * 100)}%`;
        targetIntervalMs = Math.max(0, 1000 / Math.max(0.1, Number(status.target_fps || 5)));
        const frame = status.frame_available ? `${status.frame_width}×${status.frame_height}, wiek ${status.frame_age_sec}s` : "brak świeżej klatki";
        const calibration = status.calibration
            ? `kalibracja: ${status.calibration.views_used} widoków, błąd ${Number(status.calibration.mean_reprojection_error_px).toFixed(3)} px, ${status.calibration.lens_model}`
            : "brak kalibracji";
        const profileName = status.active_profile?.name || "—";
        setStatus($("calibrationStatus"), `Kamera: ${frame}\nProfil: ${profileName}\nWidoki: ${captures}/${minimum}\n${calibration}${status.compatibility_warning ? `\nUWAGA: ${status.compatibility_warning}` : ""}`,
            status.calibration && status.calibration_compatible ? "success" : "");
        if (status.model) $("deviceMeta").textContent = `${status.model.device || "—"} · ${status.model.precision || "—"}`;
        await refreshViews();
        await refreshMetricPoints();
    } catch (error) {
        setStatus($("calibrationStatus"), error.message, "error");
    }
}

function checkerboardQuery() {
    const body = checkerboardBody();
    return new URLSearchParams({
        inner_corners_x: body.inner_corners_x,
        inner_corners_y: body.inner_corners_y,
        square_length_m: body.square_length_m,
        dpi: 300,
    }).toString();
}

$("downloadBoard").addEventListener("click", () => {
    location.href = `/api/depth3d/checkerboard.png?${checkerboardQuery()}`;
});

$("newProfile").addEventListener("click", async () => {
    const name = window.prompt("Nazwa nowego profilu:", `Profil ${new Date().toLocaleDateString("pl-PL")}`);
    if (!name) return;
    try {
        await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/profiles`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({name, checkerboard: checkerboardBody()}),
        });
        lastAppliedProfileId = null;
        await refreshStatus();
    } catch (error) { setStatus($("calibrationStatus"), error.message, "error"); }
});

$("renameProfile").addEventListener("click", async () => {
    if (!activeProfileId) return;
    const current = profileSelect.options[profileSelect.selectedIndex]?.textContent?.split(" · ")[0] || "Profil";
    const name = window.prompt("Nowa nazwa profilu:", current);
    if (!name) return;
    try {
        await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/profiles/${encodeURIComponent(activeProfileId)}`, {
            method: "PATCH",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({name}),
        });
        await refreshStatus();
    } catch (error) { setStatus($("calibrationStatus"), error.message, "error"); }
});


$("exportProfile").addEventListener("click", () => {
    if (!activeProfileId) return;
    location.href = `/api/depth3d/${encodeURIComponent(cameraSelect.value)}/profiles/${encodeURIComponent(activeProfileId)}/export.zip`;
});

$("deleteProfile").addEventListener("click", async () => {
    if (!activeProfileId || !window.confirm("Usunąć aktywny profil razem z widokami i kalibracją?")) return;
    try {
        await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/profiles/${encodeURIComponent(activeProfileId)}`, {method: "DELETE"});
        lastAppliedProfileId = null;
        await refreshStatus();
    } catch (error) { setStatus($("calibrationStatus"), error.message, "error"); }
});

profileSelect.addEventListener("change", async () => {
    if (!profileSelect.value) return;
    try {
        await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/profiles/${encodeURIComponent(profileSelect.value)}/activate`, {method: "POST"});
        lastAppliedProfileId = null;
        await refreshStatus();
    } catch (error) { setStatus($("calibrationStatus"), error.message, "error"); }
});

$("viewsList").addEventListener("click", async event => {
    const button = event.target.closest("[data-delete-view]");
    if (!button || !activeProfileId) return;
    if (!window.confirm("Usunąć ten widok? Wynik kalibracji profilu zostanie unieważniony do ponownego przeliczenia.")) return;
    try {
        await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/profiles/${encodeURIComponent(activeProfileId)}/views/${encodeURIComponent(button.dataset.deleteView)}`, {method: "DELETE"});
        await refreshStatus();
    } catch (error) { setStatus($("calibrationStatus"), error.message, "error"); }
});

$("metricPointsList").addEventListener("click", async event => {
    const button = event.target.closest("[data-delete-metric-point]");
    if (!button || !activeProfileId) return;
    if (!window.confirm("Usunąć ten punkt kontrolny? Spowoduje to unieważnienie zapisanej korekcji.")) return;
    try {
        await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/profiles/${encodeURIComponent(activeProfileId)}/depth-points/${encodeURIComponent(button.dataset.deleteMetricPoint)}`, {method: "DELETE"});
        await refreshMetricPoints();
    } catch (error) { $("metricCorrectionStatus").textContent = error.message; }
});

$("addMetricPoint").addEventListener("click", () => {
    if (!currentResult) {
        $("metricHint").textContent = "Najpierw uruchom inferencję głębi, aby było z czego pobrać predykcję.";
        return;
    }
    pendingMetricPoint = true;
    $("metricHint").textContent = "Tryb dodawania punktu aktywny — kliknij RGB albo mapę głębi.";
});

$("fitMetricCorrection").addEventListener("click", async () => {
    if (!activeProfileId) return;
    try {
        const data = await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/profiles/${encodeURIComponent(activeProfileId)}/depth-correction/fit`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({method: $("metricMethod").value}),
        });
        $("metricCorrectionStatus").textContent = correctionSummaryText(data.correction);
        $("metricHint").textContent = "Korekcja zapisana. Kolejne inferencje będą już skorygowane.";
        await refreshMetricPoints();
    } catch (error) { $("metricCorrectionStatus").textContent = error.message; }
});

$("clearMetricPoints").addEventListener("click", async () => {
    if (!activeProfileId || !window.confirm("Usunąć wszystkie punkty kontrolne i skasować korekcję dla tego profilu?")) return;
    try {
        await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/profiles/${encodeURIComponent(activeProfileId)}/depth-points`, {method: "DELETE"});
        $("metricCorrectionStatus").textContent = "Punkty usunięte. Korekcja skasowana.";
        await refreshMetricPoints();
    } catch (error) { $("metricCorrectionStatus").textContent = error.message; }
});

$("detectBoard").addEventListener("click", async () => {
    try {
        const data = await postCheckerboard(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/checkerboard/detect`);
        setStatus($("calibrationStatus"), data.found
            ? `Szachownica wykryta: ${data.corner_count}/${data.expected_corners} narożników · pokrycie ${(data.coverage_ratio * 100).toFixed(1)}% · ostrość ${data.blur_score.toFixed(0)}.`
            : `Nie wykryto pełnej szachownicy (${data.corner_count}/${data.expected_corners}).`, data.found ? "success" : "error");
    } catch (error) { setStatus($("calibrationStatus"), error.message, "error"); }
});

$("captureView").addEventListener("click", async () => {
    try {
        const data = await postCheckerboard(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/checkerboard/capture`);
        setStatus($("calibrationStatus"), data.accepted
            ? `Zapisano widok ${data.capture_count}. ${data.corner_count}/${data.expected_corners} narożników · pokrycie ${(data.coverage_ratio * 100).toFixed(1)}% · ostrość ${data.blur_score.toFixed(0)}.`
            : data.warning, data.accepted ? "success" : "");
        await refreshStatus();
    } catch (error) { setStatus($("calibrationStatus"), error.message, "error"); }
});

$("resetViews").addEventListener("click", async () => {
    if (!window.confirm("Usunąć wszystkie widoki z aktywnego profilu?")) return;
    try {
        await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/captures`, {method: "DELETE"});
        await refreshStatus();
    } catch (error) { setStatus($("calibrationStatus"), error.message, "error"); }
});

$("calibrateBtn").addEventListener("click", async () => {
    setStatus($("calibrationStatus"), "Obliczanie kalibracji…");
    try {
        const data = await postCheckerboard(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/checkerboard/calibrate`);
        const coverage = Number(data.board_spec.aggregate_coverage_ratio || 0) * 100;
        setStatus($("calibrationStatus"), `Kalibracja gotowa · błąd ${Number(data.mean_reprojection_error_px).toFixed(3)} px · ${data.views_used} widoków · pokrycie kadru ${coverage.toFixed(1)}%.`, "success");
        await refreshStatus();
    } catch (error) { setStatus($("calibrationStatus"), error.message, "error"); }
});

function drawBase64Image(canvas, context, b64, mime) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => {
            canvas.width = image.naturalWidth;
            canvas.height = image.naturalHeight;
            context.clearRect(0, 0, canvas.width, canvas.height);
            context.drawImage(image, 0, 0);
            resolve();
        };
        image.onerror = reject;
        image.src = `data:${mime};base64,${b64}`;
    });
}

function updateDisplayFps(now) {
    if (lastCompletedAt > 0) {
        const instantaneous = 1000 / Math.max(1, now - lastCompletedAt);
        displayFpsEma = displayFpsEma ? displayFpsEma * 0.78 + instantaneous * 0.22 : instantaneous;
    }
    lastCompletedAt = now;
    $("streamFpsMeta").textContent = displayFpsEma ? `${displayFpsEma.toFixed(1)} FPS` : "—";
}

function drawPersonOverlays(canvas, context, people) {
    if (!Array.isArray(people) || !people.length) return;
    const scaleX = canvas.width / Math.max(1, Number(currentResult.frame_width || canvas.width));
    const scaleY = canvas.height / Math.max(1, Number(currentResult.frame_height || canvas.height));
    context.save();
    context.font = `${Math.max(13, Math.round(canvas.width / 60))}px ui-monospace, SFMono-Regular, Consolas, monospace`;
    context.lineWidth = Math.max(2, canvas.width / 480);
    context.textBaseline = "top";
    for (const person of people) {
        const [x1, y1, x2, y2] = person.box.map(Number);
        const left = x1 * scaleX;
        const top = y1 * scaleY;
        const width = Math.max(1, (x2 - x1) * scaleX);
        const height = Math.max(1, (y2 - y1) * scaleY);
        const label = `OSOBA ${person.person_index} · Z ${Number(person.depth_z_m).toFixed(2)} m`;
        const metrics = context.measureText(label);
        const labelHeight = Math.max(20, Math.round(canvas.width / 45));
        context.strokeStyle = "#ffffff";
        context.fillStyle = "#dd211c";
        context.strokeRect(left, top, width, height);
        const labelTop = Math.max(0, top - labelHeight);
        context.fillRect(left, labelTop, metrics.width + 12, labelHeight);
        context.fillStyle = "#ffffff";
        context.fillText(label, left + 6, labelTop + 3);
    }
    context.restore();
}

function renderPersonSummary(people, warning) {
    if (warning) {
        $("peopleResult").textContent = warning;
        return;
    }
    if (!Array.isArray(people) || !people.length) {
        $("peopleResult").textContent = "Nie wykryto osoby z wystarczającą liczbą poprawnych pikseli głębi.";
        return;
    }
    $("peopleResult").textContent = people.map(person =>
        `Osoba ${person.person_index}: Z ${Number(person.depth_z_m).toFixed(2)} m · promień ${Number(person.ray_distance_m).toFixed(2)} m · pewność YOLO ${(Number(person.confidence) * 100).toFixed(0)}%`
    ).join("\n");
}

async function runInference() {
    if (inferBusy) return false;
    inferBusy = true;
    const started = performance.now();
    try {
        const data = await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/infer`, {method: "POST"});
        currentResult = data;
        await Promise.all([
            drawBase64Image(rgbCanvas, rgbCtx, data.rgb_jpeg_b64, "image/jpeg"),
            drawBase64Image(depthCanvas, depthCtx, data.depth_jpeg_b64, data.depth_image_mime || "image/jpeg"),
        ]);
        drawPersonOverlays(rgbCanvas, rgbCtx, data.person_distances);
        drawPersonOverlays(depthCanvas, depthCtx, data.person_distances);
        renderPersonSummary(data.person_distances, data.person_detection_warning);
        $("rgbEmpty").hidden = true;
        $("depthEmpty").hidden = true;
        $("rgbTitle").textContent = data.calibrated ? "RGB po korekcji dystorsji" : "RGB surowe — brak kalibracji";
        $("modelMeta").textContent = data.model_id;
        $("deviceMeta").textContent = data.device || "—";
        $("timeMeta").textContent = `${data.processing_ms.toFixed(1)} ms`;
        $("modelFpsMeta").textContent = `${data.inference_fps.toFixed(1)} FPS`;
        $("rangeMeta").textContent = `${data.min_depth_m ?? "—"}–${data.max_depth_m ?? "—"} m · med. ${data.median_depth_m ?? "—"} m`;
        $("personMeta").textContent = `${Array.isArray(data.person_distances) ? data.person_distances.length : 0} · ${Number(data.person_detection_ms || 0).toFixed(1)} ms`;
        $("pipelineMeta").textContent = `${Number(data.backend_total_ms || data.pipeline_ms || 0).toFixed(1)} ms`;
        updateDisplayFps(performance.now());
        const total = performance.now() - started;
        const correctionText = data.correction_applied && data.correction_summary ? `\nKorekcja: ${data.correction_summary.method} · RMSE ${Number(data.correction_summary.rmse_m || 0).toFixed(3)} m.` : "";
        setStatus($("depthStatus"), `${data.calibration_warning || "Kalibracja zgodna."}${correctionText}\nPełny cykl HTTP + render: ${total.toFixed(1)} ms.`, data.calibrated ? "success" : "");
        return true;
    } catch (error) {
        setStatus($("depthStatus"), error.message, "error");
        running = false;
        $("startDepth").disabled = false;
        $("stopDepth").disabled = true;
        return false;
    } finally {
        inferBusy = false;
    }
}

async function loop() {
    if (!running) return;
    const started = performance.now();
    await runInference();
    if (!running) return;
    const elapsed = performance.now() - started;
    timer = setTimeout(loop, Math.max(0, targetIntervalMs - elapsed));
}

$("startDepth").addEventListener("click", () => {
    running = true;
    lastCompletedAt = 0;
    displayFpsEma = 0;
    $("startDepth").disabled = true;
    $("stopDepth").disabled = false;
    loop();
});
$("singleDepth").addEventListener("click", runInference);
$("stopDepth").addEventListener("click", () => {
    running = false;
    clearTimeout(timer);
    $("startDepth").disabled = false;
    $("stopDepth").disabled = true;
    setStatus($("depthStatus"), "Pętla zatrzymana.");
});

function canvasPoint(event, canvas) {
    const rect = canvas.getBoundingClientRect();
    return [(event.clientX - rect.left) * canvas.width / rect.width, (event.clientY - rect.top) * canvas.height / rect.height];
}

function drawMarker(canvas, context, point) {
    context.save();
    context.strokeStyle = "#fff";
    context.fillStyle = "#DD211C";
    context.lineWidth = Math.max(2, canvas.width / 480);
    context.beginPath();
    context.arc(point[0], point[1], Math.max(6, canvas.width / 140), 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.restore();
}

async function measureAt(event) {
    if (!currentResult) return;
    const sourceCanvas = event.currentTarget;
    const point = canvasPoint(event, sourceCanvas);
    drawMarker(sourceCanvas, sourceCanvas.getContext("2d"), point);
    if (pendingMetricPoint) {
        const measuredDistance = Number($("metricDistance").value);
        if (!(measuredDistance > 0)) {
            $("metricHint").textContent = "Podaj poprawną rzeczywistą odległość w metrach.";
            pendingMetricPoint = false;
            return;
        }
        try {
            const data = await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/profiles/${encodeURIComponent(activeProfileId)}/depth-points`, {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    point,
                    radius: 6,
                    label: $("metricPointLabel").value,
                    measured_distance_m: measuredDistance,
                }),
            });
            $("metricHint").textContent = `Zapisano punkt: model ${Number(data.predicted_depth_m).toFixed(3)} m → pomiar ${measuredDistance.toFixed(3)} m.`;
            pendingMetricPoint = false;
            await refreshMetricPoints();
        } catch (error) {
            $("metricHint").textContent = error.message;
            pendingMetricPoint = false;
        }
        return;
    }
    try {
        const data = await api(`/api/depth3d/${encodeURIComponent(cameraSelect.value)}/distance`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({point, radius: 6}),
        });
        $("measureResult").textContent = `Głębokość Z: ${data.depth_z_m.toFixed(3)} m · odległość po promieniu kamery: ${data.ray_distance_m.toFixed(3)} m · piksel [${data.pixel.join(", ")}] · wiek wyniku ${data.result_age_sec.toFixed(2)} s.`;
    } catch (error) { $("measureResult").textContent = error.message; }
}

rgbCanvas.addEventListener("click", measureAt);
depthCanvas.addEventListener("click", measureAt);

cameraSelect.addEventListener("change", async () => {
    running = false;
    clearTimeout(timer);
    currentResult = null;
    activeProfileId = null;
    lastAppliedProfileId = null;
    lastViewsKey = null;
    lastMetricPointsKey = null;
    pendingMetricPoint = false;
    localStorage.setItem("perimetr:camera_id", cameraSelect.value);
    const url = new URL(location.href);
    url.searchParams.set("camera_id", cameraSelect.value);
    history.replaceState(null, "", url);
    await refreshStatus();
});

loadCameras().catch(error => setStatus($("calibrationStatus"), error.message, "error"));
setInterval(refreshStatus, 3000);
window.addEventListener("DOMContentLoaded", () => { if (window.lucide) window.lucide.createIcons(); });
