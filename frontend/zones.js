/* ============================================================
   Perimetr — Strefy niebezpieczne
   Rysowanie polygonów + strefy z markerów ArUco
   Kolory MSBP: DANGER = czerwony, WARNING = ambar (#FFB020 na wideo)
   ============================================================ */
(function () {
    "use strict";

    let cameraId = (window.Perimetr && window.Perimetr.getCameraId)
        ? window.Perimetr.getCameraId() : "cam_default";

    const stack = document.getElementById("canvasStack");
    const liveCanvas = document.getElementById("liveCanvas");
    const zoneCanvas = document.getElementById("zoneCanvas");
    const zoneCtx = zoneCanvas.getContext("2d");
    const zoneList = document.getElementById("zoneList");
    const zonesCount = document.getElementById("zonesCount");

    const drawBtn = document.getElementById("zoneDrawBtn");
    const drawPanel = document.getElementById("zoneDrawPanel");
    const nameInput = document.getElementById("zoneName");
    const severitySelect = document.getElementById("zoneSeverity");
    const warningDistanceInput = document.getElementById("zoneWarningDistance");
    const saveBtn = document.getElementById("zoneSaveBtn");
    const cancelBtn = document.getElementById("zoneCancelBtn");

    const markerZoneBtn = document.getElementById("markerZoneBtn");
    const markerPanel = document.getElementById("markerZonePanel");
    const markerName = document.getElementById("markerZoneName");
    const markerIds = document.getElementById("markerZoneIds");
    const markerSeverity = document.getElementById("markerZoneSeverity");
    const markerWarningDistance = document.getElementById("markerZoneWarningDistance");
    const markerSaveBtn = document.getElementById("markerZoneSaveBtn");
    const markerCancelBtn = document.getElementById("markerZoneCancelBtn");

    let zones = [];              // konfiguracja stref z /api/zones (marker_ids, name…)
    let livePolygons = {};       // zone_id → polygon rozwiązany w ostatniej klatce (WS)
    let liveActiveZones = [];    // autorytatywny zestaw aktywny w ostatniej klatce
    let hasLiveZoneState = false;
    let liveMarkers = [];        // ostatnio wykryte markery ArUco (z WS)
    let dynamicSafetyZones = []; // ruchome strefy WARNING/DANGER wokół maszyn
    let liveDetections = [];
    let liveDangers = [];
    let personDistances = [];
    let postureAssessments = [];
    let workerIdentifications = [];
    const postureInterpolator = window.PerimetrPostureInterpolation
        ? new window.PerimetrPostureInterpolation.TrackInterpolator({
            durationMs: 180,
        })
        : {
            update: () => false,
            sample: () => null,
            isAnimating: () => false,
            clear: () => {},
        };
    let postureAnimationFrame = null;
    let layers = (window.Perimetr && window.Perimetr.getLayers)
        ? window.Perimetr.getLayers()
        : {
            boxes: true,
            posture: true,
            zones: true,
            markers: true,
            distances: true,
        };
    let drawing = false;
    let draftPoly = [];          // [[x_norm, y_norm], ...]

    // ---------- Kolory MSBP na wideo ---------------------

    const COLOR = {
        dangerStroke: "#DD211C",
        dangerFill:   "rgba(221, 33, 28, 0.16)",
        warnStroke:   "#FFB020",
        warnFill:     "rgba(255, 176, 32, 0.10)",
        markerBg:     "#FAFAF8",
        markerFg:     "#0A0A0A",
        chipBg:       "#DD211C",
        chipFg:       "#FAFAF8",
        chipWarnBg:   "#FFB020",
        chipWarnFg:   "#0A0A0A",
        vertex:       "#DD211C",
        vertexOnWarn: "#FFB020",
        person:        "#38D996",
        vehicle:       "#5AA7FF",
        hardhat:       "#F7E14A",
        vest:          "#22D3EE",
        neutral:       "#FAFAF8",
        posture:       "#22D3EE",
        worker:        "#C084FC",
    };
    function colorsFor(sev) {
        return sev === "WARNING"
            ? { stroke: COLOR.warnStroke, fill: COLOR.warnFill, chipBg: COLOR.chipWarnBg, chipFg: COLOR.chipWarnFg }
            : { stroke: COLOR.dangerStroke, fill: COLOR.dangerFill, chipBg: COLOR.chipBg, chipFg: COLOR.chipFg };
    }

    function postureNow() {
        return window.performance && typeof window.performance.now === "function"
            ? window.performance.now() : Date.now();
    }

    function cancelPostureAnimation() {
        if (postureAnimationFrame === null) return;
        window.cancelAnimationFrame(postureAnimationFrame);
        postureAnimationFrame = null;
    }

    function schedulePostureAnimation() {
        if (!layers.posture || postureAnimationFrame !== null ||
            !postureInterpolator.isAnimating(postureNow())) return;
        postureAnimationFrame = window.requestAnimationFrame(timestamp => {
            postureAnimationFrame = null;
            drawOverlay(timestamp);
            schedulePostureAnimation();
        });
    }

    // ---------- API zon ---------------------

    async function fetchZones() {
        try {
            const r = await fetch(`/api/zones/${encodeURIComponent(cameraId)}`);
            if (!r.ok) return;
            const data = await r.json();
            zones = data.zones || [];
            renderZoneList();
            drawOverlay();
        } catch (e) {
            console.error("Failed to load zones:", e);
        }
    }

    async function saveZones(nextZones) {
        const payload = { zones: nextZones.map(z => ({
            id: z.id, name: z.name, severity: z.severity,
            polygon: z.polygon || [],
            marker_ids: z.marker_ids || [],
            warning_distance_m: Number.isFinite(Number(z.warning_distance_m)) ? Number(z.warning_distance_m) : 1.5,
            warning_distance_px: Number.isFinite(Number(z.warning_distance_px)) ? Number(z.warning_distance_px) : 60,
            active: z.active !== false,
        })) };
        const r = await fetch(`/api/zones/${encodeURIComponent(cameraId)}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!r.ok) {
            const msg = await r.text();
            throw new Error(msg || "Save failed");
        }
        const data = await r.json();
        zones = data.zones || [];
        renderZoneList();
        drawOverlay();
    }

    async function deleteZone(zoneId) {
        const next = zones.filter(z => z.id !== zoneId);
        try {
            await saveZones(next);
        } catch (e) {
            alert("Nie udało się usunąć strefy: " + e.message);
        }
    }

    async function toggleZone(zoneId) {
        const next = zones.map(z => z.id === zoneId ? { ...z, active: !z.active } : z);
        try {
            await saveZones(next);
        } catch (e) {
            alert("Nie udało się przełączyć: " + e.message);
        }
    }

    // ---------- Lista stref w bocznym panelu ---------------------

    function renderZoneList() {
        zoneList.innerHTML = "";
        zonesCount.textContent = zones.length;

        if (zones.length === 0) {
            const empty = document.createElement("div");
            empty.className = "px-alarms-empty";
            empty.textContent = "Brak stref";
            zoneList.appendChild(empty);
            return;
        }

        zones.forEach(z => {
            const row = document.createElement("div");
            row.className = "px-zone-row";

            const bar = document.createElement("span");
            bar.className = "px-zone-row-severity" + (z.severity === "WARNING" ? " warning" : "");
            row.appendChild(bar);

            const body = document.createElement("div");
            body.className = "px-zone-row-body";
            const title = document.createElement("div");
            title.className = "px-zone-row-title";
            const nm = document.createElement("span");
            nm.className = "px-zone-row-name";
            nm.textContent = z.name;
            const sev = document.createElement("span");
            sev.className = "msbp-tag " + (z.severity === "WARNING" ? "msbp-tag--warning" : "msbp-tag--red");
            sev.textContent = z.severity === "WARNING" ? "Warning" : "Danger";
            title.appendChild(nm);
            title.appendChild(sev);

            const meta = document.createElement("span");
            meta.className = "px-zone-row-meta";
            meta.textContent = z.marker_ids && z.marker_ids.length
                ? "markery " + z.marker_ids.join(" · ") + " · warning " + Number(z.warning_distance_m || 0).toFixed(1) + " m"
                : "polygon · " + (z.polygon || []).length + " pkt · warning " + Number(z.warning_distance_m || 0).toFixed(1) + " m";

            body.appendChild(title);
            body.appendChild(meta);
            row.appendChild(body);

            const toggle = document.createElement("button");
            toggle.className = "px-toggle" + (z.active !== false ? " on" : "");
            toggle.title = z.active !== false ? "Wyłącz strefę" : "Włącz strefę";
            toggle.onclick = () => toggleZone(z.id);
            row.appendChild(toggle);

            const del = document.createElement("button");
            del.className = "px-zone-del";
            del.textContent = "×";
            del.title = "Usuń strefę";
            del.onclick = () => deleteZone(z.id);
            row.appendChild(del);

            zoneList.appendChild(row);
        });
    }

    // ---------- Rysowanie polygonu na overlay ---------------------

    function polygonFor(z) {
        if (livePolygons[z.id]) return livePolygons[z.id];
        return z.polygon || [];
    }

    function zonesForOverlay() {
        // `active_zones` is resolved by the backend for the exact current
        // frame.  Prefer it after the first WS message so remote edits,
        // disabled zones and marker-zone expiry cannot leave a stale polygon
        // from the separately fetched configuration on a raw preview.
        return hasLiveZoneState ? liveActiveZones : zones;
    }

    function drawPolygon(poly, colors, opts) {
        opts = opts || {};
        const w = zoneCanvas.width, h = zoneCanvas.height;
        if (poly.length < 1) return;
        zoneCtx.save();
        zoneCtx.lineWidth = opts.lineWidth || 2.5;
        zoneCtx.setLineDash(opts.dashed ? [5, 5] : []);
        zoneCtx.strokeStyle = colors.stroke;
        zoneCtx.fillStyle = colors.fill;
        zoneCtx.beginPath();
        poly.forEach((pt, i) => {
            const px = pt[0] * w, py = pt[1] * h;
            if (i === 0) zoneCtx.moveTo(px, py);
            else zoneCtx.lineTo(px, py);
        });
        if (poly.length >= 3) { zoneCtx.closePath(); zoneCtx.fill(); }
        zoneCtx.stroke();

        if (opts.vertices) {
            zoneCtx.fillStyle = colors.stroke;
            poly.forEach(pt => {
                zoneCtx.beginPath();
                zoneCtx.arc(pt[0] * w, pt[1] * h, 4, 0, Math.PI * 2);
                zoneCtx.fill();
            });
        }

        if (opts.label) {
            const first = poly[0];
            const px = first[0] * w, py = first[1] * h;
            const label = opts.label.toUpperCase();
            zoneCtx.font = "600 10.5px 'JetBrains Mono', ui-monospace, monospace";
            const paddingX = 8, paddingY = 3, gap = 4;
            const metrics = zoneCtx.measureText(label);
            const chipW = metrics.width + paddingX * 2;
            const chipH = 18;
            const chipX = px;
            const chipY = py - chipH - gap;
            zoneCtx.fillStyle = colors.chipBg;
            zoneCtx.fillRect(chipX, chipY, chipW, chipH);
            zoneCtx.fillStyle = colors.chipFg;
            zoneCtx.textBaseline = "middle";
            zoneCtx.fillText(label, chipX + paddingX, chipY + chipH / 2 + 1);
        }
        zoneCtx.restore();
    }

    function drawMarkers() {
        if (!layers.markers) return;
        // liveMarkers przychodzą z WS w pixel space bieżącej klatki.
        // canvas ma taki sam pixel size jak klatka (renderFrame w app.js ustawia).
        liveMarkers.forEach(m => {
            const c = m.corners;
            if (!Array.isArray(c) || c.length < 3) return;
            zoneCtx.save();
            zoneCtx.lineWidth = 2;
            zoneCtx.strokeStyle = COLOR.markerBg;
            zoneCtx.fillStyle = COLOR.markerBg;
            zoneCtx.beginPath();
            zoneCtx.moveTo(c[0][0], c[0][1]);
            for (let i = 1; i < c.length; i++) zoneCtx.lineTo(c[i][0], c[i][1]);
            zoneCtx.closePath();
            zoneCtx.stroke();
            // ID pod środkiem markera
            const cx = m.center[0], cy = m.center[1];
            const label = String(m.marker_id);
            zoneCtx.font = "500 11px 'JetBrains Mono', ui-monospace, monospace";
            const paddingX = 5, paddingY = 2;
            const metrics = zoneCtx.measureText(label);
            const chipW = metrics.width + paddingX * 2;
            const chipH = 14;
            const chipX = cx - chipW / 2;
            const chipY = cy + 12;
            zoneCtx.fillStyle = "rgba(10,10,10,0.78)";
            zoneCtx.fillRect(chipX, chipY, chipW, chipH);
            zoneCtx.fillStyle = COLOR.markerBg;
            zoneCtx.textBaseline = "middle";
            zoneCtx.fillText(label, chipX + paddingX, chipY + chipH / 2 + 1);
            zoneCtx.restore();
        });

        // QR identyfikatora pracownika jest również markerem obrazu.  Jego
        // obrys znika razem z warstwą „Markery”, natomiast sam profil nadal
        // działa w logice detekcji i na karcie pracownika.
        workerIdentifications.forEach(identity => {
            const poly = identity.tag_polygon || [];
            if (identity.cached || poly.length < 3) return;
            zoneCtx.save();
            zoneCtx.lineWidth = 2;
            zoneCtx.strokeStyle = COLOR.worker;
            zoneCtx.beginPath();
            poly.forEach((pt, index) => {
                if (index === 0) zoneCtx.moveTo(pt[0], pt[1]);
                else zoneCtx.lineTo(pt[0], pt[1]);
            });
            zoneCtx.closePath();
            zoneCtx.stroke();
            zoneCtx.restore();
        });
    }

    function detectionColor(detection) {
        if (detection.category === "person") return COLOR.person;
        if (detection.category === "hardhat") return COLOR.hardhat;
        if (detection.category === "vest") return COLOR.vest;
        if (detection.category === "vehicle") return COLOR.vehicle;
        return COLOR.neutral;
    }

    function validBox(box) {
        return Array.isArray(box) && box.length === 4 &&
            box.every(value => Number.isFinite(Number(value)));
    }

    function drawChip(text, x, y, background, foreground, opts) {
        if (!text) return;
        opts = opts || {};
        const fontSize = opts.fontSize || 12;
        const paddingX = opts.paddingX || 6;
        const height = opts.height || 20;
        zoneCtx.save();
        zoneCtx.font = `600 ${fontSize}px 'JetBrains Mono', ui-monospace, monospace`;
        const width = zoneCtx.measureText(text).width + paddingX * 2;
        const chipX = Math.max(0, Math.min(x, zoneCanvas.width - width));
        const chipY = Math.max(0, Math.min(y, zoneCanvas.height - height));
        zoneCtx.fillStyle = background;
        zoneCtx.fillRect(chipX, chipY, width, height);
        zoneCtx.fillStyle = foreground;
        zoneCtx.textBaseline = "middle";
        zoneCtx.fillText(text, chipX + paddingX, chipY + height / 2 + 0.5);
        zoneCtx.restore();
    }

    function drawBoxes() {
        if (!layers.boxes) return;
        liveDetections.forEach(detection => {
            const box = detection.box;
            if (!validBox(box)) return;
            const x1 = Number(box[0]), y1 = Number(box[1]);
            const x2 = Number(box[2]), y2 = Number(box[3]);
            const color = detectionColor(detection);
            zoneCtx.save();
            zoneCtx.strokeStyle = color;
            zoneCtx.lineWidth = detection.category === "person" ? 2.5 : 2;
            zoneCtx.strokeRect(x1, y1, Math.max(0, x2 - x1), Math.max(0, y2 - y1));
            zoneCtx.restore();

            const confidence = Number(detection.confidence);
            const confidenceLabel = Number.isFinite(confidence)
                ? " " + Math.round(confidence * 100) + "%"
                : "";
            const label = (detection.class_name || detection.category || "obiekt") +
                confidenceLabel;
            drawChip(label, x1, y1 - 22, color, "#0A0A0A");
        });

        // Powiązanie QR z osobą jest metadanymi bboxa, więc respektuje ten sam
        // lokalny przełącznik.  Wyłączenie BBOX nie wyłącza identyfikacji.
        workerIdentifications.forEach(identity => {
            const box = identity.person_box;
            if (!validBox(box) || !identity.worker_id) return;
            const name = identity.full_name ||
                [identity.first_name, identity.last_name].filter(Boolean).join(" ");
            const suffix = name ? " · " + name : "";
            drawChip(
                "ID " + identity.worker_id + suffix + (identity.cached ? " ~" : ""),
                Number(box[0]),
                Number(box[1]) + 3,
                "rgba(82, 30, 128, 0.92)",
                "#FFFFFF",
                { fontSize: 11, height: 18 },
            );
        });
    }

    function formatDistanceValue(item) {
        if (item.distance_m !== null && item.distance_m !== undefined) {
            return Math.abs(Number(item.distance_m)).toFixed(2) + " m";
        }
        if (item.distance_px !== null && item.distance_px !== undefined) {
            return "pomiar pikselowy · " +
                Math.round(Math.abs(Number(item.distance_px))) + " px";
        }
        return "—";
    }

    function drawDistances() {
        if (!layers.distances) return;

        // Odległość osoby od wykrytej maszyny/pojazdu.
        liveDangers.forEach(danger => {
            const personBox = danger.person && danger.person.box;
            const hazardBox = danger.hazard && danger.hazard.box;
            if (!validBox(personBox) || !validBox(hazardBox)) return;
            const px = (Number(personBox[0]) + Number(personBox[2])) / 2;
            const py = (Number(personBox[1]) + Number(personBox[3])) / 2;
            const hx = (Number(hazardBox[0]) + Number(hazardBox[2])) / 2;
            const hy = (Number(hazardBox[1]) + Number(hazardBox[3])) / 2;
            const dangerColor = danger.severity === "DANGER"
                ? COLOR.dangerStroke : COLOR.warnStroke;
            zoneCtx.save();
            zoneCtx.strokeStyle = dangerColor;
            zoneCtx.lineWidth = 2;
            zoneCtx.setLineDash([7, 5]);
            zoneCtx.beginPath();
            zoneCtx.moveTo(px, py);
            zoneCtx.lineTo(hx, hy);
            zoneCtx.stroke();
            zoneCtx.restore();
            drawChip(
                formatDistanceValue(danger) +
                    (danger.distance_m !== null && danger.distance_m !== undefined
                        ? " od maszyny" : ""),
                (px + hx) / 2,
                (py + hy) / 2 - 10,
                dangerColor,
                danger.severity === "DANGER" ? "#FFFFFF" : "#0A0A0A",
                { fontSize: 11, height: 18 },
            );
        });

        // Najbliższa granica skonfigurowanej strefy dla każdej osoby.
        personDistances.forEach(distance => {
            const box = distance.person_box;
            if (!validBox(box)) return;
            const footX = (Number(box[0]) + Number(box[2])) / 2;
            const footY = Number(box[3]);
            const color = distance.inside ? COLOR.dangerStroke : COLOR.warnStroke;
            const metric = distance.distance_m !== null &&
                distance.distance_m !== undefined;
            const relation = metric
                ? (distance.inside ? " wewnątrz strefy" : " do strefy")
                : "";
            const label = (distance.zone_name || "STREFA") + " · " +
                formatDistanceValue(distance) + relation;
            zoneCtx.save();
            zoneCtx.strokeStyle = color;
            zoneCtx.fillStyle = color;
            zoneCtx.lineWidth = 2;
            zoneCtx.beginPath();
            zoneCtx.arc(footX, footY, 4, 0, Math.PI * 2);
            zoneCtx.fill();
            zoneCtx.beginPath();
            zoneCtx.moveTo(footX, footY + 4);
            zoneCtx.lineTo(footX, Math.min(zoneCanvas.height - 1, footY + 18));
            zoneCtx.stroke();
            zoneCtx.restore();
            drawChip(
                label,
                Number(box[0]),
                Math.min(zoneCanvas.height - 19, footY + 8),
                color,
                distance.inside ? "#FFFFFF" : "#0A0A0A",
                { fontSize: 11, height: 18 },
            );
        });
    }

    const POSTURE_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 7],
        [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
        [11, 12], [11, 13], [13, 15],
        [15, 17], [15, 19], [15, 21], [17, 19],
        [12, 14], [14, 16],
        [16, 18], [16, 20], [16, 22], [18, 20],
        [11, 23], [12, 24], [23, 24],
        [23, 25], [25, 27], [24, 26], [26, 28],
        [27, 29], [29, 31], [28, 30], [30, 32],
        [27, 31], [28, 32],
    ];

    function drawPosture(animationTimestamp) {
        if (!layers.posture) return;
        const now = Number.isFinite(Number(animationTimestamp))
            ? Number(animationTimestamp) : postureNow();
        postureAssessments.forEach(assessment => {
            const landmarks = postureInterpolator.sample(
                assessment.track_id,
                now,
            ) || assessment.pose_landmarks || [];
            if (!landmarks.length) return;
            const color = assessment.severity === "DANGER"
                ? COLOR.dangerStroke
                : assessment.severity === "WARNING"
                    ? COLOR.warnStroke : COLOR.posture;
            const points = new Map();
            landmarks.slice(0, 33).forEach((landmark, index) => {
                if (!Array.isArray(landmark) || landmark.length < 4) return;
                const x = Number(landmark[0]);
                const y = Number(landmark[1]);
                const visibility = Number(landmark[3]);
                if (!Number.isFinite(x) || !Number.isFinite(y) ||
                    !Number.isFinite(visibility) || visibility < 0.5 ||
                    x < 0 || x > 1 || y < 0 || y > 1) return;
                points.set(index, [x * zoneCanvas.width, y * zoneCanvas.height]);
            });

            zoneCtx.save();
            zoneCtx.strokeStyle = color;
            zoneCtx.fillStyle = color;
            zoneCtx.lineWidth = 2;
            POSTURE_CONNECTIONS.forEach(connection => {
                const start = points.get(connection[0]);
                const end = points.get(connection[1]);
                if (!start || !end) return;
                zoneCtx.beginPath();
                zoneCtx.moveTo(start[0], start[1]);
                zoneCtx.lineTo(end[0], end[1]);
                zoneCtx.stroke();
            });
            points.forEach(point => {
                zoneCtx.beginPath();
                zoneCtx.arc(point[0], point[1], 3, 0, Math.PI * 2);
                zoneCtx.fill();
            });
            zoneCtx.restore();

            const box = assessment.person && assessment.person.box;
            if (validBox(box)) {
                const score = Number.isFinite(Number(assessment.risk_score))
                    ? " · " + Math.round(Number(assessment.risk_score) * 100) + "%"
                    : "";
                drawChip(
                    "POSTURE #" + assessment.track_id + score,
                    Number(box[0]),
                    Number(box[1]) - 44,
                    color,
                    "#0A0A0A",
                    { fontSize: 10, height: 18 },
                );
            }
        });
    }

    function syncCanvasSize() {
        if (zoneCanvas.width !== liveCanvas.width ||
            zoneCanvas.height !== liveCanvas.height) {
            zoneCanvas.width = liveCanvas.width || 640;
            zoneCanvas.height = liveCanvas.height || 480;
        }
    }

    function drawOverlay(animationTimestamp) {
        syncCanvasSize();
        zoneCtx.clearRect(0, 0, zoneCanvas.width, zoneCanvas.height);

        if (layers.zones) {
            zonesForOverlay().forEach(z => {
                if (z.active === false) return;
                const poly = polygonFor(z);
                if (poly.length < 3) return;
                const label = z.name + " — " + (z.severity || "DANGER") +
                    " — warning " + Number(z.warning_distance_m || 0).toFixed(1) + " m";
                drawPolygon(poly, colorsFor(z.severity), { label: label });
            });

            // Dynamic zones are attached to the current machine detection.
            // The backend supplies a projected metric polygon after ground-plane
            // calibration, or a cheap image-space fallback before calibration.
            dynamicSafetyZones
                .slice()
                .sort((a, b) => (a.severity === "DANGER") - (b.severity === "DANGER"))
                .forEach(z => {
                    const poly = z.polygon || [];
                    if (poly.length < 3) return;
                    let limit = "dynamiczna";
                    if (z.threshold_m !== null && z.threshold_m !== undefined) {
                        limit = Number(z.threshold_m).toFixed(1) + " m";
                    } else if (z.threshold_px !== null && z.threshold_px !== undefined) {
                        limit = Math.round(z.threshold_px) + " px";
                    }
                    const mode = z.calibrated ? "metryczna" : "demo 2D";
                    drawPolygon(poly, colorsFor(z.severity), {
                        label: "MASZYNA · " + z.severity + " · " + limit + " · " + mode,
                        lineWidth: z.severity === "DANGER" ? 3 : 2,
                    });
                });
        }

        drawDistances();
        drawBoxes();
        // POS controls the MediaPipe skeleton independently from detector
        // BBOX. It only affects drawing; posture analysis and alerts continue.
        drawPosture(animationTimestamp);
        drawMarkers();

        if (drawing && draftPoly.length > 0) {
            drawPolygon(draftPoly, colorsFor(severitySelect.value), {
                dashed: true, vertices: true,
            });
        }
    }

    // ---------- Interakcje: rysowanie polygonu ---------------------

    function startDrawing() {
        drawing = true;
        draftPoly = [];
        drawPanel.classList.remove("hidden");
        markerPanel.classList.add("hidden");
        drawBtn.disabled = true;
        markerZoneBtn.disabled = true;
        stack.classList.add("drawing");
        nameInput.value = "";
        warningDistanceInput.value = "1.5";
        nameInput.focus();
        updateSaveEnabled();
        drawOverlay();
    }

    function cancelDrawing() {
        drawing = false;
        draftPoly = [];
        drawPanel.classList.add("hidden");
        drawBtn.disabled = false;
        markerZoneBtn.disabled = false;
        stack.classList.remove("drawing");
        drawOverlay();
    }

    async function commitDrawing() {
        if (draftPoly.length < 3) return;
        const name = (nameInput.value || "Strefa").trim().slice(0, 40);
        const severity = severitySelect.value;
        const warningDistanceM = Math.max(0, Number(warningDistanceInput.value || 0));
        const next = zones.concat([{
            name, severity, polygon: draftPoly, marker_ids: [],
            warning_distance_m: warningDistanceM,
            warning_distance_px: 60,
            active: true,
        }]);
        try {
            await saveZones(next);
            cancelDrawing();
        } catch (e) {
            alert("Nie udało się zapisać: " + e.message);
        }
    }

    function updateSaveEnabled() {
        saveBtn.disabled = draftPoly.length < 3;
    }

    function pointerToNormalized(evt) {
        const rect = zoneCanvas.getBoundingClientRect();
        const x = (evt.clientX - rect.left) / rect.width;
        const y = (evt.clientY - rect.top) / rect.height;
        return [
            Math.max(0, Math.min(1, x)),
            Math.max(0, Math.min(1, y)),
        ];
    }

    zoneCanvas.addEventListener("click", (evt) => {
        if (!drawing) return;
        evt.preventDefault();
        draftPoly.push(pointerToNormalized(evt));
        updateSaveEnabled();
        drawOverlay();
    });

    zoneCanvas.addEventListener("dblclick", (evt) => {
        if (!drawing) return;
        evt.preventDefault();
        if (draftPoly.length >= 2) {
            const last = draftPoly[draftPoly.length - 1];
            const prev = draftPoly[draftPoly.length - 2];
            if (Math.abs(last[0] - prev[0]) < 0.005 &&
                Math.abs(last[1] - prev[1]) < 0.005) {
                draftPoly.pop();
            }
        }
        commitDrawing();
    });

    document.addEventListener("keydown", (evt) => {
        if (!drawing) return;
        if (evt.key === "Enter") { evt.preventDefault(); commitDrawing(); }
        if (evt.key === "Escape") { evt.preventDefault(); cancelDrawing(); }
        if (evt.key === "Backspace" && document.activeElement !== nameInput) {
            evt.preventDefault();
            if (draftPoly.length > 0) {
                draftPoly.pop();
                updateSaveEnabled();
                drawOverlay();
            }
        }
    });

    severitySelect.addEventListener("change", drawOverlay);
    drawBtn.addEventListener("click", startDrawing);
    cancelBtn.addEventListener("click", cancelDrawing);
    saveBtn.addEventListener("click", commitDrawing);

    // ---------- Interakcje: strefa z markerów ---------------------

    function openMarkerPanel() {
        markerPanel.classList.remove("hidden");
        drawPanel.classList.add("hidden");
        drawBtn.disabled = true;
        markerZoneBtn.disabled = true;
        markerName.value = "";
        markerIds.value = "";
        markerWarningDistance.value = "1.5";
        markerName.focus();
    }
    function closeMarkerPanel() {
        markerPanel.classList.add("hidden");
        drawBtn.disabled = false;
        markerZoneBtn.disabled = false;
    }
    async function commitMarkerZone() {
        const ids = markerIds.value
            .split(/[\s,;]+/).map(s => s.trim()).filter(Boolean)
            .map(s => parseInt(s, 10)).filter(n => !isNaN(n));
        if (ids.length < 3) {
            alert("Podaj co najmniej 3 ID markerów (np. 10, 20, 30, 40).");
            return;
        }
        const next = zones.concat([{
            name: (markerName.value || "Strefa z markerów").trim().slice(0, 40),
            severity: markerSeverity.value,
            polygon: [],
            marker_ids: ids,
            warning_distance_m: Math.max(0, Number(markerWarningDistance.value || 0)),
            warning_distance_px: 60,
            active: true,
        }]);
        try {
            await saveZones(next);
            closeMarkerPanel();
        } catch (e) {
            alert("Nie udało się zapisać: " + e.message);
        }
    }
    if (markerZoneBtn) markerZoneBtn.addEventListener("click", openMarkerPanel);
    if (markerCancelBtn) markerCancelBtn.addEventListener("click", closeMarkerPanel);
    if (markerSaveBtn) markerSaveBtn.addEventListener("click", commitMarkerZone);

    // ---------- Redraw loop + WS listener ---------------------

    const observer = new MutationObserver(() => drawOverlay());
    observer.observe(liveCanvas, { attributes: true, attributeFilter: ["width", "height"] });
    window.addEventListener("resize", () => drawOverlay());

    function consumeFrame(d) {
        const active = d.active_zones || [];
        const next = {};
        active.forEach(a => { if (a.id && a.polygon) next[a.id] = a.polygon; });
        livePolygons = next;
        liveActiveZones = active;
        hasLiveZoneState = Array.isArray(d.active_zones);
        liveMarkers = d.markers || [];
        dynamicSafetyZones = d.dynamic_safety_zones || [];
        liveDetections = d.detections || [];
        liveDangers = d.active_dangers || [];
        personDistances = d.person_distances || [];
        postureAssessments = d.posture_assessments || [];
        const animationActive = postureInterpolator.update(
            postureAssessments,
            postureNow(),
            d.timestamp,
        );
        if (animationActive) schedulePostureAnimation();
        else cancelPostureAnimation();
        workerIdentifications = d.worker_identifications || [];
    }

    document.addEventListener("perimetr-frame", (evt) => {
        consumeFrame(evt.detail || {});
        drawOverlay();
    });

    // app.js fires this after the raw image has decoded and both canvases have
    // the exact current-frame dimensions.  It prevents one-frame drift when
    // switching between 16:9 USB and 4:3 phone cameras.
    document.addEventListener("perimetr-frame-rendered", (evt) => {
        consumeFrame(evt.detail || {});
        drawOverlay();
    });

    document.addEventListener("perimetr-camera-changed", (evt) => {
        cameraId = evt.detail || "cam_default";
        zones = [];
        livePolygons = {};
        liveActiveZones = [];
        hasLiveZoneState = false;
        liveMarkers = [];
        dynamicSafetyZones = [];
        liveDetections = [];
        liveDangers = [];
        personDistances = [];
        postureAssessments = [];
        postureInterpolator.clear();
        cancelPostureAnimation();
        workerIdentifications = [];
        drawOverlay();
        fetchZones();
    });

    document.addEventListener("perimetr-layers", (evt) => {
        layers = evt.detail || layers;
        if (layers.posture) schedulePostureAnimation();
        else cancelPostureAnimation();
        drawOverlay();
    });

    fetchZones();
})();
