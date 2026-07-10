/* ============================================================
   Perimetr — Strefy niebezpieczne
   Rysowanie polygonów + strefy z markerów ArUco
   Kolory MSBP: DANGER = czerwony, WARNING = ambar (#FFB020 na wideo)
   ============================================================ */
(function () {
    "use strict";

    const CAMERA_ID = "cam_default";

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
    const saveBtn = document.getElementById("zoneSaveBtn");
    const cancelBtn = document.getElementById("zoneCancelBtn");

    const markerZoneBtn = document.getElementById("markerZoneBtn");
    const markerPanel = document.getElementById("markerZonePanel");
    const markerName = document.getElementById("markerZoneName");
    const markerIds = document.getElementById("markerZoneIds");
    const markerSeverity = document.getElementById("markerZoneSeverity");
    const markerSaveBtn = document.getElementById("markerZoneSaveBtn");
    const markerCancelBtn = document.getElementById("markerZoneCancelBtn");

    let zones = [];              // konfiguracja stref z /api/zones (marker_ids, name…)
    let livePolygons = {};       // zone_id → polygon rozwiązany w ostatniej klatce (WS)
    let liveMarkers = [];        // ostatnio wykryte markery ArUco (z WS)
    let layers = { boxes: true, zones: true, markers: true, distances: true };
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
    };
    function colorsFor(sev) {
        return sev === "WARNING"
            ? { stroke: COLOR.warnStroke, fill: COLOR.warnFill, chipBg: COLOR.chipWarnBg, chipFg: COLOR.chipWarnFg }
            : { stroke: COLOR.dangerStroke, fill: COLOR.dangerFill, chipBg: COLOR.chipBg, chipFg: COLOR.chipFg };
    }

    // ---------- API zon ---------------------

    async function fetchZones() {
        try {
            const r = await fetch(`/api/zones/${CAMERA_ID}`);
            if (!r.ok) return;
            const data = await r.json();
            zones = data.zones || [];
            renderZoneList();
        } catch (e) {
            console.error("Failed to load zones:", e);
        }
    }

    async function saveZones(nextZones) {
        const payload = { zones: nextZones.map(z => ({
            id: z.id, name: z.name, severity: z.severity,
            polygon: z.polygon || [],
            marker_ids: z.marker_ids || [],
            active: z.active !== false,
        })) };
        const r = await fetch(`/api/zones/${CAMERA_ID}`, {
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
                ? "markery " + z.marker_ids.join(" · ") + " · KAM-01"
                : "polygon · " + (z.polygon || []).length + " pkt · KAM-01";

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
        const w = zoneCanvas.width, h = zoneCanvas.height;
        // liveMarkers przychodzą z WS w pixel space bieżącej klatki.
        // canvas ma taki sam pixel size jak klatka (renderFrame w app.js ustawia).
        liveMarkers.forEach(m => {
            const c = m.corners;
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
    }

    function syncCanvasSize() {
        if (zoneCanvas.width !== liveCanvas.width ||
            zoneCanvas.height !== liveCanvas.height) {
            zoneCanvas.width = liveCanvas.width || 640;
            zoneCanvas.height = liveCanvas.height || 480;
        }
    }

    function drawOverlay() {
        syncCanvasSize();
        const w = zoneCanvas.width, h = zoneCanvas.height;
        zoneCtx.clearRect(0, 0, w, h);

        if (layers.zones) {
            zones.forEach(z => {
                if (z.active === false) return;
                const poly = polygonFor(z);
                if (poly.length < 3) return;
                const label = z.marker_ids && z.marker_ids.length
                    ? z.name + " — " + (z.severity || "DANGER")
                    : z.name + " — " + (z.severity || "DANGER");
                drawPolygon(poly, colorsFor(z.severity), { label: label });
            });
        }

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
        const next = zones.concat([{
            name, severity, polygon: draftPoly, marker_ids: [], active: true,
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

    function tick() {
        if (drawing || zones.length > 0 || liveMarkers.length > 0) drawOverlay();
        requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);

    document.addEventListener("perimetr-frame", (evt) => {
        const d = evt.detail || {};
        const active = d.active_zones || [];
        const next = {};
        active.forEach(a => { if (a.id && a.polygon) next[a.id] = a.polygon; });
        livePolygons = next;
        liveMarkers = d.markers || [];
    });

    document.addEventListener("perimetr-layers", (evt) => {
        layers = evt.detail || layers;
        drawOverlay();
    });

    fetchZones();
})();
