(function () {
    "use strict";

    const CAMERA_ID = "cam_default";

    const stack = document.getElementById("canvasStack");
    const liveCanvas = document.getElementById("liveCanvas");
    const zoneCanvas = document.getElementById("zoneCanvas");
    const zoneCtx = zoneCanvas.getContext("2d");
    const zoneList = document.getElementById("zoneList");
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

    let zones = [];              // saved zones from server (name/severity/marker_ids)
    let livePolygons = {};       // zone_id → resolved polygon from latest frame
    let drawing = false;
    let draftPoly = [];          // [[x_norm, y_norm], ...]

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

    function renderZoneList() {
        zoneList.innerHTML = "";
        zones.forEach(z => {
            const row = document.createElement("div");
            row.className = "zone-row" + (z.severity === "WARNING" ? " warning" : "");
            const name = document.createElement("span");
            name.className = "zone-row-name";
            const suffix = z.marker_ids && z.marker_ids.length
                ? " · markery [" + z.marker_ids.join(",") + "]"
                : "";
            name.textContent = z.name + suffix;
            const sev = document.createElement("span");
            sev.className = "zone-row-sev";
            sev.textContent = z.severity;
            const del = document.createElement("button");
            del.type = "button";
            del.className = "danger";
            del.textContent = "×";
            del.title = "Usun strefe";
            del.onclick = () => deleteZone(z.id);
            row.appendChild(name);
            row.appendChild(sev);
            row.appendChild(del);
            zoneList.appendChild(row);
        });
    }

    async function deleteZone(zoneId) {
        const next = zones.filter(z => z.id !== zoneId);
        try {
            await saveZones(next);
        } catch (e) {
            alert("Nie udalo sie usunac strefy: " + e.message);
        }
    }

    function startDrawing() {
        drawing = true;
        draftPoly = [];
        drawPanel.classList.remove("hidden");
        drawBtn.disabled = true;
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
        stack.classList.remove("drawing");
        drawOverlay();
    }

    async function commitDrawing() {
        if (draftPoly.length < 3) return;
        const name = (nameInput.value || "Strefa").trim().slice(0, 40);
        const severity = severitySelect.value;
        const next = zones.concat([{
            name, severity, polygon: draftPoly, active: true,
        }]);
        try {
            await saveZones(next);
            cancelDrawing();
        } catch (e) {
            alert("Nie udalo sie zapisac: " + e.message);
        }
    }

    function updateSaveEnabled() {
        saveBtn.disabled = draftPoly.length < 3;
    }

    function syncCanvasSize() {
        if (zoneCanvas.width !== liveCanvas.width ||
            zoneCanvas.height !== liveCanvas.height) {
            zoneCanvas.width = liveCanvas.width || 640;
            zoneCanvas.height = liveCanvas.height || 480;
        }
    }

    function colorsFor(severity) {
        return severity === "WARNING"
            ? { stroke: "#ff9800", fill: "rgba(255, 152, 0, 0.2)" }
            : { stroke: "#ff4d4d", fill: "rgba(255, 77, 77, 0.2)" };
    }

    function drawPolygon(poly, colors, opts) {
        opts = opts || {};
        const w = zoneCanvas.width, h = zoneCanvas.height;
        if (poly.length < 1) return;
        zoneCtx.save();
        zoneCtx.lineWidth = opts.lineWidth || 2;
        zoneCtx.setLineDash(opts.dashed ? [8, 4] : []);
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
            zoneCtx.fillStyle = "#fff";
            poly.forEach(pt => {
                zoneCtx.beginPath();
                zoneCtx.arc(pt[0] * w, pt[1] * h, 4, 0, Math.PI * 2);
                zoneCtx.fill();
            });
        }
        if (opts.label) {
            const first = poly[0];
            zoneCtx.font = "600 13px system-ui, sans-serif";
            zoneCtx.fillStyle = colors.stroke;
            zoneCtx.strokeStyle = "rgba(0,0,0,0.7)";
            zoneCtx.lineWidth = 3;
            zoneCtx.strokeText(opts.label, first[0] * w + 6, first[1] * h - 6);
            zoneCtx.fillText(opts.label, first[0] * w + 6, first[1] * h - 6);
        }
        zoneCtx.restore();
    }

    function polygonFor(z) {
        // Marker-zones store no polygon on disk; use whatever backend
        // resolved for the most recent frame. Regular polygon-zones fall
        // back to their persisted polygon so they render even before the
        // first WS message arrives.
        if (livePolygons[z.id]) return livePolygons[z.id];
        return z.polygon || [];
    }

    function drawOverlay() {
        syncCanvasSize();
        const w = zoneCanvas.width, h = zoneCanvas.height;
        zoneCtx.clearRect(0, 0, w, h);

        // saved zones — always visible so user sees where geofence actually is
        zones.forEach(z => {
            if (z.active === false) return;
            const poly = polygonFor(z);
            if (poly.length < 3) return;
            const label = z.marker_ids && z.marker_ids.length
                ? z.name + " · [" + z.marker_ids.join(",") + "]"
                : z.name;
            drawPolygon(poly, colorsFor(z.severity), { label: label });
        });

        // draft polygon (while user is drawing a new one)
        if (drawing && draftPoly.length > 0) {
            drawPolygon(draftPoly, colorsFor(severitySelect.value), {
                dashed: true, vertices: true,
            });
        }
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
        // dblclick fires two clicks first — drop the duplicate 2nd vertex
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

    // Keep overlay canvas sized to live canvas + redraw whenever the underlying
    // frame changes so saved zones stay visible on top of every rendered frame.
    const observer = new MutationObserver(() => drawOverlay());
    observer.observe(liveCanvas, { attributes: true, attributeFilter: ["width", "height"] });
    window.addEventListener("resize", () => drawOverlay());

    // Overlay canvas is cleared implicitly when we resize it in syncCanvasSize.
    // A background RAF loop keeps zones visible even when the live frame stops
    // changing dimensions (which happens once camera settles into steady state).
    function tick() {
        if (drawing || zones.length > 0) drawOverlay();
        requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);

    // WebSocket streams a resolved polygon for every zone in every frame.
    // For marker-zones this is the ONLY source of the polygon (they store
    // an empty polygon on disk). We also refresh the zone list from the
    // server payload so marker_ids/name updates without a hard reload.
    document.addEventListener("perimetr-frame", function (evt) {
        const d = evt.detail || {};
        const active = d.active_zones || [];
        // Rebuild livePolygons from scratch so zones that lost visibility
        // (and dropped from the resolver) disappear from the overlay.
        const next = {};
        active.forEach(a => { if (a.id && a.polygon) next[a.id] = a.polygon; });
        livePolygons = next;
    });

    fetchZones();
})();
