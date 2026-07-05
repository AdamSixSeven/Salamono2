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

    let zones = [];              // saved zones from server
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
            polygon: z.polygon, active: z.active !== false,
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
            name.textContent = z.name;
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

    function drawOverlay() {
        syncCanvasSize();
        const w = zoneCanvas.width;
        const h = zoneCanvas.height;
        zoneCtx.clearRect(0, 0, w, h);
        if (!drawing) return;

        // draft polygon
        if (draftPoly.length > 0) {
            zoneCtx.save();
            zoneCtx.lineWidth = 2;
            zoneCtx.strokeStyle = severitySelect.value === "WARNING"
                ? "#ff9800" : "#ff4d4d";
            zoneCtx.fillStyle = severitySelect.value === "WARNING"
                ? "rgba(255, 152, 0, 0.2)" : "rgba(255, 77, 77, 0.2)";
            zoneCtx.beginPath();
            draftPoly.forEach((pt, i) => {
                const px = pt[0] * w, py = pt[1] * h;
                if (i === 0) zoneCtx.moveTo(px, py);
                else zoneCtx.lineTo(px, py);
            });
            if (draftPoly.length >= 3) {
                zoneCtx.closePath();
                zoneCtx.fill();
            }
            zoneCtx.stroke();
            // vertex dots
            zoneCtx.fillStyle = "#fff";
            draftPoly.forEach(pt => {
                zoneCtx.beginPath();
                zoneCtx.arc(pt[0] * w, pt[1] * h, 4, 0, Math.PI * 2);
                zoneCtx.fill();
            });
            zoneCtx.restore();
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

    // Keep overlay canvas sized to live canvas + redraw on each frame
    const observer = new MutationObserver(() => { if (drawing) drawOverlay(); });
    observer.observe(liveCanvas, { attributes: true, attributeFilter: ["width", "height"] });
    window.addEventListener("resize", () => { if (drawing) drawOverlay(); });

    fetchZones();
})();
