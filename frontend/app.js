(function () {
    "use strict";

    const canvas = document.getElementById("liveCanvas");
    const ctx = canvas.getContext("2d");
    const alertList = document.getElementById("alertList");
    const alarmBanner = document.getElementById("alarmBanner");
    const alarmText = document.getElementById("alarmText");
    const noSignal = document.getElementById("noSignal");
    const connectionStatus = document.getElementById("connectionStatus");
    const modeIndicator = document.getElementById("modeIndicator");
    const frameCount = document.getElementById("frameCount");
    const fpsDisplay = document.getElementById("fpsDisplay");
    const alertCount = document.getElementById("alertCount");
    const alertTotal = document.getElementById("alertTotal");
    const processingTime = document.getElementById("processingTime");
    const ppeBadge = document.getElementById("ppeBadge");
    const ppeHeadline = document.getElementById("ppeHeadline");
    const ppeHardhat = document.getElementById("ppeHardhat");
    const ppeVest = document.getElementById("ppeVest");
    const MODE_LABEL = { site: "Plac (pojazdy)", checkpoint: "Bramka (PPE)" };
    let ppeHideTimeout = null;

    let ws = null;
    let totalAlerts = 0;
    let frameIdx = 0;
    let fpsFrames = 0;
    let fpsLastTime = performance.now();
    let alarmTimeout = null;

    function connectWebSocket() {
        const protocol = location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(protocol + "//" + location.host + "/ws/live");

        ws.onopen = function () {
            connectionStatus.textContent = "Polaczono";
            connectionStatus.className = "status connected";
        };

        ws.onclose = function () {
            connectionStatus.textContent = "Rozlaczono";
            connectionStatus.className = "status disconnected";
            setTimeout(connectWebSocket, 2000);
        };

        ws.onerror = function () {
            ws.close();
        };

        ws.onmessage = function (event) {
            var data = JSON.parse(event.data);
            renderFrame(data);
            updateMode(data);
            updateAlerts(data);
            updatePPE(data);
            updateStats(data);
        };
    }

    function updateMode(data) {
        var m = data.mode || "site";
        modeIndicator.textContent = "Tryb: " + (MODE_LABEL[m] || m);
    }

    function updatePPE(data) {
        if (data.mode !== "checkpoint") {
            ppeBadge.classList.add("hidden");
            return;
        }
        var checks = data.ppe_checks || [];
        if (checks.length === 0) return;
        var check = checks[0];
        var ok = check.severity === "OK";

        ppeBadge.classList.remove("hidden", "ok", "fail");
        ppeBadge.classList.add(ok ? "ok" : "fail");
        ppeHeadline.textContent = ok ? "PPE OK" : "BRAK: " + check.missing.map(function(m){return m.toUpperCase();}).join(" + ");

        setPpeItem(ppeHardhat, check.has_hardhat);
        setPpeItem(ppeVest, check.has_vest);

        if (!ok) {
            totalAlerts++;
            var el = createPpeAlertElement(check);
            alertList.insertBefore(el, alertList.firstChild);
            while (alertList.children.length > 100) {
                alertList.removeChild(alertList.lastChild);
            }
            alertTotal.textContent = totalAlerts;
        }

        if (ppeHideTimeout) clearTimeout(ppeHideTimeout);
        ppeHideTimeout = setTimeout(function() {
            ppeBadge.classList.add("hidden");
        }, 6000);
    }

    function setPpeItem(el, ok) {
        el.classList.remove("ok", "fail");
        el.classList.add(ok ? "ok" : "fail");
        el.querySelector(".ppe-mark").textContent = ok ? "✓" : "✗";
    }

    function createPpeAlertElement(check) {
        var item = document.createElement("div");
        item.className = "alert-item";
        var date = new Date(check.timestamp * 1000);
        var timeStr = date.getHours().toString().padStart(2, "0") + ":" +
                      date.getMinutes().toString().padStart(2, "0") + ":" +
                      date.getSeconds().toString().padStart(2, "0");
        var thumbHtml = check.frame_thumbnail_url ?
            '<img class="alert-thumb" src="' + check.frame_thumbnail_url + '">' : "";
        item.innerHTML =
            thumbHtml +
            '<div class="alert-info">' +
            '<span class="alert-time">' + timeStr + '</span>' +
            '<span class="alert-rule">Brak PPE</span>' +
            '<span class="alert-severity danger">DANGER</span>' +
            '<span class="alert-details">Brak: ' + check.missing.join(", ") + '</span>' +
            '</div>';
        return item;
    }

    function renderFrame(data) {
        if (!data.frame_jpeg_b64) return;

        noSignal.classList.add("hidden");

        var img = new Image();
        img.onload = function () {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        img.src = "data:image/jpeg;base64," + data.frame_jpeg_b64;
    }

    function updateAlerts(data) {
        if (data.confirmed_alerts && data.confirmed_alerts.length > 0) {
            var count = data.confirmed_alerts.length;
            alarmBanner.classList.remove("hidden");
            alarmBanner.classList.add("pulse");
            alarmText.textContent =
                "ALARM — Wykryto " + count + " zagrozenie/a!";

            if (alarmTimeout) clearTimeout(alarmTimeout);
            alarmTimeout = setTimeout(function () {
                alarmBanner.classList.add("hidden");
                alarmBanner.classList.remove("pulse");
            }, 5000);

            for (var i = 0; i < data.confirmed_alerts.length; i++) {
                var alert = data.confirmed_alerts[i];
                totalAlerts++;
                var el = createAlertElement(alert);
                alertList.insertBefore(el, alertList.firstChild);
            }

            while (alertList.children.length > 100) {
                alertList.removeChild(alertList.lastChild);
            }

            alertTotal.textContent = totalAlerts;
        }
    }

    function createAlertElement(alert) {
        var item = document.createElement("div");
        item.className =
            "alert-item" +
            (alert.severity === "WARNING" ? " warning" : "");

        var date = new Date(alert.timestamp * 1000);
        var timeStr =
            date.getHours().toString().padStart(2, "0") +
            ":" +
            date.getMinutes().toString().padStart(2, "0") +
            ":" +
            date.getSeconds().toString().padStart(2, "0");

        var thumbHtml = "";
        if (alert.frame_thumbnail_url) {
            thumbHtml =
                '<img class="alert-thumb" src="' +
                alert.frame_thumbnail_url +
                '" alt="thumbnail">';
        }

        var severityClass =
            alert.severity === "DANGER" ? "danger" : "warning";
        var ruleLabel =
            alert.rule_name === "person_vehicle_overlap"
                ? "Osoba w strefie pojazdu"
                : "Osoba blisko pojazdu";

        var detailParts = [];
        if (alert.distance_px > 0) {
            detailParts.push(alert.distance_px.toFixed(0) + "px");
        }
        if (alert.overlap_iou > 0) {
            detailParts.push(
                "IoU: " + (alert.overlap_iou * 100).toFixed(0) + "%"
            );
        }
        detailParts.push(
            "Pewnosc: " +
                (alert.person.confidence * 100).toFixed(0) +
                "% / " +
                (alert.hazard.confidence * 100).toFixed(0) +
                "%"
        );

        item.innerHTML =
            thumbHtml +
            '<div class="alert-info">' +
            '<span class="alert-time">' + timeStr + "</span>" +
            '<span class="alert-rule">' + ruleLabel + "</span>" +
            '<span class="alert-severity ' + severityClass + '">' +
            alert.severity + "</span>" +
            '<span class="alert-details">' +
            detailParts.join(" | ") + "</span>" +
            "</div>";

        return item;
    }

    function updateStats(data) {
        frameIdx++;
        frameCount.textContent = frameIdx;
        processingTime.textContent = data.processing_ms
            ? data.processing_ms.toFixed(0)
            : "0";
        alertCount.textContent = totalAlerts;

        fpsFrames++;
        var now = performance.now();
        var elapsed = now - fpsLastTime;
        if (elapsed >= 1000) {
            var fps = (fpsFrames / elapsed) * 1000;
            fpsDisplay.textContent = fps.toFixed(1);
            fpsFrames = 0;
            fpsLastTime = now;
        }
    }

    connectWebSocket();
})();
