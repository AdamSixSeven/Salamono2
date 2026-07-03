(function () {
    "use strict";

    const canvas = document.getElementById("liveCanvas");
    const ctx = canvas.getContext("2d");
    const alertList = document.getElementById("alertList");
    const alarmBanner = document.getElementById("alarmBanner");
    const alarmText = document.getElementById("alarmText");
    const noSignal = document.getElementById("noSignal");
    const connectionStatus = document.getElementById("connectionStatus");
    const frameCount = document.getElementById("frameCount");
    const fpsDisplay = document.getElementById("fpsDisplay");
    const alertCount = document.getElementById("alertCount");
    const alertTotal = document.getElementById("alertTotal");
    const processingTime = document.getElementById("processingTime");

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
            updateAlerts(data);
            updateStats(data);
        };
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
