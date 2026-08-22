(function () {
    "use strict";

    var grid = document.getElementById("reviewGrid");
    var emptyEl = document.getElementById("reviewEmpty");
    var resultCount = document.getElementById("resultCount");
    var filterMode = document.getElementById("filterMode");
    var filterSeverity = document.getElementById("filterSeverity");
    var filterKind = document.getElementById("filterKind");
    var filterWorker = document.getElementById("filterWorker");
    var filterReviewStatus = document.getElementById("filterReviewStatus");
    var filterSince = document.getElementById("filterSince");
    var filterUntil = document.getElementById("filterUntil");
    var applyBtn = document.getElementById("applyFilters");
    var clearBtn = document.getElementById("clearFilters");
    var exportBtn = document.getElementById("exportCsv");
    var exportTrainingBtn = document.getElementById("exportTraining");
    var quickDateBtns = document.querySelectorAll(".quick-dates .chip");
    var sumTotal = document.getElementById("sumTotal");
    var sumDanger = document.getElementById("sumDanger");
    var sumWarning = document.getElementById("sumWarning");
    var sumNew = document.getElementById("sumNew");
    var sumPpe = document.getElementById("sumPpe");
    var sumZone = document.getElementById("sumZone");
    var sumSite = document.getElementById("sumSite");
    var sumPosture = document.getElementById("sumPosture");
    var sumFall = document.getElementById("sumFall");
    var sumSmoking = document.getElementById("sumSmoking");
    var sumUnknown = document.getElementById("sumUnknown");
    var lightbox = document.getElementById("lightbox");
    var lightboxClose = document.getElementById("lightboxClose");
    var lightboxImg = document.getElementById("lightboxImg");
    var lightboxVideo = document.getElementById("lightboxVideo");
    var lightboxVideoSource = document.getElementById("lightboxVideoSource");
    var lightboxVideoStatus = document.getElementById("lightboxVideoStatus");
    var lightboxTitle = document.getElementById("lightboxTitle");
    var lightboxTime = document.getElementById("lightboxTime");
    var lightboxMode = document.getElementById("lightboxMode");
    var lightboxCamera = document.getElementById("lightboxCamera");
    var lightboxRule = document.getElementById("lightboxRule");
    var lightboxSeverity = document.getElementById("lightboxSeverity");
    var lightboxWorker = document.getElementById("lightboxWorker");
    var lightboxTrack = document.getElementById("lightboxTrack");
    var lightboxAction = document.getElementById("lightboxAction");
    var lightboxSafety = document.getElementById("lightboxSafety");
    var lightboxConfidence = document.getElementById("lightboxConfidence");
    var lightboxMarker = document.getElementById("lightboxMarker");
    var lightboxReviewStatus = document.getElementById("lightboxReviewStatus");
    var lightboxDetails = document.getElementById("lightboxDetails");
    var reviewedBy = document.getElementById("reviewedBy");
    var reviewNote = document.getElementById("reviewNote");
    var reviewSaveStatus = document.getElementById("reviewSaveStatus");
    var reviewButtons = document.querySelectorAll("[data-review]");

    var MODE_LABEL = { site: "Plac", checkpoint: "Bramka" };
    var records = [];
    var activeRecord = null;

    var REVIEW_LABEL = {
        new: "Nowe",
        acknowledged: "Przyjęte",
        confirmed: "Potwierdzone",
        false_positive: "Fałszywy alarm",
        escalated: "Eskalowane",
    };

    function formatTime(ts) {
        var d = new Date(ts * 1000);
        return d.getFullYear() + "-" +
               String(d.getMonth() + 1).padStart(2, "0") + "-" +
               String(d.getDate()).padStart(2, "0") + " " +
               String(d.getHours()).padStart(2, "0") + ":" +
               String(d.getMinutes()).padStart(2, "0") + ":" +
               String(d.getSeconds()).padStart(2, "0");
    }

    function formatWorker(details) {
        if (!details) return "";
        var workerId = details.worker_id;
        var fullName = details.worker_full_name ||
            [details.worker_first_name, details.worker_last_name].filter(Boolean).join(" ");
        return [
            fullName || (workerId ? "ID " + workerId : ""),
            fullName && workerId ? "ID " + workerId : "",
            details.worker_position,
            details.worker_department,
        ].filter(Boolean).join(" · ");
    }

    function formatHistoryWorker(rec) {
        if (rec.worker) {
            var name = [rec.worker.first_name, rec.worker.last_name].filter(Boolean).join(" ");
            return name + " (ID " + rec.worker.worker_id + ")";
        }
        return formatWorker(rec.details) || "pracownik niezidentyfikowany";
    }

    function resetVideo() {
        lightboxVideo.pause();
        lightboxVideoSource.removeAttribute("src");
        lightboxVideo.load();
        lightboxVideo.classList.add("hidden");
        lightboxVideoStatus.textContent = "";
    }

    function localDatetimeToTs(v) {
        if (!v) return null;
        var d = new Date(v);
        if (isNaN(d.getTime())) return null;
        return d.getTime() / 1000;
    }

    function activeSinceUntil() {
        return {
            since: localDatetimeToTs(filterSince.value),
            until: localDatetimeToTs(filterUntil.value),
        };
    }

    function buildQuery() {
        var qs = new URLSearchParams();
        qs.set("limit", "500");
        if (filterMode.value) qs.set("mode", filterMode.value);
        if (filterSeverity.value) qs.set("severity", filterSeverity.value);
        if (filterKind.value) qs.set("kind", filterKind.value);
        if (filterWorker.value.trim()) qs.set("worker_id", filterWorker.value.trim());
        if (filterReviewStatus.value) qs.set("review_status", filterReviewStatus.value);
        var r = activeSinceUntil();
        if (r.since !== null) qs.set("since", String(r.since));
        if (r.until !== null) qs.set("until", String(r.until));
        return qs.toString();
    }

    function buildSummaryQuery() {
        var qs = new URLSearchParams();
        if (filterMode.value) qs.set("mode", filterMode.value);
        if (filterSeverity.value) qs.set("severity", filterSeverity.value);
        if (filterKind.value) qs.set("kind", filterKind.value);
        if (filterWorker.value.trim()) qs.set("worker_id", filterWorker.value.trim());
        if (filterReviewStatus.value) qs.set("review_status", filterReviewStatus.value);
        var r = activeSinceUntil();
        if (r.since !== null) qs.set("since", String(r.since));
        if (r.until !== null) qs.set("until", String(r.until));
        return qs.toString();
    }

    function fetchRecords() {
        return fetch("/api/alerts?" + buildQuery(), {
            credentials: "same-origin",
        }).then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        });
    }

    function fetchSummary() {
        return fetch("/api/alerts/summary?" + buildSummaryQuery(), {
            credentials: "same-origin",
        }).then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        });
    }

    function renderSummary(s) {
        sumTotal.textContent = s.total;
        sumDanger.textContent = s.by_severity.DANGER || 0;
        sumWarning.textContent = s.by_severity.WARNING || 0;
        sumNew.textContent = (s.by_review_status && s.by_review_status.new) || 0;
        sumPpe.textContent = s.by_kind.ppe_missing || 0;
        sumZone.textContent = (s.by_kind.zone_breach || 0) + (s.by_kind.zone_approach || 0);
        sumSite.textContent = s.by_kind.site_hazard || 0;
        sumPosture.textContent = s.by_kind.posture_anomaly || 0;
        sumFall.textContent = s.by_kind.fall_detected || 0;
        sumSmoking.textContent = s.by_kind.smoking_gesture || 0;
        sumUnknown.textContent = s.by_kind.unidentified_worker || 0;
    }

    function render(list) {
        records = list;
        grid.innerHTML = "";
        resultCount.textContent = list.length + " zdarzen";
        if (list.length === 0) {
            emptyEl.classList.remove("hidden");
            return;
        }
        emptyEl.classList.add("hidden");
        list.forEach(function (rec, idx) {
            grid.appendChild(makeCard(rec, idx));
        });
    }

    function makeCard(rec, idx) {
        var card = document.createElement("div");
        card.className = "review-card";
        card.dataset.idx = idx;

        var thumb = document.createElement("div");
        thumb.className = "thumb";
        if (rec.thumbnail_url) {
            thumb.style.backgroundImage = "url('" + rec.thumbnail_url + "')";
        } else {
            thumb.classList.add("no-image");
            thumb.textContent = "BRAK ZDJECIA";
        }

        var sev = document.createElement("span");
        sev.className = "sev-tag " + rec.severity;
        sev.textContent = rec.severity;
        thumb.appendChild(sev);

        var mode = document.createElement("span");
        mode.className = "mode-tag";
        mode.textContent = (MODE_LABEL[rec.mode] || rec.mode).toUpperCase();
        thumb.appendChild(mode);

        var review = document.createElement("span");
        review.className = "review-tag review-" + (rec.review_status || "new");
        review.textContent = REVIEW_LABEL[rec.review_status || "new"] || rec.review_status;
        thumb.appendChild(review);

        var body = document.createElement("div");
        body.className = "card-body";
        var desc = document.createElement("div");
        desc.className = "desc";
        desc.textContent = rec.description;
        var time = document.createElement("div");
        time.className = "time";
        var workerLabel = formatWorker(rec.details);
        time.textContent = formatTime(rec.timestamp) + (workerLabel ? " · " + workerLabel : "");
        body.appendChild(desc);
        body.appendChild(time);

        card.appendChild(thumb);
        card.appendChild(body);
        card.onclick = function () { openLightbox(rec); };
        return card;
    }

    function openLightbox(rec) {
        activeRecord = rec;
        lightboxImg.src = rec.snapshot_url || rec.thumbnail_url || "";
        resetVideo();
        if (rec.clip_available && rec.clip_url) {
            lightboxVideoSource.src = rec.clip_url;
            lightboxVideo.classList.remove("hidden");
            lightboxVideoStatus.textContent = "Ładowanie metadanych klipu…";
            lightboxVideo.load();
        } else {
            lightboxVideoStatus.textContent = "Brak klipu powiązanego z tym zdarzeniem";
        }
        lightboxTitle.textContent = rec.description;
        lightboxTime.textContent = formatTime(rec.timestamp);
        lightboxMode.textContent = MODE_LABEL[rec.mode] || rec.mode;
        lightboxCamera.textContent = rec.camera_id;
        lightboxRule.textContent = rec.rule_name;
        lightboxSeverity.textContent = rec.severity;
        lightboxSeverity.className = "meta-val";
        lightboxWorker.textContent = formatHistoryWorker(rec);
        lightboxTrack.textContent = rec.details.track_id ?? "—";
        lightboxAction.textContent = rec.details.action || rec.details.behavior_action || "—";
        lightboxSafety.textContent = rec.details.safety_state || rec.details.state || "—";
        var confidence = rec.details.confidence ?? rec.details.score;
        lightboxConfidence.textContent = confidence == null ? "—" : Number(confidence).toFixed(3);
        lightboxMarker.textContent = rec.details.marker_id ?? "—";
        lightboxReviewStatus.textContent = REVIEW_LABEL[rec.review_status || "new"] || rec.review_status;
        reviewedBy.value = rec.reviewed_by || "";
        reviewNote.value = rec.review_note || "";
        reviewSaveStatus.textContent = "";
        lightboxDetails.textContent = JSON.stringify(rec.details, null, 2);
        lightbox.classList.remove("hidden");
    }

    function closeLightbox() {
        lightbox.classList.add("hidden");
        lightboxImg.src = "";
        resetVideo();
        activeRecord = null;
    }

    function refresh() {
        Promise.all([fetchRecords(), fetchSummary()])
            .then(function (results) { render(results[0]); renderSummary(results[1]); })
            .catch(function (e) { resultCount.textContent = "Błąd: " + e.message; });
    }

    function setQuickRange(range) {
        quickDateBtns.forEach(function (b) {
            b.classList.toggle("active", b.dataset.range === range);
        });
        var now = new Date();
        var startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        function toLocalInput(d) {
            var pad = function (n) { return String(n).padStart(2, "0"); };
            return d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" +
                   pad(d.getDate()) + "T" + pad(d.getHours()) + ":" + pad(d.getMinutes());
        }
        if (range === "today") {
            filterSince.value = toLocalInput(startOfToday);
            filterUntil.value = "";
        } else if (range === "24h") {
            filterSince.value = toLocalInput(new Date(now.getTime() - 24 * 3600 * 1000));
            filterUntil.value = "";
        } else if (range === "7d") {
            filterSince.value = toLocalInput(new Date(now.getTime() - 7 * 24 * 3600 * 1000));
            filterUntil.value = "";
        } else {
            filterSince.value = "";
            filterUntil.value = "";
        }
        refresh();
    }

    quickDateBtns.forEach(function (btn) {
        btn.addEventListener("click", function () { setQuickRange(btn.dataset.range); });
    });

    applyBtn.onclick = refresh;
    clearBtn.onclick = function () {
        filterMode.value = "";
        filterSeverity.value = "";
        filterKind.value = "";
        filterWorker.value = "";
        filterReviewStatus.value = "";
        filterSince.value = "";
        filterUntil.value = "";
        setQuickRange("all");
    };

    function buildExportQuery() {
        var qs = new URLSearchParams();
        if (filterMode.value) qs.set("mode", filterMode.value);
        if (filterSeverity.value) qs.set("severity", filterSeverity.value);
        if (filterKind.value) qs.set("kind", filterKind.value);
        if (filterWorker.value.trim()) qs.set("worker_id", filterWorker.value.trim());
        if (filterReviewStatus.value) qs.set("review_status", filterReviewStatus.value);
        var r = activeSinceUntil();
        if (r.since !== null) qs.set("since", String(r.since));
        if (r.until !== null) qs.set("until", String(r.until));
        return qs.toString();
    }

    exportBtn.onclick = function () {
        window.location.href = "/api/reports/export.csv?" + buildExportQuery();
    };
    exportTrainingBtn.onclick = function () {
        var status = filterReviewStatus.value || "confirmed";
        if (["confirmed", "false_positive"].indexOf(status) < 0) status = "confirmed";
        var qs = new URLSearchParams();
        qs.set("review_status", status);
        var r = activeSinceUntil();
        if (r.since !== null) qs.set("since", String(r.since));
        if (r.until !== null) qs.set("until", String(r.until));
        window.location.href = "/api/reports/training.jsonl?" + qs.toString();
    };

    function saveReview(status) {
        if (!activeRecord) return;
        reviewSaveStatus.textContent = "Zapisywanie…";
        fetch("/api/alerts/" + encodeURIComponent(activeRecord.id) + "/review", {
            method: "PATCH",
            credentials: "same-origin",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                status: status,
                reviewed_by: reviewedBy.value.trim() || null,
                note: reviewNote.value.trim() || null,
            }),
        }).then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        }).then(function (updated) {
            activeRecord = updated;
            lightboxReviewStatus.textContent = REVIEW_LABEL[updated.review_status] || updated.review_status;
            reviewSaveStatus.textContent = "Zapisano";
            refresh();
        }).catch(function (e) {
            reviewSaveStatus.textContent = "Błąd: " + e.message;
        });
    }

    reviewButtons.forEach(function (button) {
        button.addEventListener("click", function () {
            saveReview(button.dataset.review);
        });
    });
    lightboxClose.onclick = closeLightbox;
    lightboxVideo.addEventListener("loadedmetadata", function () { lightboxVideoStatus.textContent = ""; });
    lightboxVideo.addEventListener("canplay", function () { lightboxVideoStatus.textContent = ""; });
    lightboxVideo.addEventListener("error", function () {
        var code = lightboxVideo.error ? lightboxVideo.error.code : "?";
        lightboxVideoStatus.textContent = "Klip istnieje, ale jego format lub kodek nie jest obsługiwany przez przeglądarkę" +
            (window.REVIEW_VIDEO_DIAGNOSTICS ? " (MediaError " + code + ", " + (lightboxVideoSource.getAttribute("src") || "") + ")" : "");
    });
    lightbox.onclick = function (e) {
        if (e.target === lightbox) closeLightbox();
    };
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") closeLightbox();
    });

    refresh();
})();
