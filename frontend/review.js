(function () {
    "use strict";

    var grid = document.getElementById("reviewGrid");
    var emptyEl = document.getElementById("reviewEmpty");
    var resultCount = document.getElementById("resultCount");
    var filterMode = document.getElementById("filterMode");
    var filterSeverity = document.getElementById("filterSeverity");
    var filterSince = document.getElementById("filterSince");
    var filterUntil = document.getElementById("filterUntil");
    var applyBtn = document.getElementById("applyFilters");
    var clearBtn = document.getElementById("clearFilters");
    var lightbox = document.getElementById("lightbox");
    var lightboxClose = document.getElementById("lightboxClose");
    var lightboxImg = document.getElementById("lightboxImg");
    var lightboxTitle = document.getElementById("lightboxTitle");
    var lightboxTime = document.getElementById("lightboxTime");
    var lightboxMode = document.getElementById("lightboxMode");
    var lightboxCamera = document.getElementById("lightboxCamera");
    var lightboxRule = document.getElementById("lightboxRule");
    var lightboxSeverity = document.getElementById("lightboxSeverity");
    var lightboxDetails = document.getElementById("lightboxDetails");

    var MODE_LABEL = { site: "Plac", checkpoint: "Bramka" };
    var records = [];

    function formatTime(ts) {
        var d = new Date(ts * 1000);
        return d.getFullYear() + "-" +
               String(d.getMonth() + 1).padStart(2, "0") + "-" +
               String(d.getDate()).padStart(2, "0") + " " +
               String(d.getHours()).padStart(2, "0") + ":" +
               String(d.getMinutes()).padStart(2, "0") + ":" +
               String(d.getSeconds()).padStart(2, "0");
    }

    function localDatetimeToTs(v) {
        if (!v) return null;
        var d = new Date(v);
        if (isNaN(d.getTime())) return null;
        return d.getTime() / 1000;
    }

    function buildQuery() {
        var qs = new URLSearchParams();
        qs.set("limit", "500");
        if (filterMode.value) qs.set("mode", filterMode.value);
        if (filterSeverity.value) qs.set("severity", filterSeverity.value);
        var since = localDatetimeToTs(filterSince.value);
        var until = localDatetimeToTs(filterUntil.value);
        if (since !== null) qs.set("since", String(since));
        if (until !== null) qs.set("until", String(until));
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

        var body = document.createElement("div");
        body.className = "card-body";
        var desc = document.createElement("div");
        desc.className = "desc";
        desc.textContent = rec.description;
        var time = document.createElement("div");
        time.className = "time";
        time.textContent = formatTime(rec.timestamp);
        body.appendChild(desc);
        body.appendChild(time);

        card.appendChild(thumb);
        card.appendChild(body);
        card.onclick = function () { openLightbox(rec); };
        return card;
    }

    function openLightbox(rec) {
        lightboxImg.src = rec.thumbnail_url || "";
        lightboxTitle.textContent = rec.description;
        lightboxTime.textContent = formatTime(rec.timestamp);
        lightboxMode.textContent = MODE_LABEL[rec.mode] || rec.mode;
        lightboxCamera.textContent = rec.camera_id;
        lightboxRule.textContent = rec.rule_name;
        lightboxSeverity.textContent = rec.severity;
        lightboxSeverity.className = "meta-val";
        lightboxDetails.textContent = JSON.stringify(rec.details, null, 2);
        lightbox.classList.remove("hidden");
    }

    function closeLightbox() {
        lightbox.classList.add("hidden");
        lightboxImg.src = "";
    }

    applyBtn.onclick = function () {
        fetchRecords().then(render).catch(function (e) {
            resultCount.textContent = "Blad: " + e.message;
        });
    };
    clearBtn.onclick = function () {
        filterMode.value = "";
        filterSeverity.value = "";
        filterSince.value = "";
        filterUntil.value = "";
        applyBtn.onclick();
    };
    lightboxClose.onclick = closeLightbox;
    lightbox.onclick = function (e) {
        if (e.target === lightbox) closeLightbox();
    };
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") closeLightbox();
    });

    applyBtn.onclick();
})();
