/**
 * Solar AI Sentinel - Interactive Dashboard Frontend Controller
 * Connects modern responsive UI with Flask backend REST API.
 */

// Global Chart Instances
let chartDcAc = null;
let chartIrrAc = null;
let chartPie = null;
let chartTimeSeries = null;
let batchScoredData = null;

document.addEventListener("DOMContentLoaded", () => {
    initTabNavigation();
    initSliderSync();
    initPresetButtons();
    initPredictionForm();
    initBatchProcessor();
    checkBackendHealth();
    loadDashboardKPIs();
    runSinglePrediction(); // Run initial prediction on default values
});

// =================================================================================================
// 1. BACKEND STATUS & KPIS
// =================================================================================================

async function checkBackendHealth() {
    const statusPill = document.getElementById("backend-status");
    const statusText = document.getElementById("status-text");

    try {
        const res = await fetch("/api/status");
        if (res.ok) {
            const data = await res.json();
            statusPill.className = "flex items-center space-x-2 bg-emerald-500/20 text-emerald-300 text-xs px-3 py-1.5 rounded-full border border-emerald-500/40";
            statusText.textContent = "Backend Connected";
        } else {
            throw new Error("HTTP error " + res.status);
        }
    } catch (err) {
        statusPill.className = "flex items-center space-x-2 bg-rose-500/20 text-rose-300 text-xs px-3 py-1.5 rounded-full border border-rose-500/40";
        statusText.textContent = "Backend Offline";
        console.warn("Backend API check failed:", err);
    }
}

async function loadDashboardKPIs() {
    try {
        const res = await fetch("/api/kpis");
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("kpi-total").textContent = Number(data.total_observations).toLocaleString();
        document.getElementById("kpi-normal").textContent = Number(data.isolation_forest.normal_count).toLocaleString();
        document.getElementById("kpi-norm-pct").textContent = data.isolation_forest.normal_percentage.toFixed(2);
        document.getElementById("kpi-anom").textContent = Number(data.isolation_forest.anomaly_count).toLocaleString();
        document.getElementById("kpi-anom-pct").textContent = data.isolation_forest.anomaly_percentage.toFixed(2);
        document.getElementById("kpi-noise").textContent = Number(data.dbscan.noise_count).toLocaleString();
        document.getElementById("kpi-noise-pct").textContent = data.dbscan.noise_percentage.toFixed(4);
    } catch (e) {
        console.error("Error loading KPIs:", e);
    }
}

// =================================================================================================
// 2. TAB NAVIGATION
// =================================================================================================

function initTabNavigation() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-target");

            // Update button styles
            tabBtns.forEach(b => {
                b.classList.remove("active", "text-blue-600", "border-blue-600");
                b.classList.add("text-slate-500", "border-transparent");
            });
            btn.classList.add("active", "text-blue-600", "border-blue-600");
            btn.classList.remove("text-slate-500", "border-transparent");

            // Show target section
            tabContents.forEach(tc => {
                if (tc.id === targetId) {
                    tc.classList.remove("hidden");
                } else {
                    tc.classList.add("hidden");
                }
            });

            // If switching to visual analytics, render charts lazily
            if (targetId === "tab-analytics" && !chartDcAc) {
                loadAndRenderCharts();
            }
        });
    });
}

// =================================================================================================
// 3. INPUT SLIDERS & SYNCHRONIZATION
// =================================================================================================

const inputPairs = [
    { range: "range-dc", num: "num-dc", echo: "echo-dc", suffix: " kW" },
    { range: "range-ac", num: "num-ac", echo: "echo-ac", suffix: " kW" },
    { range: "range-yield", num: "num-yield", echo: "echo-yield", suffix: " kWh" },
    { range: "range-amb-temp", num: "num-amb-temp", echo: "echo-amb", suffix: " °C" },
    { range: "range-mod-temp", num: "num-mod-temp", echo: "echo-mod", suffix: " °C" },
    { range: "range-irr", num: "num-irr", echo: "echo-irr", suffix: "" }
];

function initSliderSync() {
    inputPairs.forEach(pair => {
        const rangeEl = document.getElementById(pair.range);
        const numEl = document.getElementById(pair.num);
        const echoEl = document.getElementById(pair.echo);

        rangeEl.addEventListener("input", () => {
            numEl.value = rangeEl.value;
            echoEl.textContent = rangeEl.value + pair.suffix;
        });

        numEl.addEventListener("input", () => {
            rangeEl.value = numEl.value;
            echoEl.textContent = numEl.value + pair.suffix;
        });
    });
}

function setFormValues(values) {
    if (values.DC_POWER !== undefined) {
        document.getElementById("range-dc").value = values.DC_POWER;
        document.getElementById("num-dc").value = values.DC_POWER;
        document.getElementById("echo-dc").textContent = values.DC_POWER + " kW";
    }
    if (values.AC_POWER !== undefined) {
        document.getElementById("range-ac").value = values.AC_POWER;
        document.getElementById("num-ac").value = values.AC_POWER;
        document.getElementById("echo-ac").textContent = values.AC_POWER + " kW";
    }
    if (values.DAILY_YIELD !== undefined) {
        document.getElementById("range-yield").value = values.DAILY_YIELD;
        document.getElementById("num-yield").value = values.DAILY_YIELD;
        document.getElementById("echo-yield").textContent = values.DAILY_YIELD + " kWh";
    }
    if (values.AMBIENT_TEMPERATURE !== undefined) {
        document.getElementById("range-amb-temp").value = values.AMBIENT_TEMPERATURE;
        document.getElementById("num-amb-temp").value = values.AMBIENT_TEMPERATURE;
        document.getElementById("echo-amb").textContent = values.AMBIENT_TEMPERATURE + " °C";
    }
    if (values.MODULE_TEMPERATURE !== undefined) {
        document.getElementById("range-mod-temp").value = values.MODULE_TEMPERATURE;
        document.getElementById("num-mod-temp").value = values.MODULE_TEMPERATURE;
        document.getElementById("echo-mod").textContent = values.MODULE_TEMPERATURE + " °C";
    }
    if (values.IRRADIATION !== undefined) {
        document.getElementById("range-irr").value = values.IRRADIATION;
        document.getElementById("num-irr").value = values.IRRADIATION;
        document.getElementById("echo-irr").textContent = values.IRRADIATION;
    }
}

function getFormValues() {
    return {
        DC_POWER: parseFloat(document.getElementById("num-dc").value),
        AC_POWER: parseFloat(document.getElementById("num-ac").value),
        DAILY_YIELD: parseFloat(document.getElementById("num-yield").value),
        AMBIENT_TEMPERATURE: parseFloat(document.getElementById("num-amb-temp").value),
        MODULE_TEMPERATURE: parseFloat(document.getElementById("num-mod-temp").value),
        IRRADIATION: parseFloat(document.getElementById("num-irr").value)
    };
}

// =================================================================================================
// 4. PRESETS BUTTONS
// =================================================================================================

async function initPresetButtons() {
    const container = document.getElementById("preset-buttons");
    try {
        const res = await fetch("/api/presets");
        if (!res.ok) return;
        const presets = await res.json();

        container.innerHTML = "";
        presets.forEach((preset, idx) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "preset-btn" + (idx === 0 ? " active" : "");
            btn.innerHTML = `<span class="block truncate">${preset.name}</span>`;
            btn.title = preset.description;

            btn.addEventListener("click", () => {
                document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                setFormValues(preset.values);
                runSinglePrediction();
            });

            container.appendChild(btn);
        });
    } catch (e) {
        console.error("Error loading presets:", e);
    }
}

// =================================================================================================
// 5. SINGLE OBSERVATION PREDICTION
// =================================================================================================

function initPredictionForm() {
    const form = document.getElementById("prediction-form");
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        runSinglePrediction();
    });
}

async function runSinglePrediction() {
    const btn = document.getElementById("btn-predict");
    const payload = getFormValues();

    try {
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-1"></i> Analyzing...`;

        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Inference error");
        }

        const data = await res.json();
        updateResultUI(data);

    } catch (err) {
        alert("Prediction Failed: " + err.message);
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-bolt mr-1"></i> Run AI Anomaly Inference`;
    }
}

function updateResultUI(data) {
    const card = document.getElementById("result-card");
    const iconWrapper = document.getElementById("status-icon-wrapper");
    const icon = document.getElementById("status-icon");
    const title = document.getElementById("status-title");
    const subtitle = document.getElementById("status-subtitle");
    const scoreVal = document.getElementById("result-score");
    const codeVal = document.getElementById("result-code");
    const scoreBar = document.getElementById("score-bar");
    const diagnosis = document.getElementById("result-diagnosis");

    const isAnomaly = data.is_anomaly;
    const score = data.anomaly_score;
    const scoreFormatted = (score >= 0 ? "+" : "") + score.toFixed(4);

    scoreVal.textContent = scoreFormatted;
    codeVal.textContent = `Code: ${data.anomaly_code}`;
    diagnosis.textContent = data.engineering_diagnosis;

    // Normalize score [-0.5, +0.5] to [0%, 100%]
    let pct = ((score + 0.5) / 1.0) * 100;
    pct = Math.max(5, Math.min(95, pct));
    scoreBar.style.width = pct + "%";

    if (isAnomaly) {
        card.className = "bg-white rounded-2xl p-6 shadow-sm border status-anomaly-card transition-all duration-300";
        iconWrapper.className = "w-20 h-20 rounded-full mx-auto flex items-center justify-center text-3xl mb-3 bg-rose-100 text-rose-600 shadow-inner animate-bounce";
        icon.className = "fa-solid fa-triangle-exclamation";
        title.textContent = "ANOMALY DETECTED";
        title.className = "text-2xl font-extrabold text-rose-600 tracking-tight";
        subtitle.textContent = "Multi-sensor pattern deviates from nominal generation manifold";
        scoreBar.className = "bg-rose-500 h-full rounded-full transition-all duration-500";
    } else {
        card.className = "bg-white rounded-2xl p-6 shadow-sm border status-normal-card transition-all duration-300";
        iconWrapper.className = "w-20 h-20 rounded-full mx-auto flex items-center justify-center text-3xl mb-3 bg-emerald-100 text-emerald-600 shadow-inner";
        icon.className = "fa-solid fa-check";
        title.textContent = "NOMINAL GENERATION";
        title.className = "text-2xl font-extrabold text-emerald-600 tracking-tight";
        subtitle.textContent = "Telemetry adheres to nominal power conversion curve";
        scoreBar.className = "bg-emerald-500 h-full rounded-full transition-all duration-500";
    }
}

// =================================================================================================
// 6. VISUAL ANALYTICS CHARTS (CHART.JS)
// =================================================================================================

async function loadAndRenderCharts() {
    try {
        const res = await fetch("/api/charts/data");
        if (!res.ok) return;
        const data = await res.json();

        // 1. DC vs AC Scatter
        const ctxDcAc = document.getElementById("chart-dc-ac").getContext("2d");
        chartDcAc = new Chart(ctxDcAc, {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: "Normal",
                        data: data.dc_vs_ac.normal,
                        backgroundColor: "rgba(59, 130, 246, 0.4)",
                        pointRadius: 2.5
                    },
                    {
                        label: "Anomaly",
                        data: data.dc_vs_ac.anomaly,
                        backgroundColor: "rgba(239, 68, 68, 0.8)",
                        pointRadius: 4.5,
                        pointStyle: "crossRot"
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: "DC Power (kW)", font: { size: 10 } } },
                    y: { title: { display: true, text: "AC Power (kW)", font: { size: 10 } } }
                },
                plugins: { legend: { position: "top", labels: { boxWidth: 10, font: { size: 10 } } } }
            }
        });

        // 2. Irradiation vs AC Scatter
        const ctxIrrAc = document.getElementById("chart-irr-ac").getContext("2d");
        chartIrrAc = new Chart(ctxIrrAc, {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: "Normal",
                        data: data.irr_vs_ac.normal,
                        backgroundColor: "rgba(16, 185, 129, 0.4)",
                        pointRadius: 2.5
                    },
                    {
                        label: "Anomaly",
                        data: data.irr_vs_ac.anomaly,
                        backgroundColor: "rgba(239, 68, 68, 0.85)",
                        pointRadius: 4.5,
                        pointStyle: "crossRot"
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: "Solar Irradiation", font: { size: 10 } } },
                    y: { title: { display: true, text: "AC Power (kW)", font: { size: 10 } } }
                },
                plugins: { legend: { position: "top", labels: { boxWidth: 10, font: { size: 10 } } } }
            }
        });

        // 3. Proportion Doughnut
        const ctxPie = document.getElementById("chart-pie").getContext("2d");
        chartPie = new Chart(ctxPie, {
            type: "doughnut",
            data: {
                labels: ["Normal Observations (95.00%)", "Flagged Anomalies (5.00%)"],
                datasets: [{
                    data: [65335, 3439],
                    backgroundColor: ["#10b981", "#ef4444"],
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "bottom", labels: { font: { size: 11 } } }
                }
            }
        });

        // 4. Time Series Curve
        const ctxTs = document.getElementById("chart-timeseries").getContext("2d");
        chartTimeSeries = new Chart(ctxTs, {
            type: "line",
            data: {
                labels: data.timeseries.labels.map(l => l.split(" ")[1] || l),
                datasets: [
                    {
                        label: "AC Power (kW)",
                        data: data.timeseries.ac_power,
                        borderColor: "rgba(59, 130, 246, 0.8)",
                        borderWidth: 1.5,
                        fill: false,
                        pointRadius: 0
                    },
                    {
                        label: "Anomaly Flagged",
                        data: data.timeseries.anomalies,
                        borderColor: "transparent",
                        backgroundColor: "rgba(239, 68, 68, 0.9)",
                        pointRadius: 4,
                        showLine: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: "Timestamp (15-min)", font: { size: 10 } }, ticks: { maxTicksLimit: 12 } },
                    y: { title: { display: true, text: "AC Power (kW)", font: { size: 10 } } }
                },
                plugins: { legend: { position: "top", labels: { boxWidth: 10, font: { size: 10 } } } }
            }
        });

    } catch (err) {
        console.error("Error loading chart data:", err);
    }
}

// =================================================================================================
// 7. BATCH CSV PROCESSOR
// =================================================================================================

function initBatchProcessor() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("batch-file-input");
    const filenameEl = document.getElementById("selected-filename");
    const btnProcess = document.getElementById("btn-batch-process");
    const btnDownload = document.getElementById("btn-batch-download");

    let selectedFile = null;

    dropzone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("border-blue-500", "bg-blue-50/50");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("border-blue-500", "bg-blue-50/50");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("border-blue-500", "bg-blue-50/50");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        filenameEl.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        filenameEl.classList.remove("hidden");
        btnProcess.disabled = false;
    }

    btnProcess.addEventListener("click", async () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            btnProcess.disabled = true;
            btnProcess.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-1.5"></i> Scoring...`;

            const res = await fetch("/api/predict/batch", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || "Batch processing error");
            }

            const data = await res.json();
            batchScoredData = data.preview;

            // Update KPI counters
            document.getElementById("batch-total").textContent = data.summary.total_samples.toLocaleString();
            document.getElementById("batch-normal").textContent = data.summary.normal_count.toLocaleString();
            document.getElementById("batch-anom").textContent = `${data.summary.anomaly_count.toLocaleString()} (${data.summary.anomaly_percentage}%)`;

            // Render table
            const tbody = document.getElementById("batch-table-body");
            tbody.innerHTML = "";
            data.preview.forEach((row, i) => {
                const tr = document.createElement("tr");
                const isAnom = row.anomaly_code === -1;
                tr.className = isAnom ? "bg-rose-50/50" : "hover:bg-slate-50";
                tr.innerHTML = `
                    <td class="px-4 py-2 font-mono">${i + 1}</td>
                    <td class="px-4 py-2 font-mono">${Number(row.DC_POWER || 0).toFixed(1)}</td>
                    <td class="px-4 py-2 font-mono">${Number(row.AC_POWER || 0).toFixed(1)}</td>
                    <td class="px-4 py-2 font-mono">${Number(row.IRRADIATION || 0).toFixed(3)}</td>
                    <td class="px-4 py-2 font-bold ${isAnom ? 'text-rose-600' : 'text-emerald-600'}">
                        ${isAnom ? '<i class="fa-solid fa-triangle-exclamation mr-1"></i> Anomaly' : '<i class="fa-solid fa-check mr-1"></i> Normal'}
                    </td>
                    <td class="px-4 py-2 font-mono font-semibold">${row.anomaly_score}</td>
                `;
                tbody.appendChild(tr);
            });

            document.getElementById("batch-results").classList.remove("hidden");
            btnDownload.classList.remove("hidden");

        } catch (err) {
            alert("Error in batch processing: " + err.message);
            console.error(err);
        } finally {
            btnProcess.disabled = false;
            btnProcess.innerHTML = `<i class="fa-solid fa-play mr-1.5"></i> Score Batch Data`;
        }
    });

    btnDownload.addEventListener("click", () => {
        if (!batchScoredData || batchScoredData.length === 0) return;
        
        // Convert to CSV
        const headers = Object.keys(batchScoredData[0]);
        const csvRows = [headers.join(",")];
        for (const row of batchScoredData) {
            const values = headers.map(h => JSON.stringify(row[h] ?? ""));
            csvRows.push(values.join(","));
        }
        const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "scored_telemetry_predictions.csv";
        a.click();
        URL.revokeObjectURL(url);
    });
}
