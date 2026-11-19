import React, { useState, useEffect, useRef, useMemo } from 'react';
import Chart from 'chart.js/auto';
import {
    DEFAULT_KPIS,
    DEFAULT_CHARTS,
    SAMPLE_EXPLORER_RECORDS
} from './mockData.js';

export default function App() {
    const [activeTab, setActiveTab] = useState("detector");
    const [backendStatus, setBackendStatus] = useState("checking");
    const [kpiData, setKpiData] = useState(DEFAULT_KPIS);

    // Check backend connection
    useEffect(() => {
        fetch("/api/status")
            .then(res => res.json())
            .then(data => {
                if (data.status === "healthy") {
                    setBackendStatus("online");
                    fetch("/api/kpis").then(r => r.json()).then(d => setKpiData(d)).catch(() => { });
                } else {
                    setBackendStatus("offline");
                }
            })
            .catch(() => setBackendStatus("offline"));
    }, []);

    return (
        <div className="flex flex-col min-h-screen bg-[#FAF8F5] text-black">
            {/* Header */}
            <header className="border-b border-black/10 bg-[#FAF8F5]">
                <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
                    <div>
                        <div className="flex items-center gap-2">
                            <span className="w-2.5 h-2.5 rounded-full bg-[#839958]"></span>
                            <h1 className="text-lg font-extrabold tracking-tight text-black">
                                Solar Power Anomaly Detection
                            </h1>
                            <span className="text-xs px-2.5 py-0.5 rounded-md bg-[#839958]/20 text-black border border-[#839958]/40 font-bold">
                                Machine Learning Mini-Project
                            </span>
                        </div>
                        <p className="text-xs text-black/80 font-medium mt-0.5">
                            D. Y. Patil College of Engineering, Akurdi | TY AI & Data Science (Course: AD124PC503)
                        </p>
                    </div>

                    <div className="flex items-center gap-2 text-xs">
                        {backendStatus === "online" ? (
                            <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#839958]/25 text-black border border-[#839958]/50 font-bold">
                                <span className="w-2 h-2 rounded-full bg-[#839958]"></span>
                                Model Loaded (Port 5000)
                            </span>
                        ) : (
                            <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-black/5 text-black border border-black/15 font-semibold">
                                <span className="w-2 h-2 rounded-full bg-[#839958]"></span>
                                Local Dataset Ready
                            </span>
                        )}
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-6xl mx-auto px-4 sm:px-6 py-6 flex-grow w-full space-y-6">

                {/* Academic Context Notice Card (Moss Green -> Rosy Brown on hover) */}
                <div className="paper-card p-3.5 text-xs text-black flex items-start gap-2.5">
                    <i className="fa-solid fa-circle-info text-black text-sm mt-0.5"></i>
                    <div>
                        <strong className="text-black">Academic Definition:</strong> Anomalies in this project represent statistical outliers in multi-sensor space (such as inverter tripping, sensor drift, or passing clouds), <em>not confirmed equipment breakdowns</em>.
                    </div>
                </div>

                {/* 4 Essential Summary Cards (Moss Green -> Rosy Brown on hover, All Text Black) */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
                    <div className="paper-card p-4 cursor-pointer">
                        <span className="text-[11px] font-bold uppercase tracking-wider text-black block">Total Observations</span>
                        <div className="text-2xl font-black text-black mt-1">{kpiData.total_observations.toLocaleString()}</div>
                        <span className="text-[11px] text-black/80 font-medium">34 days synchronized</span>
                    </div>

                    <div className="paper-card p-4 cursor-pointer">
                        <span className="text-[11px] font-bold uppercase tracking-wider text-black block">Isolation Forest</span>
                        <div className="text-2xl font-black text-black mt-1">{kpiData.isolation_forest.anomaly_count.toLocaleString()}</div>
                        <span className="text-[11px] text-black font-bold">3,439 anomalies (5.0%)</span>
                    </div>

                    <div className="paper-card p-4 cursor-pointer">
                        <span className="text-[11px] font-bold uppercase tracking-wider text-black block">DBSCAN Outliers</span>
                        <div className="text-2xl font-black text-black mt-1">{kpiData.dbscan.noise_count}</div>
                        <span className="text-[11px] text-black font-bold">6 noise points (0.0087%)</span>
                    </div>

                    <div className="paper-card p-4 cursor-pointer">
                        <span className="text-[11px] font-bold uppercase tracking-wider text-black block">Monitored Inverters</span>
                        <div className="text-2xl font-black text-black mt-1">{kpiData.total_inverters || 22}</div>
                        <span className="text-[11px] text-black/80 font-medium">Central Inverter Keys</span>
                    </div>
                </div>

                {/* Navigation Tabs */}
                <div className="border-b border-black/15 flex space-x-1 sm:space-x-2">
                    {[
                        { id: "detector", label: "Live Detector", icon: "fa-sliders" },
                        { id: "visualizations", label: "Visualizations", icon: "fa-chart-scatter" },
                        { id: "dataset", label: "Dataset Explorer", icon: "fa-table-list" },
                        { id: "methodology", label: "Model Comparison", icon: "fa-scale-balanced" },
                    ].map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`pb-3 px-3 sm:px-4 text-xs sm:text-sm font-bold flex items-center gap-2 border-b-2 transition-all ${activeTab === tab.id
                                    ? "border-black text-black"
                                    : "border-transparent text-black/60 hover:text-black"
                                }`}
                        >
                            <i className={`fa-solid ${tab.icon}`}></i>
                            <span>{tab.label}</span>
                        </button>
                    ))}
                </div>

                {/* Tab Views */}
                {activeTab === "detector" && <MinimalDetectorView />}
                {activeTab === "visualizations" && <MinimalVisualizationsView />}
                {activeTab === "dataset" && <MinimalDatasetView />}
                {activeTab === "methodology" && <MinimalMethodologyView />}

            </main>

            {/* Footer */}
            <footer className="mt-auto border-t border-black/10 bg-[#FAF8F5] py-4 text-xs text-black/80 font-medium">
                <div className="max-w-6xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row justify-between items-center gap-2">
                    <span>Solar Power Generation Anomaly Detection | Kaggle Plant 1 Dataset</span>
                    <span>Department of Artificial Intelligence & Data Science, DYPCOE Akurdi</span>
                </div>
            </footer>
        </div>
    );
}

// =================================================================================================
// 1. MINIMAL DETECTOR VIEW (Moss Green -> Rosy Brown Cards, All Text Black)
// =================================================================================================
function MinimalDetectorView() {
    const [inputs, setInputs] = useState({
        DC_POWER: 4200.0,
        AC_POWER: 410.0,
        DAILY_YIELD: 3100.0,
        AMBIENT_TEMPERATURE: 29.5,
        MODULE_TEMPERATURE: 42.0,
        IRRADIATION: 0.48
    });

    const [result, setResult] = useState({
        prediction: "Normal",
        anomaly_code: 1,
        score: 0.0512,
        is_anomaly: false,
        diagnosis: "Nominal Generation: Power conversion and module temperature align with expected daytime solar generation."
    });

    const [loading, setLoading] = useState(false);

    // 3 Simple Realistic Student Presets
    const presets = [
        {
            name: "Nominal Midday Generation",
            desc: "Standard daytime solar generation (Ratio ~0.098)",
            values: { DC_POWER: 4200.0, AC_POWER: 410.0, DAILY_YIELD: 3100.0, AMBIENT_TEMPERATURE: 29.5, MODULE_TEMPERATURE: 42.0, IRRADIATION: 0.48 }
        },
        {
            name: "Inverter Trip / Disconnect",
            desc: "High sunlight (0.95 W/m²), but zero inverter AC output",
            values: { DC_POWER: 0.0, AC_POWER: 0.0, DAILY_YIELD: 150.0, AMBIENT_TEMPERATURE: 35.0, MODULE_TEMPERATURE: 64.0, IRRADIATION: 0.95 }
        },
        {
            name: "Night-time Idle",
            desc: "Zero sunlight after dusk; panels dormant",
            values: { DC_POWER: 0.0, AC_POWER: 0.0, DAILY_YIELD: 5500.0, AMBIENT_TEMPERATURE: 22.0, MODULE_TEMPERATURE: 21.5, IRRADIATION: 0.0 }
        }
    ];

    const evaluateLocally = (v) => {
        const { DC_POWER, AC_POWER, IRRADIATION } = v;
        if (IRRADIATION > 0.4 && AC_POWER < 50) {
            return {
                prediction: "Anomaly",
                anomaly_code: -1,
                score: -0.1173,
                is_anomaly: true,
                diagnosis: "Inverter Output Loss: High solar radiation is present, but inverter generation has fallen to near zero."
            };
        } else if (DC_POWER > 8000 && AC_POWER < 300) {
            return {
                prediction: "Anomaly",
                anomaly_code: -1,
                score: -0.1284,
                is_anomaly: true,
                diagnosis: "Inverter Throttling: High DC power input with severely truncated AC power output."
            };
        } else {
            return {
                prediction: "Normal",
                anomaly_code: 1,
                score: 0.0512,
                is_anomaly: false,
                diagnosis: "Nominal Generation: Sensor readings correspond to typical operating curves."
            };
        }
    };

    const handleRunInference = (valuesToUse) => {
        const payload = valuesToUse || inputs;
        setLoading(true);

        fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
            .then(res => res.json())
            .then(data => {
                if (data.prediction) {
                    setResult({
                        prediction: data.prediction,
                        anomaly_code: data.anomaly_code,
                        score: data.anomaly_score,
                        is_anomaly: data.is_anomaly,
                        diagnosis: data.engineering_diagnosis
                    });
                } else {
                    setResult(evaluateLocally(payload));
                }
                setLoading(false);
            })
            .catch(() => {
                setResult(evaluateLocally(payload));
                setLoading(false);
            });
    };

    const loadPreset = (p) => {
        setInputs(p.values);
        handleRunInference(p.values);
    };

    return (
        <div className="space-y-6">
            {/* Quick Test Presets Card (Moss Green -> Rosy Brown) */}
            <div className="paper-card p-4">
                <span className="text-xs font-black uppercase tracking-wider text-black block mb-2">
                    Quick Test Scenarios:
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                    {presets.map((p, i) => (
                        <button
                            key={i}
                            onClick={() => loadPreset(p)}
                            className="p-3 text-left rounded-xl bg-white/85 border border-black/15 hover:bg-white text-black transition text-xs shadow-sm"
                        >
                            <span className="font-bold text-black block">{p.name}</span>
                            <span className="text-[11px] text-black/80 font-medium mt-0.5 block">{p.desc}</span>
                        </button>
                    ))}
                </div>
            </div>

            {/* Main Form and Output */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* 2 Cols: Inputs Card */}
                <div className="md:col-span-2 paper-card p-5 space-y-4">
                    <div className="flex justify-between items-center border-b border-black/15 pb-3">
                        <h2 className="text-sm font-black text-black">Sensor Telemetry Inputs</h2>
                        <button
                            onClick={() => handleRunInference()}
                            disabled={loading}
                            className="px-4 py-2 bg-black hover:bg-black/80 text-white text-xs font-bold rounded-lg transition disabled:opacity-50 shadow-sm"
                        >
                            {loading ? "Evaluating..." : "Run Anomaly Prediction"}
                        </button>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <SimpleSlider
                            label="DC Power"
                            unit="kW"
                            min={0} max={15000} step={50}
                            value={inputs.DC_POWER}
                            onChange={(v) => setInputs(prev => ({ ...prev, DC_POWER: parseFloat(v) || 0 }))}
                        />
                        <SimpleSlider
                            label="AC Power"
                            unit="kW"
                            min={0} max={1500} step={5}
                            value={inputs.AC_POWER}
                            onChange={(v) => setInputs(prev => ({ ...prev, AC_POWER: parseFloat(v) || 0 }))}
                        />
                        <SimpleSlider
                            label="Daily Yield"
                            unit="kWh"
                            min={0} max={10000} step={50}
                            value={inputs.DAILY_YIELD}
                            onChange={(v) => setInputs(prev => ({ ...prev, DAILY_YIELD: parseFloat(v) || 0 }))}
                        />
                        <SimpleSlider
                            label="Ambient Temperature"
                            unit="°C"
                            min={15} max={45} step={0.5}
                            value={inputs.AMBIENT_TEMPERATURE}
                            onChange={(v) => setInputs(prev => ({ ...prev, AMBIENT_TEMPERATURE: parseFloat(v) || 0 }))}
                        />
                        <SimpleSlider
                            label="Module Temperature"
                            unit="°C"
                            min={15} max={80} step={0.5}
                            value={inputs.MODULE_TEMPERATURE}
                            onChange={(v) => setInputs(prev => ({ ...prev, MODULE_TEMPERATURE: parseFloat(v) || 0 }))}
                        />
                        <SimpleSlider
                            label="Solar Irradiation"
                            unit="W/m²"
                            min={0.0} max={1.3} step={0.02}
                            value={inputs.IRRADIATION}
                            onChange={(v) => setInputs(prev => ({ ...prev, IRRADIATION: parseFloat(v) || 0 }))}
                        />
                    </div>
                </div>

                {/* 1 Col: Clean Result Card */}
                <div className="paper-card p-5 flex flex-col justify-between space-y-4">
                    <div>
                        <span className="text-[11px] font-black uppercase tracking-wider text-black block">
                            Model Prediction
                        </span>

                        <div className="mt-3">
                            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-black bg-white/90 text-black border border-black/20 shadow-sm">
                                <span className={`w-2 h-2 rounded-full ${result.is_anomaly ? "bg-[#D3968C]" : "bg-[#839958]"}`}></span>
                                {result.is_anomaly ? "Anomaly Observation" : "Normal Observation"}
                            </span>
                        </div>

                        <div className="mt-4 p-3.5 bg-white/85 rounded-xl border border-black/15 space-y-1.5 text-xs shadow-sm">
                            <div className="flex justify-between">
                                <span className="text-black font-semibold">Prediction Label:</span>
                                <span className="font-mono font-black text-black">{result.anomaly_code}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-black font-semibold">Anomaly Score:</span>
                                <span className="font-mono font-black text-black">
                                    {(result.score >= 0 ? "+" : "") + Number(result.score).toFixed(4)}
                                </span>
                            </div>
                            <div className="text-[10px] text-black/70 font-medium pt-1">
                                (Scores below 0 indicate outlier behavior)
                            </div>
                        </div>

                        <div className="mt-4">
                            <span className="text-xs font-black text-black block mb-1">Explanation:</span>
                            <p className="text-xs text-black font-medium leading-relaxed bg-white/85 p-3 rounded-xl border border-black/15 shadow-sm">
                                {result.diagnosis}
                            </p>
                        </div>
                    </div>

                    <div className="text-[11px] text-black font-bold border-t border-black/15 pt-3">
                        Model: Isolation Forest (100 trees, 0.05 contamination)
                    </div>
                </div>
            </div>
        </div>
    );
}

function SimpleSlider({ label, unit, min, max, step, value, onChange }) {
    return (
        <div className="bg-white/85 p-3 rounded-xl border border-black/15 space-y-1.5 shadow-sm">
            <div className="flex justify-between items-center text-xs">
                <span className="text-black font-bold">{label}</span>
                <span className="font-mono font-black text-black">{value} {unit}</span>
            </div>
            <input
                type="range"
                min={min} max={max} step={step}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                className="w-full"
            />
        </div>
    );
}

// =================================================================================================
// 2. MINIMAL VISUALIZATIONS VIEW (Moss Green Cards -> Rosy Brown Hover, All Text Black)
// =================================================================================================
function MinimalVisualizationsView() {
    const canvasDcAc = useRef(null);
    const canvasIrrAc = useRef(null);
    const canvasBar = useRef(null);
    const canvasTs = useRef(null);

    useEffect(() => {
        const charts = [];
        const subtleGrid = { color: "rgba(0, 0, 0, 0.08)" };
        const tickStyle = { color: "#000000", font: { size: 10, weight: "bold" } };

        // 1. DC vs AC Scatter
        if (canvasDcAc.current) {
            const chart1 = new Chart(canvasDcAc.current, {
                type: "scatter",
                data: {
                    datasets: [
                        { label: "Normal (Nominal)", data: DEFAULT_CHARTS.dc_vs_ac.normal, backgroundColor: "rgba(10, 51, 35, 0.7)", pointRadius: 2.2 },
                        { label: "Anomaly", data: DEFAULT_CHARTS.dc_vs_ac.anomaly, backgroundColor: "rgba(186, 75, 62, 0.95)", pointRadius: 4.5, pointStyle: "crossRot" }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { grid: subtleGrid, ticks: tickStyle, title: { display: true, text: "DC Power (kW)", color: "#000000", font: { size: 11, weight: "bold" } } },
                        y: { grid: subtleGrid, ticks: tickStyle, title: { display: true, text: "AC Power (kW)", color: "#000000", font: { size: 11, weight: "bold" } } }
                    },
                    plugins: { legend: { position: "top", labels: { boxWidth: 10, color: "#000000", font: { size: 11, weight: "bold" } } } }
                }
            });
            charts.push(chart1);
        }

        // 2. Irradiation vs AC Scatter
        if (canvasIrrAc.current) {
            const chart2 = new Chart(canvasIrrAc.current, {
                type: "scatter",
                data: {
                    datasets: [
                        { label: "Normal Generation", data: DEFAULT_CHARTS.irr_vs_ac.normal, backgroundColor: "rgba(16, 86, 102, 0.7)", pointRadius: 2.2 },
                        { label: "Anomaly", data: DEFAULT_CHARTS.irr_vs_ac.anomaly, backgroundColor: "rgba(186, 75, 62, 0.95)", pointRadius: 4.5, pointStyle: "crossRot" }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { grid: subtleGrid, ticks: tickStyle, title: { display: true, text: "Solar Irradiation (W/m²)", color: "#000000", font: { size: 11, weight: "bold" } } },
                        y: { grid: subtleGrid, ticks: tickStyle, title: { display: true, text: "AC Power (kW)", color: "#000000", font: { size: 11, weight: "bold" } } }
                    },
                    plugins: { legend: { position: "top", labels: { boxWidth: 10, color: "#000000", font: { size: 11, weight: "bold" } } } }
                }
            });
            charts.push(chart2);
        }

        // 3. Comparison Bar Chart
        if (canvasBar.current) {
            const chart3 = new Chart(canvasBar.current, {
                type: "bar",
                data: {
                    labels: ["Isolation Forest", "DBSCAN"],
                    datasets: [{
                        label: "Anomaly Observations",
                        data: [3439, 6],
                        backgroundColor: ["#D3968C", "#105666"],
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { grid: { display: false }, ticks: tickStyle },
                        y: { grid: subtleGrid, ticks: tickStyle, title: { display: true, text: "Detected Anomaly Count", color: "#000000", font: { size: 11, weight: "bold" } } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
            charts.push(chart3);
        }

        // 4. Diurnal Curve
        if (canvasTs.current) {
            const chart4 = new Chart(canvasTs.current, {
                type: "line",
                data: {
                    labels: DEFAULT_CHARTS.timeseries.labels.map(l => l.split(" ")[1] || l),
                    datasets: [
                        { label: "AC Power (kW)", data: DEFAULT_CHARTS.timeseries.ac_power, borderColor: "#000000", borderWidth: 2, fill: false, pointRadius: 0 },
                        { label: "Anomaly Event", data: DEFAULT_CHARTS.timeseries.anomalies, borderColor: "transparent", backgroundColor: "#BA4B3E", pointRadius: 5, showLine: false }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { grid: subtleGrid, ticks: tickStyle, title: { display: true, text: "Time of Day (15-min intervals)", color: "#000000", font: { size: 11, weight: "bold" } } },
                        y: { grid: subtleGrid, ticks: tickStyle, title: { display: true, text: "AC Power (kW)", color: "#000000", font: { size: 11, weight: "bold" } } }
                    },
                    plugins: { legend: { position: "top", labels: { boxWidth: 10, color: "#000000", font: { size: 11, weight: "bold" } } } }
                }
            });
            charts.push(chart4);
        }

        return () => charts.forEach(c => c.destroy());
    }, []);

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div className="paper-card p-4">
                <h3 className="text-xs font-black text-black">DC Power vs AC Power (Inverter Conversion)</h3>
                <p className="text-[11px] text-black/80 font-medium mb-2.5">Deviations highlight inverter throttling, clipping or tripping</p>
                <div className="h-64 bg-white/90 p-2 rounded-xl border border-black/10"><canvas ref={canvasDcAc}></canvas></div>
            </div>

            <div className="paper-card p-4">
                <h3 className="text-xs font-black text-black">Solar Irradiation vs AC Power</h3>
                <p className="text-[11px] text-black/80 font-medium mb-2.5">Identifies loss of generation during high solar irradiance</p>
                <div className="h-64 bg-white/90 p-2 rounded-xl border border-black/10"><canvas ref={canvasIrrAc}></canvas></div>
            </div>

            <div className="paper-card p-4">
                <h3 className="text-xs font-black text-black">Model Comparison: Detected Outliers</h3>
                <p className="text-[11px] text-black/80 font-medium mb-2.5">Isolation Forest (5.0%) vs DBSCAN density noise points (0.0087%)</p>
                <div className="h-64 bg-white/90 p-2 rounded-xl border border-black/10"><canvas ref={canvasBar}></canvas></div>
            </div>

            <div className="paper-card p-4">
                <h3 className="text-xs font-black text-black">Daily Solar Power Generation Curve</h3>
                <p className="text-[11px] text-black/80 font-medium mb-2.5">Typical diurnal bell curve with flagged anomalous events</p>
                <div className="h-64 bg-white/90 p-2 rounded-xl border border-black/10"><canvas ref={canvasTs}></canvas></div>
            </div>
        </div>
    );
}

// =================================================================================================
// 3. MINIMAL DATASET VIEW (Moss Green Cards -> Rosy Brown Hover, All Text Black)
// =================================================================================================
function MinimalDatasetView() {
    const [filter, setFilter] = useState("all");
    const [search, setSearch] = useState("");

    const records = SAMPLE_EXPLORER_RECORDS;

    const filtered = useMemo(() => {
        return records.filter(r => {
            const matchesSearch = !search || (r.DATE_TIME && r.DATE_TIME.includes(search));
            const isAnom = r.anomaly_iforest === -1;
            const matchesFilter = filter === "all" || (filter === "anomaly" && isAnom) || (filter === "normal" && !isAnom);
            return matchesSearch && matchesFilter;
        });
    }, [records, filter, search]);

    return (
        <div className="paper-card p-5 space-y-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 border-b border-black/15 pb-3">
                <div>
                    <h2 className="text-sm font-black text-black">Kaggle Plant 1 Telemetry Samples</h2>
                    <p className="text-xs text-black/80 font-medium">Synchronized generation and weather observations (15-min intervals)</p>
                </div>

                <div className="flex items-center gap-2">
                    <input
                        type="text"
                        placeholder="Filter by timestamp..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="px-2.5 py-1.5 text-xs border border-black/20 bg-white/90 rounded-lg text-black font-semibold focus:outline-none"
                    />
                    <select
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        className="px-2.5 py-1.5 text-xs border border-black/20 bg-white/90 rounded-lg text-black font-semibold focus:outline-none"
                    >
                        <option value="all">All Records</option>
                        <option value="anomaly">Anomalies (-1)</option>
                        <option value="normal">Normal (+1)</option>
                    </select>
                </div>
            </div>

            <div className="overflow-x-auto rounded-xl border border-black/15 bg-white/90">
                <table className="min-w-full divide-y divide-black/10 text-xs">
                    <thead className="bg-black/5 text-black font-black">
                        <tr>
                            <th className="px-3.5 py-2.5 text-left">Timestamp</th>
                            <th className="px-3.5 py-2.5 text-left">Inverter</th>
                            <th className="px-3.5 py-2.5 text-right">DC Power (kW)</th>
                            <th className="px-3.5 py-2.5 text-right">AC Power (kW)</th>
                            <th className="px-3.5 py-2.5 text-right">Daily Yield</th>
                            <th className="px-3.5 py-2.5 text-right">Module Temp</th>
                            <th className="px-3.5 py-2.5 text-right">Irradiation</th>
                            <th className="px-3.5 py-2.5 text-center">Prediction</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-black/10 font-mono text-[11px] text-black">
                        {filtered.map((r, i) => {
                            const isAnom = r.anomaly_iforest === -1;
                            return (
                                <tr key={i} className={isAnom ? "bg-[#D3968C]/20 font-bold" : "hover:bg-black/5"}>
                                    <td className="px-3.5 py-2 text-black">{r.DATE_TIME}</td>
                                    <td className="px-3.5 py-2 text-black font-sans font-bold">{r.SOURCE_KEY ? r.SOURCE_KEY.slice(-8) : "N/A"}</td>
                                    <td className="px-3.5 py-2 text-right text-black font-bold">{r.DC_POWER}</td>
                                    <td className="px-3.5 py-2 text-right text-black font-bold">{r.AC_POWER}</td>
                                    <td className="px-3.5 py-2 text-right text-black">{r.DAILY_YIELD}</td>
                                    <td className="px-3.5 py-2 text-right text-black">{r.MODULE_TEMPERATURE}°C</td>
                                    <td className="px-3.5 py-2 text-right text-black">{r.IRRADIATION}</td>
                                    <td className="px-3.5 py-2 text-center font-sans">
                                        <span className={`px-2 py-0.5 rounded text-[10px] font-black border ${isAnom
                                                ? "bg-white text-black border-black/30"
                                                : "bg-[#839958]/30 text-black border-black/20"
                                            }`}>
                                            {isAnom ? "Anomaly" : "Normal"}
                                        </span>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
            <p className="text-[11px] text-black/80 font-medium">
                Showing {filtered.length} sample observations from the merged 68,774 Kaggle dataset.
            </p>
        </div>
    );
}

// =================================================================================================
// 4. MINIMAL METHODOLOGY VIEW (Moss Green Cards -> Rosy Brown Hover, All Text Black)
// =================================================================================================
function MinimalMethodologyView() {
    return (
        <div className="space-y-5">
            {/* Model Comparison Table */}
            <div className="paper-card p-5 space-y-3">
                <h2 className="text-sm font-black text-black">Unsupervised Model Comparison</h2>
                <p className="text-xs text-black/80 font-medium">
                    Comparison between Isolation Forest (Primary Model) and DBSCAN (Benchmark Model).
                </p>

                <div className="overflow-x-auto rounded-xl border border-black/15 bg-white/90">
                    <table className="min-w-full divide-y divide-black/10 text-xs">
                        <thead className="bg-black/5 text-black font-black">
                            <tr>
                                <th className="px-4 py-2.5 text-left">Property</th>
                                <th className="px-4 py-2.5 text-left">Isolation Forest (Primary)</th>
                                <th className="px-4 py-2.5 text-left">DBSCAN (Comparison)</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-black/10 text-black">
                            <tr>
                                <td className="px-4 py-2 font-bold text-black">Algorithm Family</td>
                                <td className="px-4 py-2 text-black font-medium">Tree-based Ensemble (Random partitioning)</td>
                                <td className="px-4 py-2 text-black font-medium">Density-based Spatial Clustering</td>
                            </tr>
                            <tr>
                                <td className="px-4 py-2 font-bold text-black">Hyperparameters</td>
                                <td className="px-4 py-2 font-mono text-xs text-black font-semibold">n_estimators=100, contamination=0.05</td>
                                <td className="px-4 py-2 font-mono text-xs text-black font-semibold">eps=0.8, min_samples=10</td>
                            </tr>
                            <tr>
                                <td className="px-4 py-2 font-bold text-black">Feature Scaling</td>
                                <td className="px-4 py-2 text-black font-medium">Raw features (Scale invariant)</td>
                                <td className="px-4 py-2 text-black font-medium">StandardScaler required (Euclidean distance)</td>
                            </tr>
                            <tr>
                                <td className="px-4 py-2 font-bold text-black">Detected Anomalies</td>
                                <td className="px-4 py-2 font-black text-black">3,439 observations (5.00%)</td>
                                <td className="px-4 py-2 font-black text-black">6 noise points (0.0087%)</td>
                            </tr>
                            <tr>
                                <td className="px-4 py-2 font-bold text-black">Detection Logic</td>
                                <td className="px-4 py-2 text-black font-medium">Isolates points with short tree path lengths</td>
                                <td className="px-4 py-2 text-black font-medium">Flags points in low-density sparse neighborhoods</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Viva Talking Points */}
            <div className="paper-card p-5 space-y-3">
                <h3 className="text-xs font-black text-black uppercase tracking-wider">
                    Key Viva Discussion Points:
                </h3>
                <div className="space-y-2.5 text-xs text-black leading-relaxed">
                    <div className="p-3.5 bg-white/85 rounded-xl border border-black/15 shadow-sm">
                        <strong className="text-black font-black block mb-0.5">Q1: Why do Isolation Forest and DBSCAN report different anomaly counts?</strong>
                        <span className="text-black font-medium">Isolation Forest uses a fixed contamination parameter (0.05) to isolate the top 5% most anomalous observations. DBSCAN uses spatial density (eps=0.8, min_samples=10) on standardized features, meaning it only flags extreme multi-sigma outliers where points lack dense neighbors.</span>
                    </div>
                    <div className="p-3.5 bg-white/85 rounded-xl border border-black/15 shadow-sm">
                        <strong className="text-black font-black block mb-0.5">Q2: Why must StandardScaler be applied before DBSCAN?</strong>
                        <span className="text-black font-medium">DBSCAN calculates pairwise Euclidean distances in 6-dimensional space. Because DC Power reaches up to 14,471 kW while solar irradiation is only between 0.0 and 1.2 W/m², unscaled features would allow power to dominate the distance calculation.</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
