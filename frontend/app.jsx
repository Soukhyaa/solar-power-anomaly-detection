const { useState, useEffect, useRef, useMemo } = React;

// Main App Component
function App() {
    const [activeTab, setActiveTab] = useState("simulator");
    const [backendStatus, setBackendStatus] = useState("checking");
    const [kpiData, setKpiData] = useState(null);
    const [inverters, setInverters] = useState([]);
    const [featureStats, setFeatureStats] = useState([]);
    const [presets, setPresets] = useState([]);

    useEffect(() => {
        // Fetch System Health
        fetch("/api/status")
            .then(res => res.json())
            .then(data => setBackendStatus(data.status === "healthy" ? "online" : "offline"))
            .catch(() => setBackendStatus("offline"));

        // Fetch KPIs
        fetch("/api/kpis")
            .then(res => res.json())
            .then(data => setKpiData(data))
            .catch(err => console.error("Error fetching KPIs:", err));

        // Fetch Inverters
        fetch("/api/inverters")
            .then(res => res.json())
            .then(data => setInverters(data))
            .catch(err => console.error("Error fetching inverters:", err));

        // Fetch Feature Statistics
        fetch("/api/feature-stats")
            .then(res => res.json())
            .then(data => setFeatureStats(data))
            .catch(err => console.error("Error fetching feature stats:", err));

        // Fetch Presets
        fetch("/api/presets")
            .then(res => res.json())
            .then(data => setPresets(data))
            .catch(err => console.error("Error fetching presets:", err));
    }, []);

    return (
        <div className="flex flex-col min-h-screen">
            <Navbar status={backendStatus} />

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-grow w-full space-y-8">
                <HeroHeader kpiData={kpiData} />
                <KPIGrid kpiData={kpiData} invertersCount={inverters.length} />

                {/* Tab Navigation */}
                <div className="border-b border-slate-200">
                    <nav className="flex space-x-2 sm:space-x-6 overflow-x-auto pb-1" aria-label="Tabs">
                        {[
                            { id: "simulator", label: "Real-Time AI Simulator", icon: "fa-sliders" },
                            { id: "analytics", label: "Visual Analytics", icon: "fa-chart-line" },
                            { id: "inverters", label: "Inverter Matrix (22)", icon: "fa-solar-panel" },
                            { id: "explorer", label: "Telemetry Explorer", icon: "fa-table" },
                            { id: "stats", label: "Feature Statistics", icon: "fa-chart-simple" },
                            { id: "batch", label: "Batch CSV Scanner", icon: "fa-file-csv" },
                            { id: "architecture", label: "Architecture & Viva", icon: "fa-graduation-cap" },
                        ].map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`py-3 px-2 sm:px-3 border-b-2 font-medium text-xs sm:text-sm whitespace-nowrap flex items-center gap-2 transition ${
                                    activeTab === tab.id
                                        ? "text-blue-600 border-blue-600 font-bold"
                                        : "text-slate-500 border-transparent hover:text-slate-700 hover:border-slate-300"
                                }`}
                            >
                                <i className={`fa-solid ${tab.icon}`}></i>
                                <span>{tab.label}</span>
                            </button>
                        ))}
                    </nav>
                </div>

                {/* Tab Content Views */}
                {activeTab === "simulator" && <SimulatorView presets={presets} />}
                {activeTab === "analytics" && <AnalyticsView />}
                {activeTab === "inverters" && <InverterMatrixView inverters={inverters} />}
                {activeTab === "explorer" && <TelemetryExplorerView inverters={inverters} />}
                {activeTab === "stats" && <FeatureStatsView stats={featureStats} />}
                {activeTab === "batch" && <BatchScannerView />}
                {activeTab === "architecture" && <ArchitectureVivaView />}
            </main>

            <Footer />
        </div>
    );
}

// -------------------------------------------------------------------------------------------------
// NAVBAR COMPONENT
// -------------------------------------------------------------------------------------------------
function Navbar({ status }) {
    return (
        <nav className="bg-gradient-to-r from-slate-900 via-indigo-950 to-blue-900 text-white shadow-lg sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-16">
                    <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-xl bg-amber-400 flex items-center justify-center text-slate-900 shadow-md">
                            <i className="fa-solid fa-solar-panel text-xl"></i>
                        </div>
                        <div>
                            <span className="text-xl font-bold tracking-tight">Solar AI Sentinel</span>
                            <span className="hidden md:inline-block ml-2 px-2.5 py-0.5 text-xs font-semibold bg-blue-500/30 text-blue-200 rounded-full border border-blue-400/30">
                                React 18 Edition
                            </span>
                        </div>
                    </div>

                    <div className="flex items-center space-x-4">
                        <div className="hidden lg:flex flex-col text-right text-xs text-slate-300">
                            <span className="font-medium text-slate-100">D. Y. Patil College of Engineering, Akurdi</span>
                            <span className="text-slate-400">TY AI & Data Science | Course: AD124PC503</span>
                        </div>
                        <div className={`flex items-center space-x-2 text-xs px-3 py-1.5 rounded-full border ${
                            status === "online" 
                                ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40" 
                                : "bg-rose-500/20 text-rose-300 border-rose-500/40"
                        }`}>
                            <span className={`w-2 h-2 rounded-full ${status === "online" ? "bg-emerald-400 animate-pulse" : "bg-rose-400"}`}></span>
                            <span className="font-medium">{status === "online" ? "React + Flask API Connected" : "Connecting..."}</span>
                        </div>
                    </div>
                </div>
            </div>
        </nav>
    );
}

// -------------------------------------------------------------------------------------------------
// HERO BANNER COMPONENT
// -------------------------------------------------------------------------------------------------
function HeroHeader({ kpiData }) {
    return (
        <header className="bg-white rounded-2xl p-6 sm:p-8 shadow-sm border border-slate-200/80 relative overflow-hidden">
            <div className="absolute -right-8 -bottom-8 opacity-5 pointer-events-none">
                <i className="fa-solid fa-sun text-9xl text-amber-500"></i>
            </div>
            <div className="max-w-3xl">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 text-blue-700 text-xs font-semibold mb-3 border border-blue-100">
                    <i className="fa-brands fa-react text-blue-500"></i> Pure React 18 Architecture | Unsupervised Machine Learning
                </div>
                <h1 className="text-2xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
                    Solar Power Anomaly Detection Dashboard
                </h1>
                <p className="mt-2.5 text-slate-600 text-xs sm:text-sm leading-relaxed">
                    Automated multi-sensor health monitoring analyzing <strong>68,774 observations</strong> across 
                    <strong> 22 inverters</strong> over 34 days of operation. Evaluates power conversion efficiency, 
                    solar irradiance correlation, and thermal dynamics using <strong>Isolation Forest</strong> and <strong>DBSCAN</strong>.
                </p>
                <div className="mt-4 p-3 bg-amber-50/90 border border-amber-200/80 rounded-xl flex items-start gap-3 text-xs text-amber-900">
                    <i className="fa-solid fa-triangle-exclamation text-amber-600 text-base mt-0.5"></i>
                    <div>
                        <strong>Academic Evaluation Clarification:</strong> Anomalies represent mathematical outliers in multi-sensor space, 
                        <em>not verified equipment breakdowns</em>. Domain expertise is required to distinguish inverter trips and clipping from cloud transits or sensor drift.
                    </div>
                </div>
            </div>
        </header>
    );
}

// -------------------------------------------------------------------------------------------------
// LIVE KPI GRID COMPONENT
// -------------------------------------------------------------------------------------------------
function KPIGrid({ kpiData, invertersCount }) {
    if (!kpiData) {
        return <div className="p-8 text-center text-slate-400"><i className="fa-solid fa-spinner fa-spin mr-2"></i>Loading live telemetry KPIs...</div>;
    }

    const { total_observations, isolation_forest, dbscan, plant_extremes } = kpiData;

    return (
        <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500 block">Total Records</span>
                <div className="mt-2 text-xl font-extrabold text-slate-900">{total_observations.toLocaleString()}</div>
                <span className="text-[11px] text-slate-400">15-min intervals</span>
            </div>

            <div className="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500 block">Nominal Operation</span>
                <div className="mt-2 text-xl font-extrabold text-emerald-600">{isolation_forest.normal_count.toLocaleString()}</div>
                <span className="text-[11px] text-emerald-600 font-semibold">{isolation_forest.normal_percentage}% nominal</span>
            </div>

            <div className="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500 block">IF Anomalies</span>
                <div className="mt-2 text-xl font-extrabold text-rose-600">{isolation_forest.anomaly_count.toLocaleString()}</div>
                <span className="text-[11px] text-rose-600 font-semibold">{isolation_forest.anomaly_percentage}% (n=100)</span>
            </div>

            <div className="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500 block">DBSCAN Noise</span>
                <div className="mt-2 text-xl font-extrabold text-indigo-600">{dbscan.noise_count}</div>
                <span className="text-[11px] text-indigo-600 font-semibold">{dbscan.noise_percentage}% (eps=0.8)</span>
            </div>

            <div className="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500 block">Inverters Monitored</span>
                <div className="mt-2 text-xl font-extrabold text-slate-900">{invertersCount || 22}</div>
                <span className="text-[11px] text-slate-400">Plant 1 Arrays</span>
            </div>

            <div className="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500 block">Peak DC Generation</span>
                <div className="mt-2 text-xl font-extrabold text-amber-600">{plant_extremes.peak_dc_power.toLocaleString()} <span className="text-xs">kW</span></div>
                <span className="text-[11px] text-slate-400">Peak AC: {plant_extremes.peak_ac_power} kW</span>
            </div>
        </section>
    );
}

// -------------------------------------------------------------------------------------------------
// 1. SIMULATOR VIEW (REAL-TIME REACT INTERFERENCE)
// -------------------------------------------------------------------------------------------------
function SimulatorView({ presets }) {
    const [formValues, setFormValues] = useState({
        DC_POWER: 4200.0,
        AC_POWER: 410.0,
        DAILY_YIELD: 3100.0,
        AMBIENT_TEMPERATURE: 29.5,
        MODULE_TEMPERATURE: 42.0,
        IRRADIATION: 0.48
    });
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [activePreset, setActivePreset] = useState(0);

    const handleSliderChange = (field, val) => {
        setFormValues(prev => ({ ...prev, [field]: parseFloat(val) }));
    };

    const runInference = (overrideValues) => {
        const payload = overrideValues || formValues;
        setLoading(true);
        fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
            .then(res => res.json())
            .then(data => {
                setResult(data);
                setLoading(false);
            })
            .catch(err => {
                console.error("Inference error:", err);
                setLoading(false);
            });
    };

    useEffect(() => {
        runInference();
    }, []);

    const applyPreset = (preset, idx) => {
        setActivePreset(idx);
        setFormValues(preset.values);
        runInference(preset.values);
    };

    return (
        <section className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left 7 Columns: Telemetry Controls */}
                <div className="lg:col-span-7 bg-white rounded-2xl p-6 shadow-sm border border-slate-200 space-y-6">
                    <div>
                        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                            <i className="fa-solid fa-gauge-high text-blue-600"></i> Multi-Sensor Telemetry Simulator
                        </h2>
                        <p className="text-xs text-slate-500 mt-1">
                            Adjust multi-sensor telemetry or select standard test scenarios to evaluate Isolation Forest real-time inference.
                        </p>
                    </div>

                    {/* Presets */}
                    <div>
                        <span className="block text-xs font-bold text-slate-600 uppercase tracking-wider mb-2">
                            Demonstration Test Presets:
                        </span>
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                            {presets.map((p, i) => (
                                <button
                                    key={i}
                                    type="button"
                                    onClick={() => applyPreset(p, i)}
                                    className={`preset-btn text-left p-2.5 rounded-lg border text-xs font-semibold transition ${
                                        activePreset === i
                                            ? "bg-blue-50 border-blue-500 text-blue-700 shadow-sm"
                                            : "bg-white border-slate-200 text-slate-700 hover:bg-slate-50"
                                    }`}
                                >
                                    <span className="block font-bold truncate">{p.name}</span>
                                    <span className="block text-[10px] text-slate-400 truncate">{p.description}</span>
                                </button>
                            ))}
                        </div>
                    </div>

                    <hr className="border-slate-100" />

                    {/* Sliders Grid */}
                    <form onSubmit={(e) => { e.preventDefault(); runInference(); }} className="space-y-4">
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            <SliderField
                                label="DC Power (kW)"
                                min={0} max={15000} step={50}
                                value={formValues.DC_POWER}
                                unit=" kW"
                                onChange={(val) => handleSliderChange("DC_POWER", val)}
                            />
                            <SliderField
                                label="AC Power (kW)"
                                min={0} max={1500} step={10}
                                value={formValues.AC_POWER}
                                unit=" kW"
                                onChange={(val) => handleSliderChange("AC_POWER", val)}
                            />
                            <SliderField
                                label="Daily Yield (kWh)"
                                min={0} max={10000} step={50}
                                value={formValues.DAILY_YIELD}
                                unit=" kWh"
                                onChange={(val) => handleSliderChange("DAILY_YIELD", val)}
                            />
                            <SliderField
                                label="Ambient Temp (°C)"
                                min={15} max={50} step={0.5}
                                value={formValues.AMBIENT_TEMPERATURE}
                                unit=" °C"
                                onChange={(val) => handleSliderChange("AMBIENT_TEMPERATURE", val)}
                            />
                            <SliderField
                                label="Module Temp (°C)"
                                min={15} max={75} step={0.5}
                                value={formValues.MODULE_TEMPERATURE}
                                unit=" °C"
                                onChange={(val) => handleSliderChange("MODULE_TEMPERATURE", val)}
                            />
                            <SliderField
                                label="Solar Irradiation"
                                min={0} max={1.5} step={0.02}
                                value={formValues.IRRADIATION}
                                unit=""
                                onChange={(val) => handleSliderChange("IRRADIATION", val)}
                            />
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full py-3 px-4 rounded-xl font-bold text-white bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-md hover:shadow-lg transition flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <><i className="fa-solid fa-spinner fa-spin mr-1"></i> Running Inference...</>
                            ) : (
                                <><i className="fa-solid fa-bolt mr-1"></i> Run AI Anomaly Inference</>
                            )}
                        </button>
                    </form>
                </div>

                {/* Right 5 Columns: Result Diagnosis Card */}
                <div className="lg:col-span-5 space-y-6">
                    {result ? (
                        <div className={`bg-white rounded-2xl p-6 shadow-sm border transition-all duration-300 ${
                            result.is_anomaly ? "status-anomaly-card border-rose-300" : "status-normal-card border-emerald-300"
                        }`}>
                            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Inference Status</span>
                                <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600 font-mono font-bold">
                                    Code: {result.anomaly_code}
                                </span>
                            </div>

                            <div className="my-6 text-center">
                                <div className={`w-20 h-20 rounded-full mx-auto flex items-center justify-center text-3xl mb-3 shadow-inner ${
                                    result.is_anomaly ? "bg-rose-100 text-rose-600 animate-bounce" : "bg-emerald-100 text-emerald-600"
                                }`}>
                                    <i className={`fa-solid ${result.is_anomaly ? "fa-triangle-exclamation" : "fa-check"}`}></i>
                                </div>
                                <h3 className={`text-2xl font-extrabold tracking-tight ${
                                    result.is_anomaly ? "text-rose-600" : "text-emerald-600"
                                }`}>
                                    {result.is_anomaly ? "ANOMALY DETECTED" : "NOMINAL GENERATION"}
                                </h3>
                                <p className="text-xs text-slate-500 mt-1">
                                    {result.is_anomaly ? "Multi-sensor pattern deviates from nominal operating manifold" : "Telemetry adheres to nominal power conversion curve"}
                                </p>
                            </div>

                            {/* Score Meter */}
                            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/80 space-y-2">
                                <div className="flex justify-between items-center text-xs">
                                    <span className="font-semibold text-slate-700">Isolation Forest Score:</span>
                                    <span className="font-mono font-bold text-slate-900 text-sm">
                                        {(result.anomaly_score >= 0 ? "+" : "") + result.anomaly_score.toFixed(4)}
                                    </span>
                                </div>
                                <div className="w-full bg-slate-200 h-2.5 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full rounded-full transition-all duration-500 ${result.is_anomaly ? "bg-rose-500" : "bg-emerald-500"}`}
                                        style={{ width: `${Math.max(5, Math.min(95, ((result.anomaly_score + 0.5) / 1.0) * 100))}%` }}
                                    ></div>
                                </div>
                                <div className="flex justify-between text-[10px] text-slate-400">
                                    <span>-0.50 (Extreme Anomaly)</span>
                                    <span>0.00 (Boundary)</span>
                                    <span>+0.50 (Nominal)</span>
                                </div>
                            </div>

                            {/* Engineering Diagnosis */}
                            <div className="mt-4 p-4 rounded-xl bg-blue-50/70 border border-blue-200/70">
                                <h4 className="text-xs font-bold text-blue-900 flex items-center gap-1.5 uppercase tracking-wider">
                                    <i className="fa-solid fa-stethoscope text-blue-600"></i> Engineering Diagnosis:
                                </h4>
                                <p className="text-xs text-blue-800 mt-1.5 leading-relaxed">
                                    {result.engineering_diagnosis}
                                </p>
                            </div>
                        </div>
                    ) : null}

                    {/* Sensor Subspace Echo */}
                    <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-200 text-xs">
                        <span className="font-bold text-slate-700 uppercase tracking-wider block mb-2">Evaluated Sensor Subspace:</span>
                        <div className="grid grid-cols-2 gap-2 text-slate-600">
                            <div>DC Power: <span className="font-mono font-bold text-slate-900">{formValues.DC_POWER} kW</span></div>
                            <div>AC Power: <span className="font-mono font-bold text-slate-900">{formValues.AC_POWER} kW</span></div>
                            <div>Daily Yield: <span className="font-mono font-bold text-slate-900">{formValues.DAILY_YIELD} kWh</span></div>
                            <div>Ambient Temp: <span className="font-mono font-bold text-slate-900">{formValues.AMBIENT_TEMPERATURE} °C</span></div>
                            <div>Module Temp: <span className="font-mono font-bold text-slate-900">{formValues.MODULE_TEMPERATURE} °C</span></div>
                            <div>Irradiation: <span className="font-mono font-bold text-slate-900">{formValues.IRRADIATION}</span></div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}

function SliderField({ label, min, max, step, value, unit, onChange }) {
    return (
        <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200/80">
            <div className="flex justify-between items-center mb-1">
                <label className="text-xs font-semibold text-slate-700">{label}</label>
                <input
                    type="number"
                    min={min} max={max} step={step}
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    className="w-20 text-xs font-mono font-bold text-right p-1 rounded border border-slate-300"
                />
            </div>
            <input
                type="range"
                min={min} max={max} step={step}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                className="w-full accent-blue-600"
            />
            <span className="text-[10px] text-slate-400">Current: {value}{unit}</span>
        </div>
    );
}

// -------------------------------------------------------------------------------------------------
// 2. VISUAL ANALYTICS VIEW (CHART.JS IN REACT)
// -------------------------------------------------------------------------------------------------
function AnalyticsView() {
    const [chartData, setChartData] = useState(null);
    const canvasDcAc = useRef(null);
    const canvasIrrAc = useRef(null);
    const canvasPie = useRef(null);
    const canvasTs = useRef(null);
    const canvasInv = useRef(null);

    useEffect(() => {
        fetch("/api/charts/data")
            .then(res => res.json())
            .then(data => setChartData(data))
            .catch(err => console.error("Error loading chart data:", err));
    }, []);

    useEffect(() => {
        if (!chartData) return;

        const charts = [];

        // 1. DC vs AC
        if (canvasDcAc.current) {
            const chart1 = new Chart(canvasDcAc.current, {
                type: "scatter",
                data: {
                    datasets: [
                        { label: "Normal", data: chartData.dc_vs_ac.normal, backgroundColor: "rgba(59, 130, 246, 0.4)", pointRadius: 2.5 },
                        { label: "Anomaly", data: chartData.dc_vs_ac.anomaly, backgroundColor: "rgba(239, 68, 68, 0.8)", pointRadius: 4.5, pointStyle: "crossRot" }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: "DC Power (kW)", font: { size: 10 } } },
                        y: { title: { display: true, text: "AC Power (kW)", font: { size: 10 } } }
                    },
                    plugins: { legend: { position: "top", labels: { boxWidth: 10, font: { size: 10 } } } }
                }
            });
            charts.push(chart1);
        }

        // 2. Irradiation vs AC
        if (canvasIrrAc.current) {
            const chart2 = new Chart(canvasIrrAc.current, {
                type: "scatter",
                data: {
                    datasets: [
                        { label: "Normal", data: chartData.irr_vs_ac.normal, backgroundColor: "rgba(16, 185, 129, 0.4)", pointRadius: 2.5 },
                        { label: "Anomaly", data: chartData.irr_vs_ac.anomaly, backgroundColor: "rgba(239, 68, 68, 0.85)", pointRadius: 4.5, pointStyle: "crossRot" }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: "Solar Irradiation", font: { size: 10 } } },
                        y: { title: { display: true, text: "AC Power (kW)", font: { size: 10 } } }
                    },
                    plugins: { legend: { position: "top", labels: { boxWidth: 10, font: { size: 10 } } } }
                }
            });
            charts.push(chart2);
        }

        // 3. Proportion Doughnut
        if (canvasPie.current) {
            const chart3 = new Chart(canvasPie.current, {
                type: "doughnut",
                data: {
                    labels: ["Normal (95.00%)", "Anomalies (5.00%)"],
                    datasets: [{ data: [65335, 3439], backgroundColor: ["#10b981", "#ef4444"], hoverOffset: 4 }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { position: "bottom", labels: { font: { size: 11 } } } }
                }
            });
            charts.push(chart3);
        }

        // 4. Time Series Curve
        if (canvasTs.current) {
            const chart4 = new Chart(canvasTs.current, {
                type: "line",
                data: {
                    labels: chartData.timeseries.labels.map(l => l.split(" ")[1] || l),
                    datasets: [
                        { label: "AC Power (kW)", data: chartData.timeseries.ac_power, borderColor: "rgba(59, 130, 246, 0.8)", borderWidth: 1.5, fill: false, pointRadius: 0 },
                        { label: "Anomaly Flagged", data: chartData.timeseries.anomalies, borderColor: "transparent", backgroundColor: "rgba(239, 68, 68, 0.9)", pointRadius: 4, showLine: false }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: "15-min Intervals", font: { size: 10 } }, ticks: { maxTicksLimit: 10 } },
                        y: { title: { display: true, text: "AC Power (kW)", font: { size: 10 } } }
                    },
                    plugins: { legend: { position: "top", labels: { boxWidth: 10, font: { size: 10 } } } }
                }
            });
            charts.push(chart4);
        }

        // 5. Inverters Anomaly Ranking Bar Chart
        if (canvasInv.current && chartData.inverters_chart) {
            const chart5 = new Chart(canvasInv.current, {
                type: "bar",
                data: {
                    labels: chartData.inverters_chart.labels,
                    datasets: [{
                        label: "Anomaly Count",
                        data: chartData.inverters_chart.counts,
                        backgroundColor: "#f43f5e",
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: "Inverter Key (Last 8 Chars)", font: { size: 10 } } },
                        y: { title: { display: true, text: "Anomalies Logged", font: { size: 10 } } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
            charts.push(chart5);
        }

        return () => charts.forEach(c => c.destroy());
    }, [chartData]);

    if (!chartData) {
        return <div className="p-12 text-center text-slate-400"><i className="fa-solid fa-spinner fa-spin mr-2"></i>Loading multi-sensor visualization streams...</div>;
    }

    return (
        <section className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
                    <h3 className="text-base font-bold text-slate-900 flex justify-between">
                        <span>DC vs AC Power (Inverter Conversion Curve)</span>
                        <span className="text-xs font-normal text-slate-400">kW</span>
                    </h3>
                    <p className="text-xs text-slate-500 mb-4">Deviation from diagonal highlights inverter conversion drops & clipping</p>
                    <div className="h-72"><canvas ref={canvasDcAc}></canvas></div>
                </div>

                <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
                    <h3 className="text-base font-bold text-slate-900 flex justify-between">
                        <span>Solar Irradiation vs AC Output</span>
                        <span className="text-xs font-normal text-slate-400">Sun vs Generation</span>
                    </h3>
                    <p className="text-xs text-slate-500 mb-4">Points near bottom right represent critical inverter disconnection</p>
                    <div className="h-72"><canvas ref={canvasIrrAc}></canvas></div>
                </div>

                <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
                    <h3 className="text-base font-bold text-slate-900">Isolation Forest Anomaly Proportion</h3>
                    <p className="text-xs text-slate-500 mb-4">Target contamination rate set to 5.0% (3,439 anomalies)</p>
                    <div className="h-72 flex items-center justify-center"><canvas ref={canvasPie}></canvas></div>
                </div>

                <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
                    <h3 className="text-base font-bold text-slate-900">Diurnal Generation Time Series</h3>
                    <p className="text-xs text-slate-500 mb-4">Consecutive 15-minute intervals showing flagged spikes</p>
                    <div className="h-72"><canvas ref={canvasTs}></canvas></div>
                </div>

                <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 lg:col-span-2">
                    <h3 className="text-base font-bold text-slate-900">Top 10 Inverters with Highest Flagged Anomalies</h3>
                    <p className="text-xs text-slate-500 mb-4">Cross-inverter comparison helps maintenance teams prioritize string inspections</p>
                    <div className="h-64"><canvas ref={canvasInv}></canvas></div>
                </div>
            </div>
        </section>
    );
}

// -------------------------------------------------------------------------------------------------
// 3. INVERTER MATRIX VIEW (22 INVERTERS)
// -------------------------------------------------------------------------------------------------
function InverterMatrixView({ inverters }) {
    const [searchTerm, setSearchTerm] = useState("");
    const [sortBy, setSortBy] = useState("anomaly_count");

    const filteredInverters = useMemo(() => {
        let list = inverters.filter(inv => inv.inverter_id.toLowerCase().includes(searchTerm.toLowerCase()));
        if (sortBy === "anomaly_count") list.sort((a, b) => b.anomaly_count - a.anomaly_count);
        if (sortBy === "anomaly_rate") list.sort((a, b) => b.anomaly_rate - a.anomaly_rate);
        if (sortBy === "efficiency") list.sort((a, b) => b.efficiency - a.efficiency);
        return list;
    }, [inverters, searchTerm, sortBy]);

    return (
        <section className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 space-y-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                        <i className="fa-solid fa-solar-panel text-blue-600"></i> Inverter Fleet Telemetry & Health Matrix
                    </h2>
                    <p className="text-xs text-slate-500 mt-1">
                        Comprehensive individual performance metrics across all 22 central inverters over 34 operational days.
                    </p>
                </div>

                <div className="flex items-center gap-3 w-full sm:w-auto">
                    <input
                        type="text"
                        placeholder="Search inverter ID..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="p-2 text-xs border rounded-lg w-full sm:w-48"
                    />
                    <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value)}
                        className="p-2 text-xs border rounded-lg"
                    >
                        <option value="anomaly_count">Sort by Anomalies</option>
                        <option value="anomaly_rate">Sort by Anomaly Rate</option>
                        <option value="efficiency">Sort by Efficiency</option>
                    </select>
                </div>
            </div>

            <div className="overflow-x-auto rounded-xl border border-slate-200">
                <table className="min-w-full divide-y divide-slate-200 text-xs">
                    <thead className="bg-slate-50 font-bold text-slate-600">
                        <tr>
                            <th className="px-4 py-3 text-left">Inverter ID</th>
                            <th className="px-4 py-3 text-left">Total Records</th>
                            <th className="px-4 py-3 text-left">Flagged Anomalies</th>
                            <th className="px-4 py-3 text-left">Anomaly Rate</th>
                            <th className="px-4 py-3 text-left">Avg DC (kW)</th>
                            <th className="px-4 py-3 text-left">Avg AC (kW)</th>
                            <th className="px-4 py-3 text-left">Conversion Eff.</th>
                            <th className="px-4 py-3 text-left">Health Status</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 bg-white">
                        {filteredInverters.map((inv, idx) => (
                            <tr key={idx} className="hover:bg-slate-50 transition">
                                <td className="px-4 py-2.5 font-mono font-bold text-slate-900">{inv.inverter_id}</td>
                                <td className="px-4 py-2.5 font-mono">{inv.total_records.toLocaleString()}</td>
                                <td className="px-4 py-2.5 font-mono font-bold text-rose-600">{inv.anomaly_count}</td>
                                <td className="px-4 py-2.5 font-mono font-bold">{inv.anomaly_rate}%</td>
                                <td className="px-4 py-2.5 font-mono">{inv.avg_dc}</td>
                                <td className="px-4 py-2.5 font-mono">{inv.avg_ac}</td>
                                <td className="px-4 py-2.5 font-mono">{inv.efficiency}%</td>
                                <td className="px-4 py-2.5">
                                    <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold ${
                                        inv.badge === "emerald" 
                                            ? "bg-emerald-100 text-emerald-800"
                                            : inv.badge === "amber"
                                            ? "bg-amber-100 text-amber-800"
                                            : "bg-rose-100 text-rose-800"
                                    }`}>
                                        {inv.status}
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </section>
    );
}

// -------------------------------------------------------------------------------------------------
// 4. TELEMETRY EXPLORER VIEW (PAGINATED & SEARCHABLE 68,774 ROWS)
// -------------------------------------------------------------------------------------------------
function TelemetryExplorerView({ inverters }) {
    const [page, setPage] = useState(1);
    const [filter, setFilter] = useState("all");
    const [inverter, setInverter] = useState("");
    const [search, setSearch] = useState("");
    const [data, setData] = useState({ records: [], total: 0, pages: 1 });
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        setLoading(true);
        const params = new URLSearchParams({ page, limit: 25, filter, inverter, search });
        fetch(`/api/data/records?${params.toString()}`)
            .then(res => res.json())
            .then(resData => {
                setData(resData);
                setLoading(false);
            })
            .catch(err => {
                console.error("Error loading records:", err);
                setLoading(false);
            });
    }, [page, filter, inverter, search]);

    return (
        <section className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 space-y-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                        <i className="fa-solid fa-table text-blue-600"></i> Telemetry Dataset Explorer
                    </h2>
                    <p className="text-xs text-slate-500 mt-1">
                        Inspect, filter, and search across all <strong>{data.total.toLocaleString()}</strong> synchronized sensor observations.
                    </p>
                </div>

                <div className="flex flex-wrap items-center gap-2.5 w-full sm:w-auto">
                    <input
                        type="text"
                        placeholder="Search timestamp..."
                        value={search}
                        onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                        className="p-2 text-xs border rounded-lg w-full sm:w-36"
                    />
                    <select
                        value={filter}
                        onChange={(e) => { setFilter(e.target.value); setPage(1); }}
                        className="p-2 text-xs border rounded-lg"
                    >
                        <option value="all">All Telemetry</option>
                        <option value="anomalies">Anomalies Only (-1)</option>
                        <option value="normal">Normal Only (+1)</option>
                        <option value="dbscan_noise">DBSCAN Noise (6)</option>
                    </select>

                    <select
                        value={inverter}
                        onChange={(e) => { setInverter(e.target.value); setPage(1); }}
                        className="p-2 text-xs border rounded-lg"
                    >
                        <option value="">All 22 Inverters</option>
                        {inverters.map(inv => (
                            <option key={inv.inverter_id} value={inv.inverter_id}>
                                {inv.inverter_id.slice(-8)} ({inv.anomaly_count} anom)
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {loading ? (
                <div className="p-12 text-center text-slate-400"><i className="fa-solid fa-spinner fa-spin mr-2"></i>Loading telemetry rows...</div>
            ) : (
                <div className="overflow-x-auto rounded-xl border border-slate-200">
                    <table className="min-w-full divide-y divide-slate-200 text-xs">
                        <thead className="bg-slate-50 font-bold text-slate-600">
                            <tr>
                                <th className="px-4 py-2.5 text-left">Timestamp</th>
                                <th className="px-4 py-2.5 text-left">Inverter Key</th>
                                <th className="px-4 py-2.5 text-left">DC Power (kW)</th>
                                <th className="px-4 py-2.5 text-left">AC Power (kW)</th>
                                <th className="px-4 py-2.5 text-left">Yield (kWh)</th>
                                <th className="px-4 py-2.5 text-left">Module Temp</th>
                                <th className="px-4 py-2.5 text-left">Irradiation</th>
                                <th className="px-4 py-2.5 text-left">Status</th>
                                <th className="px-4 py-2.5 text-left">IF Score</th>
                                <th className="px-4 py-2.5 text-left">DBSCAN</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 bg-white">
                            {data.records.map((r, i) => (
                                <tr key={i} className={r.anomaly_code === -1 ? "bg-rose-50/50 hover:bg-rose-50" : "hover:bg-slate-50"}>
                                    <td className="px-4 py-2 font-mono text-slate-700">{r.date_time}</td>
                                    <td className="px-4 py-2 font-mono">{r.inverter_id.slice(-8)}</td>
                                    <td className="px-4 py-2 font-mono font-bold">{r.dc_power}</td>
                                    <td className="px-4 py-2 font-mono font-bold">{r.ac_power}</td>
                                    <td className="px-4 py-2 font-mono">{r.daily_yield}</td>
                                    <td className="px-4 py-2 font-mono">{r.module_temp}°C</td>
                                    <td className="px-4 py-2 font-mono">{r.irradiation}</td>
                                    <td className="px-4 py-2">
                                        <span className={`px-2 py-0.5 rounded font-bold ${
                                            r.anomaly_code === -1 ? "bg-rose-100 text-rose-700" : "bg-emerald-100 text-emerald-700"
                                        }`}>
                                            {r.anomaly_label}
                                        </span>
                                    </td>
                                    <td className="px-4 py-2 font-mono">{r.anomaly_score}</td>
                                    <td className="px-4 py-2 font-mono">
                                        {r.dbscan_noise ? <span className="text-rose-600 font-bold">Noise (-1)</span> : `Cluster ${r.dbscan_cluster}`}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Pagination Controls */}
            <div className="flex justify-between items-center text-xs text-slate-500 pt-2">
                <span>Page {data.page} of {data.pages} ({data.total.toLocaleString()} rows)</span>
                <div className="flex space-x-2">
                    <button
                        disabled={data.page <= 1}
                        onClick={() => setPage(p => Math.max(1, p - 1))}
                        className="px-3 py-1.5 border rounded-lg bg-white disabled:opacity-40"
                    >
                        Previous
                    </button>
                    <button
                        disabled={data.page >= data.pages}
                        onClick={() => setPage(p => Math.min(data.pages, p + 1))}
                        className="px-3 py-1.5 border rounded-lg bg-white disabled:opacity-40"
                    >
                        Next
                    </button>
                </div>
            </div>
        </section>
    );
}

// -------------------------------------------------------------------------------------------------
// 5. FEATURE STATISTICS & DISTRIBUTIONS VIEW
// -------------------------------------------------------------------------------------------------
function FeatureStatsView({ stats }) {
    if (!stats || stats.length === 0) {
        return <div className="p-8 text-center text-slate-400"><i className="fa-solid fa-spinner fa-spin mr-2"></i>Loading feature statistics...</div>;
    }

    return (
        <section className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 space-y-6">
            <div>
                <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                    <i className="fa-solid fa-chart-simple text-blue-600"></i> Numerical Sensor Feature Distributions & Centroids
                </h2>
                <p className="text-xs text-slate-500 mt-1">
                    Parametric summary of all six selected telemetry dimensions, comparing nominal operational baselines against anomaly centroids.
                </p>
            </div>

            <div className="overflow-x-auto rounded-xl border border-slate-200">
                <table className="min-w-full divide-y divide-slate-200 text-xs">
                    <thead className="bg-slate-50 font-bold text-slate-600">
                        <tr>
                            <th className="px-4 py-3 text-left">Feature Name</th>
                            <th className="px-4 py-3 text-left">Mean</th>
                            <th className="px-4 py-3 text-left">Std Dev</th>
                            <th className="px-4 py-3 text-left">Min</th>
                            <th className="px-4 py-3 text-left">Median (50%)</th>
                            <th className="px-4 py-3 text-left">Max</th>
                            <th className="px-4 py-3 text-left text-emerald-700 bg-emerald-50/50">Normal Mean</th>
                            <th className="px-4 py-3 text-left text-rose-700 bg-rose-50/50">Anomaly Mean</th>
                            <th className="px-4 py-3 text-left">Anomaly / Normal Ratio</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 bg-white">
                        {stats.map((s, idx) => (
                            <tr key={idx} className="hover:bg-slate-50">
                                <td className="px-4 py-3 font-mono font-bold text-slate-900">{s.feature}</td>
                                <td className="px-4 py-3 font-mono">{s.mean.toLocaleString()}</td>
                                <td className="px-4 py-3 font-mono">{s.std.toLocaleString()}</td>
                                <td className="px-4 py-3 font-mono">{s.min}</td>
                                <td className="px-4 py-3 font-mono font-semibold">{s.median.toLocaleString()}</td>
                                <td className="px-4 py-3 font-mono font-bold text-slate-900">{s.max.toLocaleString()}</td>
                                <td className="px-4 py-3 font-mono font-bold text-emerald-700 bg-emerald-50/30">{s.normal_mean.toLocaleString()}</td>
                                <td className="px-4 py-3 font-mono font-bold text-rose-700 bg-rose-50/30">{s.anomaly_mean.toLocaleString()}</td>
                                <td className="px-4 py-3 font-mono font-bold">
                                    <span className={s.deviation_ratio > 2.0 ? "text-rose-600" : "text-slate-700"}>
                                        {s.deviation_ratio}x
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div className="p-4 rounded-xl bg-blue-50/70 border border-blue-200/70 text-xs text-blue-900 leading-relaxed">
                <strong>Key Physical Finding for Viva:</strong> Anomalous observations demonstrate a <strong>3.69x</strong> higher DC Power centroid (10,239 kW vs 2,773 kW) 
                and a <strong>3.68x</strong> higher AC Power centroid (999 kW vs 271 kW) alongside elevated module surface temperatures (54.4°C vs 30.0°C). 
                Isolation Forest naturally identifies midday peak periods where inverters experience thermal derating, string imbalances, and clipping.
            </div>
        </section>
    );
}

// -------------------------------------------------------------------------------------------------
// 6. BATCH CSV SCANNER VIEW
// -------------------------------------------------------------------------------------------------
function BatchScannerView() {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [batchResult, setBatchResult] = useState(null);

    const handleFileChange = (e) => {
        if (e.target.files.length > 0) setFile(e.target.files[0]);
    };

    const handleBatchProcess = () => {
        if (!file) return;
        setLoading(true);
        const formData = new FormData();
        formData.append("file", file);

        fetch("/api/predict/batch", {
            method: "POST",
            body: formData
        })
            .then(res => res.json())
            .then(data => {
                setBatchResult(data);
                setLoading(false);
            })
            .catch(err => {
                alert("Batch scoring failed: " + err.message);
                setLoading(false);
            });
    };

    const downloadScoredCSV = () => {
        if (!batchResult || !batchResult.preview || batchResult.preview.length === 0) return;
        const headers = Object.keys(batchResult.preview[0]);
        const rows = [headers.join(",")];
        batchResult.preview.forEach(r => {
            rows.push(headers.map(h => JSON.stringify(r[h] ?? "")).join(","));
        });
        const blob = new Blob([rows.join("\n")], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "scored_solar_telemetry.csv";
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <section className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 space-y-6">
            <div>
                <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                    <i className="fa-solid fa-file-csv text-blue-600"></i> Batch Telemetry CSV Processor
                </h2>
                <p className="text-xs text-slate-500 mt-1">
                    Upload any custom multi-row CSV file containing solar telemetry to score hundreds of records with Isolation Forest simultaneously.
                </p>
            </div>

            <div className="border-2 border-dashed border-slate-300 rounded-2xl p-8 text-center hover:border-blue-500 transition cursor-pointer">
                <input type="file" accept=".csv" onChange={handleFileChange} id="csv-upload" className="hidden" />
                <label htmlFor="csv-upload" className="cursor-pointer">
                    <div className="w-12 h-12 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center mx-auto mb-3 text-xl">
                        <i className="fa-solid fa-cloud-arrow-up"></i>
                    </div>
                    <span className="text-sm font-semibold text-slate-700 block">Click to Select or Drop Telemetry CSV File</span>
                    <span className="text-xs text-slate-400 mt-1 block">Compatible columns: DC_POWER, AC_POWER, DAILY_YIELD, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION</span>
                    {file && <span className="text-xs font-mono font-bold text-blue-600 mt-2 block">Selected: {file.name} ({(file.size / 1024).toFixed(1)} KB)</span>}
                </label>
            </div>

            <div className="flex gap-4">
                <button
                    onClick={handleBatchProcess}
                    disabled={!file || loading}
                    className="py-2.5 px-6 rounded-xl font-bold text-white bg-blue-600 hover:bg-blue-700 transition disabled:opacity-50"
                >
                    {loading ? <><i className="fa-solid fa-spinner fa-spin mr-1.5"></i> Scoring Telemetry...</> : <><i className="fa-solid fa-play mr-1.5"></i> Score Batch Data</>}
                </button>

                {batchResult && (
                    <button
                        onClick={downloadScoredCSV}
                        className="py-2.5 px-6 rounded-xl font-bold text-slate-700 bg-slate-100 hover:bg-slate-200 transition"
                    >
                        <i className="fa-solid fa-download mr-1.5"></i> Download Scored CSV
                    </button>
                )}
            </div>

            {batchResult && (
                <div className="space-y-4 pt-4 border-t border-slate-100">
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                            <span className="text-xs text-slate-500 font-semibold uppercase">Total Scored Rows</span>
                            <div className="text-xl font-bold text-slate-900 mt-1">{batchResult.summary.total_samples.toLocaleString()}</div>
                        </div>
                        <div className="p-4 bg-emerald-50 rounded-xl border border-emerald-200">
                            <span className="text-xs text-emerald-700 font-semibold uppercase">Nominal Rows</span>
                            <div className="text-xl font-bold text-emerald-600 mt-1">{batchResult.summary.normal_count.toLocaleString()}</div>
                        </div>
                        <div className="p-4 bg-rose-50 rounded-xl border border-rose-200">
                            <span className="text-xs text-rose-700 font-semibold uppercase">Flagged Anomalies</span>
                            <div className="text-xl font-bold text-rose-600 mt-1">{batchResult.summary.anomaly_count.toLocaleString()} ({batchResult.summary.anomaly_percentage}%)</div>
                        </div>
                    </div>

                    <div className="overflow-x-auto rounded-xl border border-slate-200">
                        <table className="min-w-full divide-y divide-slate-200 text-xs">
                            <thead className="bg-slate-50 font-bold text-slate-600">
                                <tr>
                                    <th className="px-4 py-2.5 text-left">#</th>
                                    <th className="px-4 py-2.5 text-left">DC Power</th>
                                    <th className="px-4 py-2.5 text-left">AC Power</th>
                                    <th className="px-4 py-2.5 text-left">Irradiation</th>
                                    <th className="px-4 py-2.5 text-left">Prediction</th>
                                    <th className="px-4 py-2.5 text-left">Anomaly Score</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100 bg-white">
                                {batchResult.preview.slice(0, 15).map((row, i) => (
                                    <tr key={i} className={row.anomaly_code === -1 ? "bg-rose-50/50" : "hover:bg-slate-50"}>
                                        <td className="px-4 py-2 font-mono">{i + 1}</td>
                                        <td className="px-4 py-2 font-mono">{Number(row.DC_POWER || 0).toFixed(1)}</td>
                                        <td className="px-4 py-2 font-mono">{Number(row.AC_POWER || 0).toFixed(1)}</td>
                                        <td className="px-4 py-2 font-mono">{Number(row.IRRADIATION || 0).toFixed(3)}</td>
                                        <td className="px-4 py-2 font-bold">
                                            {row.anomaly_code === -1 ? (
                                                <span className="text-rose-600"><i className="fa-solid fa-triangle-exclamation mr-1"></i> Anomaly</span>
                                            ) : (
                                                <span className="text-emerald-600"><i className="fa-solid fa-check mr-1"></i> Normal</span>
                                            )}
                                        </td>
                                        <td className="px-4 py-2 font-mono font-semibold">{row.anomaly_score}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </section>
    );
}

// -------------------------------------------------------------------------------------------------
// 7. ARCHITECTURE & VIVA DEFENSE VIEW
// -------------------------------------------------------------------------------------------------
function ArchitectureVivaView() {
    return (
        <section className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 space-y-4">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-amber-100 text-amber-700 flex items-center justify-center font-bold">
                            <i className="fa-solid fa-tree text-lg"></i>
                        </div>
                        <div>
                            <h3 className="text-base font-bold text-slate-900">Isolation Forest (Primary Model)</h3>
                            <span className="text-xs text-slate-500">Tree-Based Recursive Partitioning</span>
                        </div>
                    </div>
                    <p className="text-xs text-slate-600 leading-relaxed">
                        Isolates anomalies rather than profiling normal data points. Subsampled decision trees recursively split attributes; anomalies have shorter average path lengths h(x).
                    </p>
                    <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200/80 font-mono text-xs text-slate-700 space-y-1">
                        <div><strong>Anomaly Score:</strong> s(x, n) = 2^(-E(h(x)) / c(n))</div>
                        <div><strong>Parameters:</strong> n_estimators = 100, contamination = 0.05</div>
                        <div><strong>Feature Scaling:</strong> Not required (scale-invariant tree cuts)</div>
                        <div><strong>Detected Anomalies:</strong> 3,439 (5.0004%)</div>
                    </div>
                </div>

                <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 space-y-4">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-blue-100 text-blue-700 flex items-center justify-center font-bold">
                            <i className="fa-solid fa-diagram-project text-lg"></i>
                        </div>
                        <div>
                            <h3 className="text-base font-bold text-slate-900">DBSCAN (Comparison Model)</h3>
                            <span className="text-xs text-slate-500">Density-Based Spatial Clustering of Applications with Noise</span>
                        </div>
                    </div>
                    <p className="text-xs text-slate-600 leading-relaxed">
                        Groups points residing in continuous high-density manifolds. Points in sparse regions with fewer than min_samples neighbors inside eps-neighborhood are labeled noise (-1).
                    </p>
                    <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200/80 font-mono text-xs text-slate-700 space-y-1">
                        <div><strong>Distance Metric:</strong> d(u, v) = sqrt(sum (u_i - v_i)^2)</div>
                        <div><strong>Parameters:</strong> eps = 0.8, min_samples = 10</div>
                        <div><strong>Feature Scaling:</strong> Mandatory (StandardScaler, Mean=0, Std=1)</div>
                        <div><strong>Detected Noise Points:</strong> 6 (0.0087%)</div>
                    </div>
                </div>
            </div>

            <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 space-y-4">
                <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <i className="fa-solid fa-clipboard-question text-indigo-600"></i> Examiner Viva Voce Reference & Defenses
                </h3>

                <div className="space-y-3">
                    <details className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 group">
                        <summary className="font-bold text-xs sm:text-sm text-slate-800 cursor-pointer flex justify-between items-center">
                            <span>Q1: What is the main objective of this machine learning system?</span>
                            <i className="fa-solid fa-chevron-down text-slate-400 group-open:rotate-180 transition"></i>
                        </summary>
                        <p className="text-xs text-slate-600 mt-2.5 leading-relaxed">
                            To develop an unsupervised system detecting unusual operational patterns and sensor anomalies in solar photovoltaic plant data without requiring prior failure labels.
                        </p>
                    </details>

                    <details className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 group">
                        <summary className="font-bold text-xs sm:text-sm text-slate-800 cursor-pointer flex justify-between items-center">
                            <span>Q2: Why are anomalies defined as "unusual observations" rather than "equipment faults"?</span>
                            <i className="fa-solid fa-chevron-down text-slate-400 group-open:rotate-180 transition"></i>
                        </summary>
                        <p className="text-xs text-slate-600 mt-2.5 leading-relaxed">
                            Unsupervised models identify statistical outliers in multi-sensor space. Deviations may result from inverter clipping, cloud transits, high panel temperature, or sensor lag. Physical verification by domain engineers is necessary to confirm actual equipment faults.
                        </p>
                    </details>

                    <details className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 group">
                        <summary className="font-bold text-xs sm:text-sm text-slate-800 cursor-pointer flex justify-between items-center">
                            <span>Q3: Why was feature scaling applied to DBSCAN but not Isolation Forest?</span>
                            <i className="fa-solid fa-chevron-down text-slate-400 group-open:rotate-180 transition"></i>
                        </summary>
                        <p className="text-xs text-slate-600 mt-2.5 leading-relaxed">
                            DBSCAN calculates Euclidean distance in 6D space. Without standardization, DC Power (~10,000) completely suppresses Irradiation (~0.8). Isolation Forest relies on tree-based monotonic splits, making it invariant to feature scale.
                        </p>
                    </details>

                    <details className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 group">
                        <summary className="font-bold text-xs sm:text-sm text-slate-800 cursor-pointer flex justify-between items-center">
                            <span>Q4: Why did Isolation Forest detect 3,439 anomalies while DBSCAN detected only 6?</span>
                            <i className="fa-solid fa-chevron-down text-slate-400 group-open:rotate-180 transition"></i>
                        </summary>
                        <p className="text-xs text-slate-600 mt-2.5 leading-relaxed">
                            Isolation Forest was configured with contamination=0.05, mathematically designating the 5% shortest-path points as anomalies. DBSCAN evaluates continuous density in 6D space without a preset quota; only 6 observations were isolated from all dense clusters under eps=0.8.
                        </p>
                    </details>
                </div>
            </div>
        </section>
    );
}

// -------------------------------------------------------------------------------------------------
// FOOTER COMPONENT
// -------------------------------------------------------------------------------------------------
function Footer() {
    return (
        <footer className="bg-slate-900 text-slate-400 py-6 border-t border-slate-800 mt-auto text-xs">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row justify-between items-center gap-4">
                <div>
                    <span className="font-bold text-slate-200">Solar AI Sentinel Dashboard</span> | React 18 + Flask Architecture
                </div>
                <div className="text-center sm:text-right text-slate-400">
                    Department of AI & Data Science | D. Y. Patil College of Engineering, Akurdi, Pune
                </div>
            </div>
        </footer>
    );
}

// Mount the React Application
const rootElement = document.getElementById("root");
ReactDOM.createRoot(rootElement).render(<App />);
