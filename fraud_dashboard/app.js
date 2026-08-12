document.addEventListener('DOMContentLoaded', () => {
    // 1. Tab Navigation
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // 2. Threshold Slider Math Simulation
    const slider = document.getElementById('threshold-slider');
    const sliderVal = document.getElementById('slider-val');
    const kpiThreshold = document.getElementById('kpi-threshold');

    const meterRecall = document.getElementById('meter-recall');
    const meterPrecision = document.getElementById('meter-precision');
    const meterFpr = document.getElementById('meter-fpr');
    const meterSavings = document.getElementById('meter-savings');

    const barRecall = document.getElementById('bar-recall');
    const barPrecision = document.getElementById('bar-precision');
    const barFpr = document.getElementById('bar-fpr');
    const barSavings = document.getElementById('bar-savings');

    function updateThresholdMetrics(tau) {
        sliderVal.textContent = parseFloat(tau).toFixed(3);
        kpiThreshold.textContent = parseFloat(tau).toFixed(3);

        // Simulation curves fitted to Precision-Recall relationship
        // At tau = 0.428 -> Recall = 94.7%, Precision = 88.4%, FPR = 3.2%, Savings = 80.5 Cr
        let recall = Math.min(99.5, Math.max(50.0, 94.7 + (0.428 - tau) * 45.0));
        let precision = Math.min(98.0, Math.max(45.0, 88.4 - (0.428 - tau) * 35.0));
        let fpr = Math.min(38.0, Math.max(0.4, 3.2 + (0.428 - tau) * 15.0));
        let savings = Math.min(85.0, (recall / 100.0) * 85.0);

        meterRecall.textContent = recall.toFixed(1) + '%';
        meterPrecision.textContent = precision.toFixed(1) + '%';
        meterFpr.textContent = fpr.toFixed(1) + '%';
        meterSavings.textContent = '₹' + savings.toFixed(1) + ' Cr';

        barRecall.style.width = recall.toFixed(1) + '%';
        barPrecision.style.width = precision.toFixed(1) + '%';
        barFpr.style.width = fpr.toFixed(1) + '%';
        barSavings.style.width = ((savings / 85.0) * 100).toFixed(1) + '%';
    }

    if (slider) {
        slider.addEventListener('input', (e) => {
            updateThresholdMetrics(e.target.value);
        });
    }

    // 3. Real-Time Risk Scorer Form
    const btnScore = document.getElementById('btn-score');
    const resScore = document.getElementById('res-score');
    const expList = document.getElementById('local-exp-list');

    if (btnScore) {
        btnScore.addEventListener('click', () => {
            const amount = parseFloat(document.getElementById('inp-amount').value) || 0;
            const avg = parseFloat(document.getElementById('inp-avg').value) || 1;
            const hour = parseInt(document.getElementById('inp-hour').value) || 0;
            const vel1h = parseInt(document.getElementById('inp-vel1h').value) || 0;
            const dist = parseFloat(document.getElementById('inp-dist').value) || 0;
            const pos = document.getElementById('inp-pos').value;

            const ratio = (amount / avg).toFixed(1);
            const isNight = [22, 23, 0, 1, 2, 3].includes(hour);

            // Heuristic scoring logic simulating XGBoost model + SHAP explanation
            let risk = 15.0;
            if (ratio > 5.0) risk += 35.0;
            if (ratio > 15.0) risk += 25.0;
            if (isNight) risk += 15.0;
            if (vel1h >= 3) risk += 12.0;
            if (dist > 200) risk += 10.0;
            if (pos === 'CNP') risk += 12.0;

            risk = Math.min(99.4, Math.max(2.1, risk));

            resScore.textContent = risk.toFixed(1) + '%';

            expList.innerHTML = `
                <li><strong>Amount Anomaly:</strong> Transaction of ₹${amount.toLocaleString()} is ${ratio}x relative to cardholder's 30-day baseline (₹${avg.toLocaleString()}).</li>
                <li><strong>Temporal Window:</strong> Executed at ${hour}:00 hrs (${isNight ? 'High Risk Night Window' : 'Normal Day Window'}) via ${pos} entry mode.</li>
                <li><strong>Short-term Velocity:</strong> ${vel1h} transaction(s) recorded in the last 1 hour.</li>
                <li><strong>Geographic Location:</strong> Transaction point is ${dist} km away from registered home address.</li>
                <li><strong>Final Verdict:</strong> Probability (${risk.toFixed(1)}%) ${risk >= 42.8 ? '<span style="color:#ef4444;font-weight:700;">EXCEEDS THRESHOLD (&tau;*=0.428) — BLOCK TRANSACTION</span>' : '<span style="color:#10b981;font-weight:700;">BELOW THRESHOLD — APPROVE</span>'}.</li>
            `;
        });
    }

    // 4. Render Charts via Chart.js
    // Chart 1: Models Comparison Bar Chart
    const ctxModels = document.getElementById('chart-models-bar');
    if (ctxModels) {
        new Chart(ctxModels, {
            type: 'bar',
            data: {
                labels: ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR'],
                datasets: [
                    {
                        label: 'Logistic Regression',
                        data: [0.624, 0.782, 0.694, 0.865, 0.712],
                        backgroundColor: 'rgba(148, 163, 184, 0.6)'
                    },
                    {
                        label: 'Random Forest',
                        data: [0.841, 0.812, 0.826, 0.942, 0.887],
                        backgroundColor: 'rgba(59, 130, 246, 0.7)'
                    },
                    {
                        label: 'XGBoost Classifier',
                        data: [0.884, 0.947, 0.914, 0.978, 0.952],
                        backgroundColor: 'rgba(16, 185, 129, 0.85)'
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true, max: 1.0, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { labels: { color: '#f0f4f8' } }
                }
            }
        });
    }

    // Chart 2: EDA Temporal Fraud Rate by Hour
    const ctxHour = document.getElementById('chart-eda-hour');
    if (ctxHour) {
        new Chart(ctxHour, {
            type: 'line',
            data: {
                labels: Array.from({length: 24}, (_, i) => i + ':00'),
                datasets: [{
                    label: 'Fraud Rate (%)',
                    data: [18.2, 22.4, 25.1, 19.8, 12.3, 6.1, 3.2, 2.8, 4.1, 5.2, 6.0, 5.8, 7.1, 6.9, 7.4, 8.2, 9.0, 11.2, 14.5, 16.8, 19.1, 21.5, 23.8, 20.2],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { title: { display: true, text: 'Fraud Rate (%)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { grid: { display: false } }
                },
                plugins: { legend: { labels: { color: '#f0f4f8' } } }
            }
        });
    }

    // Chart 3: EDA POS Entry Mode Fraud Rate
    const ctxPos = document.getElementById('chart-eda-pos');
    if (ctxPos) {
        new Chart(ctxPos, {
            type: 'bar',
            data: {
                labels: ['CNP (Card Not Present)', 'SWIPE', 'CHIP'],
                datasets: [{
                    label: 'Fraud Rate (%)',
                    data: [28.4, 8.7, 3.1],
                    backgroundColor: ['#ef4444', '#f59e0b', '#3b82f6']
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { title: { display: true, text: 'Fraud Rate (%)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { grid: { display: false } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }
});
