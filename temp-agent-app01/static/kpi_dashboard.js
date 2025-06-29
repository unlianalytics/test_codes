document.addEventListener('DOMContentLoaded', function() {
    console.log('[KPI Dashboard] Initializing...');
    
    // Initialize dashboard
    initializeDashboard();
    
    // Update data every 30 seconds
    setInterval(updateDashboardData, 30000);
    
    function initializeDashboard() {
        console.log('[KPI Dashboard] Loading initial data...');
        
        // Update last update time
        updateLastUpdate();
        
        // Load data points count
        loadDataPointsCount();
        
        // Initialize with sample data (replace with real API calls)
        updateKpiValues();
        
        console.log('[KPI Dashboard] Initialization complete');
    }
    
    function updateLastUpdate() {
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        document.getElementById('lastUpdate').textContent = timeString;
    }
    
    function loadDataPointsCount() {
        // This would typically come from your backend
        // For now, using a sample value
        document.getElementById('dataPoints').textContent = '2,847';
    }
    
    function updateKpiValues() {
        // This would typically fetch real data from your backend
        // For now, using sample data with some randomization
        
        // Signal Quality
        document.getElementById('rsrp-value').textContent = `${-85 + Math.random() * 10} dBm`;
        document.getElementById('sinr-value').textContent = `${18 + Math.random() * 5} dB`;
        document.getElementById('rsrq-value').textContent = `${-8 + Math.random() * 3} dB`;
        
        // Coverage
        document.getElementById('coverage-area').textContent = `${98 + Math.random() * 2}%`;
        document.getElementById('avg-distance').textContent = `${1.2 + Math.random() * 0.5} km`;
        document.getElementById('cell-range').textContent = `${2.8 + Math.random() * 0.4} km`;
        
        // Throughput
        document.getElementById('avg-speed').textContent = `${45 + Math.random() * 15} Mbps`;
        document.getElementById('peak-speed').textContent = `${78 + Math.random() * 20} Mbps`;
        document.getElementById('bandwidth').textContent = '20 MHz';
        
        // Interference
        document.getElementById('interference-ratio').textContent = `${12 + Math.random() * 8}%`;
        document.getElementById('noise-floor').textContent = `${-95 + Math.random() * 5} dBm`;
        document.getElementById('sir-value').textContent = `${15 + Math.random() * 5} dB`;
        
        // Mobility
        document.getElementById('handover-success').textContent = `${99 + Math.random() * 1}%`;
        document.getElementById('location-updates').textContent = `${1245 + Math.random() * 200}/hr`;
        document.getElementById('cell-changes').textContent = `${89 + Math.random() * 20}/hr`;
        
        // Capacity
        document.getElementById('load-value').textContent = `${67 + Math.random() * 15}%`;
        document.getElementById('utilization').textContent = `${72 + Math.random() * 10}%`;
        document.getElementById('congestion').textContent = `${8 + Math.random() * 5}%`;
    }
    
    function updateDashboardData() {
        console.log('[KPI Dashboard] Updating data...');
        updateKpiValues();
        updateLastUpdate();
    }
});

// Global functions for modal handling
let currentChart = null;

function showKpiDetails(category) {
    console.log('[KPI Dashboard] Showing details for:', category);
    
    const modal = document.getElementById('kpiModal');
    const modalTitle = document.getElementById('modalTitle');
    const kpiDetails = document.getElementById('kpiDetails');
    
    // Set modal title based on category
    const titles = {
        'signal-quality': 'Signal Quality KPIs',
        'coverage': 'Coverage KPIs',
        'throughput': 'Throughput KPIs',
        'interference': 'Interference KPIs',
        'mobility': 'Mobility KPIs',
        'capacity': 'Capacity KPIs'
    };
    
    modalTitle.textContent = titles[category] || 'KPI Details';
    
    // Show modal
    modal.style.display = 'flex';
    
    // Create chart
    createKpiChart(category);
    
    // Load detailed information
    loadKpiDetails(category);
}

function closeKpiModal() {
    console.log('[KPI Dashboard] Closing modal');
    
    const modal = document.getElementById('kpiModal');
    modal.style.display = 'none';
    
    // Destroy current chart if it exists
    if (currentChart) {
        currentChart.destroy();
        currentChart = null;
    }
}

function createKpiChart(category) {
    const ctx = document.getElementById('kpiChart').getContext('2d');
    
    // Destroy existing chart
    if (currentChart) {
        currentChart.destroy();
    }
    
    // Chart configuration based on category
    const chartConfig = getChartConfig(category);
    
    currentChart = new Chart(ctx, chartConfig);
}

function getChartConfig(category) {
    const baseConfig = {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'KPI Trends'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    };
    
    // Generate sample data based on category
    const timeLabels = [];
    const now = new Date();
    for (let i = 23; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 60 * 60 * 1000);
        timeLabels.push(time.getHours() + ':00');
    }
    
    baseConfig.data.labels = timeLabels;
    
    switch (category) {
        case 'signal-quality':
            baseConfig.data.datasets = [
                {
                    label: 'RSRP (dBm)',
                    data: generateRandomData(-90, -75, 24),
                    borderColor: '#e20074',
                    backgroundColor: 'rgba(226, 0, 116, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'SINR (dB)',
                    data: generateRandomData(15, 25, 24),
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    tension: 0.4
                }
            ];
            break;
            
        case 'coverage':
            baseConfig.data.datasets = [
                {
                    label: 'Coverage Area (%)',
                    data: generateRandomData(95, 100, 24),
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    tension: 0.4
                }
            ];
            break;
            
        case 'throughput':
            baseConfig.data.datasets = [
                {
                    label: 'Average Speed (Mbps)',
                    data: generateRandomData(40, 60, 24),
                    borderColor: '#ffc107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Peak Speed (Mbps)',
                    data: generateRandomData(70, 90, 24),
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    tension: 0.4
                }
            ];
            break;
            
        case 'interference':
            baseConfig.data.datasets = [
                {
                    label: 'Interference Ratio (%)',
                    data: generateRandomData(8, 18, 24),
                    borderColor: '#6f42c1',
                    backgroundColor: 'rgba(111, 66, 193, 0.1)',
                    tension: 0.4
                }
            ];
            break;
            
        case 'mobility':
            baseConfig.data.datasets = [
                {
                    label: 'Handover Success Rate (%)',
                    data: generateRandomData(98, 100, 24),
                    borderColor: '#17a2b8',
                    backgroundColor: 'rgba(23, 162, 184, 0.1)',
                    tension: 0.4
                }
            ];
            break;
            
        case 'capacity':
            baseConfig.data.datasets = [
                {
                    label: 'Load (%)',
                    data: generateRandomData(60, 80, 24),
                    borderColor: '#fd7e14',
                    backgroundColor: 'rgba(253, 126, 20, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Utilization (%)',
                    data: generateRandomData(65, 85, 24),
                    borderColor: '#20c997',
                    backgroundColor: 'rgba(32, 201, 151, 0.1)',
                    tension: 0.4
                }
            ];
            break;
    }
    
    return baseConfig;
}

function generateRandomData(min, max, count) {
    const data = [];
    for (let i = 0; i < count; i++) {
        data.push(min + Math.random() * (max - min));
    }
    return data;
}

function loadKpiDetails(category) {
    const kpiDetails = document.getElementById('kpiDetails');
    
    const details = {
        'signal-quality': `
            <h3>Signal Quality Analysis</h3>
            <p><strong>RSRP (Reference Signal Received Power):</strong> Measures the power of the reference signal received by the UE. Values above -85 dBm indicate good signal strength.</p>
            <p><strong>SINR (Signal-to-Interference-plus-Noise Ratio):</strong> Indicates the quality of the signal. Values above 15 dB represent excellent signal quality.</p>
            <p><strong>RSRQ (Reference Signal Received Quality):</strong> Measures the quality of the received reference signal. Values above -10 dB indicate good signal quality.</p>
        `,
        'coverage': `
            <h3>Coverage Analysis</h3>
            <p><strong>Coverage Area:</strong> Percentage of the target area with adequate signal strength. Values above 95% indicate excellent coverage.</p>
            <p><strong>Average Distance:</strong> Mean distance between user equipment and serving cell. Lower values indicate better coverage density.</p>
            <p><strong>Cell Range:</strong> Maximum effective range of the cell. Important for network planning and optimization.</p>
        `,
        'throughput': `
            <h3>Throughput Analysis</h3>
            <p><strong>Average Speed:</strong> Mean data transfer rate experienced by users. Critical for user experience assessment.</p>
            <p><strong>Peak Speed:</strong> Maximum achievable data transfer rate under optimal conditions.</p>
            <p><strong>Bandwidth:</strong> Available frequency spectrum for data transmission.</p>
        `,
        'interference': `
            <h3>Interference Analysis</h3>
            <p><strong>Interference Ratio:</strong> Percentage of interference affecting signal quality. Lower values are better.</p>
            <p><strong>Noise Floor:</strong> Background noise level in the system. Critical for signal quality assessment.</p>
            <p><strong>SIR (Signal-to-Interference Ratio):</strong> Ratio of desired signal power to interference power.</p>
        `,
        'mobility': `
            <h3>Mobility Analysis</h3>
            <p><strong>Handover Success Rate:</strong> Percentage of successful handovers between cells. Critical for mobile user experience.</p>
            <p><strong>Location Updates:</strong> Frequency of location registration updates. Indicates network activity level.</p>
            <p><strong>Cell Changes:</strong> Number of cell transitions per hour. Important for mobility management.</p>
        `,
        'capacity': `
            <h3>Capacity Analysis</h3>
            <p><strong>Load:</strong> Current network load percentage. Values above 80% may indicate capacity issues.</p>
            <p><strong>Utilization:</strong> Resource utilization rate. Important for capacity planning.</p>
            <p><strong>Congestion:</strong> Percentage of time the network experiences congestion. Lower values are better.</p>
        `
    };
    
    kpiDetails.innerHTML = details[category] || '<p>No detailed information available for this category.</p>';
}

// Close modal when clicking outside
document.addEventListener('click', function(e) {
    const modal = document.getElementById('kpiModal');
    if (e.target === modal) {
        closeKpiModal();
    }
});

// Close modal with Escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeKpiModal();
    }
}); 