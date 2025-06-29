document.addEventListener('DOMContentLoaded', function() {
    console.log('[KPI Dashboard] Initializing enhanced version with AI analysis...');
    
    // Initialize dashboard
    initializeDashboard();
    
    function initializeDashboard() {
        updateDashboardStats();
        setInterval(updateDashboardStats, 30000); // Update every 30 seconds
    }
    
    function updateDashboardStats() {
        const now = new Date();
        document.getElementById('lastUpdate').textContent = 'Just now';
        document.getElementById('dataPoints').textContent = Math.floor(Math.random() * 1000) + 500;
    }
    
    // Enhanced KPI Details Modal with AI Analysis
    window.showKpiDetails = async function(category) {
        console.log('[KPI] Showing details for:', category);
        
        const modal = document.getElementById('kpiModal');
        const modalTitle = document.getElementById('modalTitle');
        const kpiDetails = document.getElementById('kpiDetails');
        
        // Set modal title
        const titles = {
            'signal-quality': 'Signal Quality Analytics',
            'coverage': 'Coverage Analysis',
            'throughput': 'Throughput Performance',
            'interference': 'Interference Analysis',
            'mobility': 'Mobility Performance',
            'capacity': 'Capacity Utilization'
        };
        
        modalTitle.textContent = titles[category] || 'KPI Details';
        
        // Show modal
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        
        // Create enhanced chart
        createEnhancedChart(category);
        
        // Populate details with loading state
        populateKpiDetails(category);
        
        // Generate AI analysis
        await generateAIAnalysis(category);
    };
    
    window.closeKpiModal = function() {
        const modal = document.getElementById('kpiModal');
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
        
        // Destroy existing chart
        const canvas = document.getElementById('kpiChart');
        if (canvas && window.currentChart) {
            window.currentChart.destroy();
            window.currentChart = null;
        }
    };
    
    // AI Analysis Generation
    async function generateAIAnalysis(category) {
        const analysisContainer = document.getElementById('aiAnalysis');
        if (!analysisContainer) return;
        
        // Show loading state
        analysisContainer.innerHTML = `
            <div class="ai-analysis-loading">
                <div class="loading-spinner"></div>
                <p>🤖 AI Agent is analyzing your ${category.replace('-', ' ')} data...</p>
            </div>
        `;
        
        try {
            // Get current KPI data for context
            const kpiData = getCurrentKpiData(category);
            
            // Create AI prompt
            const prompt = createAIPrompt(category, kpiData);
            
            console.log('[AI] Sending analysis request for:', category);
            
            // Call AI endpoint
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `message=${encodeURIComponent(prompt)}`
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.response) {
                // Display AI analysis
                analysisContainer.innerHTML = `
                    <div class="ai-analysis-content">
                        <div class="ai-header">
                            <span class="ai-icon">🤖</span>
                            <span class="ai-title">AI Analysis</span>
                        </div>
                        <div class="ai-text">
                            ${formatAIAnalysis(data.response)}
                        </div>
                        <div class="ai-footer">
                            <span class="ai-timestamp">Generated just now</span>
                            <button class="refresh-analysis-btn" onclick="refreshAIAnalysis('${category}')">
                                🔄 Refresh Analysis
                            </button>
                        </div>
                    </div>
                `;
            } else {
                throw new Error('No response from AI');
            }
            
        } catch (error) {
            console.error('[AI] Analysis error:', error);
            analysisContainer.innerHTML = `
                <div class="ai-analysis-error">
                    <div class="error-icon">⚠️</div>
                    <p>Unable to generate AI analysis at this time.</p>
                    <button class="retry-btn" onclick="refreshAIAnalysis('${category}')">
                        🔄 Retry Analysis
                    </button>
                </div>
            `;
        }
    }
    
    // Refresh AI Analysis
    window.refreshAIAnalysis = async function(category) {
        await generateAIAnalysis(category);
    };
    
    function getCurrentKpiData(category) {
        // Get current values from the dashboard
        const data = {};
        
        switch(category) {
            case 'signal-quality':
                data.rsrp = document.getElementById('rsrp-value')?.textContent || '-85 dBm';
                data.sinr = document.getElementById('sinr-value')?.textContent || '18.5 dB';
                data.rsrq = document.getElementById('rsrq-value')?.textContent || '-8.2 dB';
                break;
            case 'coverage':
                data.coverageArea = document.getElementById('coverage-area')?.textContent || '98.5%';
                data.avgDistance = document.getElementById('avg-distance')?.textContent || '1.2 km';
                data.cellRange = document.getElementById('cell-range')?.textContent || '2.8 km';
                break;
            case 'throughput':
                data.avgSpeed = document.getElementById('avg-speed')?.textContent || '45.2 Mbps';
                data.peakSpeed = document.getElementById('peak-speed')?.textContent || '78.9 Mbps';
                data.bandwidth = document.getElementById('bandwidth')?.textContent || '20 MHz';
                break;
            case 'interference':
                data.interferenceRatio = document.getElementById('interference-ratio')?.textContent || '12.3%';
                data.noiseFloor = document.getElementById('noise-floor')?.textContent || '-95 dBm';
                data.sir = document.getElementById('sir-value')?.textContent || '15.2 dB';
                break;
            case 'mobility':
                data.handoverSuccess = document.getElementById('handover-success')?.textContent || '99.2%';
                data.locationUpdates = document.getElementById('location-updates')?.textContent || '1,245/hr';
                data.cellChanges = document.getElementById('cell-changes')?.textContent || '89/hr';
                break;
            case 'capacity':
                data.load = document.getElementById('load-value')?.textContent || '67.8%';
                data.utilization = document.getElementById('utilization')?.textContent || '72.3%';
                data.congestion = document.getElementById('congestion')?.textContent || '8.5%';
                break;
        }
        
        return data;
    }
    
    function createAIPrompt(category, kpiData) {
        const prompts = {
            'signal-quality': `As an RF engineering expert, analyze the current signal quality metrics for our LTE network:
- RSRP: ${kpiData.rsrp}
- SINR: ${kpiData.sinr}
- RSRQ: ${kpiData.rsrq}

Please provide a comprehensive analysis including:
1. Overall signal quality assessment
2. Performance trends and implications
3. Potential issues or areas of concern
4. Recommendations for optimization
5. Comparison with industry benchmarks

Format your response as a professional analysis report suitable for network engineers.`,
            
            'coverage': `As an RF engineering expert, analyze the current coverage performance for our LTE network:
- Coverage Area: ${kpiData.coverageArea}
- Average Distance: ${kpiData.avgDistance}
- Cell Range: ${kpiData.cellRange}

Please provide a comprehensive analysis including:
1. Overall coverage assessment
2. Coverage gaps and weak areas
3. Cell planning implications
4. Recommendations for coverage improvement
5. Indoor vs outdoor coverage considerations

Format your response as a professional analysis report suitable for network engineers.`,
            
            'throughput': `As an RF engineering expert, analyze the current throughput performance for our LTE network:
- Average Speed: ${kpiData.avgSpeed}
- Peak Speed: ${kpiData.peakSpeed}
- Bandwidth: ${kpiData.bandwidth}

Please provide a comprehensive analysis including:
1. Overall throughput performance assessment
2. Speed distribution and user experience
3. Capacity utilization analysis
4. Bottlenecks and optimization opportunities
5. Recommendations for performance improvement

Format your response as a professional analysis report suitable for network engineers.`,
            
            'interference': `As an RF engineering expert, analyze the current interference situation for our LTE network:
- Interference Ratio: ${kpiData.interferenceRatio}
- Noise Floor: ${kpiData.noiseFloor}
- SIR: ${kpiData.sir}

Please provide a comprehensive analysis including:
1. Overall interference assessment
2. Sources of interference
3. Impact on network performance
4. Interference mitigation strategies
5. Recommendations for interference reduction

Format your response as a professional analysis report suitable for network engineers.`,
            
            'mobility': `As an RF engineering expert, analyze the current mobility performance for our LTE network:
- Handover Success Rate: ${kpiData.handoverSuccess}
- Location Updates: ${kpiData.locationUpdates}
- Cell Changes: ${kpiData.cellChanges}

Please provide a comprehensive analysis including:
1. Overall mobility performance assessment
2. Handover efficiency analysis
3. Mobility patterns and trends
4. Potential mobility issues
5. Recommendations for mobility optimization

Format your response as a professional analysis report suitable for network engineers.`,
            
            'capacity': `As an RF engineering expert, analyze the current capacity utilization for our LTE network:
- Load: ${kpiData.load}
- Utilization: ${kpiData.utilization}
- Congestion: ${kpiData.congestion}

Please provide a comprehensive analysis including:
1. Overall capacity assessment
2. Resource utilization patterns
3. Congestion analysis and impact
4. Capacity planning considerations
5. Recommendations for capacity optimization

Format your response as a professional analysis report suitable for network engineers.`
        };
        
        return prompts[category] || prompts['signal-quality'];
    }
    
    function formatAIAnalysis(text) {
        // Convert plain text to formatted HTML with proper paragraphs
        return text
            .split('\n\n')
            .map(paragraph => `<p>${paragraph.trim()}</p>`)
            .join('');
    }
    
    function createEnhancedChart(category) {
        const canvas = document.getElementById('kpiChart');
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart
        if (window.currentChart) {
            window.currentChart.destroy();
        }
        
        // Generate enhanced data based on category
        const chartData = generateChartData(category);
        
        // Enhanced chart configuration
        window.currentChart = new Chart(ctx, {
            type: chartData.type,
            data: chartData.data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: chartData.title,
                        font: {
                            size: 18,
                            weight: 'bold'
                        },
                        color: '#333'
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            padding: 20,
                            font: {
                                size: 14
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: '#e20074',
                        borderWidth: 2,
                        cornerRadius: 8,
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y + (chartData.unit || '');
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: chartData.xAxisLabel,
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: chartData.yAxisLabel,
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 6,
                        hoverRadius: 8
                    },
                    line: {
                        borderWidth: 3
                    },
                    bar: {
                        borderWidth: 2
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }
    
    function generateChartData(category) {
        const timeLabels = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'];
        
        switch(category) {
            case 'signal-quality':
                return {
                    type: 'line',
                    title: 'Signal Quality Metrics Over Time',
                    xAxisLabel: 'Time (24h)',
                    yAxisLabel: 'Signal Strength (dBm/dB)',
                    unit: '',
                    data: {
                        labels: timeLabels,
                        datasets: [
                            {
                                label: 'RSRP',
                                data: [-75, -78, -82, -85, -88, -83, -77],
                                borderColor: '#e20074',
                                backgroundColor: 'rgba(226, 0, 116, 0.1)',
                                fill: true,
                                tension: 0.4
                            },
                            {
                                label: 'SINR',
                                data: [22, 20, 18, 16, 14, 17, 19],
                                borderColor: '#28a745',
                                backgroundColor: 'rgba(40, 167, 69, 0.1)',
                                fill: true,
                                tension: 0.4
                            },
                            {
                                label: 'RSRQ',
                                data: [-6, -7, -8, -9, -10, -8, -7],
                                borderColor: '#007bff',
                                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                                fill: true,
                                tension: 0.4
                            }
                        ]
                    }
                };
                
            case 'coverage':
                return {
                    type: 'bar',
                    title: 'Coverage Performance Metrics',
                    xAxisLabel: 'Time Periods',
                    yAxisLabel: 'Coverage Percentage (%)',
                    unit: '%',
                    data: {
                        labels: timeLabels,
                        datasets: [
                            {
                                label: 'Coverage Area',
                                data: [99.2, 98.8, 98.5, 98.2, 98.0, 98.3, 98.7],
                                backgroundColor: 'rgba(226, 0, 116, 0.8)',
                                borderColor: '#e20074',
                                borderWidth: 2
                            },
                            {
                                label: 'Cell Range',
                                data: [95.5, 94.8, 94.2, 93.8, 93.5, 94.0, 94.5],
                                backgroundColor: 'rgba(40, 167, 69, 0.8)',
                                borderColor: '#28a745',
                                borderWidth: 2
                            }
                        ]
                    }
                };
                
            case 'throughput':
                return {
                    type: 'line',
                    title: 'Throughput Performance Analysis',
                    xAxisLabel: 'Time (24h)',
                    yAxisLabel: 'Speed (Mbps)',
                    unit: ' Mbps',
                    data: {
                        labels: timeLabels,
                        datasets: [
                            {
                                label: 'Average Speed',
                                data: [52, 48, 45, 42, 38, 44, 49],
                                borderColor: '#e20074',
                                backgroundColor: 'rgba(226, 0, 116, 0.1)',
                                fill: true,
                                tension: 0.4
                            },
                            {
                                label: 'Peak Speed',
                                data: [85, 82, 78, 75, 72, 76, 80],
                                borderColor: '#ffc107',
                                backgroundColor: 'rgba(255, 193, 7, 0.1)',
                                fill: true,
                                tension: 0.4
                            }
                        ]
                    }
                };
                
            case 'interference':
                return {
                    type: 'radar',
                    title: 'Interference Analysis Dashboard',
                    xAxisLabel: '',
                    yAxisLabel: '',
                    unit: '',
                    data: {
                        labels: ['Interference Ratio', 'Noise Floor', 'SIR', 'C/I Ratio', 'Adjacent Channel', 'Co-channel'],
                        datasets: [
                            {
                                label: 'Current Values',
                                data: [12.3, 85, 15.2, 18.5, 8.7, 11.2],
                                borderColor: '#e20074',
                                backgroundColor: 'rgba(226, 0, 116, 0.2)',
                                pointBackgroundColor: '#e20074',
                                pointBorderColor: '#fff',
                                pointHoverBackgroundColor: '#fff',
                                pointHoverBorderColor: '#e20074'
                            },
                            {
                                label: 'Target Values',
                                data: [8.0, 90, 20.0, 25.0, 5.0, 8.0],
                                borderColor: '#28a745',
                                backgroundColor: 'rgba(40, 167, 69, 0.2)',
                                pointBackgroundColor: '#28a745',
                                pointBorderColor: '#fff',
                                pointHoverBackgroundColor: '#fff',
                                pointHoverBorderColor: '#28a745'
                            }
                        ]
                    }
                };
                
            case 'mobility':
                return {
                    type: 'doughnut',
                    title: 'Mobility Performance Distribution',
                    xAxisLabel: '',
                    yAxisLabel: '',
                    unit: '',
                    data: {
                        labels: ['Successful Handovers', 'Failed Handovers', 'Location Updates', 'Cell Changes'],
                        datasets: [
                            {
                                data: [99.2, 0.8, 85.5, 14.5],
                                backgroundColor: [
                                    '#28a745',
                                    '#dc3545',
                                    '#007bff',
                                    '#ffc107'
                                ],
                                borderColor: [
                                    '#1e7e34',
                                    '#c82333',
                                    '#0056b3',
                                    '#e0a800'
                                ],
                                borderWidth: 3
                            }
                        ]
                    }
                };
                
            case 'capacity':
                return {
                    type: 'bar',
                    title: 'Capacity Utilization Analysis',
                    xAxisLabel: 'Time Periods',
                    yAxisLabel: 'Utilization Percentage (%)',
                    unit: '%',
                    data: {
                        labels: timeLabels,
                        datasets: [
                            {
                                label: 'Load',
                                data: [65, 68, 72, 75, 78, 74, 70],
                                backgroundColor: 'rgba(226, 0, 116, 0.8)',
                                borderColor: '#e20074',
                                borderWidth: 2
                            },
                            {
                                label: 'Utilization',
                                data: [70, 73, 76, 79, 82, 78, 74],
                                backgroundColor: 'rgba(40, 167, 69, 0.8)',
                                borderColor: '#28a745',
                                borderWidth: 2
                            },
                            {
                                label: 'Congestion',
                                data: [5, 7, 9, 12, 15, 11, 8],
                                backgroundColor: 'rgba(220, 53, 69, 0.8)',
                                borderColor: '#dc3545',
                                borderWidth: 2
                            }
                        ]
                    }
                };
                
            default:
                return {
                    type: 'line',
                    title: 'KPI Performance',
                    xAxisLabel: 'Time',
                    yAxisLabel: 'Value',
                    unit: '',
                    data: {
                        labels: timeLabels,
                        datasets: [{
                            label: 'Performance',
                            data: [80, 85, 82, 88, 90, 87, 85],
                            borderColor: '#e20074',
                            backgroundColor: 'rgba(226, 0, 116, 0.1)',
                            fill: true
                        }]
                    }
                };
        }
    }
    
    function populateKpiDetails(category) {
        const kpiDetails = document.getElementById('kpiDetails');
        
        const details = {
            'signal-quality': {
                title: 'Signal Quality Metrics',
                items: [
                    { label: 'Average RSRP', value: '-85 dBm', status: 'good' },
                    { label: 'Average SINR', value: '18.5 dB', status: 'excellent' },
                    { label: 'Average RSRQ', value: '-8.2 dB', status: 'good' },
                    { label: 'Signal Stability', value: '95.2%', status: 'excellent' },
                    { label: 'Weak Signal Areas', value: '2.3%', status: 'warning' },
                    { label: 'Strong Signal Areas', value: '78.9%', status: 'excellent' }
                ]
            },
            'coverage': {
                title: 'Coverage Analysis',
                items: [
                    { label: 'Total Coverage Area', value: '98.5%', status: 'excellent' },
                    { label: 'Average Cell Range', value: '2.8 km', status: 'good' },
                    { label: 'Coverage Gaps', value: '1.5%', status: 'warning' },
                    { label: 'Indoor Coverage', value: '92.3%', status: 'good' },
                    { label: 'Outdoor Coverage', value: '99.1%', status: 'excellent' },
                    { label: 'Edge Coverage', value: '85.7%', status: 'good' }
                ]
            },
            'throughput': {
                title: 'Throughput Performance',
                items: [
                    { label: 'Average Speed', value: '45.2 Mbps', status: 'good' },
                    { label: 'Peak Speed', value: '78.9 Mbps', status: 'excellent' },
                    { label: 'Bandwidth Utilization', value: '72.3%', status: 'good' },
                    { label: 'Latency', value: '15.2 ms', status: 'excellent' },
                    { label: 'Packet Loss', value: '0.3%', status: 'excellent' },
                    { label: 'Jitter', value: '2.1 ms', status: 'good' }
                ]
            },
            'interference': {
                title: 'Interference Analysis',
                items: [
                    { label: 'Interference Ratio', value: '12.3%', status: 'warning' },
                    { label: 'Noise Floor', value: '-95 dBm', status: 'good' },
                    { label: 'Signal-to-Interference', value: '15.2 dB', status: 'good' },
                    { label: 'Carrier-to-Interference', value: '18.5 dB', status: 'excellent' },
                    { label: 'Adjacent Channel', value: '8.7%', status: 'good' },
                    { label: 'Co-channel', value: '11.2%', status: 'warning' }
                ]
            },
            'mobility': {
                title: 'Mobility Performance',
                items: [
                    { label: 'Handover Success Rate', value: '99.2%', status: 'excellent' },
                    { label: 'Location Updates', value: '1,245/hr', status: 'good' },
                    { label: 'Cell Changes', value: '89/hr', status: 'good' },
                    { label: 'Handover Delay', value: '45 ms', status: 'excellent' },
                    { label: 'Failed Handovers', value: '0.8%', status: 'good' },
                    { label: 'Mobility Efficiency', value: '94.7%', status: 'excellent' }
                ]
            },
            'capacity': {
                title: 'Capacity Utilization',
                items: [
                    { label: 'Current Load', value: '67.8%', status: 'good' },
                    { label: 'Utilization Rate', value: '72.3%', status: 'good' },
                    { label: 'Congestion Level', value: '8.5%', status: 'warning' },
                    { label: 'Available Capacity', value: '27.7%', status: 'good' },
                    { label: 'Peak Usage', value: '89.2%', status: 'warning' },
                    { label: 'Capacity Efficiency', value: '91.5%', status: 'excellent' }
                ]
            }
        };
        
        const categoryDetails = details[category] || details['signal-quality'];
        
        kpiDetails.innerHTML = `
            <h3>${categoryDetails.title}</h3>
            <div class="kpi-details-grid">
                ${categoryDetails.items.map(item => `
                    <div class="kpi-detail-item">
                        <div class="kpi-detail-label">${item.label}</div>
                        <div class="kpi-detail-value" style="color: ${getStatusColor(item.status)}">${item.value}</div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    function getStatusColor(status) {
        const colors = {
            'excellent': '#007bff',
            'good': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545'
        };
        return colors[status] || '#333';
    }
    
    // Close modal when clicking outside
    document.getElementById('kpiModal').addEventListener('click', function(e) {
        if (e.target === this) {
            closeKpiModal();
        }
    });
    
    // Close modal with Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeKpiModal();
        }
    });
    
    console.log('[KPI Dashboard] Enhanced initialization with AI analysis complete!');
}); 