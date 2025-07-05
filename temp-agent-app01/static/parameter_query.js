// Parameter Query JavaScript - Following Alarm Checker Pattern
let currentData = [];
let managedObjects = [];
let currentArea = 'houston';

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    console.log('[Parameter Query] Initializing...');
    loadSiteNames();
    loadManagedObjects();
    setupEventListeners();
});

// Tab switching function (same as Alarm Checker)
function showTab(tabId) {
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    
    if(tabId === 'houston') {
        document.querySelectorAll('.tab-button')[0].classList.add('active');
        document.getElementById('houston').classList.add('active');
        currentArea = 'houston';
    } else if(tabId === 'dallas') {
        document.querySelectorAll('.tab-button')[1].classList.add('active');
        document.getElementById('dallas').classList.add('active');
        currentArea = 'dallas';
    } else if(tabId === 'austin') {
        document.querySelectorAll('.tab-button')[2].classList.add('active');
        document.getElementById('austin').classList.add('active');
        currentArea = 'austin';
    } else if(tabId === 'area-x') {
        document.querySelectorAll('.tab-button')[3].classList.add('active');
        document.getElementById('area-x').classList.add('active');
        currentArea = 'area-x';
    }
    
    // Clear results when switching tabs
    hideResults();
    console.log('[Parameter Query] Switched to area:', currentArea);
}

// Load site names from the same CSV as Alarm Checker
let allSites = [];
async function loadSiteNames() {
    try {
        const response = await fetch('/api/parameter-site-names');
        const data = await response.json();
        
        allSites = data.sites || [];
        console.log('[Parameter Query] Loaded', allSites.length, 'sites');
    } catch (error) {
        console.error('[Parameter Query] Error loading site names:', error);
        showError('Failed to load site names. Please refresh the page.');
    }
}

// Load managed objects from the CSV mapping
async function loadManagedObjects() {
    try {
        const response = await fetch('/api/parameter-managed-objects');
        const data = await response.json();
        
        managedObjects = data.managed_objects;
        const managedObjectSelect = document.getElementById('managed-object-select');
        managedObjectSelect.innerHTML = '<option value="">Select a managed object...</option>';
        
        managedObjects.forEach(mo => {
            const option = document.createElement('option');
            option.value = mo.managed_object;
            option.textContent = `${mo.managed_object} - ${mo.query_description}`;
            managedObjectSelect.appendChild(option);
        });
        
        console.log('[Parameter Query] Loaded', managedObjects.length, 'managed objects');
    } catch (error) {
        console.error('[Parameter Query] Error loading managed objects:', error);
        showError('Failed to load managed objects. Please refresh the page.');
    }
}

// Setup event listeners
function setupEventListeners() {
    const siteInput = document.getElementById('site-input');
    const managedObjectSelect = document.getElementById('managed-object-select');
    const ossIpInput = document.getElementById('oss-ip-input');
    const autocompleteList = document.getElementById('site-autocomplete');
    
    // Autocomplete functionality
    siteInput.addEventListener('input', function() {
        const query = siteInput.value.trim();
        if (query.length === 0) {
            autocompleteList.style.display = 'none';
            return;
        }
        const matches = filterSites(query);
        showAutocompleteList(matches);
    });
    
    siteInput.addEventListener('blur', function() {
        setTimeout(() => { autocompleteList.style.display = 'none'; }, 150);
    });
    
    siteInput.addEventListener('focus', function() {
        const query = siteInput.value.trim();
        if (query.length > 0) {
            const matches = filterSites(query);
            showAutocompleteList(matches);
        }
    });
    
    // Update OSS IP when site is selected
    siteInput.addEventListener('change', function() {
        updateOssIp();
        updateFetchButton();
    });
    
    // Enable/disable fetch button based on selections
    function updateFetchButton() {
        const fetchButton = document.getElementById('fetch-parameter-btn');
        const siteSelected = siteInput.value !== '';
        const managedObjectSelected = managedObjectSelect.value !== '';
        
        fetchButton.disabled = !(siteSelected && managedObjectSelected);
    }
    
    managedObjectSelect.addEventListener('change', updateFetchButton);
}

// Filter sites for autocomplete
function filterSites(query) {
    return allSites.filter(site => 
        site.site_name.toLowerCase().includes(query.toLowerCase())
    );
}

// Show autocomplete list
function showAutocompleteList(matches) {
    const list = document.getElementById('site-autocomplete');
    list.innerHTML = '';
    
    if (matches.length === 0) {
        list.style.display = 'none';
        return;
    }
    
    matches.forEach(site => {
        const item = document.createElement('li');
        item.className = 'autocomplete-item';
        item.textContent = site.site_name;
        item.onclick = function() {
            document.getElementById('site-input').value = site.site_name;
            list.style.display = 'none';
            updateOssIp();
            updateFetchButton();
        };
        list.appendChild(item);
    });
    
    list.style.display = 'block';
}

// Update OSS IP based on selected site
function updateOssIp() {
    const siteInput = document.getElementById('site-input');
    const ossIpInput = document.getElementById('oss-ip-input');
    const selectedSiteName = siteInput.value.trim();
    
    const selectedSite = allSites.find(site => site.site_name === selectedSiteName);
    if (selectedSite) {
        ossIpInput.value = selectedSite.oss_ip_address;
    } else {
        ossIpInput.value = '';
    }
}

// Update fetch button state
function updateFetchButton() {
    const fetchButton = document.getElementById('fetch-parameter-btn');
    const siteInput = document.getElementById('site-input');
    const managedObjectSelect = document.getElementById('managed-object-select');
    
    const siteSelected = siteInput.value !== '';
    const managedObjectSelected = managedObjectSelect.value !== '';
    
    fetchButton.disabled = !(siteSelected && managedObjectSelected);
}

// Fetch parameter data from Oracle
async function fetchParameterData() {
    const siteInput = document.getElementById('site-input');
    const managedObjectSelect = document.getElementById('managed-object-select');
    const ossIpInput = document.getElementById('oss-ip-input');
    
    const siteName = siteInput.value;
    const managedObject = managedObjectSelect.value;
    const ossIpAddress = ossIpInput.value;
    
    if (!siteName || !managedObject || !ossIpAddress) {
        showError('Please select both a site and managed object.');
        return;
    }
    
    // Show loading state
    showLoading(true);
    hideError();
    hideResults();
    
    try {
        const response = await fetch('/api/fetch-parameter-data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                site_name: siteName,
                managed_object: managedObject,
                oss_ip_address: ossIpAddress
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            if (data.success) {
                currentData = data.data || [];
                displayResults(currentData);
                console.log(`[Parameter Query] Fetched ${currentData.length} records for ${currentArea}`);
            } else {
                showError(data.error || 'Failed to fetch parameter data.');
            }
        } else {
            showError(data.error || 'Failed to fetch parameter data.');
        }
    } catch (error) {
        console.error('[Parameter Query] Error fetching data:', error);
        showError('Network error. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Display results in table
function displayResults(data) {
    const resultsSection = document.getElementById('parameter-results-section');
    const resultsContainer = document.getElementById('parameter-results-container');
    
    if (!data || data.length === 0) {
        resultsSection.style.display = 'block';
        resultsContainer.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #666;">
                <i class="fas fa-info-circle" style="font-size: 3rem; margin-bottom: 20px; color: #e20074;"></i>
                <p>No parameter data found for the selected criteria.</p>
            </div>
        `;
        return;
    }
    
    // Generate table headers from first row
    const headers = Object.keys(data[0]);
    
    // Create table HTML
    let tableHTML = `
        <table class="parameter-table">
            <thead>
                <tr>
    `;
    
    headers.forEach(header => {
        tableHTML += `<th>${header.replace(/_/g, ' ').toUpperCase()}</th>`;
    });
    
    tableHTML += `
                </tr>
            </thead>
            <tbody>
    `;
    
    // Generate table rows
    data.forEach(row => {
        tableHTML += '<tr>';
        headers.forEach(header => {
            tableHTML += `<td>${row[header] || ''}</td>`;
        });
        tableHTML += '</tr>';
    });
    
    tableHTML += `
            </tbody>
        </table>
    `;
    
    // Show results
    resultsSection.style.display = 'block';
    resultsContainer.innerHTML = tableHTML;
}

// Download parameter data to CSV
function downloadParameterCSV() {
    if (!currentData || currentData.length === 0) {
        showError('No data to export.');
        return;
    }
    
    try {
        // Get headers from first row
        const headers = Object.keys(currentData[0]);
        
        // Create CSV content
        let csvContent = headers.join(',') + '\n';
        
        currentData.forEach(row => {
            const values = headers.map(header => {
                const value = row[header] || '';
                // Escape commas and quotes in CSV
                if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                    return `"${value.replace(/"/g, '""')}"`;
                }
                return value;
            });
            csvContent += values.join(',') + '\n';
        });
        
        // Create and download file
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `parameter_data_${currentArea}_${new Date().toISOString().slice(0, 10)}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        console.log(`[Parameter Query] Exported ${currentData.length} records for ${currentArea}`);
    } catch (error) {
        console.error('[Parameter Query] Error exporting CSV:', error);
        showError('Failed to export CSV file.');
    }
}

// Show/hide loading spinner
function showLoading(show) {
    const fetchButton = document.getElementById('fetch-parameter-btn');
    
    if (show) {
        fetchButton.disabled = true;
        fetchButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Fetching...';
    } else {
        fetchButton.disabled = false;
        fetchButton.innerHTML = '<i class="fas fa-search"></i> Fetch Parameter Data';
    }
}

// Show error message
function showError(message) {
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

// Hide error message
function hideError() {
    const errorSection = document.getElementById('errorSection');
    errorSection.style.display = 'none';
}

// Hide results
function hideResults() {
    const resultsSection = document.getElementById('parameter-results-section');
    resultsSection.style.display = 'none';
}

console.log('[Parameter Query] Script loaded successfully'); 