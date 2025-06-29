// Parameter Search Functionality - Clean Version
// Version: 2.0
// Purpose: LTE and NR parameter search with autocomplete

document.addEventListener('DOMContentLoaded', function() {
    console.log('[Parameter Search] Initializing clean version...');
    
    // Tab switching logic
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => {
                btn.classList.remove('active');
                btn.setAttribute('aria-selected', 'false');
            });
            tabContents.forEach(content => {
                content.classList.remove('active');
                content.style.display = 'none';
            });
            
            // Add active class to clicked button
            this.classList.add('active');
            this.setAttribute('aria-selected', 'true');
            
            // Show corresponding content
            const tab = this.getAttribute('data-tab');
            const content = document.getElementById(tab + '-content');
            if (content) {
                content.classList.add('active');
                content.style.display = '';
            }
            
            console.log('[Tab] Switched to:', tab);
        });
    });

    // Initialize autocomplete
    initializeAutocomplete();

    // LTE Parameter Search Functions
    window.searchLteparManagedObjects = async function() {
        console.log('[LTE] Starting search...');
        const input = document.getElementById('lteparSearchInput');
        const dropdown = document.getElementById('lteparSearchManagedObjectFilter');
        
        if (!input || !dropdown) {
            console.error('[LTE] Required elements not found:', { input: !!input, dropdown: !!dropdown });
            return;
        }
        
        const abbr = input.value.trim();
        console.log('[LTE] Searching for:', abbr);
        
        if (!abbr) {
            console.log('[LTE] Search input is empty.');
            return;
        }
        
        // Clear dropdown
        dropdown.innerHTML = '<option value="">Select Managed Object</option>';
        
        try {
            const url = `/api/ltepar-search-managed-objects?abbreviated_name=${encodeURIComponent(abbr)}`;
            console.log('[LTE] Fetching:', url);
            
            const response = await fetch(url);
            const data = await response.json();
            
            console.log('[LTE] API Response:', data);
            
            if (data.error) {
                console.error('[LTE] API Error:', data.error);
                dropdown.innerHTML = '<option value="">API Error: ' + data.error + '</option>';
                return;
            }
            
            const managedObjects = data.managed_objects || [];
            console.log('[LTE] Found managed objects:', managedObjects);
            
            if (managedObjects.length === 0) {
                dropdown.innerHTML = '<option value="">No managed objects found</option>';
                return;
            }
            
            // Populate dropdown
            managedObjects.forEach(obj => {
                const option = document.createElement('option');
                option.value = obj;
                option.textContent = obj;
                dropdown.appendChild(option);
            });
            
            // Auto-select if only one
            if (managedObjects.length === 1) {
                console.log('[LTE] Auto-selecting single managed object');
                dropdown.value = managedObjects[0];
                fetchAndDisplayLteparParameterDetails();
            }
            
        } catch (error) {
            console.error('[LTE] Fetch error:', error);
            dropdown.innerHTML = '<option value="">Network Error</option>';
        }
    };

    // Fetch and display LTE parameter details
    window.fetchAndDisplayLteparParameterDetails = async function() {
        console.log('[LTE] Fetching parameter details...');
        const input = document.getElementById('lteparSearchInput');
        const dropdown = document.getElementById('lteparSearchManagedObjectFilter');
        const detailsDiv = document.getElementById('lteparParameterDetails');
        
        if (!input || !dropdown || !detailsDiv) {
            console.error('[LTE] Required elements not found for details');
            return;
        }
        
        const abbr = input.value.trim();
        const managedObject = dropdown.value;
        
        if (!abbr || !managedObject) {
            console.log('[LTE] Missing required values:', { abbr, managedObject });
            return;
        }
        
        try {
            const url = `/api/ltepar-parameter-details?abbreviated_name=${encodeURIComponent(abbr)}&managed_object=${encodeURIComponent(managedObject)}`;
            console.log('[LTE] Fetching details:', url);
            
            const response = await fetch(url);
            const data = await response.json();
            
            console.log('[LTE] Details response:', data);
            
            if (data.error) {
                detailsDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                return;
            }
            
            if (!data.details || data.details.length === 0) {
                detailsDiv.innerHTML = '<p>No details found for this combination.</p>';
                return;
            }
            
            // Display as form
            const form = document.createElement('form');
            form.className = 'parameter-form';
            
            const row = data.details[0];
            for (const [key, value] of Object.entries(row)) {
                const group = document.createElement('div');
                group.className = 'form-group';
                group.innerHTML = `<label class="form-label"><strong>${key}</strong></label><div class="form-value">${value}</div>`;
                form.appendChild(group);
            }
            
            detailsDiv.innerHTML = '';
            detailsDiv.appendChild(form);
            
        } catch (error) {
            console.error('[LTE] Details fetch error:', error);
            detailsDiv.innerHTML = '<p style="color: red;">Network Error: Unable to fetch details.</p>';
        }
    };

    // NR Parameter Search Functions
    window.searchNrparManagedObjects = async function() {
        console.log('[NR] Starting search...');
        const input = document.getElementById('nrparSearchInput');
        const dropdown = document.getElementById('nrparSearchManagedObjectFilter');
        
        if (!input || !dropdown) {
            console.error('[NR] Required elements not found');
            return;
        }
        
        const abbr = input.value.trim();
        console.log('[NR] Searching for:', abbr);
        
        if (!abbr) {
            console.log('[NR] Search input is empty.');
            return;
        }
        
        dropdown.innerHTML = '<option value="">Select Managed Object</option>';
        
        try {
            const url = `/api/nrpar-search-managed-objects?abbreviated_name=${encodeURIComponent(abbr)}`;
            console.log('[NR] Fetching:', url);
            
            const response = await fetch(url);
            const data = await response.json();
            
            console.log('[NR] API Response:', data);
            
            const managedObjects = data.managed_objects || [];
            
            if (managedObjects.length === 0) {
                dropdown.innerHTML = '<option value="">No managed objects found</option>';
                return;
            }
            
            managedObjects.forEach(obj => {
                const option = document.createElement('option');
                option.value = obj;
                option.textContent = obj;
                dropdown.appendChild(option);
            });
            
            if (managedObjects.length === 1) {
                console.log('[NR] Auto-selecting single managed object');
                dropdown.value = managedObjects[0];
                fetchAndDisplayNrparParameterDetails();
            }
            
        } catch (error) {
            console.error('[NR] Fetch error:', error);
            dropdown.innerHTML = '<option value="">Network Error</option>';
        }
    };

    // Fetch and display NR parameter details
    window.fetchAndDisplayNrparParameterDetails = async function() {
        console.log('[NR] Fetching parameter details...');
        const input = document.getElementById('nrparSearchInput');
        const dropdown = document.getElementById('nrparSearchManagedObjectFilter');
        const detailsDiv = document.getElementById('nrparParameterDetails');
        
        if (!input || !dropdown || !detailsDiv) {
            console.error('[NR] Required elements not found for details');
            return;
        }
        
        const abbr = input.value.trim();
        const managedObject = dropdown.value;
        
        if (!abbr || !managedObject) {
            console.log('[NR] Missing required values');
            return;
        }
        
        try {
            const url = `/api/nrpar-parameter-details?abbreviated_name=${encodeURIComponent(abbr)}&managed_object=${encodeURIComponent(managedObject)}`;
            console.log('[NR] Fetching details:', url);
            
            const response = await fetch(url);
            const data = await response.json();
            
            console.log('[NR] Details response:', data);
            
            if (!data.details || data.details.length === 0) {
                detailsDiv.innerHTML = '<p>No details found for this combination.</p>';
                return;
            }
            
            const form = document.createElement('form');
            form.className = 'parameter-form';
            
            const row = data.details[0];
            for (const [key, value] of Object.entries(row)) {
                const group = document.createElement('div');
                group.className = 'form-group';
                group.innerHTML = `<label class="form-label"><strong>${key}</strong></label><div class="form-value">${value}</div>`;
                form.appendChild(group);
            }
            
            detailsDiv.innerHTML = '';
            detailsDiv.appendChild(form);
            
        } catch (error) {
            console.error('[NR] Details fetch error:', error);
            detailsDiv.innerHTML = '<p style="color: red;">Network Error: Unable to fetch details.</p>';
        }
    };

    // Event Listeners
    function setupEventListeners() {
        console.log('[Setup] Setting up event listeners...');
        
        // LTE Search Button
        const lteSearchBtn = document.getElementById('lteparSearchButton');
        if (lteSearchBtn) {
            lteSearchBtn.addEventListener('click', searchLteparManagedObjects);
            console.log('[Setup] LTE search button listener added');
        } else {
            console.warn('[Setup] LTE search button not found');
        }
        
        // NR Search Button
        const nrSearchBtn = document.getElementById('nrparSearchButton');
        if (nrSearchBtn) {
            nrSearchBtn.addEventListener('click', searchNrparManagedObjects);
            console.log('[Setup] NR search button listener added');
        } else {
            console.warn('[Setup] NR search button not found');
        }
        
        // LTE Managed Object Dropdown
        const lteDropdown = document.getElementById('lteparSearchManagedObjectFilter');
        if (lteDropdown) {
            lteDropdown.addEventListener('change', fetchAndDisplayLteparParameterDetails);
            console.log('[Setup] LTE dropdown listener added');
        } else {
            console.warn('[Setup] LTE dropdown not found');
        }
        
        // NR Managed Object Dropdown
        const nrDropdown = document.getElementById('nrparSearchManagedObjectFilter');
        if (nrDropdown) {
            nrDropdown.addEventListener('change', fetchAndDisplayNrparParameterDetails);
            console.log('[Setup] NR dropdown listener added');
        } else {
            console.warn('[Setup] NR dropdown not found');
        }
    }

    // Initialize autocomplete
    function initializeAutocomplete() {
        console.log('[Autocomplete] Initializing...');
        
        const lteInput = document.getElementById('lteparSearchInput');
        const nrInput = document.getElementById('nrparSearchInput');
        
        if (lteInput) {
            createAutocomplete(lteInput, 'lte');
            console.log('[Autocomplete] LTE autocomplete created');
        }
        
        if (nrInput) {
            createAutocomplete(nrInput, 'nr');
            console.log('[Autocomplete] NR autocomplete created');
        }
    }

    function createAutocomplete(inputElement, type) {
        let autocompleteDropdown = null;
        let currentFocus = -1;

        function createAutocompleteDropdown() {
            if (autocompleteDropdown) {
                autocompleteDropdown.remove();
            }
            
            autocompleteDropdown = document.createElement('div');
            autocompleteDropdown.className = 'autocomplete-dropdown';
            autocompleteDropdown.style.cssText = `
                position: absolute;
                border: 1px solid #d4d4d4;
                border-top: none;
                z-index: 99;
                top: 100%;
                left: 0;
                right: 0;
                background-color: white;
                max-height: 150px;
                overflow-y: auto;
            `;
            
            inputElement.parentNode.style.position = 'relative';
            inputElement.parentNode.appendChild(autocompleteDropdown);
        }

        async function fetchAutocompleteSuggestions(query) {
            if (query.length < 2) return [];
            
            try {
                const endpoint = type === 'lte' ? '/api/ltepar-abbreviated-autocomplete' : '/api/nrpar-abbreviated-autocomplete';
                const response = await fetch(`${endpoint}?query=${encodeURIComponent(query)}`);
                const data = await response.json();
                return data.abbreviated_names || [];
            } catch (error) {
                console.error(`[Autocomplete ${type.toUpperCase()}] Error:`, error);
                return [];
            }
        }

        function displayAutocompleteSuggestions(suggestions) {
            if (!autocompleteDropdown) {
                createAutocompleteDropdown();
            }
            
            autocompleteDropdown.innerHTML = '';
            
            if (suggestions.length === 0) {
                autocompleteDropdown.style.display = 'none';
                return;
            }
            
            suggestions.forEach((suggestion, index) => {
                const item = document.createElement('div');
                item.className = 'autocomplete-item';
                item.style.cssText = `
                    padding: 10px;
                    cursor: pointer;
                    border-bottom: 1px solid #d4d4d4;
                `;
                item.textContent = suggestion;
                
                item.addEventListener('click', function() {
                    inputElement.value = suggestion;
                    hideAutocomplete();
                    // Trigger search
                    if (type === 'lte') {
                        searchLteparManagedObjects();
                    } else {
                        searchNrparManagedObjects();
                    }
                });
                
                item.addEventListener('mouseenter', function() {
                    removeActiveClass();
                    this.classList.add('autocomplete-active');
                    this.style.backgroundColor = '#e9e9e9';
                });
                
                autocompleteDropdown.appendChild(item);
            });
            
            autocompleteDropdown.style.display = 'block';
        }

        function removeActiveClass() {
            const items = autocompleteDropdown.querySelectorAll('.autocomplete-item');
            items.forEach(item => {
                item.classList.remove('autocomplete-active');
                item.style.backgroundColor = '';
            });
        }

        function hideAutocomplete() {
            if (autocompleteDropdown) {
                autocompleteDropdown.style.display = 'none';
            }
            currentFocus = -1;
        }

        // Event listeners
        inputElement.addEventListener('input', async function() {
            const query = this.value.trim();
            const suggestions = await fetchAutocompleteSuggestions(query);
            displayAutocompleteSuggestions(suggestions);
        });

        inputElement.addEventListener('keydown', function(e) {
            const items = autocompleteDropdown ? autocompleteDropdown.querySelectorAll('.autocomplete-item') : [];
            
            if (e.key === 'ArrowDown') {
                currentFocus++;
                addActiveClass(items);
            } else if (e.key === 'ArrowUp') {
                currentFocus--;
                addActiveClass(items);
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (currentFocus > -1 && items[currentFocus]) {
                    items[currentFocus].click();
                } else {
                    // Trigger search with current input
                    if (type === 'lte') {
                        searchLteparManagedObjects();
                    } else {
                        searchNrparManagedObjects();
                    }
                }
            } else if (e.key === 'Escape') {
                hideAutocomplete();
            }
        });

        function addActiveClass(items) {
            if (!items.length) return;
            
            removeActiveClass();
            
            if (currentFocus >= items.length) currentFocus = 0;
            if (currentFocus < 0) currentFocus = items.length - 1;
            
            items[currentFocus].classList.add('autocomplete-active');
            items[currentFocus].style.backgroundColor = '#e9e9e9';
        }

        // Hide autocomplete when clicking outside
        document.addEventListener('click', function(e) {
            if (!inputElement.contains(e.target) && !autocompleteDropdown?.contains(e.target)) {
                hideAutocomplete();
            }
        });
    }

    // Setup everything
    setupEventListeners();
    
    console.log('[Parameter Search] Initialization complete!');
});