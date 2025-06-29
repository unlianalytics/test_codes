// Parameter Search Functionality
// Version: 1.0
// Purpose: LTE and NR parameter search with autocomplete
// Protected from chat functionality interference

(function() {
    'use strict';
    
    // Register this feature
    if (typeof CodeProtection !== 'undefined') {
        CodeProtection.registerFeature('parameter-search');
    }
    
    document.addEventListener('DOMContentLoaded', function() {
        console.log('[Parameter Search] Initializing...');
        
        // Tab switching logic
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                tabButtons.forEach(btn => {
                    btn.classList.remove('active');
                    btn.setAttribute('aria-selected', 'false');
                });
                tabContents.forEach(content => {
                    content.classList.remove('active');
                    content.style.display = 'none';
                });
                this.classList.add('active');
                this.setAttribute('aria-selected', 'true');
                const tab = this.getAttribute('data-tab');
                const content = document.getElementById(tab + '-content');
                if (content) {
                    content.classList.add('active');
                    content.style.display = '';
                }
                // LTE parameters tab logic
                if (tab === 'ltepar') {
                    console.log('[LTEpar] Tab clicked. Populating Managed Object dropdown...');
                    populateLteparManagedObjects();
                }
                // NR parameters tab logic
                if (tab === 'nrpar') {
                    console.log('[NRpar] Tab clicked. Resetting NR parameter search UI.');
                    resetNrparTab();
                }
            });
        });

        // Initialize autocomplete for both tabs
        initializeAutocomplete();

        // LTEpar Managed Object dropdown population
        window.populateLteparManagedObjects = async function() {
            const dropdown = document.getElementById('lteparManagedObjectFilter');
            const abbrDropdown = document.getElementById('lteparAbbreviatedNameFilter');
            console.log('[LTEpar] Dropdown element:', dropdown);
            if (!dropdown) {
                console.error('[LTEpar] Dropdown element not found!');
                return;
            }
            dropdown.innerHTML = '<option value="">All Managed Objects</option>';
            if (abbrDropdown) {
                abbrDropdown.innerHTML = '<option value="">All Abbreviated Names</option>';
            }
            try {
                console.log('[LTEpar] Fetching /api/ltepar-managed-objects ...');
                const res = await fetch('/api/ltepar-managed-objects');
                const data = await res.json();
                console.log('[LTEpar] API response:', data);
                (data.managed_objects || []).forEach(obj => {
                    const opt = document.createElement('option');
                    opt.value = obj;
                    opt.textContent = obj;
                    dropdown.appendChild(opt);
                    console.log('[LTEpar] Added option:', obj);
                });
                console.log('[LTEpar] Dropdown population complete.');
            } catch (err) {
                console.error('[LTEpar] Error fetching or populating:', err);
                dropdown.innerHTML = '<option value="">API Error</option>';
            }
        };

        // LTEpar Abbreviated Name dropdown population
        async function populateLteparAbbreviatedNames() {
            const managedObject = document.getElementById('lteparManagedObjectFilter').value;
            const abbrDropdown = document.getElementById('lteparAbbreviatedNameFilter');
            if (!abbrDropdown) return;
            abbrDropdown.innerHTML = '<option value="">All Abbreviated Names</option>';
            if (!managedObject) return;
            try {
                console.log('[LTEpar] Fetching /api/ltepar-abbreviated-names?managed_object=' + managedObject);
                const res = await fetch('/api/ltepar-abbreviated-names?managed_object=' + encodeURIComponent(managedObject));
                const data = await res.json();
                console.log('[LTEpar] Abbreviated names API response:', data);
                (data.abbreviated_names || []).forEach(name => {
                    const opt = document.createElement('option');
                    opt.value = name;
                    opt.textContent = name;
                    abbrDropdown.appendChild(opt);
                    console.log('[LTEpar] Added abbreviated name option:', name);
                });
            } catch (err) {
                console.error('[LTEpar] Error fetching abbreviated names:', err);
                abbrDropdown.innerHTML = '<option value="">API Error</option>';
            }
        }

        // LTEpar Parameter Search logic
        window.searchLteparManagedObjects = async function() {
            const input = document.getElementById('lteparSearchInput');
            const dropdown = document.getElementById('lteparSearchManagedObjectFilter');
            const abbr = input.value.trim();
            dropdown.innerHTML = '<option value="">Select Managed Object</option>';
            if (!abbr) {
                console.log('[LTEpar] Search input is empty.');
                return;
            }
            try {
                console.log('[LTEpar] Fetching /api/ltepar-search-managed-objects?abbreviated_name=' + abbr);
                const res = await fetch('/api/ltepar-search-managed-objects?abbreviated_name=' + encodeURIComponent(abbr));
                const data = await res.json();
                console.log('[LTEpar] Search managed objects API response:', data);
                
                const managedObjects = data.managed_objects || [];
                
                if (managedObjects.length === 0) {
                    console.log('[LTEpar] No managed objects found.');
                    dropdown.innerHTML = '<option value="">No managed objects found</option>';
                    return;
                }
                
                // Populate dropdown with managed objects
                managedObjects.forEach(obj => {
                    const opt = document.createElement('option');
                    opt.value = obj;
                    opt.textContent = obj;
                    dropdown.appendChild(opt);
                    console.log('[LTEpar] Added search managed object option:', obj);
                });
                
                // If only one managed object, auto-select it and proceed
                if (managedObjects.length === 1) {
                    console.log('[LTEpar] Single managed object found, auto-selecting and proceeding...');
                    dropdown.value = managedObjects[0];
                    // Trigger the parameter details fetch automatically
                    fetchAndDisplayLteparParameterDetails();
                }
                
            } catch (err) {
                console.error('[LTEpar] Error fetching search managed objects:', err);
                dropdown.innerHTML = '<option value="">API Error</option>';
            }
        };

        // Event listeners
        const lteparManagedObjectFilter = document.getElementById('lteparManagedObjectFilter');
        if (lteparManagedObjectFilter) {
            lteparManagedObjectFilter.addEventListener('change', populateLteparAbbreviatedNames);
        }
        const lteparSearchButton = document.getElementById('lteparSearchButton');
        if (lteparSearchButton) {
            lteparSearchButton.addEventListener('click', searchLteparManagedObjects);
        }
        const lteparSearchManagedObjectFilter = document.getElementById('lteparSearchManagedObjectFilter');
        if (lteparSearchManagedObjectFilter) {
            lteparSearchManagedObjectFilter.addEventListener('change', fetchAndDisplayLteparParameterDetails);
        }

        // Fetch and display parameter details as a form
        window.fetchAndDisplayLteparParameterDetails = async function() {
            const abbrInput = document.getElementById('lteparSearchInput');
            const managedObjectDropdown = document.getElementById('lteparSearchManagedObjectFilter');
            const detailsDiv = document.getElementById('lteparParameterDetails');
            const abbr = abbrInput.value.trim();
            const managedObject = managedObjectDropdown.value;
            detailsDiv.innerHTML = '';
            if (!abbr || !managedObject) {
                console.log('[LTEpar] Abbreviated name or Managed Object not selected.');
                return;
            }
            try {
                console.log(`[LTEpar] Fetching /api/ltepar-parameter-details?abbreviated_name=${abbr}&managed_object=${managedObject}`);
                const res = await fetch(`/api/ltepar-parameter-details?abbreviated_name=${encodeURIComponent(abbr)}&managed_object=${encodeURIComponent(managedObject)}`);
                const data = await res.json();
                console.log('[LTEpar] Parameter details API response:', data);
                if (!data.details || !data.details.length) {
                    detailsDiv.innerHTML = '<p>No details found for this combination.</p>';
                    return;
                }
                // Display as a form
                const form = document.createElement('form');
                form.className = 'parameter-form';
                const row = data.details[0];
                for (const [key, value] of Object.entries(row)) {
                    const group = document.createElement('div');
                    group.className = 'form-group';
                    group.innerHTML = `<label class="form-label"><strong>${key}</strong></label><div class="form-value">${value}</div>`;
                    form.appendChild(group);
                }
                detailsDiv.appendChild(form);
            } catch (err) {
                console.error('[LTEpar] Error fetching parameter details:', err);
                detailsDiv.innerHTML = '<p style="color:red">API Error: Unable to fetch details.</p>';
            }
        };

        // NR parameters: search and display logic
        async function searchNrparManagedObjects() {
            const input = document.getElementById('nrparSearchInput');
            const dropdown = document.getElementById('nrparSearchManagedObjectFilter');
            const abbr = input.value.trim();
            dropdown.innerHTML = '<option value="">Select Managed Object</option>';
            if (!abbr) {
                console.log('[NRpar] Search input is empty.');
                return;
            }
            try {
                console.log('[NRpar] Fetching /api/nrpar-search-managed-objects?abbreviated_name=' + abbr);
                const res = await fetch('/api/nrpar-search-managed-objects?abbreviated_name=' + encodeURIComponent(abbr));
                const data = await res.json();
                console.log('[NRpar] Search managed objects API response:', data);
                
                const managedObjects = data.managed_objects || [];
                
                if (managedObjects.length === 0) {
                    console.log('[NRpar] No managed objects found.');
                    dropdown.innerHTML = '<option value="">No managed objects found</option>';
                    return;
                }
                
                // Populate dropdown with managed objects
                managedObjects.forEach(obj => {
                    const opt = document.createElement('option');
                    opt.value = obj;
                    opt.textContent = obj;
                    dropdown.appendChild(opt);
                    console.log('[NRpar] Added search managed object option:', obj);
                });
                
                // If only one managed object, auto-select it and proceed
                if (managedObjects.length === 1) {
                    console.log('[NRpar] Single managed object found, auto-selecting and proceeding...');
                    dropdown.value = managedObjects[0];
                    // Trigger the parameter details fetch automatically
                    fetchAndDisplayNrparParameterDetails();
                }
                
            } catch (err) {
                console.error('[NRpar] Error fetching search managed objects:', err);
                dropdown.innerHTML = '<option value="">API Error</option>';
            }
        }

        async function fetchAndDisplayNrparParameterDetails() {
            const abbrInput = document.getElementById('nrparSearchInput');
            const managedObjectDropdown = document.getElementById('nrparSearchManagedObjectFilter');
            const detailsDiv = document.getElementById('nrparParameterDetails');
            const abbr = abbrInput.value.trim();
            const managedObject = managedObjectDropdown.value;
            detailsDiv.innerHTML = '';
            if (!abbr || !managedObject) {
                console.log('[NRpar] Abbreviated name or Managed Object not selected.');
                return;
            }
            try {
                console.log(`[NRpar] Fetching /api/nrpar-parameter-details?abbreviated_name=${abbr}&managed_object=${managedObject}`);
                const res = await fetch(`/api/nrpar-parameter-details?abbreviated_name=${encodeURIComponent(abbr)}&managed_object=${encodeURIComponent(managedObject)}`);
                const data = await res.json();
                console.log('[NRpar] Parameter details API response:', data);
                if (!data.details || !data.details.length) {
                    detailsDiv.innerHTML = '<p>No details found for this combination.</p>';
                    return;
                }
                // Display as a form
                const form = document.createElement('form');
                form.className = 'parameter-form';
                const row = data.details[0];
                for (const [key, value] of Object.entries(row)) {
                    const group = document.createElement('div');
                    group.className = 'form-group';
                    group.innerHTML = `<label class="form-label"><strong>${key}</strong></label><div class="form-value">${value}</div>`;
                    form.appendChild(group);
                }
                detailsDiv.appendChild(form);
            } catch (err) {
                console.error('[NRpar] Error fetching parameter details:', err);
                detailsDiv.innerHTML = '<p style="color:red">API Error: Unable to fetch details.</p>';
            }
        }

        // Reset NR parameters tab UI
        function resetNrparTab() {
            document.getElementById('nrparSearchInput').value = '';
            document.getElementById('nrparSearchManagedObjectFilter').innerHTML = '<option value="">Select Managed Object</option>';
            document.getElementById('nrparParameterDetails').innerHTML = '';
        }

        // NR parameters: event listeners
        const nrparSearchButton = document.getElementById('nrparSearchButton');
        if (nrparSearchButton) {
            nrparSearchButton.addEventListener('click', searchNrparManagedObjects);
        }
        const nrparSearchManagedObjectFilter = document.getElementById('nrparSearchManagedObjectFilter');
        if (nrparSearchManagedObjectFilter) {
            nrparSearchManagedObjectFilter.addEventListener('change', fetchAndDisplayNrparParameterDetails);
        }

        // ===== AUTCOMPLETE/TYPEAHEAD FUNCTIONALITY =====
        
        function initializeAutocomplete() {
            // Initialize autocomplete for LTE parameters
            const lteparSearchInput = document.getElementById('lteparSearchInput');
            if (lteparSearchInput) {
                createAutocomplete(lteparSearchInput, 'ltepar');
            }

            // Initialize autocomplete for NR parameters
            const nrparSearchInput = document.getElementById('nrparSearchInput');
            if (nrparSearchInput) {
                createAutocomplete(nrparSearchInput, 'nrpar');
            }
        }

        function createAutocomplete(inputElement, type) {
            let autocompleteList = null;
            let currentFocus = -1;
            let debounceTimer = null;

            // Create autocomplete dropdown
            function createAutocompleteDropdown() {
                if (autocompleteList) {
                    autocompleteList.remove();
                }
                
                autocompleteList = document.createElement('div');
                autocompleteList.className = 'autocomplete-list';
                autocompleteList.style.cssText = `
                    position: absolute;
                    top: 100%;
                    left: 0;
                    right: 0;
                    background: white;
                    border: 2px solid #e20074;
                    border-top: none;
                    border-radius: 0 0 8px 8px;
                    max-height: 200px;
                    overflow-y: auto;
                    z-index: 1000;
                    box-shadow: 0 4px 12px rgba(226, 0, 116, 0.15);
                    display: none;
                `;
                
                inputElement.parentNode.style.position = 'relative';
                inputElement.parentNode.appendChild(autocompleteList);
            }

            // Fetch autocomplete suggestions
            async function fetchAutocompleteSuggestions(query) {
                if (!query || query.length < 2) {
                    hideAutocomplete();
                    return;
                }

                try {
                    const endpoint = type === 'ltepar' 
                        ? '/api/ltepar-abbreviated-autocomplete' 
                        : '/api/nrpar-abbreviated-autocomplete';
                    
                    const response = await fetch(`${endpoint}?query=${encodeURIComponent(query)}`);
                    const data = await response.json();
                    
                    displayAutocompleteSuggestions(data.abbreviated_names || []);
                } catch (error) {
                    console.error(`[${type.toUpperCase()}] Autocomplete error:`, error);
                    hideAutocomplete();
                }
            }

            // Display autocomplete suggestions
            function displayAutocompleteSuggestions(suggestions) {
                if (!autocompleteList) {
                    createAutocompleteDropdown();
                }

                autocompleteList.innerHTML = '';
                
                if (suggestions.length === 0) {
                    hideAutocomplete();
                    return;
                }

                suggestions.slice(0, 10).forEach((suggestion, index) => {
                    const item = document.createElement('div');
                    item.className = 'autocomplete-item';
                    item.style.cssText = `
                        padding: 12px 16px;
                        cursor: pointer;
                        border-bottom: 1px solid #f0f0f0;
                        font-family: 'Roboto', Arial, sans-serif;
                        font-size: 14px;
                        color: #333;
                        transition: background-color 0.2s;
                    `;
                    item.textContent = suggestion;
                    
                    item.addEventListener('mouseenter', () => {
                        removeActiveClass();
                        item.style.backgroundColor = '#fff0fa';
                        currentFocus = index;
                    });
                    
                    item.addEventListener('click', () => {
                        inputElement.value = suggestion;
                        hideAutocomplete();
                        // Trigger search for the selected item
                        if (type === 'ltepar') {
                            searchLteparManagedObjects();
                        } else {
                            searchNrparManagedObjects();
                        }
                    });
                    
                    autocompleteList.appendChild(item);
                });

                autocompleteList.style.display = 'block';
            }

            // Remove active class from all items
            function removeActiveClass() {
                const items = autocompleteList.querySelectorAll('.autocomplete-item');
                items.forEach(item => {
                    item.style.backgroundColor = '';
                });
            }

            // Hide autocomplete dropdown
            function hideAutocomplete() {
                if (autocompleteList) {
                    autocompleteList.style.display = 'none';
                }
                currentFocus = -1;
            }

            // Handle keyboard navigation
            function handleKeydown(e) {
                if (!autocompleteList || autocompleteList.style.display === 'none') {
                    return;
                }

                const items = autocompleteList.querySelectorAll('.autocomplete-item');
                
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    currentFocus++;
                    if (currentFocus >= items.length) currentFocus = 0;
                    addActiveClass(items);
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    currentFocus--;
                    if (currentFocus < 0) currentFocus = items.length - 1;
                    addActiveClass(items);
                } else if (e.key === 'Enter') {
                    e.preventDefault();
                    if (currentFocus > -1 && items[currentFocus]) {
                        items[currentFocus].click();
                    }
                } else if (e.key === 'Escape') {
                    hideAutocomplete();
                }
            }

            // Add active class to current item
            function addActiveClass(items) {
                removeActiveClass();
                if (items[currentFocus]) {
                    items[currentFocus].style.backgroundColor = '#fff0fa';
                }
            }

            // Event listeners
            inputElement.addEventListener('input', (e) => {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(() => {
                    fetchAutocompleteSuggestions(e.target.value);
                }, 300); // 300ms debounce
            });

            inputElement.addEventListener('keydown', handleKeydown);
            inputElement.addEventListener('blur', () => {
                // Delay hiding to allow for clicks on suggestions
                setTimeout(hideAutocomplete, 150);
            });

            inputElement.addEventListener('focus', () => {
                if (inputElement.value.length >= 2) {
                    fetchAutocompleteSuggestions(inputElement.value);
                }
            });

            // Hide autocomplete when clicking outside
            document.addEventListener('click', (e) => {
                if (!inputElement.contains(e.target) && !autocompleteList?.contains(e.target)) {
                    hideAutocomplete();
                }
            });
        }
        
        console.log('[Parameter Search] Initialization complete');
    });
})();