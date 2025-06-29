// Shared utilities and protection system
// Version: 1.0
// Purpose: Protect existing functionality when adding new features

window.CodeProtection = {
    // Track which features are loaded
    loadedFeatures: new Set(),
    
    // Register a feature as loaded
    registerFeature: function(featureName) {
        this.loadedFeatures.add(featureName);
        console.log(`[CodeProtection] Feature registered: ${featureName}`);
    },
    
    // Check if a feature is loaded
    isFeatureLoaded: function(featureName) {
        return this.loadedFeatures.has(featureName);
    },
    
    // Validate critical functions exist
    validateFeature: function(featureName, requiredFunctions) {
        const missing = [];
        requiredFunctions.forEach(func => {
            if (typeof window[func] === 'undefined') {
                missing.push(func);
            }
        });
        
        if (missing.length > 0) {
            console.error(`[CodeProtection] Missing functions for ${featureName}:`, missing);
            return false;
        }
        
        console.log(`[CodeProtection] Feature ${featureName} validation passed`);
        return true;
    },
    
    // Backup critical functions
    backupFunction: function(functionName, originalFunction) {
        if (typeof window[functionName] !== 'undefined') {
            window[`${functionName}_BACKUP`] = window[functionName];
            console.log(`[CodeProtection] Backed up function: ${functionName}`);
        }
        window[functionName] = originalFunction;
    },
    
    // Restore backed up function
    restoreFunction: function(functionName) {
        if (typeof window[`${functionName}_BACKUP`] !== 'undefined') {
            window[functionName] = window[`${functionName}_BACKUP`];
            console.log(`[CodeProtection] Restored function: ${functionName}`);
        }
    },
    
    // Emergency recovery - restore all backed up functions
    emergencyRecovery: function() {
        console.log('[CodeProtection] Starting emergency recovery...');
        const backupFunctions = Object.keys(window).filter(key => key.endsWith('_BACKUP'));
        backupFunctions.forEach(backupKey => {
            const originalKey = backupKey.replace('_BACKUP', '');
            this.restoreFunction(originalKey);
        });
        console.log('[CodeProtection] Emergency recovery completed');
    }
};

// Global error handler to catch missing functions
window.addEventListener('error', function(e) {
    console.error('[CodeProtection] JavaScript error detected:', e.error);
    console.error('[CodeProtection] Error location:', e.filename, 'line:', e.lineno);
    
    // If it's a missing function error, try to restore
    if (e.error && e.error.message && e.error.message.includes('is not defined')) {
        console.warn('[CodeProtection] Attempting to restore missing function...');
        CodeProtection.emergencyRecovery();
    }
});

// Page load validation
document.addEventListener('DOMContentLoaded', function() {
    console.log('[CodeProtection] Page loaded, validating features...');
    
    // Check which page we're on and validate accordingly
    const currentPage = window.location.pathname;
    
    if (currentPage.includes('parameter-search')) {
        CodeProtection.registerFeature('parameter-search');
        CodeProtection.validateFeature('parameter-search', [
            'populateLteparManagedObjects',
            'searchLteparManagedObjects',
            'fetchAndDisplayLteparParameterDetails'
        ]);
    }
    
    if (currentPage.includes('rf-ai-agent')) {
        CodeProtection.registerFeature('rf-ai-agent');
        // Validate chat functions exist
        CodeProtection.validateFeature('rf-ai-agent', [
            'sendMessage',
            'addMessageToChat'
        ]);
    }
    
    if (currentPage.includes('kpi-dashboard')) {
        CodeProtection.registerFeature('kpi-dashboard');
        CodeProtection.validateFeature('kpi-dashboard', [
            'showKpiDetails',
            'closeKpiModal',
            'createKpiChart'
        ]);
    }
    
    console.log('[CodeProtection] Feature validation complete');
});

console.log('[CodeProtection] Protection system loaded'); 