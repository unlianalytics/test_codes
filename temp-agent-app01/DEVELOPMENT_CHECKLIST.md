# Development Checklist - Code Protection & Best Practices

## Before Making Changes:
- [ ] Identify which feature you're working on
- [ ] Check which JavaScript file contains that feature
- [ ] Verify no other features are in the same file
- [ ] Create a backup of the current working state
- [ ] Check browser console for any existing errors
- [ ] **NEW**: Verify API endpoints match between frontend and backend
- [ ] **NEW**: Check if CSV files are properly loaded and accessible

## During Development:
- [ ] Only modify the specific feature's JavaScript file
- [ ] Test the feature you're working on
- [ ] Test that other features still work
- [ ] Check browser console for any errors
- [ ] **UPDATED**: Use clean, linear code structure (avoid complex nesting)
- [ ] **NEW**: Add comprehensive console logging for debugging
- [ ] **NEW**: Implement defensive programming (check elements exist)
- [ ] **NEW**: Test API endpoints directly via browser URL

## After Making Changes:
- [ ] Test all features on the page
- [ ] Verify no console errors
- [ ] Check that existing functionality still works
- [ ] Update this checklist if needed
- [ ] Commit changes with descriptive messages
- [ ] **NEW**: Test both LTE and NR parameter search functionality
- [ ] **NEW**: Verify autocomplete/typeahead features work

## Emergency Recovery:
If something breaks:
1. **UPDATED**: Check browser console for detailed error messages
2. **UPDATED**: Look for missing DOM elements or API endpoints
3. **NEW**: Test API endpoints directly (e.g., `/api/debug-csv-structure`)
4. **NEW**: Clear browser cache and restart server
5. **NEW**: Check CSV file structure and column names
6. Use CodeProtection.emergencyRecovery() if available
7. Restore from backup if needed
8. Check which feature caused the issue

## Feature Isolation Rules:
- ✅ RF AI Agent → rf_ai_agent.js
- ✅ Parameter Search → script.js  
- ✅ KPI Dashboard → kpi_dashboard.js
- ✅ Shared utilities → shared.js

## API Endpoint Reference:
### Parameter Search Endpoints:
- **LTE**: `/api/ltepar-search-managed-objects`, `/api/ltepar-parameter-details`, `/api/ltepar-abbreviated-autocomplete`
- **NR**: `/api/nrpar-search-managed-objects`, `/api/nrpar-parameter-details`, `/api/nrpar-abbreviated-autocomplete`
- **Debug**: `/api/debug-csv-structure`, `/api/test-lte-search`, `/api/clear-lte-cache`

### Chat Endpoints:
- **Main Chat**: `/chat`
- **Test Chat**: `/test-chat`
- **Simple Chat**: `/simple-chat`

## CSV Structure Requirements:
- **Required Columns**: "Abbreviated name", "Managed object"
- **Encoding**: latin1
- **Location**: `csv_files/Parameters_LTE.csv`, `csv_files/Parameters_NR.csv`
- **Validation**: Use `/api/debug-csv-structure` to verify

## Testing Checklist:
- [ ] RF AI Agent chat works with "Thinking, partner..." animation
- [ ] Parameter search works (LTE & NR tabs)
- [ ] Autocomplete/typeahead works for both LTE and NR
- [ ] Single managed object auto-selects and proceeds
- [ ] KPI Dashboard works (when activated)
- [ ] No JavaScript conflicts between features
- [ ] All features load independently
- [ ] **NEW**: API endpoints return correct data format
- [ ] **NEW**: CSV files load without errors
- [ ] **NEW**: Console shows proper debug messages

## Common Issues & Solutions:
### API Errors:
- **Check**: API endpoint URLs match exactly
- **Check**: CSV column names are correct ("Abbreviated name", "Managed object")
- **Check**: CSV files exist and are readable
- **Debug**: Use `/api/debug-csv-structure` endpoint

### JavaScript Errors:
- **Check**: DOM elements exist before using them
- **Check**: No conflicts with other JavaScript files
- **Check**: Event listeners are properly attached
- **Debug**: Check browser console for detailed error messages

### CSV Loading Issues:
- **Check**: File paths are correct
- **Check**: File encoding is latin1
- **Check**: Column names match exactly (case-sensitive)
- **Debug**: Use debug endpoints to verify structure

## Code Quality Standards:
- [ ] Use defensive programming (check elements exist)
- [ ] Add comprehensive console logging
- [ ] Implement proper error handling
- [ ] Use clean, linear code structure
- [ ] Avoid complex nested functions
- [ ] Test API endpoints independently
- [ ] Verify CSV data structure

## Recent Fixes Applied:
- ✅ **Fixed LTE parameter search API errors** - Corrected endpoint URLs
- ✅ **Fixed NR parameter search responsiveness** - Cleaned up JavaScript structure
- ✅ **Added comprehensive logging** - Better debugging capabilities
- ✅ **Implemented defensive programming** - Prevents null reference errors
- ✅ **Removed CodeProtection conflicts** - Simplified code architecture
- ✅ **Added auto-selection for single results** - Better UX
- ✅ **Fixed CSV column name handling** - Robust column mapping

## Next Steps:
- [ ] Activate KPI Dashboard with card/grid buttons
- [ ] Implement additional parameter search features
- [ ] Add more robust error handling
- [ ] Consider implementing feature flags for easier testing 