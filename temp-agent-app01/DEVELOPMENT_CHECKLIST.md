# Development Checklist - Code Protection

## Before Making Changes:
- [ ] Identify which feature you're working on
- [ ] Check which JavaScript file contains that feature
- [ ] Verify no other features are in the same file
- [ ] Create a backup of the current working state
- [ ] Check browser console for any existing errors

## During Development:
- [ ] Only modify the specific feature's JavaScript file
- [ ] Test the feature you're working on
- [ ] Test that other features still work
- [ ] Check browser console for any errors
- [ ] Verify CodeProtection system is working

## After Making Changes:
- [ ] Test all features on the page
- [ ] Verify no console errors
- [ ] Check that existing functionality still works
- [ ] Update this checklist if needed
- [ ] Commit changes with descriptive messages

## Emergency Recovery:
If something breaks:
1. Check browser console for errors
2. Look for missing functions
3. Use CodeProtection.emergencyRecovery() if available
4. Restore from backup if needed
5. Check which feature caused the issue

## Feature Isolation Rules:
- ✅ RF AI Agent → rf_ai_agent.js
- ✅ Parameter Search → script.js  
- ✅ KPI Dashboard → kpi_dashboard.js
- ✅ Shared utilities → shared.js

## Testing Checklist:
- [ ] RF AI Agent chat works
- [ ] Parameter search works (LTE & NR)
- [ ] Autocomplete works
- [ ] KPI Dashboard works
- [ ] No JavaScript conflicts
- [ ] All features load independently 