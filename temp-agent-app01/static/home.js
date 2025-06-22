// Home page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log('T-Mobile Multi-App Platform loaded successfully!');
    
    // Add click handlers for app cards
    const appCards = document.querySelectorAll('.app-card:not(.coming-soon)');
    
    appCards.forEach(card => {
        card.addEventListener('click', function() {
            // Add a subtle click effect
            this.style.transform = 'scale(0.98)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
        });
    });
    
    // Add hover effects for coming soon cards
    const comingSoonCards = document.querySelectorAll('.app-card.coming-soon');
    
    comingSoonCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.opacity = '0.8';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.opacity = '0.7';
        });
    });
    
    // Platform status check (simulated)
    checkPlatformStatus();
});

function checkPlatformStatus() {
    // Simulate platform status check
    setTimeout(() => {
        const statusElement = document.querySelector('.platform-status');
        if (statusElement) {
            statusElement.innerHTML = `
                <span class="status-icon">🟢</span>
                <span class="status-text">All systems operational - ${new Date().toLocaleTimeString()}</span>
            `;
        }
    }, 1000);
}

// Add smooth scrolling for better UX
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Customer Support Modal Functions
function openSupportModal() {
    const modal = document.getElementById('supportModal');
    modal.style.display = 'block';
    document.body.style.overflow = 'hidden'; // Prevent background scrolling
}

function closeSupportModal() {
    const modal = document.getElementById('supportModal');
    modal.style.display = 'none';
    document.body.style.overflow = 'auto'; // Restore scrolling
}

// Close modal when clicking outside of it
window.onclick = function(event) {
    const modal = document.getElementById('supportModal');
    if (event.target === modal) {
        closeSupportModal();
    }
}

// Close modal with Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeSupportModal();
    }
}); 