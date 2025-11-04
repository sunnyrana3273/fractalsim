// Notification state
let lastNotificationId = 0;
let notifications = [];

// DOM elements
const notificationsContainer = document.getElementById('notifications-container');
const totalCountElement = document.getElementById('total-count');
const handStatusElement = document.getElementById('hand-status');
const statusIndicator = document.getElementById('status-indicator');
const clearSquaresBtn = document.getElementById('clear-squares-btn');

// Fetch notifications from server
async function fetchNotifications() {
    try {
        const response = await fetch('/api/notifications');
        const data = await response.json();
        
        // Update total count
        totalCountElement.textContent = data.total_count;
        
        // Check for new notifications
        if (data.notifications && data.notifications.length > 0) {
            const latestNotification = data.notifications[data.notifications.length - 1];
            
            // Only add if it's a new notification
            if (latestNotification.id > lastNotificationId) {
                addNotification(latestNotification);
                lastNotificationId = latestNotification.id;
                
                // Update hand status
                updateHandStatus(latestNotification.type);
            }
        }
    } catch (error) {
        console.error('Error fetching notifications:', error);
    }
}

// Add notification to UI
function addNotification(notification) {
    // Remove "no notifications" message if present
    const noNotificationsMsg = notificationsContainer.querySelector('.no-notifications');
    if (noNotificationsMsg) {
        noNotificationsMsg.remove();
    }
    
    // Create notification element
    const notificationElement = document.createElement('div');
    notificationElement.className = `notification-item hand-${notification.type.toLowerCase()}`;
    
    const timeString = new Date(notification.timestamp * 1000).toLocaleTimeString();
    
    notificationElement.innerHTML = `
        <div class="notification-header">
            <div>
                <span class="notification-number">#${notification.id}</span>
                <span class="notification-type">Hand ${notification.type}</span>
            </div>
            <span class="notification-time">${timeString}</span>
        </div>
        <div class="notification-message">${notification.message}</div>
    `;
    
    // Add to top of container
    notificationsContainer.insertBefore(notificationElement, notificationsContainer.firstChild);
    
    // Keep only last 10 notifications visible
    const notificationItems = notificationsContainer.querySelectorAll('.notification-item');
    if (notificationItems.length > 10) {
        notificationItems[notificationItems.length - 1].remove();
    }
    
    // Animate in
    notificationElement.style.animation = 'slideIn 0.3s ease-out';
}

// Update hand status indicator
function updateHandStatus(status) {
    handStatusElement.textContent = `Hand ${status}`;
    statusIndicator.className = 'status-indicator';
    
    if (status === 'OPEN') {
        statusIndicator.classList.add('hand-open');
    } else if (status === 'CLOSED') {
        statusIndicator.classList.add('hand-closed');
    }
}

// Poll for updates every 500ms
setInterval(fetchNotifications, 500);

// Initial fetch
fetchNotifications();

// Handle camera feed errors
const cameraFeed = document.getElementById('camera-feed');
cameraFeed.addEventListener('error', () => {
    cameraFeed.src = '/video_feed?' + new Date().getTime();
});

// Clear squares button
clearSquaresBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/api/clear_squares', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        if (data.success) {
            // Visual feedback - briefly change button style
            clearSquaresBtn.textContent = '✓ Cleared!';
            clearSquaresBtn.classList.add('cleared');
            
            setTimeout(() => {
                clearSquaresBtn.textContent = 'Clear Squares';
                clearSquaresBtn.classList.remove('cleared');
            }, 1500);
        }
    } catch (error) {
        console.error('Error clearing squares:', error);
        alert('Failed to clear squares. Please try again.');
    }
});

