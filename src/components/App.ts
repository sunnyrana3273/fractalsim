import type { Notification } from '../types';
import { fetchNotifications, clearSquares, clearNotifications, toggleShapeMode, getShapeMode } from '../utils/api';

class HandTrackerApp {
    private lastNotificationId = 0;
    private notifications: Notification[] = [];
    
    private notificationsContainer!: HTMLElement;
    private totalCountElement!: HTMLElement;
    private handStatusElement!: HTMLElement;
    private statusIndicator!: HTMLElement;
    private clearSquaresBtn!: HTMLButtonElement;
    private clearNotificationsBtn!: HTMLButtonElement;
    private toggleShapeBtn!: HTMLButtonElement;
    private cameraFeed!: HTMLImageElement;
    private currentShapeMode: string = 'square';
    
    constructor() {
        this.initialize();
    }
    
    private initialize(): void {
        this.createHTML();
        this.attachEventListeners();
        this.startPolling();
        this.handleCameraFeedErrors();
    }
    
    private createHTML(): void {
        const app = document.getElementById('app');
        if (!app) throw new Error('App element not found');
        
        app.innerHTML = `
            <div class="container">
                <header>
                    <h1>👋 jorkin it</h1>
                    <div class="stats">
                        <div class="stat-item">
                            <span class="stat-label">Total Gestures:</span>
                            <span class="stat-value" id="total-count">0</span>
                        </div>
                        <button id="toggle-shape-btn" class="clear-btn">Square</button>
                        <button id="clear-squares-btn" class="clear-btn">Clear</button>
                    </div>
                </header>
                
                <div class="main-content">
                    <div class="camera-section">
                        <div class="camera-container">
                            <img id="camera-feed" src="/video_feed" alt="Camera Feed">
                            <div class="camera-overlay">
                                <div class="status-indicator" id="status-indicator">
                                    <span id="hand-status">Waiting for hand...</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="notifications-section">
                        <div class="notifications-header-section">
                            <h2>Recent Gestures</h2>
                            <button id="clear-notifications-btn" class="clear-notifications-btn">Clear</button>
                        </div>
                        <div class="notifications-container" id="notifications-container">
                            <p class="no-notifications">No gestures detected yet. Open and close your hand to see notifications here!</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Get DOM elements
        this.notificationsContainer = document.getElementById('notifications-container')!;
        this.totalCountElement = document.getElementById('total-count')!;
        this.handStatusElement = document.getElementById('hand-status')!;
        this.statusIndicator = document.getElementById('status-indicator')!;
        this.clearSquaresBtn = document.getElementById('clear-squares-btn') as HTMLButtonElement;
        this.clearNotificationsBtn = document.getElementById('clear-notifications-btn') as HTMLButtonElement;
        this.toggleShapeBtn = document.getElementById('toggle-shape-btn') as HTMLButtonElement;
        this.cameraFeed = document.getElementById('camera-feed') as HTMLImageElement;
        
        // Load initial shape mode
        this.loadShapeMode();
    }
    
    private attachEventListeners(): void {
        this.clearSquaresBtn.addEventListener('click', () => this.handleClearSquares());
        this.clearNotificationsBtn.addEventListener('click', () => this.handleClearNotifications());
        this.toggleShapeBtn.addEventListener('click', () => this.handleToggleShape());
    }
    
    private async loadShapeMode(): Promise<void> {
        try {
            const data = await getShapeMode();
            this.currentShapeMode = data.shape_mode;
            this.updateToggleButton();
        } catch (error) {
            console.error('Error loading shape mode:', error);
        }
    }
    
    private updateToggleButton(): void {
        this.toggleShapeBtn.textContent = this.currentShapeMode === 'circle' ? 'Circle' : 'Square';
    }
    
    private async handleToggleShape(): Promise<void> {
        try {
            const data = await toggleShapeMode();
            if (data.success) {
                this.currentShapeMode = data.shape_mode;
                this.updateToggleButton();
            }
        } catch (error) {
            console.error('Error toggling shape mode:', error);
        }
    }
    
    private async fetchNotifications(): Promise<void> {
        try {
            const data = await fetchNotifications();
            
            // Update total count
            this.totalCountElement.textContent = data.total_count.toString();
            
            // Check for new notifications
            if (data.notifications && data.notifications.length > 0) {
                const latestNotification = data.notifications[data.notifications.length - 1];
                
                // Only add if it's a new notification
                if (latestNotification.id > this.lastNotificationId) {
                    this.addNotification(latestNotification);
                    this.lastNotificationId = latestNotification.id;
                    
                    // Update hand status
                    this.updateHandStatus(latestNotification.type);
                }
            }
        } catch (error) {
            console.error('Error fetching notifications:', error);
        }
    }
    
    private addNotification(notification: Notification): void {
        // Remove "no notifications" message if present
        const noNotificationsMsg = this.notificationsContainer.querySelector('.no-notifications');
        if (noNotificationsMsg) {
            noNotificationsMsg.remove();
        }
        
        // Create notification element
        const notificationElement = document.createElement('div');
        notificationElement.className = `notification-item hand-${notification.type.toLowerCase()}`;
        
        notificationElement.innerHTML = `
            <div class="notification-header">
                <div>
                    <span class="notification-number">#${notification.id}</span>
                    <span class="notification-type">Hand ${notification.type}</span>
                </div>
            </div>
            <div class="notification-message">${notification.message}</div>
        `;
        
        // Add to top of container
        this.notificationsContainer.insertBefore(notificationElement, this.notificationsContainer.firstChild);
        
        // Keep only last 10 notifications visible
        const notificationItems = this.notificationsContainer.querySelectorAll('.notification-item');
        if (notificationItems.length > 10) {
            notificationItems[notificationItems.length - 1].remove();
        }
        
        // Animate in
        notificationElement.style.animation = 'slideIn 0.3s ease-out';
    }
    
    private updateHandStatus(status: 'OPEN' | 'CLOSED'): void {
        this.handStatusElement.textContent = `Hand ${status}`;
        this.statusIndicator.className = 'status-indicator';
        
        if (status === 'OPEN') {
            this.statusIndicator.classList.add('hand-open');
        } else if (status === 'CLOSED') {
            this.statusIndicator.classList.add('hand-closed');
        }
    }
    
    private async handleClearSquares(): Promise<void> {
        try {
            const data = await clearSquares();
            if (data.success) {
                // Visual feedback - briefly change button style
                this.clearSquaresBtn.textContent = '✓ Cleared!';
                this.clearSquaresBtn.classList.add('cleared');
                
                setTimeout(() => {
                    this.clearSquaresBtn.textContent = 'Clear Squares';
                    this.clearSquaresBtn.classList.remove('cleared');
                }, 1500);
            }
        } catch (error) {
            console.error('Error clearing squares:', error);
            alert('Failed to clear squares. Please try again.');
        }
    }
    
    private async handleClearNotifications(): Promise<void> {
        try {
            const data = await clearNotifications();
            if (data.success) {
                // Clear notifications from UI
                this.notificationsContainer.innerHTML = '<p class="no-notifications">No gestures detected yet. Open and close your hand to see notifications here!</p>';
                
                // Reset notification tracking
                this.lastNotificationId = 0;
                this.notifications = [];
                
                // Visual feedback - briefly change button style
                const originalText = this.clearNotificationsBtn.textContent;
                this.clearNotificationsBtn.textContent = '✓ Cleared';
                this.clearNotificationsBtn.classList.add('cleared');
                
                setTimeout(() => {
                    this.clearNotificationsBtn.textContent = originalText;
                    this.clearNotificationsBtn.classList.remove('cleared');
                }, 1500);
            }
        } catch (error) {
            console.error('Error clearing notifications:', error);
            alert('Failed to clear notifications. Please try again.');
        }
    }
    
    private handleCameraFeedErrors(): void {
        this.cameraFeed.addEventListener('error', () => {
            this.cameraFeed.src = `/video_feed?${new Date().getTime()}`;
        });
    }
    
    private startPolling(): void {
        // Initial fetch
        this.fetchNotifications();
        
        // Poll for updates every 500ms
        setInterval(() => this.fetchNotifications(), 500);
    }
}

export default HandTrackerApp;

