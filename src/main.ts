import './styles.css';
import HandTrackerApp from './components/App';

// Initialize the app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new HandTrackerApp();
    });
} else {
    new HandTrackerApp();
}


