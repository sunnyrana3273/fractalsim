import type { NotificationsResponse, StatsResponse, ClearSquaresResponse, ClearNotificationsResponse } from '../types';

const API_BASE = '/api';

export async function fetchNotifications(): Promise<NotificationsResponse> {
    const response = await fetch(`${API_BASE}/notifications`);
    if (!response.ok) {
        throw new Error('Failed to fetch notifications');
    }
    return response.json();
}

export async function fetchStats(): Promise<StatsResponse> {
    const response = await fetch(`${API_BASE}/stats`);
    if (!response.ok) {
        throw new Error('Failed to fetch stats');
    }
    return response.json();
}

export async function clearSquares(): Promise<ClearSquaresResponse> {
    const response = await fetch(`${API_BASE}/clear_squares`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    });
    if (!response.ok) {
        throw new Error('Failed to clear squares');
    }
    return response.json();
}

export async function clearNotifications(): Promise<ClearNotificationsResponse> {
    const response = await fetch(`${API_BASE}/clear_notifications`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    });
    if (!response.ok) {
        throw new Error('Failed to clear notifications');
    }
    return response.json();
}

