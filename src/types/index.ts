export interface Notification {
    id: number;
    type: 'OPEN' | 'CLOSED';
    timestamp: number;
    message: string;
}

export interface NotificationsResponse {
    notifications: Notification[];
    total_count: number;
}

export interface StatsResponse {
    total_gestures: number;
    hand_detected: boolean;
}

export interface ClearSquaresResponse {
    success: boolean;
    message: string;
}

export interface ClearNotificationsResponse {
    success: boolean;
    message: string;
}

