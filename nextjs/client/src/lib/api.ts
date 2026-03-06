const API_BASE = '/api'; // Proxied via Vite to http://localhost:3001

function getToken(): string | null {
    return sessionStorage.getItem('sw_token');
}

async function request<T>(
    method: string,
    path: string,
    body?: unknown,
    options: RequestInit = {}
): Promise<T> {
    const token = getToken();
    const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
    };

    const response = await fetch(`${API_BASE}${path}`, {
        method,
        headers,
        credentials: 'include',
        body: body !== undefined ? JSON.stringify(body) : undefined,
        ...options,
    });

    if (response.status === 401) {
        // Session expired — clear and redirect to login
        sessionStorage.removeItem('sw_token');
        sessionStorage.removeItem('sw-user');
        window.location.href = '/login';
        throw new Error('Session expired');
    }

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
    }

    return data as T;
}

export const api = {
    get: <T>(path: string) => request<T>('GET', path),
    post: <T>(path: string, body?: unknown) => request<T>('POST', path, body),
    put: <T>(path: string, body?: unknown) => request<T>('PUT', path, body),
    delete: <T>(path: string) => request<T>('DELETE', path),
};

// ─── Auth ───────────────────────────────────────────
export const authApi = {
    login: (id: string, password: string) =>
        api.post<{ token: string; user: { id: string; name: string; email: string; role: 'admin' | 'operator'; assignedPlants: string[] } }>('/auth/login', { id, password }),
    logout: () => api.post('/auth/logout'),
    resetPassword: (currentPassword: string, newPassword: string, confirmPassword: string) =>
        api.post('/auth/reset-password', { currentPassword, newPassword, confirmPassword }),
    forceReset: (operator_id: string, new_password: string) =>
        api.post('/auth/force-reset', { operator_id, new_password }),
};

// ─── Operator ───────────────────────────────────────
export const operatorApi = {
    getDashboard: () => api.get<any>('/operator/dashboard'),
    getPlants: () => api.get<any[]>('/operator/plants'),
    getInverterGrid: (plantId: string, blockId: string) =>
        api.get<any[]>(`/operator/plants/${plantId}/blocks/${blockId}/inverters`),
    getInverter: (id: string) => api.get<any>(`/operator/inverters/${id}`),
    getReadings: (id: string, range = '24h') =>
        api.get<any[]>(`/operator/inverters/${id}/readings?range=${range}`),
    getFaults: (id: string) => api.get<any[]>(`/operator/inverters/${id}/faults`),
    getAlerts: () => api.get<any[]>('/operator/alerts'),
    acknowledgeAlert: (id: string) => api.post(`/operator/alerts/${id}/acknowledge`),
};

// ─── Admin ──────────────────────────────────────────
export const adminApi = {
    getDashboard: () => api.get<any>('/admin/dashboard'),

    // Plants
    getPlants: () => api.get<any[]>('/admin/plants'),
    createPlant: (data: any) => api.post('/admin/plants', data),
    updatePlant: (id: string, data: any) => api.put(`/admin/plants/${id}`, data),
    deletePlant: (id: string) => api.delete(`/admin/plants/${id}`),

    // Blocks
    getBlocks: (plantId: string) => api.get<any[]>(`/admin/plants/${plantId}/blocks`),
    createBlock: (plantId: string, name: string) => api.post(`/admin/plants/${plantId}/blocks`, { name }),
    updateBlock: (id: string, name: string) => api.put(`/admin/blocks/${id}`, { name }),
    deleteBlock: (id: string) => api.delete(`/admin/blocks/${id}`),

    // Inverters
    getInverters: (blockId: string) => api.get<any[]>(`/admin/blocks/${blockId}/inverters`),
    createInverter: (blockId: string, data: any) => api.post(`/admin/blocks/${blockId}/inverters`, data),
    updateInverter: (id: string, data: any) => api.put(`/admin/inverters/${id}`, data),
    deleteInverter: (id: string) => api.delete(`/admin/inverters/${id}`),

    // Operators
    getOperators: () => api.get<any[]>('/admin/operators'),
    createOperator: (data: any) => api.post('/admin/operators', data),
    updateOperator: (id: string, data: any) => api.put(`/admin/operators/${id}`, data),
    assignPlants: (id: string, plant_ids: string[]) => api.put(`/admin/operators/${id}/assign-plants`, { plant_ids }),
    deactivateOperator: (id: string) => api.put(`/admin/operators/${id}/deactivate`),
    activateOperator: (id: string) => api.put(`/admin/operators/${id}/activate`),
    deleteOperator: (id: string) => api.delete(`/admin/operators/${id}`),
    resetOperatorPassword: (id: string, new_password: string) =>
        api.post(`/admin/operators/${id}/reset-password`, { new_password }),

    // Alerts, Audit, Settings
    getAlerts: (params?: Record<string, string>) => {
        const qs = params ? '?' + new URLSearchParams(params).toString() : '';
        return api.get<any[]>(`/admin/alerts${qs}`);
    },
    getAuditLogs: (params?: Record<string, string>) => {
        const qs = params ? '?' + new URLSearchParams(params).toString() : '';
        return api.get<any[]>(`/admin/audit-logs${qs}`);
    },
    getSettings: () => api.get<any>('/admin/settings'),
    updateSettings: (data: any) => api.put('/admin/settings', data),
};

// ─── Chatbot ────────────────────────────────────────
export const chatbotApi = {
    query: (message: string, context?: any, conversationHistory?: any[]) =>
        api.post<{ response: string; context_used: boolean }>('/chatbot/query', { message, context, conversationHistory }),
};
