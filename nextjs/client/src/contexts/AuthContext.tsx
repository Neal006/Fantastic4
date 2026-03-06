import React, { createContext, useContext, useState, useCallback } from 'react';
import { authApi } from '@/lib/api';

export type UserRole = 'admin' | 'operator';

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  assignedPlants: string[];
}

interface AuthContextType {
  user: AuthUser | null;
  login: (userId: string, password: string) => Promise<boolean>;
  logout: () => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(() => {
    const stored = sessionStorage.getItem('sw-user');
    if (stored) {
      try { return JSON.parse(stored); } catch { return null; }
    }
    return null;
  });

  const login = useCallback(async (userId: string, password: string): Promise<boolean> => {
    try {
      const result = await authApi.login(userId, password);
      const u = result.user;
      const authUser: AuthUser = {
        id: u.id,
        name: u.name,
        email: u.email,
        role: u.role,
        assignedPlants: u.assignedPlants || [],
      };
      // Store token and user
      sessionStorage.setItem('sw_token', result.token);
      sessionStorage.setItem('sw-user', JSON.stringify(authUser));
      setUser(authUser);
      return true;
    } catch {
      return false;
    }
  }, []);

  const logout = useCallback(async () => {
    try { await authApi.logout(); } catch { /* ignore */ }
    setUser(null);
    sessionStorage.removeItem('sw_token');
    sessionStorage.removeItem('sw-user');
  }, []);

  return (
    <AuthContext.Provider value={{ user, login, logout, isAuthenticated: !!user }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be inside AuthProvider');
  return ctx;
}
