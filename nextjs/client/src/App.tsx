import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider } from "next-themes";
import { AuthProvider } from "@/contexts/AuthContext";
import ProtectedRoute from "@/components/ProtectedRoute";
import OperatorLayout from "@/components/layouts/OperatorLayout";
import AdminLayout from "@/components/layouts/AdminLayout";
import LoginPage from "@/pages/LoginPage";
import OperatorDashboard from "@/pages/operator/OperatorDashboard";
import InverterGridPage from "@/pages/operator/InverterGridPage";
import OperatorAlerts from "@/pages/operator/OperatorAlerts";
import ChatbotPage from "@/pages/operator/ChatbotPage";
import ProfilePage from "@/pages/operator/ProfilePage";
import AdminDashboard from "@/pages/admin/AdminDashboard";
import PlantManagement from "@/pages/admin/PlantManagement";
import PlantDetailPage from "@/pages/admin/PlantDetailPage";
import OperatorManagement from "@/pages/admin/OperatorManagement";
import LiveMonitoring from "@/pages/admin/LiveMonitoring";
import AdminAlerts from "@/pages/admin/AdminAlerts";
import AuditLogs from "@/pages/admin/AuditLogs";
import SettingsPage from "@/pages/admin/SettingsPage";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false}>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Sonner />
        <AuthProvider>
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Navigate to="/login" replace />} />
              <Route path="/login" element={<LoginPage />} />

              <Route path="/operator" element={<ProtectedRoute allowedRoles={['operator']}><OperatorLayout /></ProtectedRoute>}>
                <Route index element={<OperatorDashboard />} />
                <Route path="plant/:plantId/block/:blockId" element={<InverterGridPage />} />
                <Route path="alerts" element={<OperatorAlerts />} />
                <Route path="chatbot" element={<ChatbotPage />} />
                <Route path="profile" element={<ProfilePage />} />
              </Route>

              <Route path="/admin" element={<ProtectedRoute allowedRoles={['admin']}><AdminLayout /></ProtectedRoute>}>
                <Route index element={<AdminDashboard />} />
                <Route path="plants" element={<PlantManagement />} />
                <Route path="plants/:plantId" element={<PlantDetailPage />} />
                <Route path="operators" element={<OperatorManagement />} />
                <Route path="monitoring" element={<LiveMonitoring />} />
                <Route path="alerts" element={<AdminAlerts />} />
                <Route path="audit-logs" element={<AuditLogs />} />
                <Route path="settings" element={<SettingsPage />} />
                <Route path="profile" element={<ProfilePage />} />
              </Route>

              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </AuthProvider>
      </TooltipProvider>
    </QueryClientProvider>
  </ThemeProvider>
);

export default App;
