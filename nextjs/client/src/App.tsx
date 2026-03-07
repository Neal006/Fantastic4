import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { ThemeProvider } from "next-themes";
import { AuthProvider } from "@/contexts/AuthContext";
import ProtectedRoute from "@/components/ProtectedRoute";
import OperatorLayout from "@/components/layouts/OperatorLayout";
import AdminLayout from "@/components/layouts/AdminLayout";
import LoginPage from "@/pages/LoginPage";
import LandingPage from "@/pages/LandingPage";
import OperatorDashboard from "@/pages/operator/OperatorDashboard";
import InverterGridPage from "@/pages/operator/InverterGridPage";
import OperatorAlerts from "@/pages/operator/OperatorAlerts";
import ChatbotPage from "@/pages/operator/ChatbotPage";
import MLPredictPage from "@/pages/operator/MLPredictPage";
import ProfilePage from "@/pages/operator/ProfilePage";
import AdminDashboard from "@/pages/admin/AdminDashboard";
import PlantManagement from "@/pages/admin/PlantManagement";
import PlantDetailPage from "@/pages/admin/PlantDetailPage";
import OperatorManagement from "@/pages/admin/OperatorManagement";
import LiveMonitoring from "@/pages/admin/LiveMonitoring";
import AdminAlerts from "@/pages/admin/AdminAlerts";
import AuditLogs from "@/pages/admin/AuditLogs";
import SettingsPage from "@/pages/admin/SettingsPage";
import ImpactPage from "@/pages/admin/ImpactPage";
import NotFound from "./pages/NotFound";
import { AnimatePresence, motion } from "framer-motion";

const queryClient = new QueryClient();

const PageWrapper = ({ children }: { children: React.ReactNode }) => (
  <motion.div
    initial={{ opacity: 0, y: 16 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -16 }}
    transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
  >
    {children}
  </motion.div>
);

const AnimatedRoutes = () => {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<PageWrapper><LandingPage /></PageWrapper>} />
        <Route path="/login" element={<PageWrapper><LoginPage /></PageWrapper>} />

        <Route path="/operator" element={<ProtectedRoute allowedRoles={['operator']}><PageWrapper><OperatorLayout /></PageWrapper></ProtectedRoute>}>
          <Route index element={<OperatorDashboard />} />
          <Route path="plant/:plantId/block/:blockId" element={<InverterGridPage />} />
          <Route path="alerts" element={<OperatorAlerts />} />
          <Route path="chatbot" element={<ChatbotPage />} />
          <Route path="ml-predict" element={<MLPredictPage />} />
          <Route path="profile" element={<ProfilePage />} />
        </Route>

        <Route path="/admin" element={<ProtectedRoute allowedRoles={['admin']}><PageWrapper><AdminLayout /></PageWrapper></ProtectedRoute>}>
          <Route index element={<AdminDashboard />} />
          <Route path="plants" element={<PlantManagement />} />
          <Route path="plants/:plantId" element={<PlantDetailPage />} />
          <Route path="operators" element={<OperatorManagement />} />
          <Route path="monitoring" element={<LiveMonitoring />} />
          <Route path="alerts" element={<AdminAlerts />} />
          <Route path="audit-logs" element={<AuditLogs />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="profile" element={<ProfilePage />} />
          <Route path="impact" element={<ImpactPage />} />
          <Route path="ml-predict" element={<MLPredictPage />} />
        </Route>

        <Route path="*" element={<NotFound />} />
      </Routes>
    </AnimatePresence>
  );
};

const App = () => (
  <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false}>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Sonner />
        <AuthProvider>
          <BrowserRouter>
            <AnimatedRoutes />
          </BrowserRouter>
        </AuthProvider>
      </TooltipProvider>
    </QueryClientProvider>
  </ThemeProvider>
);

export default App;
