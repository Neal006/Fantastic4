import { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useQuery } from '@tanstack/react-query';
import { authApi, operatorApi } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Loader2 } from 'lucide-react';
import { toast } from 'sonner';

export default function ProfilePage() {
  const { user } = useAuth();
  const [currentPw, setCurrentPw] = useState('');
  const [newPw, setNewPw] = useState('');
  const [confirmPw, setConfirmPw] = useState('');
  const [saving, setSaving] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Fetch the plant list to show plant names (only for operators)
  const { data: plants = [] } = useQuery({
    queryKey: ['operator-plants-profile'],
    queryFn: () => operatorApi.getPlants(),
    enabled: user?.role === 'operator',
  });

  if (!user) return null;

  const plantMap = Object.fromEntries((plants as any[]).map((p: any) => [p.id, p.name]));

  const handleReset = async (e: React.FormEvent) => {
    e.preventDefault();
    const errs: Record<string, string> = {};
    if (!currentPw) errs.currentPw = 'Required';
    if (newPw.length < 8) errs.newPw = 'Minimum 8 characters';
    if (newPw !== confirmPw) errs.confirmPw = 'Passwords do not match';
    setErrors(errs);
    if (Object.keys(errs).length > 0) return;

    setSaving(true);
    try {
      await authApi.resetPassword(currentPw, newPw, confirmPw);
      toast.success('Password updated successfully');
      setCurrentPw(''); setNewPw(''); setConfirmPw('');
    } catch (e: any) {
      toast.error(e.message || 'Failed to update password');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6 max-w-2xl">
      <h1 className="text-2xl font-bold">Profile</h1>

      <Card className="rounded-xl shadow-sm">
        <CardHeader><CardTitle className="text-lg">Account Information</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label className="text-xs text-muted-foreground">Name</Label>
              <p className="font-medium">{user.name}</p>
            </div>
            <div>
              <Label className="text-xs text-muted-foreground">Email</Label>
              <p className="font-medium">{user.email}</p>
            </div>
            <div>
              <Label className="text-xs text-muted-foreground">Role</Label>
              <p className="font-medium capitalize">{user.role}</p>
            </div>
            {user.role === 'operator' && user.assignedPlants?.length > 0 && (
              <div>
                <Label className="text-xs text-muted-foreground">Assigned Plants</Label>
                <div className="flex flex-wrap gap-1 mt-1">
                  {user.assignedPlants.map(pId => (
                    <Badge key={pId} variant="secondary" className="text-xs">
                      {plantMap[pId] || pId}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="rounded-xl shadow-sm">
        <CardHeader><CardTitle className="text-lg">Change Password</CardTitle></CardHeader>
        <CardContent>
          <form onSubmit={handleReset} className="space-y-3">
            <div>
              <Label>Current Password</Label>
              <Input type="password" value={currentPw} onChange={e => setCurrentPw(e.target.value)} />
              {errors.currentPw && <p className="text-xs text-destructive mt-1">{errors.currentPw}</p>}
            </div>
            <div>
              <Label>New Password</Label>
              <Input type="password" value={newPw} onChange={e => setNewPw(e.target.value)} />
              {errors.newPw && <p className="text-xs text-destructive mt-1">{errors.newPw}</p>}
            </div>
            <div>
              <Label>Confirm Password</Label>
              <Input type="password" value={confirmPw} onChange={e => setConfirmPw(e.target.value)} />
              {errors.confirmPw && <p className="text-xs text-destructive mt-1">{errors.confirmPw}</p>}
            </div>
            <Button type="submit" disabled={saving}>
              {saving ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Update Password
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
