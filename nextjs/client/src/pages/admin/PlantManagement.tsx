import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';
import { toast } from 'sonner';
import { Plus, Loader2 } from 'lucide-react';

const statusBadge = (status: string) => {
  if (status === 'active') return <Badge className="bg-sw-healthy text-primary-foreground text-xs">Active</Badge>;
  if (status === 'maintenance') return <Badge className="bg-sw-warning text-primary-foreground text-xs">Maintenance</Badge>;
  return <Badge variant="secondary" className="text-xs">Decommissioned</Badge>;
};

export default function PlantManagement() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [addOpen, setAddOpen] = useState(false);
  const [name, setName] = useState('');
  const [location, setLocation] = useState('');
  const [status, setStatus] = useState('active');

  const { data: plants = [], isLoading } = useQuery({
    queryKey: ['admin-plants'],
    queryFn: () => adminApi.getPlants(),
  });

  const createMutation = useMutation({
    mutationFn: (data: any) => adminApi.createPlant(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['admin-plants'] });
      qc.invalidateQueries({ queryKey: ['admin-dashboard'] });
      setAddOpen(false);
      setName(''); setLocation(''); setStatus('active');
      toast.success('Plant created successfully');
    },
    onError: (e: any) => toast.error(e.message),
  });

  const decommissionMutation = useMutation({
    mutationFn: (id: string) => adminApi.deletePlant(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['admin-plants'] });
      toast.success('Plant decommissioned');
    },
    onError: (e: any) => toast.error(e.message),
  });

  const handleAdd = () => {
    if (!name.trim() || !location.trim()) { toast.error('Name and location are required'); return; }
    createMutation.mutate({ name: name.trim(), location: location.trim(), status });
  };

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Plant Management</h1>
        <Dialog open={addOpen} onOpenChange={setAddOpen}>
          <DialogTrigger asChild>
            <Button size="sm"><Plus className="h-4 w-4 mr-1" />Add Plant</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader><DialogTitle>Add New Plant</DialogTitle></DialogHeader>
            <div className="space-y-3">
              <div><Label>Name</Label><Input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Solar Park X" /></div>
              <div><Label>Location</Label><Input value={location} onChange={e => setLocation(e.target.value)} placeholder="e.g. Jaipur, Rajasthan" /></div>
              <div><Label>Status</Label>
                <Select value={status} onValueChange={setStatus}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="active">Active</SelectItem>
                    <SelectItem value="maintenance">Maintenance</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button className="w-full" onClick={handleAdd} disabled={createMutation.isPending}>
                {createMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                Create Plant
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <div className="rounded-xl border overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Plant Name</TableHead>
              <TableHead>Location</TableHead>
              <TableHead className="text-right">Blocks</TableHead>
              <TableHead className="text-right">Inverters</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {(plants as any[]).length === 0 ? (
              <TableRow><TableCell colSpan={6} className="text-center py-8 text-muted-foreground">No plants found</TableCell></TableRow>
            ) : (plants as any[]).map((plant: any) => (
              <TableRow key={plant.id} className="hover:bg-muted/50">
                <TableCell className="font-medium">{plant.name}</TableCell>
                <TableCell className="text-sm text-muted-foreground">{plant.location}</TableCell>
                <TableCell className="text-right">{plant.block_count ?? 0}</TableCell>
                <TableCell className="text-right">{plant.inverter_count ?? 0}</TableCell>
                <TableCell>{statusBadge(plant.status)}</TableCell>
                <TableCell>
                  <div className="flex gap-1">
                    <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={() => navigate(`/admin/plants/${plant.id}`)}>
                      Manage
                    </Button>
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button size="sm" variant="ghost" className="h-7 text-xs text-destructive" disabled={plant.status === 'decommissioned'}>
                          Decommission
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Decommission {plant.name}?</AlertDialogTitle>
                          <AlertDialogDescription>This will mark the plant as decommissioned. Historical data is preserved.</AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogAction className="bg-destructive text-destructive-foreground hover:bg-destructive/90" onClick={() => decommissionMutation.mutate(plant.id)}>
                            Decommission
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
