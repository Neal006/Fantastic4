import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { toast } from 'sonner';
import { Plus, Key, Loader2 } from 'lucide-react';

function generatePassword() {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789';
  return Array.from({ length: 12 }, () => chars[Math.floor(Math.random() * chars.length)]).join('');
}

export default function OperatorManagement() {
  const qc = useQueryClient();
  const [addOpen, setAddOpen] = useState(false);
  const [assignOpen, setAssignOpen] = useState<string | null>(null);
  const [newName, setNewName] = useState('');
  const [newEmail, setNewEmail] = useState('');
  const [newPw, setNewPw] = useState('');
  const [newPlants, setNewPlants] = useState<string[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const { data: operators = [], isLoading } = useQuery({
    queryKey: ['admin-operators'],
    queryFn: () => adminApi.getOperators(),
    refetchInterval: 30000,
  });

  const { data: allPlants = [] } = useQuery({
    queryKey: ['admin-plants'],
    queryFn: () => adminApi.getPlants(),
  });

  const createMutation = useMutation({
    mutationFn: (data: any) => adminApi.createOperator(data),
    onSuccess: (data: any) => {
      qc.invalidateQueries({ queryKey: ['admin-operators'] });
      qc.invalidateQueries({ queryKey: ['admin-dashboard'] });
      // Assign plants if any selected
      if (newPlants.length && data?.id) {
        adminApi.assignPlants(data.id, newPlants);
      }
      setAddOpen(false);
      setNewName(''); setNewEmail(''); setNewPw(''); setNewPlants([]);
      toast.success(`Operator created. Temp password: ${newPw}`);
    },
    onError: (e: any) => toast.error(e.message),
  });

  const assignMutation = useMutation({
    mutationFn: ({ id, plant_ids }: { id: string; plant_ids: string[] }) => adminApi.assignPlants(id, plant_ids),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['admin-operators'] }); toast.success('Plants assigned'); setAssignOpen(null); },
    onError: (e: any) => toast.error(e.message),
  });

  const deactivateMutation = useMutation({
    mutationFn: (id: string) => adminApi.deactivateOperator(id),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['admin-operators'] }); toast.success('Operator deactivated'); },
    onError: (e: any) => toast.error(e.message),
  });

  const activateMutation = useMutation({
    mutationFn: (id: string) => adminApi.activateOperator(id),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['admin-operators'] }); toast.success('Operator activated'); },
    onError: (e: any) => toast.error(e.message),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => adminApi.deleteOperator(id),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['admin-operators'] }); qc.invalidateQueries({ queryKey: ['admin-dashboard'] }); toast.success('Operator deleted'); },
    onError: (e: any) => toast.error(e.message),
  });

  const handleAdd = () => {
    const errs: Record<string, string> = {};
    if (!newName.trim()) errs.name = 'Required';
    if (!newEmail.trim() || !newEmail.includes('@')) errs.email = 'Valid email required';
    if (newPw.length < 8) errs.pw = 'Min 8 characters';
    setErrors(errs);
    if (Object.keys(errs).length > 0) return;
    createMutation.mutate({ name: newName.trim(), email: newEmail.trim(), password: newPw });
  };

  const activePlants = (allPlants as any[]).filter((p: any) => p.status === 'active');

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Operator Management</h1>
        <Dialog open={addOpen} onOpenChange={setAddOpen}>
          <DialogTrigger asChild><Button size="sm"><Plus className="h-4 w-4 mr-1" />Add Operator</Button></DialogTrigger>
          <DialogContent className="max-w-md">
            <DialogHeader><DialogTitle>Add Operator</DialogTitle></DialogHeader>
            <div className="space-y-3">
              <div>
                <Label>Full Name</Label>
                <Input value={newName} onChange={e => setNewName(e.target.value)} />
                {errors.name && <p className="text-xs text-destructive mt-1">{errors.name}</p>}
              </div>
              <div>
                <Label>Email</Label>
                <Input value={newEmail} onChange={e => setNewEmail(e.target.value)} type="email" />
                {errors.email && <p className="text-xs text-destructive mt-1">{errors.email}</p>}
              </div>
              <div>
                <Label>Temporary Password</Label>
                <div className="flex gap-2">
                  <Input value={newPw} onChange={e => setNewPw(e.target.value)} />
                  <Button type="button" variant="outline" size="sm" onClick={() => setNewPw(generatePassword())}><Key className="h-3.5 w-3.5" /></Button>
                </div>
                {errors.pw && <p className="text-xs text-destructive mt-1">{errors.pw}</p>}
              </div>
              <div>
                <Label>Assign Plants</Label>
                <div className="space-y-2 mt-1">
                  {activePlants.map((p: any) => (
                    <div key={p.id} className="flex items-center gap-2">
                      <Checkbox checked={newPlants.includes(p.id)} onCheckedChange={() => setNewPlants(prev => prev.includes(p.id) ? prev.filter(x => x !== p.id) : [...prev, p.id])} />
                      <span className="text-sm">{p.name}</span>
                    </div>
                  ))}
                </div>
              </div>
              <Button className="w-full" onClick={handleAdd} disabled={createMutation.isPending}>
                {createMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                Create Operator
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <div className="rounded-xl border overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Email</TableHead>
              <TableHead>Assigned Plants</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Last Login</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {(operators as any[]).length === 0 ? (
              <TableRow><TableCell colSpan={6} className="text-center py-8 text-muted-foreground">No operators found</TableCell></TableRow>
            ) : (operators as any[]).map((op: any) => (
              <TableRow key={op.id} className="hover:bg-muted/50">
                <TableCell className="font-medium">{op.name}</TableCell>
                <TableCell className="text-sm">{op.email}</TableCell>
                <TableCell>
                  <div className="flex flex-wrap gap-1">
                    {op.assigned_plant_names ? op.assigned_plant_names.split(', ').map((pn: string, i: number) => (
                      <Badge key={i} variant="secondary" className="text-xs">{pn}</Badge>
                    )) : <span className="text-xs text-muted-foreground">None</span>}
                  </div>
                </TableCell>
                <TableCell>
                  <Badge className={op.is_active ? 'bg-sw-healthy text-primary-foreground text-xs' : 'bg-muted text-muted-foreground text-xs'}>
                    {op.is_active ? 'Active' : 'Inactive'}
                  </Badge>
                </TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  {op.last_login ? new Date(op.last_login).toLocaleString() : 'Never'}
                </TableCell>
                <TableCell>
                  <div className="flex gap-1">
                    {/* Assign Plants Dialog */}
                    <Dialog open={assignOpen === op.id} onOpenChange={(o) => setAssignOpen(o ? op.id : null)}>
                      <DialogTrigger asChild><Button size="sm" variant="ghost" className="h-7 text-xs">Assign</Button></DialogTrigger>
                      <DialogContent>
                        <DialogHeader><DialogTitle>Assign Plants — {op.name}</DialogTitle></DialogHeader>
                        <AssignPlantsList
                          current={op.assignedPlantIds || []}
                          plants={activePlants}
                          onSave={(plantIds) => assignMutation.mutate({ id: op.id, plant_ids: plantIds })}
                        />
                      </DialogContent>
                    </Dialog>
                    <Button
                      size="sm" variant="ghost" className="h-7 text-xs"
                      onClick={() => op.is_active ? deactivateMutation.mutate(op.id) : activateMutation.mutate(op.id)}
                    >
                      {op.is_active ? 'Deactivate' : 'Activate'}
                    </Button>
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button size="sm" variant="ghost" className="h-7 text-xs text-destructive">Delete</Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Delete {op.name}?</AlertDialogTitle>
                          <AlertDialogDescription>This is permanent and cannot be undone.</AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogAction className="bg-destructive text-destructive-foreground" onClick={() => deleteMutation.mutate(op.id)}>Delete</AlertDialogAction>
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

function AssignPlantsList({ current, plants, onSave }: { current: string[]; plants: any[]; onSave: (plants: string[]) => void }) {
  const [selected, setSelected] = useState<string[]>(current);
  const toggle = (id: string) => setSelected(prev => prev.includes(id) ? prev.filter(p => p !== id) : [...prev, id]);

  return (
    <div className="space-y-3">
      {plants.map((p: any) => (
        <div key={p.id} className="flex items-center gap-2">
          <Checkbox checked={selected.includes(p.id)} onCheckedChange={() => toggle(p.id)} />
          <span className="text-sm">{p.name}</span>
        </div>
      ))}
      <Button className="w-full" onClick={() => onSave(selected)}>Save Assignments</Button>
    </div>
  );
}
