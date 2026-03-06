import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { toast } from 'sonner';
import { Plus, ChevronDown, ArrowLeft, Loader2, Trash2 } from 'lucide-react';

export default function PlantDetailPage() {
  const { plantId } = useParams<{ plantId: string }>();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [addBlockOpen, setAddBlockOpen] = useState(false);
  const [blockName, setBlockName] = useState('');
  const [expandedBlocks, setExpandedBlocks] = useState<string[]>([]);
  const [addInvOpen, setAddInvOpen] = useState<string | null>(null);
  const [invName, setInvName] = useState('');
  const [invCapacity, setInvCapacity] = useState('50');

  // Fetch the plant list to get plant info
  const { data: plants = [] } = useQuery({
    queryKey: ['admin-plants'],
    queryFn: () => adminApi.getPlants(),
  });

  const plant = (plants as any[]).find((p: any) => p.id === plantId);

  // Fetch blocks for this plant
  const { data: blocks = [], isLoading: blocksLoading } = useQuery({
    queryKey: ['admin-blocks', plantId],
    queryFn: () => adminApi.getBlocks(plantId!),
    enabled: !!plantId,
  });

  // Fetch inverters for all blocks (individual queries keyed by blockId)
  const [blockInverters, setBlockInverters] = useState<Record<string, any[]>>({});

  const loadInverters = async (blockId: string) => {
    if (blockInverters[blockId]) return; // already loaded
    try {
      const invs = await adminApi.getInverters(blockId);
      setBlockInverters(prev => ({ ...prev, [blockId]: invs }));
    } catch { /* ignore */ }
  };

  const addBlockMutation = useMutation({
    mutationFn: (name: string) => adminApi.createBlock(plantId!, name),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['admin-blocks', plantId] });
      qc.invalidateQueries({ queryKey: ['admin-plants'] });
      setAddBlockOpen(false);
      setBlockName('');
      toast.success('Block added');
    },
    onError: (e: any) => toast.error(e.message),
  });

  const deleteBlockMutation = useMutation({
    mutationFn: (id: string) => adminApi.deleteBlock(id),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['admin-blocks', plantId] }); toast.success('Block deleted'); },
    onError: (e: any) => toast.error(e.message),
  });

  const addInvMutation = useMutation({
    mutationFn: ({ blockId, name, capacity }: { blockId: string; name: string; capacity: number }) =>
      adminApi.createInverter(blockId, { name, capacity_kw: capacity }),
    onSuccess: (_, vars) => {
      setBlockInverters(prev => { const copy = { ...prev }; delete copy[vars.blockId]; return copy; });
      setAddInvOpen(null);
      setInvName('');
      setInvCapacity('50');
      toast.success('Inverter added');
    },
    onError: (e: any) => toast.error(e.message),
  });

  const deleteInvMutation = useMutation({
    mutationFn: ({ id, blockId }: { id: string; blockId: string }) => adminApi.deleteInverter(id),
    onSuccess: (_, vars) => {
      setBlockInverters(prev => ({ ...prev, [vars.blockId]: (prev[vars.blockId] || []).filter((i: any) => i.id !== vars.id) }));
      toast.success('Inverter removed');
    },
    onError: (e: any) => toast.error(e.message),
  });

  const toggleBlock = (id: string) => {
    setExpandedBlocks(prev => {
      const next = prev.includes(id) ? prev.filter(b => b !== id) : [...prev, id];
      if (!prev.includes(id)) loadInverters(id);
      return next;
    });
  };

  if (!plantId) return <p>Invalid URL</p>;

  return (
    <div className="space-y-6">
      <Button variant="ghost" size="sm" onClick={() => navigate('/admin/plants')} className="mb-2">
        <ArrowLeft className="h-4 w-4 mr-1" />Back to Plants
      </Button>

      {plant && (
        <Card className="rounded-xl shadow-sm">
          <CardContent className="p-5">
            <div className="flex items-start justify-between">
              <div>
                <h1 className="text-xl font-bold">{plant.name}</h1>
                <p className="text-sm text-muted-foreground">{plant.location}</p>
              </div>
              <Badge className={plant.status === 'active' ? 'bg-sw-healthy text-primary-foreground' : 'bg-sw-warning text-primary-foreground'}>
                {plant.status}
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Blocks ({(blocks as any[]).length})</h2>
        <Dialog open={addBlockOpen} onOpenChange={setAddBlockOpen}>
          <DialogTrigger asChild>
            <Button size="sm"><Plus className="h-4 w-4 mr-1" />Add Block</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader><DialogTitle>Add Block</DialogTitle></DialogHeader>
            <div className="space-y-3">
              <div><Label>Block Name</Label><Input value={blockName} onChange={e => setBlockName(e.target.value)} placeholder="e.g. Block 4" /></div>
              <Button className="w-full" onClick={() => { if (!blockName.trim()) { toast.error('Name required'); return; } addBlockMutation.mutate(blockName.trim()); }} disabled={addBlockMutation.isPending}>
                {addBlockMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                Add Block
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {blocksLoading ? (
        <div className="flex items-center justify-center h-32"><Loader2 className="h-6 w-6 animate-spin text-primary" /></div>
      ) : (
        <div className="space-y-3">
          {(blocks as any[]).map((block: any) => (
            <Card key={block.id} className="rounded-xl shadow-sm">
              <Collapsible open={expandedBlocks.includes(block.id)} onOpenChange={() => toggleBlock(block.id)}>
                <CollapsibleTrigger className="w-full">
                  <CardHeader className="py-3 px-5 flex flex-row items-center justify-between">
                    <CardTitle className="text-sm font-medium">{block.name}</CardTitle>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="text-xs">{block.inverter_count ?? 0} inverters</Badge>
                      <ChevronDown className={`h-4 w-4 transition-transform ${expandedBlocks.includes(block.id) ? 'rotate-180' : ''}`} />
                    </div>
                  </CardHeader>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <CardContent className="pt-0 px-5 pb-4 space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-muted-foreground">Inverters in this block</span>
                      <div className="flex gap-2">
                        <Dialog open={addInvOpen === block.id} onOpenChange={(o) => setAddInvOpen(o ? block.id : null)}>
                          <DialogTrigger asChild>
                            <Button size="sm" variant="outline" className="h-7 text-xs"><Plus className="h-3 w-3 mr-1" />Add Inverter</Button>
                          </DialogTrigger>
                          <DialogContent>
                            <DialogHeader><DialogTitle>Add Inverter to {block.name}</DialogTitle></DialogHeader>
                            <div className="space-y-3">
                              <div><Label>Inverter Name</Label><Input value={invName} onChange={e => setInvName(e.target.value)} placeholder="e.g. INV-B1-01" /></div>
                              <div><Label>Capacity (kW)</Label><Input type="number" value={invCapacity} onChange={e => setInvCapacity(e.target.value)} /></div>
                              <Button className="w-full" disabled={addInvMutation.isPending} onClick={() => {
                                if (!invName.trim()) { toast.error('Name required'); return; }
                                addInvMutation.mutate({ blockId: block.id, name: invName.trim(), capacity: Number(invCapacity) });
                              }}>
                                {addInvMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                                Add Inverter
                              </Button>
                            </div>
                          </DialogContent>
                        </Dialog>
                        <AlertDialog>
                          <AlertDialogTrigger asChild>
                            <Button size="sm" variant="ghost" className="h-7 text-xs text-destructive"><Trash2 className="h-3 w-3" /></Button>
                          </AlertDialogTrigger>
                          <AlertDialogContent>
                            <AlertDialogHeader>
                              <AlertDialogTitle>Delete {block.name}?</AlertDialogTitle>
                              <AlertDialogDescription>All inverters in this block must be removed first.</AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                              <AlertDialogCancel>Cancel</AlertDialogCancel>
                              <AlertDialogAction className="bg-destructive text-destructive-foreground" onClick={() => deleteBlockMutation.mutate(block.id)}>Delete</AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>
                      </div>
                    </div>

                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Name</TableHead>
                          <TableHead>Capacity</TableHead>
                          <TableHead>Status</TableHead>
                          <TableHead></TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {!(blockInverters[block.id]) ? (
                          <TableRow><TableCell colSpan={4} className="text-center py-4"><Loader2 className="h-4 w-4 animate-spin mx-auto" /></TableCell></TableRow>
                        ) : blockInverters[block.id].length === 0 ? (
                          <TableRow><TableCell colSpan={4} className="text-center text-muted-foreground py-4">No inverters</TableCell></TableRow>
                        ) : blockInverters[block.id].map((inv: any) => (
                          <TableRow key={inv.id}>
                            <TableCell className="font-mono text-sm">{inv.name}</TableCell>
                            <TableCell>{inv.capacity_kw} kW</TableCell>
                            <TableCell>
                              <Badge variant="secondary" className="text-xs">{inv.current_category || 'unknown'}</Badge>
                            </TableCell>
                            <TableCell>
                              <Button size="sm" variant="ghost" className="h-7 text-xs text-destructive" onClick={() => deleteInvMutation.mutate({ id: inv.id, blockId: block.id })}>
                                Remove
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </CollapsibleContent>
              </Collapsible>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
