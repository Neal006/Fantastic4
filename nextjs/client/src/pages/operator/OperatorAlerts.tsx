import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { operatorApi } from '@/lib/api';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Loader2, ExternalLink } from 'lucide-react';
import { toast } from 'sonner';
import { useNavigate } from 'react-router-dom';

export default function OperatorAlerts() {
  const [filterType, setFilterType] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const qc = useQueryClient();
  const navigate = useNavigate();

  const { data: alerts = [], isLoading } = useQuery({
    queryKey: ['operator-alerts'],
    queryFn: () => operatorApi.getAlerts(),
    refetchInterval: 20000,
  });

  const ackMutation = useMutation({
    mutationFn: (id: string) => operatorApi.acknowledgeAlert(id),
    onSuccess: () => {
      toast.success('Alert acknowledged');
      qc.invalidateQueries({ queryKey: ['operator-alerts'] });
      qc.invalidateQueries({ queryKey: ['operator-dashboard'] });
    },
    onError: (e: any) => toast.error(e.message),
  });

  const filtered = alerts
    .filter((a: any) => filterType === 'all' || a.type === filterType)
    .filter((a: any) => filterStatus === 'all' || (filterStatus === 'pending' ? !a.acknowledged : a.acknowledged));

  const sevBadge = (type: string) => {
    if (type === 'critical') return <Badge className="bg-sw-critical text-primary-foreground text-xs">Critical</Badge>;
    if (type === 'warning') return <Badge className="bg-sw-warning text-primary-foreground text-xs">Warning</Badge>;
    return <Badge variant="secondary" className="text-xs">Info</Badge>;
  };

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Alerts & Notifications</h1>

      <div className="flex flex-wrap gap-2">
        <Select value={filterType} onValueChange={setFilterType}>
          <SelectTrigger className="w-36 h-8 text-xs"><SelectValue placeholder="Type" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value="critical">Critical</SelectItem>
            <SelectItem value="warning">Warning</SelectItem>
            <SelectItem value="info">Info</SelectItem>
          </SelectContent>
        </Select>
        <Select value={filterStatus} onValueChange={setFilterStatus}>
          <SelectTrigger className="w-36 h-8 text-xs"><SelectValue placeholder="Status" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="pending">Pending</SelectItem>
            <SelectItem value="acknowledged">Acknowledged</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="rounded-xl border overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Time</TableHead>
              <TableHead>Inverter</TableHead>
              <TableHead>Location</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Message</TableHead>
              <TableHead>Status</TableHead>
              <TableHead></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.length === 0 ? (
              <TableRow><TableCell colSpan={7} className="text-center py-8 text-muted-foreground">No alerts found</TableCell></TableRow>
            ) : filtered.map((alert: any) => (
              <TableRow
                key={alert.id}
                className="hover:bg-muted/50 cursor-pointer group"
                onClick={() => navigate(
                  `/operator/plant/${alert.plant_id}/block/${alert.block_id}`,
                  { state: { selectInverterId: alert.inverter_id } }
                )}
              >
                <TableCell className="text-xs">{new Date(alert.created_at).toLocaleString()}</TableCell>
                <TableCell className="font-mono text-sm">
                  <span className="flex items-center gap-1">
                    {alert.inverter_name}
                    <ExternalLink className="h-3 w-3 opacity-0 group-hover:opacity-60 transition-opacity" />
                  </span>
                </TableCell>
                <TableCell className="text-xs">{alert.plant_name} › {alert.block_name}</TableCell>
                <TableCell>{sevBadge(alert.type)}</TableCell>
                <TableCell className="text-sm max-w-[200px] truncate">{alert.message}</TableCell>
                <TableCell>
                  <Badge variant={alert.acknowledged ? 'secondary' : 'destructive'} className="text-xs">
                    {alert.acknowledged ? 'acknowledged' : 'pending'}
                  </Badge>
                </TableCell>
                <TableCell>
                  {!alert.acknowledged && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 text-xs"
                      disabled={ackMutation.isPending}
                      onClick={(e) => { e.stopPropagation(); ackMutation.mutate(alert.id); }}
                    >
                      Acknowledge
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
