# AWS RDS MySQL Setup Guide

This guide explains how to configure the NextJS server to connect to AWS RDS MySQL instance.

## AWS RDS Instance Details

- **Endpoint**: `hacamined.ch8i62460w7g.ap-south-1.rds.amazonaws.com`
- **Port**: `3306`
- **Region**: `ap-south-1` (Mumbai)
- **Database**: `hackamined`
- **Username**: `Admin`
- **Password**: `fANTASTIC!4123`

## Setup Instructions

### 1. Create `.env` File

Copy the `env.example` file to `.env` in the `nextjs/server` directory:

```bash
cd nextjs/server
cp env.example .env
```

The `.env` file is already pre-configured with AWS RDS credentials:

```env
DB_HOST=hacamined.ch8i62460w7g.ap-south-1.rds.amazonaws.com
DB_PORT=3306
DB_USER=Admin
DB_PASSWORD=fANTASTIC!4123
DB_NAME=hackamined
DB_SSL=true
```

### 2. Verify Database Schema

Ensure the `hackamined` database exists on the RDS instance with the required tables:

- `users`
- `plants`
- `blocks`
- `inverters`
- `inverter_readings`
- `alerts`
- `chat_logs`

### 3. Security Group Configuration

**IMPORTANT**: Ensure your AWS RDS security group allows inbound connections:

1. Go to AWS RDS Console → Your RDS Instance → Connectivity & Security
2. Click on the VPC Security Group
3. Add an inbound rule:
   - **Type**: MySQL/Aurora
   - **Protocol**: TCP
   - **Port**: 3306
   - **Source**: Your IP address or `0.0.0.0/0` (for testing only - restrict in production)

### 4. SSL/TLS Connection

The connection is configured to use SSL/TLS for secure communication with AWS RDS:

```javascript
ssl: {
    rejectUnauthorized: false // AWS RDS uses Amazon's CA
}
```

For production, consider downloading the AWS RDS CA certificate bundle:
- [AWS RDS CA Bundle](https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem)

### 5. Test Connection

Start the server and verify the connection:

```bash
cd nextjs/server
npm install
node index.js
```

You should see:
```
✅ MySQL connected successfully
🚀 Server running on port 3001
```

If connection fails, check:
- Security group rules
- RDS instance status (must be "Available")
- Network connectivity
- Credentials

## Connection Configuration

The database connection is configured in `db/connection.js`:

```javascript
const pool = mysql.createPool({
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    port: process.env.DB_PORT || 3306,
    ssl: process.env.DB_SSL === 'true' ? {
        rejectUnauthorized: false
    } : undefined,
    connectTimeout: 20000,
    connectionLimit: 10,
});
```

## Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DB_HOST` | RDS endpoint | localhost | ✅ |
| `DB_PORT` | MySQL port | 3306 | ❌ |
| `DB_USER` | Database user | root | ✅ |
| `DB_PASSWORD` | Database password | root | ✅ |
| `DB_NAME` | Database name | hackamined | ✅ |
| `DB_SSL` | Enable SSL/TLS | false | ✅ (for RDS) |
| `GENAI_URL` | GenAI server URL | http://localhost:8000 | ✅ |
| `PORT` | Server port | 3001 | ❌ |
| `ALLOWED_ORIGINS` | CORS origins (comma-separated) | localhost URLs | ❌ |

## Deployment Checklist

- [ ] RDS instance is running and accessible
- [ ] Security group allows inbound MySQL connections
- [ ] Database `hackamined` exists
- [ ] All required tables are created
- [ ] `.env` file is configured with RDS credentials
- [ ] `DB_SSL=true` is set
- [ ] GENAI_URL points to deployed GenAI server
- [ ] ALLOWED_ORIGINS includes your frontend URL
- [ ] Connection test passes

## Troubleshooting

### Connection Timeout

**Error**: `ER_CONNECTION_TIMED_OUT` or `connect ETIMEDOUT`

**Solutions**:
1. Check security group inbound rules
2. Verify RDS instance is publicly accessible (if connecting from outside VPC)
3. Check VPC routing tables and network ACLs
4. Increase `connectTimeout` in `connection.js`

### Access Denied

**Error**: `ER_ACCESS_DENIED_ERROR`

**Solutions**:
1. Verify username and password
2. Check user permissions: `GRANT ALL PRIVILEGES ON hackamined.* TO 'Admin'@'%';`
3. Ensure user can connect from your IP

### SSL/TLS Issues

**Error**: `SSL connection error`

**Solutions**:
1. Set `DB_SSL=true` in `.env`
2. Download AWS RDS CA bundle if needed
3. Verify RDS instance has SSL/TLS enabled

### Database Not Found

**Error**: `ER_BAD_DB_ERROR: Unknown database 'hackamined'`

**Solutions**:
1. Create database: `CREATE DATABASE hackamined;`
2. Run schema migration scripts
3. Verify `DB_NAME` in `.env`

## Performance Optimization

For production deployment:

1. **Connection Pooling**: Already configured with 10 connections
2. **Indexes**: Ensure proper indexes on frequently queried columns
3. **Query Optimization**: Use EXPLAIN to analyze slow queries
4. **Read Replicas**: Consider AWS RDS read replicas for read-heavy workloads
5. **Monitoring**: Enable AWS CloudWatch for RDS metrics

## Security Best Practices

1. **Rotate Credentials**: Change the default password
2. **Restrict Access**: Limit security group to specific IPs
3. **Use IAM Authentication**: Consider AWS IAM database authentication
4. **Enable Encryption**: Use RDS encryption at rest
5. **Backup**: Enable automated backups
6. **Audit Logging**: Enable RDS audit logs

## Support

For issues related to:
- **AWS RDS**: Check AWS RDS documentation or AWS Support
- **Application**: Check application logs in `nextjs/server`
- **Database Schema**: Refer to schema migration files
