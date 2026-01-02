# Edge-Net Genesis Node Deployment Guide

This guide covers deploying the Edge-Net Genesis bootstrap node to various platforms.

## Quick Start

### Local Development

```bash
# Run directly with Node.js
npm run genesis:prod

# Or with Docker
npm run genesis:docker
npm run genesis:docker:logs
```

### Production Deployment

```bash
# Build the Docker image
npm run genesis:docker:build

# Push to your registry
docker tag ruvector/edge-net-genesis:latest your-registry/edge-net-genesis:latest
docker push your-registry/edge-net-genesis:latest
```

## Architecture

The Genesis Node provides:

1. **WebSocket Signaling** (port 8787) - WebRTC connection establishment
2. **DHT Routing** - Kademlia-based peer discovery
3. **Ledger Sync** - CRDT-based state synchronization
4. **Firebase Registration** - Optional bootstrap node discovery
5. **Health Endpoints** (port 8788) - Kubernetes-compatible probes
6. **Prometheus Metrics** - Observable monitoring

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GENESIS_PORT` | 8787 | WebSocket signaling port |
| `GENESIS_HOST` | 0.0.0.0 | Bind address |
| `HEALTH_PORT` | 8788 | Health check/metrics port |
| `GENESIS_DATA` | /data/genesis | Persistent data directory |
| `GENESIS_NODE_ID` | (auto) | Fixed node ID for persistence |
| `LOG_LEVEL` | info | Log verbosity: debug, info, warn, error |
| `LOG_FORMAT` | json | Log format: json, text |
| `METRICS_ENABLED` | true | Enable Prometheus metrics |
| `MAX_CONN_PER_IP` | 50 | Rate limit: connections per IP |
| `MAX_MSG_PER_SEC` | 100 | Rate limit: messages per second |
| `FIREBASE_API_KEY` | - | Firebase API key (optional) |
| `FIREBASE_PROJECT_ID` | - | Firebase project ID (optional) |

### Using .env File

```bash
cp deploy/.env.example deploy/.env
# Edit deploy/.env with your values
docker-compose -f deploy/docker-compose.yml --env-file deploy/.env up -d
```

## Deployment Platforms

### Docker Standalone

```bash
# Build
docker build -t ruvector/edge-net-genesis:latest -f deploy/Dockerfile .

# Run
docker run -d \
  --name edge-net-genesis \
  -p 8787:8787 \
  -p 8788:8788 \
  -v genesis-data:/data/genesis \
  -e LOG_LEVEL=info \
  ruvector/edge-net-genesis:latest

# View logs
docker logs -f edge-net-genesis

# Check health
curl http://localhost:8788/health
```

### Docker Compose

```bash
# Start single node
docker-compose -f deploy/docker-compose.yml up -d

# Start cluster (2 nodes)
docker-compose -f deploy/docker-compose.yml --profile cluster up -d

# Start with monitoring (Prometheus + Grafana)
docker-compose -f deploy/docker-compose.yml --profile monitoring up -d

# View logs
docker-compose -f deploy/docker-compose.yml logs -f

# Stop
docker-compose -f deploy/docker-compose.yml down
```

### Kubernetes

```yaml
# genesis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edge-net-genesis
  labels:
    app: edge-net-genesis
spec:
  replicas: 2
  selector:
    matchLabels:
      app: edge-net-genesis
  template:
    metadata:
      labels:
        app: edge-net-genesis
    spec:
      containers:
      - name: genesis
        image: ruvector/edge-net-genesis:latest
        ports:
        - containerPort: 8787
          name: websocket
        - containerPort: 8788
          name: health
        env:
        - name: LOG_FORMAT
          value: "json"
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8788
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8788
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: genesis-data
          mountPath: /data/genesis
      volumes:
      - name: genesis-data
        persistentVolumeClaim:
          claimName: genesis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: edge-net-genesis
spec:
  selector:
    app: edge-net-genesis
  ports:
  - name: websocket
    port: 8787
    targetPort: 8787
  - name: health
    port: 8788
    targetPort: 8788
  type: LoadBalancer
```

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/edge-net-genesis

# Deploy
gcloud run deploy edge-net-genesis \
  --image gcr.io/PROJECT_ID/edge-net-genesis \
  --platform managed \
  --region us-central1 \
  --port 8787 \
  --memory 512Mi \
  --min-instances 1 \
  --max-instances 10 \
  --allow-unauthenticated
```

### AWS ECS (Fargate)

```json
{
  "family": "edge-net-genesis",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [{
    "name": "genesis",
    "image": "your-ecr-repo/edge-net-genesis:latest",
    "portMappings": [
      {"containerPort": 8787, "protocol": "tcp"},
      {"containerPort": 8788, "protocol": "tcp"}
    ],
    "environment": [
      {"name": "LOG_FORMAT", "value": "json"},
      {"name": "LOG_LEVEL", "value": "info"}
    ],
    "healthCheck": {
      "command": ["CMD-SHELL", "curl -f http://localhost:8788/health || exit 1"],
      "interval": 30,
      "timeout": 10,
      "retries": 3
    },
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/edge-net-genesis",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "genesis"
      }
    }
  }]
}
```

### Fly.io

```toml
# fly.toml
app = "edge-net-genesis"

[build]
  dockerfile = "deploy/Dockerfile"

[env]
  LOG_FORMAT = "json"
  LOG_LEVEL = "info"

[[services]]
  internal_port = 8787
  protocol = "tcp"

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443

  [[services.ports]]
    handlers = ["http"]
    port = 80

[[services]]
  internal_port = 8788
  protocol = "tcp"

  [[services.tcp_checks]]
    grace_period = "5s"
    interval = "30s"
    restart_limit = 3
    timeout = "10s"

[mounts]
  source = "genesis_data"
  destination = "/data/genesis"
```

```bash
fly launch
fly deploy
fly scale count 2
```

### Railway

```json
{
  "build": {
    "dockerfilePath": "deploy/Dockerfile"
  },
  "deploy": {
    "startCommand": "node deploy/genesis-prod.js",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 10
  }
}
```

### DigitalOcean App Platform

```yaml
# .do/app.yaml
name: edge-net-genesis
services:
- name: genesis
  dockerfile_path: deploy/Dockerfile
  http_port: 8788
  instance_count: 2
  instance_size_slug: basic-xxs
  envs:
  - key: LOG_FORMAT
    value: "json"
  health_check:
    http_path: /health
```

## Monitoring

### Health Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Basic liveness check |
| `GET /ready` | Readiness check (WebSocket accepting connections) |
| `GET /status` | Detailed status JSON |
| `GET /metrics` | Prometheus metrics |

### Example Health Check

```bash
# Basic health
curl http://localhost:8788/health
# {"status":"healthy","timestamp":1704067200000}

# Readiness
curl http://localhost:8788/ready
# {"ready":true,"timestamp":1704067200000}

# Full status
curl http://localhost:8788/status
# {
#   "nodeId": "abc123...",
#   "isRunning": true,
#   "uptime": 3600000,
#   "connections": 42,
#   "peers": {"total": 150, "rooms": 3},
#   ...
# }
```

### Prometheus Metrics

```
# HELP genesis_connections_total Total WebSocket connections
# TYPE genesis_connections_total counter
genesis_connections_total 1523

# HELP genesis_peers_registered Current registered peers
# TYPE genesis_peers_registered gauge
genesis_peers_registered 42

# HELP genesis_signals_relayed WebRTC signals relayed
# TYPE genesis_signals_relayed counter
genesis_signals_relayed 8432
```

### Grafana Dashboard

Import the Prometheus data source and create dashboards for:
- Connection rate and active connections
- Peer registration and churn
- Signal relay latency
- DHT routing table size
- Error rates

## Security Considerations

1. **Rate Limiting** - Built-in protection against connection/message floods
2. **Non-root User** - Container runs as unprivileged user
3. **Network Isolation** - Use private networks where possible
4. **TLS Termination** - Use a reverse proxy (nginx, traefik) for TLS
5. **Firebase Rules** - If using Firebase, configure security rules

### TLS with Nginx

```nginx
upstream genesis {
    server localhost:8787;
}

server {
    listen 443 ssl;
    server_name genesis.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/genesis.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/genesis.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://genesis;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 86400;
    }
}
```

## Scaling

### Horizontal Scaling

Genesis nodes can run in parallel. Peers will discover each other through:
1. Firebase bootstrap registration (if enabled)
2. DHT peer exchange
3. Direct peer-to-peer connections

### Load Balancing

For WebSocket, use sticky sessions or connection-aware load balancing:

```nginx
upstream genesis_cluster {
    ip_hash;  # Sticky sessions by IP
    server genesis-1:8787;
    server genesis-2:8787;
    server genesis-3:8787;
}
```

## Troubleshooting

### Common Issues

1. **Connection refused**
   - Check firewall rules for ports 8787 and 8788
   - Verify container is running: `docker ps`

2. **WebSocket upgrade failed**
   - Ensure proxy supports WebSocket upgrade
   - Check `Connection: upgrade` header is passed

3. **High memory usage**
   - Reduce `MAX_CONN_PER_IP` limit
   - Enable peer pruning (default: 5 min timeout)

4. **Peers not discovering each other**
   - Verify Firebase credentials if using bootstrap registration
   - Check network connectivity between nodes

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=debug npm run genesis:prod

# Or in Docker
docker run -e LOG_LEVEL=debug ruvector/edge-net-genesis:latest
```

### Health Check Script

```bash
# Use the included health check script
node deploy/health-check.js --host localhost --port 8788

# JSON output for automation
node deploy/health-check.js --json
```

## Support

- GitHub Issues: https://github.com/ruvnet/ruvector/issues
- Documentation: https://github.com/ruvnet/ruvector/tree/main/examples/edge-net
