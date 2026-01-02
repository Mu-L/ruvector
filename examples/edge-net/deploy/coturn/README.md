# Edge-Net coturn TURN/STUN Server Deployment

Deploy a self-hosted TURN server for reliable NAT traversal in Edge-Net P2P networks.

```
+------------------------------------------------------------------+
|                    NAT TRAVERSAL ARCHITECTURE                     |
+------------------------------------------------------------------+
|                                                                  |
|     Peer A (Symmetric NAT)          Peer B (Symmetric NAT)       |
|     +------------------+            +------------------+          |
|     |  Browser/Node.js |            |  Browser/Node.js |          |
|     |     WebRTC       |            |     WebRTC       |          |
|     +--------+---------+            +---------+--------+          |
|              |                                |                   |
|              v                                v                   |
|     +--------+---------+            +---------+--------+          |
|     |    NAT Router    |            |    NAT Router    |          |
|     +--------+---------+            +---------+--------+          |
|              |                                |                   |
|              +--------->  TURN  <-------------+                   |
|                          Server                                  |
|                      (coturn)                                    |
|                                                                  |
|     Direct P2P fails due to symmetric NAT                        |
|     TURN server relays traffic between peers                     |
|                                                                  |
+------------------------------------------------------------------+
```

## Why Self-Host TURN?

| Factor | Free TURN Services | Self-Hosted coturn |
|--------|-------------------|-------------------|
| **Bandwidth** | 500MB-1GB/month | Unlimited |
| **Latency** | Variable | Optimized location |
| **Reliability** | Best-effort | SLA-backed |
| **Privacy** | Third-party relay | Your infrastructure |
| **Cost at scale** | Expensive | Predictable |

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose
- Server with public IP address
- Ports 3478/UDP, 3478/TCP, 5349/TCP open

### 2. Deploy

```bash
# Clone and navigate to deployment directory
cd examples/edge-net/deploy/coturn

# Configure environment
cp .env.example .env

# Edit .env with your settings
nano .env

# Deploy
docker-compose up -d

# Check logs
docker-compose logs -f coturn
```

### 3. Configure Edge-Net

Set environment variables for your Edge-Net nodes:

```bash
export EDGE_NET_TURN_URL="turn:your-server.example.com:3478"
export EDGE_NET_TURN_USERNAME="edgenet"
export EDGE_NET_TURN_CREDENTIAL="your-secure-password"
```

Or in JavaScript:

```javascript
import { EdgeNet } from '@ruvector/edge-net';

const node = await EdgeNet.init({
  siteId: 'my-site',
  network: {
    webrtc: {
      iceServers: [
        { urls: 'stun:your-server.example.com:3478' },
        {
          urls: 'turn:your-server.example.com:3478',
          username: 'edgenet',
          credential: 'your-secure-password',
        },
        {
          urls: 'turn:your-server.example.com:3478?transport=tcp',
          username: 'edgenet',
          credential: 'your-secure-password',
        },
        {
          urls: 'turns:your-server.example.com:5349',
          username: 'edgenet',
          credential: 'your-secure-password',
        },
      ],
    },
  },
});
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TURN_REALM` | Authentication realm | `edge-net.local` |
| `TURN_USER` | TURN username | `edgenet` |
| `TURN_PASSWORD` | TURN password | `changeme` |
| `EXTERNAL_IP` | Server public IP | Auto-detect |
| `MIN_PORT` | Relay port range start | `49152` |
| `MAX_PORT` | Relay port range end | `65535` |

### Firewall Rules

```bash
# UFW (Ubuntu)
sudo ufw allow 3478/udp comment 'STUN/TURN UDP'
sudo ufw allow 3478/tcp comment 'STUN/TURN TCP'
sudo ufw allow 5349/tcp comment 'TURN TLS'
sudo ufw allow 49152:65535/udp comment 'TURN relay ports'

# iptables
iptables -A INPUT -p udp --dport 3478 -j ACCEPT
iptables -A INPUT -p tcp --dport 3478 -j ACCEPT
iptables -A INPUT -p tcp --dport 5349 -j ACCEPT
iptables -A INPUT -p udp --dport 49152:65535 -j ACCEPT

# Cloud provider security groups
# - Allow inbound UDP 3478
# - Allow inbound TCP 3478
# - Allow inbound TCP 5349
# - Allow inbound UDP 49152-65535
```

### TLS Configuration (Recommended for Production)

1. Obtain TLS certificates (Let's Encrypt or other CA):

```bash
# Using certbot
sudo certbot certonly --standalone -d turn.example.com

# Copy certificates
mkdir -p certs
cp /etc/letsencrypt/live/turn.example.com/fullchain.pem certs/turn.crt
cp /etc/letsencrypt/live/turn.example.com/privkey.pem certs/turn.key
chmod 600 certs/turn.key
```

2. Update `turnserver.conf`:

```conf
cert=/etc/coturn/certs/turn.crt
pkey=/etc/coturn/certs/turn.key
```

3. Restart coturn:

```bash
docker-compose restart coturn
```

## Testing

### Test with trickle-ice

Visit [trickle-ice.stuntman.ericsson.com](https://trickle-ice.stuntman.ericsson.com/) and add your server:

1. STUN test: `stun:your-server.example.com:3478`
2. TURN test: `turn:your-server.example.com:3478` (with credentials)

### Test with Edge-Net Diagnostics

```javascript
import { testIceConnectivity, createIceConfig } from '@ruvector/edge-net/webrtc';

// Test with custom TURN server
const diagnostics = await testIceConnectivity({
  iceServers: [
    { urls: 'stun:your-server.example.com:3478' },
    {
      urls: 'turn:your-server.example.com:3478',
      username: 'edgenet',
      credential: 'your-password',
    },
  ],
});

console.log(diagnostics.formatReport());
```

### Test with turnutils

```bash
# Install turnutils (part of coturn package)
apt-get install coturn

# Test STUN
turnutils_stunclient your-server.example.com

# Test TURN
turnutils_uclient -T -u edgenet -w your-password your-server.example.com
```

## Production Deployment

### Cloud Providers

#### AWS EC2

```bash
# Launch t3.small instance with Ubuntu 22.04
# Configure security group with TURN ports
# Assign Elastic IP

# SSH and deploy
ssh ubuntu@<elastic-ip>
git clone https://github.com/ruvnet/ruvector.git
cd ruvector/examples/edge-net/deploy/coturn
cp .env.example .env
echo "EXTERNAL_IP=<elastic-ip>" >> .env
docker-compose up -d
```

#### Google Cloud

```bash
# Create VM
gcloud compute instances create edge-net-turn \
  --machine-type=e2-small \
  --zone=us-central1-a \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud

# Configure firewall
gcloud compute firewall-rules create turn-udp \
  --allow=udp:3478,udp:49152-65535 \
  --target-tags=turn-server

gcloud compute firewall-rules create turn-tcp \
  --allow=tcp:3478,tcp:5349 \
  --target-tags=turn-server

# Add tag to instance
gcloud compute instances add-tags edge-net-turn \
  --tags=turn-server \
  --zone=us-central1-a
```

#### DigitalOcean

```bash
# Create droplet via CLI or web console
doctl compute droplet create edge-net-turn \
  --region=nyc1 \
  --size=s-1vcpu-1gb \
  --image=ubuntu-22-04-x64

# Configure firewall
doctl compute firewall create turn-server \
  --inbound-rules="protocol:udp,ports:3478,address:0.0.0.0/0 protocol:tcp,ports:3478,address:0.0.0.0/0 protocol:tcp,ports:5349,address:0.0.0.0/0 protocol:udp,ports:49152-65535,address:0.0.0.0/0" \
  --droplet-ids=<droplet-id>
```

### High Availability

For production deployments requiring high availability:

```yaml
# docker-compose.ha.yml
version: '3.8'

services:
  coturn-1:
    extends:
      file: docker-compose.yml
      service: coturn
    environment:
      - REDIS_URL=redis://redis:6379

  coturn-2:
    extends:
      file: docker-compose.yml
      service: coturn
    environment:
      - REDIS_URL=redis://redis:6379

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  haproxy:
    image: haproxy:2.8-alpine
    ports:
      - "3478:3478/udp"
      - "3478:3478/tcp"
      - "5349:5349/tcp"
    volumes:
      - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
```

### Monitoring

Enable Prometheus metrics:

```bash
docker-compose --profile monitoring up -d
```

Grafana dashboard available at [grafana.com/dashboards/15061](https://grafana.com/grafana/dashboards/15061-coturn/).

## Security Best Practices

1. **Strong Credentials**: Use long, random passwords for TURN authentication
2. **TLS Everywhere**: Enable TURNS (TURN over TLS) for encrypted relay
3. **Rate Limiting**: Configure per-user quotas in `turnserver.conf`
4. **IP Allowlists**: Restrict relay to known peer networks if possible
5. **Regular Updates**: Keep coturn image updated for security patches
6. **Monitoring**: Set up alerts for unusual traffic patterns

## Troubleshooting

### No relay candidates

```
Symptoms: Only host/srflx candidates, no relay candidates
Cause: TURN server unreachable or auth failure
```

Fix:
1. Check firewall rules allow UDP 3478
2. Verify credentials in environment variables
3. Test with `turnutils_uclient`

### Connection timeouts

```
Symptoms: ICE gathering times out
Cause: Firewall blocking UDP or misconfigured ports
```

Fix:
1. Enable TCP transport as fallback
2. Open relay port range in firewall
3. Use TURNS on port 443 if UDP blocked

### Authentication failures

```
Symptoms: 401 Unauthorized in TURN requests
Cause: Incorrect credentials or realm mismatch
```

Fix:
1. Verify username/password match server config
2. Check realm setting matches
3. Try regenerating credentials

### High latency

```
Symptoms: Relay connections have >200ms latency
Cause: Server geographically distant from peers
```

Fix:
1. Deploy TURN servers in multiple regions
2. Use DNS-based routing to nearest server
3. Consider edge deployment (Cloudflare Spectrum, etc.)

## Alternative TURN Providers

If self-hosting is not feasible, consider these managed services:

| Provider | Free Tier | Paid Pricing | Notes |
|----------|-----------|--------------|-------|
| [Metered.ca](https://metered.ca) | 500MB/month | $0.40/GB | Easy setup, global |
| [Twilio](https://twilio.com/stun-turn) | None | $0.40/GB | Enterprise features |
| [Xirsys](https://xirsys.com) | 500MB/month | $24.99/mo | Global CDN |
| [Cloudflare Calls](https://developers.cloudflare.com/calls/) | 1000 min/mo | Custom | Low latency |

## Resources

- [coturn Documentation](https://github.com/coturn/coturn/wiki)
- [WebRTC TURN Server Guide](https://webrtc.org/getting-started/turn-server)
- [RFC 5766 - TURN](https://tools.ietf.org/html/rfc5766)
- [RFC 8656 - TURN-bis](https://tools.ietf.org/html/rfc8656)
