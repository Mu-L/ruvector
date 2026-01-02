#!/usr/bin/env node
/**
 * @ruvector/edge-net Genesis Node Health Check
 *
 * Standalone health check script for use with container orchestrators,
 * load balancers, and monitoring systems.
 *
 * Usage:
 *   node deploy/health-check.js                    # Check localhost:8788
 *   node deploy/health-check.js --host 10.0.0.1   # Check specific host
 *   node deploy/health-check.js --port 9000       # Check specific port
 *   node deploy/health-check.js --endpoint ready  # Check readiness
 *   node deploy/health-check.js --json            # JSON output
 *
 * Exit codes:
 *   0 - Healthy
 *   1 - Unhealthy or error
 *
 * @module @ruvector/edge-net/deploy/health-check
 */

import http from 'http';

// Parse arguments
const args = process.argv.slice(2);
const config = {
    host: 'localhost',
    port: 8788,
    endpoint: 'health',
    timeout: 5000,
    json: false,
};

for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
        case '--host':
        case '-h':
            config.host = args[++i];
            break;
        case '--port':
        case '-p':
            config.port = parseInt(args[++i]);
            break;
        case '--endpoint':
        case '-e':
            config.endpoint = args[++i];
            break;
        case '--timeout':
        case '-t':
            config.timeout = parseInt(args[++i]);
            break;
        case '--json':
        case '-j':
            config.json = true;
            break;
        case '--help':
            console.log(`
Genesis Node Health Check

Usage: node health-check.js [options]

Options:
  --host, -h <host>       Host to check (default: localhost)
  --port, -p <port>       Port to check (default: 8788)
  --endpoint, -e <path>   Endpoint to check: health, ready, status, metrics (default: health)
  --timeout, -t <ms>      Request timeout in milliseconds (default: 5000)
  --json, -j              Output JSON format
  --help                  Show this help

Examples:
  node health-check.js
  node health-check.js --host genesis.example.com --port 8788
  node health-check.js --endpoint ready
  node health-check.js --json

Exit Codes:
  0 - Healthy/Ready
  1 - Unhealthy/Not Ready/Error
`);
            process.exit(0);
    }
}

function checkHealth() {
    return new Promise((resolve, reject) => {
        const startTime = Date.now();

        const req = http.get({
            hostname: config.host,
            port: config.port,
            path: `/${config.endpoint}`,
            timeout: config.timeout,
        }, (res) => {
            let data = '';

            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                const latency = Date.now() - startTime;

                try {
                    const parsed = JSON.parse(data);
                    resolve({
                        healthy: res.statusCode === 200,
                        statusCode: res.statusCode,
                        latency,
                        data: parsed,
                    });
                } catch {
                    resolve({
                        healthy: res.statusCode === 200,
                        statusCode: res.statusCode,
                        latency,
                        data: data,
                    });
                }
            });
        });

        req.on('error', (err) => {
            reject({
                healthy: false,
                error: err.message,
                latency: Date.now() - startTime,
            });
        });

        req.on('timeout', () => {
            req.destroy();
            reject({
                healthy: false,
                error: 'Request timeout',
                latency: config.timeout,
            });
        });
    });
}

async function main() {
    try {
        const result = await checkHealth();

        if (config.json) {
            console.log(JSON.stringify({
                ...result,
                host: config.host,
                port: config.port,
                endpoint: config.endpoint,
                timestamp: new Date().toISOString(),
            }, null, 2));
        } else {
            if (result.healthy) {
                console.log(`OK - ${config.host}:${config.port}/${config.endpoint} (${result.latency}ms)`);

                if (result.data?.ready !== undefined) {
                    console.log(`  Ready: ${result.data.ready}`);
                }
                if (result.data?.status) {
                    console.log(`  Status: ${result.data.status}`);
                }
            } else {
                console.log(`FAIL - ${config.host}:${config.port}/${config.endpoint}`);
                console.log(`  Status Code: ${result.statusCode}`);
            }
        }

        process.exit(result.healthy ? 0 : 1);

    } catch (error) {
        if (config.json) {
            console.log(JSON.stringify({
                healthy: false,
                host: config.host,
                port: config.port,
                endpoint: config.endpoint,
                error: error.error || error.message,
                timestamp: new Date().toISOString(),
            }, null, 2));
        } else {
            console.log(`ERROR - ${config.host}:${config.port}/${config.endpoint}`);
            console.log(`  Error: ${error.error || error.message}`);
        }

        process.exit(1);
    }
}

main();
