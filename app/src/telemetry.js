// Latency and confidence metrics in CloudWatch embedded metric format (EMF).
// Web Functions extract EMF from stdout; console.* output is wrapped by the runtime and
// isn't parsed, so write the JSON line directly.
export function emitMetrics({ jevMs, totalMs, confidence, action, source }) {
  process.stdout.write(JSON.stringify({
    _aws: {
      Timestamp: Date.now(),
      CloudWatchMetrics: [{
        Namespace: 'JevTriage',
        Dimensions: [['Source'], ['Source', 'Action']],
        Metrics: [
          { Name: 'JevLatency', Unit: 'Milliseconds' },
          { Name: 'RequestLatency', Unit: 'Milliseconds' },
          { Name: 'QueueConfidence', Unit: 'None' },
        ],
      }],
    },
    Source: source,
    Action: action ?? 'none',
    JevLatency: jevMs,
    RequestLatency: totalMs,
    QueueConfidence: confidence,
  }) + '\n');
}
