// The routing policy lives in code, not in the model: Jev answers narrow questions,
// and these thresholds decide what the system does with the answers.
export const AUTO_ROUTE_THRESHOLD = Number(process.env.AUTO_ROUTE_THRESHOLD ?? 0.8);
export const AUTO_REPLY_THRESHOLD = Number(process.env.AUTO_REPLY_THRESHOLD ?? 0.9);

export function route(decision, threshold = AUTO_ROUTE_THRESHOLD) {
  if (decision.confidence < threshold) {
    return { action: 'human_review', reason: `queue confidence ${decision.confidence.toFixed(2)} < ${threshold}` };
  }
  if (decision.queue === 'other') {
    return { action: 'out_of_scope_reply', reason: 'confidently not a banking support request' };
  }
  if (decision.selfServeP >= AUTO_REPLY_THRESHOLD) {
    return { action: 'auto_reply_with_article', queue: decision.queue, reason: 'confident queue and self-serve' };
  }
  return { action: 'route_to_queue', queue: decision.queue, reason: 'confident queue' };
}
