# 3-minute video script (target 2:45, hard limit 3:00)
Record screen + voice. Do a dry run first. **Show the server URL bar and the `s390x` line.**

| Time | Show on screen | Say |
|---|---|---|
| 0:00-0:25 | Title slide, then one shortage statistic with its source (**find one real stat and cite it**) | "Hospitals find out about shortages after they happen, and sending patient logistics data to outside clouds adds compliance risk. ResQ-Stream predicts stockouts hours ahead and reroutes stock before they happen, with data staying on IBM Z." |
| 0:25-0:55 | Flowchart from README | "Telemetry flows in; a temporal graph neural network scores every hospital and item for stockout in the next 6 hours; a min-cost-flow optimizer turns those scores into a ranked transfer list. Everything runs on IBM LinuxONE." |
| 0:55-1:20 | Jupyter on the VM: Cell 1 showing `s390x`, Cell 2 showing `risk scores from: torch` | "This is our LinuxONE instance. The model, trained on a standard machine, runs natively here. We even fixed a big-endian byte-order issue to port it. We completed both IBM labs first." |
| 1:20-2:15 | Dashboard: press Play, pause on a surge, point to red cells, a transfer, then the chart | "This replays 30 days of simulated data the model never saw. Red outlines are stockouts with ResQ-Stream; small 'base: OUT' tags are stockouts without it. When risk climbs, the optimizer sends stock from distributors or neighbouring hospitals, only on lanes that arrive in time." |
| 2:15-2:40 | Results table (notebook Cell 4 or README) | "Stockout episodes drop from 186 to 39, and unmet demand falls 87 percent, on simulated data. The optimizer decides in about 5 milliseconds." |
| 2:40-3:00 | Limitations + next steps slide | "Data is simulated, and our gradient-boosting baseline beat the GNN on accuracy, so both feed the same optimizer. Next: real-time Kafka streaming, Telum acceleration, and federated learning so hospitals share nothing but model updates." |

Checklist: under 3:00 | LinuxONE instance visible | tech stack named (PyTorch, NetworkX, s390x) | limitations stated | cite the statistic.
