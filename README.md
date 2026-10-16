# IT Support Ticket Router — fine-tuning Llama 3.1 8B Instruct

End-to-end workflow: dataset → Google Colab (Unsloth LoRA) → GGUF → Ollama on your Mac → test.

## Files

| File | Purpose |
|---|---|
| `NetworkTeam.json`, `SecurityTeam.json`, `SoftwareTeam.json` | Training data, 500 rows each (1,500 total) |
| `test_set.json` | 100 held-out tickets (10 per employee) that are **not** in the training data; used to measure accuracy |
| `colab_finetune_llama31_itsupport.ipynb` | Colab notebook: train, evaluate, export GGUF |
| `Modelfile.llama31-itsupport` | Ollama Modelfile for the exported GGUF |
| `test_routing.py` | Scores the Ollama model on `test_set.json`, or routes one ticket with `--ask` |

## Who handles what (ownership is strict, no overlap)

| Team | Employee | Owns |
|---|---|---|
| Network | **Shyam Bahadur** | Wi-Fi / access points / wireless controller / guest portal, VoIP & SIP phones, video-conference rooms & AV, IPTV/multicast |
| Network | **Ram Bahadur** | VPN (client & site-to-site), firewall rules / NAT / HA, web proxy & PAC, DNS records & resolution |
| Network | **Madan Bahadur** | Physical layer: cabling, wall jacks, patch panels, fiber/SFP, switch hardware/ports/PoE/err-disable/STP, wired 802.1X, racks/UPS/PDU, server-room environment |
| Network | **Hari Bahadur** | DHCP & IP addressing, subnets/VLAN routing, routers (BGP/OSPF/HSRP), WAN/ISP circuits, latency/packet loss/bandwidth, SNMP/NetFlow/Nagios |
| Security | **Krishna Prasad** | Identity & access: accounts, passwords, lockouts, MFA/tokens, SSO, AD groups, folder/app permissions, onboarding/offboarding, privileged access, badges |
| Security | **Govinda Prasad** | Certificates/PKI, vulnerability scans & CVE/patch tracking, pen tests, secure code / architecture / vendor reviews, encryption keys, hardening findings |
| Security | **Shiva Prasad** | Threat detection & incident response: EDR/AV, SIEM, DLP, CASB, email gateway, **phishing** (reports, simulations, awareness), malware/ransomware, suspicious logins, IDS/WAF alerts, lost/stolen device wipe, MDM compliance |
| Software | **Gita Kumari** | Business apps & SaaS: Microsoft 365, Zoom, Slack app, Jira/Confluence/ServiceNow, Salesforce, SAP, Tableau/Power BI, Adobe, browser policy, licenses & installs |
| Software | **Sita Kumari** | Developer tools & local dev: IDEs, Git client, package managers, local builds, test frameworks, Docker Desktop on laptops, Postman |
| Software | **Rita Kumari** | Backend infra & DevOps: database servers, Kubernetes, CI/CD pipelines, cloud (AWS/Azure/GCP), Terraform/Ansible, queues, Nginx/Tomcat, Datadog/Grafana/Prometheus |

When the same product appears for two people, the split is by the *nature* of the issue, e.g.
VPN won't connect → Ram, VPN MFA prompt → Krishna, VPN certificate expired → Govinda, VPN brute-force alert → Shiva.

Data format (Alpaca style, same as before):

```json
{
  "instruction": "Route this support issue to the correct team and employee, and suggest the first steps to resolve it.",
  "input": "Wi-Fi keeps dropping in the lobby.",
  "output": "Team: Network | Assigned to Shyam Bahadur: Check AP coverage, channel interference, and client logs for the lobby on the wireless controller."
}
```

About 35% of inputs are wrapped in natural phrasings ("Who should handle this issue: …", "URGENT: …") so the model answers correctly however you ask.

## Step 1 — Base model locally (optional, for comparison)

You already have Llama 3.1 8B in Ollama as `hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M`. If you want the official tag too:

```bash
ollama pull llama3.1:8b
```

Note: Colab does **not** use this local copy. The notebook downloads the base weights from Hugging Face itself (`unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, not gated, no token needed).

## Step 2 — Train in Google Colab

1. Open https://colab.research.google.com → `File → Upload notebook` → pick `colab_finetune_llama31_itsupport.ipynb`.
2. `Runtime → Change runtime type → T4 GPU`.
3. Run cells top to bottom. When the upload cell asks, select `NetworkTeam.json`, `SecurityTeam.json`, `SoftwareTeam.json`, `test_set.json`.
4. Training: ~280 steps, ~15 min on a T4. Section 6 prints team/employee accuracy on the held-out test set.
5. Section 8 exports `Q4_K_M` GGUF (~4.9 GB, ~20 min). Section 9 copies it to `My Drive/llama31-itsupport/`.

## Step 3 — Download and load into Ollama

1. Download `llama31-itsupport.Q4_K_M.gguf` from Google Drive into this folder (`ITSupportTeam/`).
2. Build and run:

```bash
cd ~/Projects/FineTuneAI/ITSupportTeam
ollama create it-support -f Modelfile.llama31-itsupport
ollama run it-support "Who should handle this: the wall jack at my desk has no link light?"
```

Expected: `Team: Network | Assigned to Madan Bahadur: Check the switch port ...`

### Fallback if GGUF export fails on the free Colab tier

Download `lora_model.zip` from Drive instead, unzip it here, and build on top of the base model that Ollama already has:

```
FROM hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M
ADAPTER ./lora_model
SYSTEM """Route this support issue to the correct team and employee, and suggest the first steps to resolve it."""
PARAMETER temperature 0
```

Accuracy is usually slightly lower this way because the adapter was trained against a different 4-bit quantization of the base.

## Step 4 — Test

```bash
# Score on the 100 held-out tickets (prints per-employee accuracy and every mistake)
python3 test_routing.py

# Route a single ticket
python3 test_routing.py --ask "Employee got a text pretending to be IT asking for their password"

# Compare against the untuned base model
python3 test_routing.py --model hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M \
  --system "Route this support issue to the correct team and employee, and suggest the first steps to resolve it. Answer as: Team: <team> | Assigned to <name>: <steps>"
```

## Tuning tips

- Low accuracy for one employee → look at the `Mistakes` list; add 20–40 rows for the confused topic to that employee's file and retrain.
- Model answers chit-chat with a routing line → that is by design (the system prompt is fixed to routing). Use a different Ollama model for general chat.
- Want JSON output instead → change the `output` field format in the JSON files to `{"team": "...", "employee": "...", "steps": "..."}` and retrain; the notebook and Modelfile do not need to change (update the regexes in `test_routing.py`).
