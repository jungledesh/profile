#!/usr/bin/env bash
# Hero's Journey demo load generator — all acts use MODE=rag.
#
# Usage:
#   MODE=rag ./load.sh                    # default: Act 1 (CONCURRENCY=4 LAMBDA=4)
#   CONCURRENCY=50 LAMBDA=0.5 MODE=rag ./load.sh   # Acts 2-4
#
# Tuning env vars (rag mode):
#   CONCURRENCY=N   concurrent workers (default: 4)
#   LAMBDA=N        mean think time in seconds per worker (default: 4)
#   MAX_TOKENS=N    max output tokens per request (default: random 80-480)
#   MODEL=NAME      served model name (default: Qwen3.6-27B)
#   VLLM_URL=URL    vLLM endpoint (default: http://localhost:8000)
#
# Each request = ~3K-token shared system prompt + ~900-token doc + ~30-token instruction
# ≈ 4,000 tokens total. System prompt is identical across all workers (prefix cache target).
#
# Runs forever. Kill with Ctrl-C.

set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${MODEL:-llama3}"
MODE="${MODE:-r1}"

post() {
  local prompt="$1" max_tokens="$2"
  jq -n \
    --arg model "$MODEL" \
    --arg prompt "$prompt" \
    --argjson max_tokens "$max_tokens" \
    '{model:$model, messages:[{role:"user",content:$prompt}],
      max_tokens:$max_tokens, temperature:0, stream:false}' \
  | curl -sS -o /dev/null \
      -H "Content-Type: application/json" \
      -d @- \
      "${VLLM_URL}/v1/chat/completions" || true
}

new_uuid() {
  if command -v uuidgen >/dev/null 2>&1; then
    uuidgen
  elif [[ -r /proc/sys/kernel/random/uuid ]]; then
    cat /proc/sys/kernel/random/uuid
  else
    echo "req-$(date +%s)-$RANDOM"
  fi
}

long_context() {
  local chunks="${CONTEXT_CHUNKS:-40}"
  python3 -c "
chunk = 'Distributed inference, GPU scheduling, KV cache growth, prefill-decode balance, memory pressure, and latency spikes under multi-tenant workloads. '
print(''.join(f'[{i:03d}] ' + chunk for i in range(1, int('${chunks}') + 1)))
"
}

LONG_CTX="$(long_context)"

load_r1() {
  # Continuous low-concurrency load — no sleep gaps.
  # CONCURRENCY controls batch size. Keep small to starve the GPU each decode
  # step without burst/idle cycles that skew NVML averages.
  local concurrency="${CONCURRENCY:-2}"
  local max_tokens="${MAX_TOKENS:-600}"
  local prompt="Write a detailed essay on GPU architecture and how tensor cores accelerate matrix multiplication."
  while true; do
    for i in $(seq 1 "$concurrency"); do
      post "$prompt" "$max_tokens" &
    done
    wait
  done
}

load_seq() {
  # 1 sequential request at a time — maximum under-batching.
  # Each decode step processes exactly 1 token, exposing raw weight-load cost.
  local max_tokens="${MAX_TOKENS:-600}"
  local prompt="Write a detailed essay on GPU architecture and how tensor cores accelerate matrix multiplication."
  while true; do
    post "$prompt" "$max_tokens"
  done
}

load_r2() {
  # High concurrency, long context, unique prompts — no sleep gaps.
  # Unique IDs bust prefix cache so KV blocks are not reused across requests.
  # See r2 prerequisite note at the top of this file.
  local concurrency="${CONCURRENCY:-8}"
  local max_tokens="${MAX_TOKENS:-256}"
  while true; do
    for ((i = 0; i < concurrency; i++)); do
      local uid
      uid="$(new_uuid)"
      post "[REQ-${uid}]
${LONG_CTX}

Summarise the above. List 10 risks and 10 recommendations." "$max_tokens" &
    done
    wait
  done
}

load_r5() {
  # Keeps exactly CONCURRENCY requests in flight at all times.
  # No batch gaps — as each request completes, a new one fires immediately.
  # This saturates max_num_seqs and builds a persistent wait queue.
  # Keep prompts short so KV stays healthy (r2 should not fire).
  local concurrency="${CONCURRENCY:-512}"
  local max_tokens="${MAX_TOKENS:-4096}"
  local prompt="Explain in detail what a transformer model is, how attention works, and why it replaced RNNs for sequence modeling tasks."
  for ((i = 0; i < concurrency; i++)); do
    while true; do
      post "$prompt" "$max_tokens"
    done &
  done
  wait
}

load_demo() {
  # Coding assistant scenario: all requests share a long system prompt (prefix),
  # each user asks a different code question. Realistic mix — 70% shared prefix
  # tokens, 30% unique. Concurrency tuned to fill max_num_seqs=32 and spill into
  # the queue so R5 fires after R3 is addressed.
  local concurrency="${CONCURRENCY:-50}"
  local max_tokens="${MAX_TOKENS:-300}"

  # Long shared system prompt — same across every request (prefix cache target)
  local system_prompt="You are an expert software engineer and coding assistant. \
You write clean, efficient, well-documented code. You follow best practices for \
the language in use, prefer clarity over cleverness, and always explain your \
reasoning. When asked to review code, you identify bugs, suggest improvements, \
and explain the trade-offs. You are familiar with Python, Rust, Go, TypeScript, \
and C++. You respond concisely unless asked for detail."

  # Varied code questions — different each request to avoid trivial cache hits
  # on the question itself, while the system prompt stays shared
  local questions=(
    "Write a Python function that flattens a nested list of arbitrary depth."
    "Review this Rust code for memory safety issues: fn get(v: &Vec<i32>, i: usize) -> i32 { v[i] }"
    "Implement a rate limiter in Go using a token bucket algorithm."
    "What is the difference between a mutex and a semaphore? Give a code example in C++."
    "Write a TypeScript utility type that makes all nested properties optional."
    "Explain why this Python code has a bug: def append(x, lst=[]): lst.append(x); return lst"
    "Write an async Rust function that retries an HTTP request up to 3 times with exponential backoff."
    "How do I implement a trie in Python? Show insert and search."
    "What does 'move semantics' mean in Rust? Give a before/after example."
    "Write a Go function that finds all duplicate elements in a slice."
    "Implement a debounce function in TypeScript."
    "Explain the difference between heap and stack allocation with C++ examples."
    "Write a Python decorator that measures and logs function execution time."
    "How do I avoid data races in Rust when sharing state across threads?"
    "Write a binary search implementation in Go with proper error handling."
  )
  local n=${#questions[@]}

  for ((i = 0; i < concurrency; i++)); do
    (
      idx=0
      while true; do
        local q="${questions[$((idx % n))]}"
        idx=$((idx + 1))
        local combined="${system_prompt}

User: ${q}"
        post "$combined" "$max_tokens"
      done
    ) &
  done
  wait
}

load_rag() {
  # Hero's Journey demo — 4-act RAG scenario for A100 SXM 80GB / Qwen3.6-27B BF16.
  #
  # Each request = shared 3K-token system prompt + unique ~900-token doc + instruction
  # ≈ 4,000 tokens total. The shared prefix is the prefix-cache target for Act 4.
  #
  # Act 1 defaults (under-batching):   CONCURRENCY=4  LAMBDA=4
  # Act 2 defaults (KV pressure):      CONCURRENCY=50 LAMBDA=0.5
  # Act 3 defaults (concurrency sat):  CONCURRENCY=50 LAMBDA=0.5  (vLLM restarted)
  # Act 4 defaults (sweet spot):       CONCURRENCY=50 LAMBDA=0.5  (prefix caching on)
  local workers="${CONCURRENCY:-4}"
  local mean_think="${LAMBDA:-4}"

  # Shared system prompt — identical across every request.
  # ~3,000 tokens; combined with the unique doc (~900 tok) + instruction (~30 tok) ≈ 4K total.
  # Prefix caching in Act 4 deduplicates this prefix across all concurrent users.
  local RAG_SYSTEM_PROMPT
  RAG_SYSTEM_PROMPT=$(cat <<'SYSTEM_EOF'
You are a senior enterprise document intelligence assistant deployed in a Retrieval-Augmented Generation (RAG) pipeline for a large enterprise organization. Your function is to analyze complex enterprise document excerpts and generate accurate, structured, and immediately actionable responses for knowledge workers across legal, finance, engineering, compliance, executive, and operations functions.

Your outputs inform consequential decisions: contract commitments, capital allocation, regulatory filings, system architecture changes, and personnel actions. Accuracy is non-negotiable. Omissions have consequences. Fabricated facts are catastrophic.

## Role and Scope

You analyze the full range of enterprise document types: master services agreements, software license agreements, non-disclosure agreements, statements of work, addenda and order forms; quarterly and annual financial reports, earnings releases, management discussion and analysis sections, and board-level financial summaries; technical architecture specifications, system design documents, runbooks, post-mortem reports, and incident timelines; ISO 27001, SOC 2 Type II, PCI-DSS, and HIPAA compliance assessments and audit findings; merger and acquisition due diligence packages, investment committee memos, and term sheet analyses; engineering roadmaps, project status reports, and dependency maps; clinical trial protocols, IRB submissions, and regulatory correspondence; supply chain risk assessments and vendor management reports; and internal policy documents covering HR, security, procurement, and finance.

For each document type, you apply the appropriate analytical framework, use the correct domain vocabulary, and identify the signals that matter most to the implied audience.

## Analytical Frameworks by Document Type

### Legal and Contractual Documents

Identify: the parties and their roles; the scope of services or license grant; payment terms including amounts, schedules, and conditions; the limitation of liability clause including cap amount, carve-outs, and trigger conditions; indemnification obligations flowing in each direction and the events that trigger them; intellectual property ownership, license grants, and work-for-hire provisions; termination rights, notice periods, and post-termination obligations; governing law and dispute resolution mechanism; and representations, warranties, and covenants that carry ongoing compliance obligations.

Limitation of liability clauses typically cap exposure at fees paid in the prior twelve months and exclude consequential, indirect, special, and exemplary damages. Deviations from this standard are material. Indemnification clauses that cover IP infringement, data breaches, and gross negligence asymmetrically create unequal risk between parties. Identify when a contract lacks standard protections that peer agreements typically include, such as data processing addenda, business continuity obligations, or audit rights.

### Financial and Accounting Documents

Identify: revenue and its growth rate year-over-year and quarter-over-quarter; gross profit and gross margin; operating income and operating margin; net income and diluted earnings per share; free cash flow calculated as operating cash flow minus capital expenditure; cash and equivalents versus total debt; and forward guidance ranges with the assumptions underlying them.

Standard warning signs: gross margin compression despite revenue growth indicates pricing pressure or rising COGS; free cash flow declining faster than net income indicates working capital deterioration or elevated capex; deferred revenue declining signals renewal pipeline weakness; revenue concentrated in three or fewer customers indicates key-account risk; and guidance ranges narrower than historical variance suggest either management confidence or deliberate sandbagging to manage consensus expectations.

Perform basic financial calculations when relevant: year-over-year growth rates, margin calculations, cash runway at stated burn rate, and customer concentration percentages. Show the arithmetic. Flag internal inconsistencies between numbers reported in different sections of the same document.

### Technical Architecture and Engineering Documents

Identify: system components and their functions; interfaces and data flows between components; stated performance characteristics and capacity limits; single points of failure and redundancy provisions; dependencies on external systems, third-party services, or specific hardware; operational assumptions that may not hold under production load; SLAs and SLOs with clarity on whether they are aspirational or contractually binding; and known limitations, deferred work, and technical debt explicitly acknowledged in the document.

Flag: critical components without a documented failover path; capacity limits within twenty percent of current utilization; external dependencies with no fallback that would cause cascading failure; operational assumptions that contradict known production behavior; and deferred work items that will become blocking dependencies given stated growth trajectories.

### Compliance and Audit Documents

Identify: the compliance framework and scope; findings by control domain; severity classification of each finding as critical, significant deficiency, or observation; the remediation owner and deadline for each finding; the current status of findings from prior audit cycles; and the overall compliance posture relative to the applicable standard.

ISO 27001 domains most frequently cited in enterprise audits: A.9 Access Control (privileged account review failures, orphaned accounts with active credentials); A.10 Cryptography (deprecated TLS versions in use, overdue certificate rotation); A.12 Operations Security (incomplete log retention, patch cadence exceeding SLA); A.15 Supplier Relationships (expired third-party SOC 2 reports, undocumented vendor assessments); A.16 Incident Management (mean time to contain exceeding SLA, escalation tree degradation after personnel changes). Recurring findings across multiple audit cycles indicate systemic non-remediation rather than isolated failures.

Flag: remediation deadlines that have passed without documented completion; findings recurring across multiple consecutive audit cycles; expired third-party certification reports creating contractual or regulatory exposure; and control failures that directly affect customer data, payment processing, or access to production systems.

### Incident Post-Mortems

Identify: the complete timeline from alert to containment to resolution; time-to-detect and time-to-contain against stated SLAs; business impact including error rate, affected request volume, revenue impact, and customers breaching their SLAs; the root cause and the causal chain; contributing factors that amplified impact or delayed detection; and action items with their owners, deadlines, and priority levels.

Common post-mortem failure patterns: alert thresholds misconfigured causing late detection; canary deployments covering an insufficient fraction of traffic; runbooks that are outdated and do not describe the current failure mode; on-call rotation changes that broke escalation chains without documentation updates; staging environments diverged from production configuration such that pre-deployment testing does not catch production regressions; absence of configuration drift detection between environments.

Flag: action items without assigned owners or deadlines; root causes appearing identically across two or more prior incidents indicating systematic non-remediation; contributing factors implicating process failures rather than isolated human error.

### Mergers, Acquisitions, and Investment Documents

Identify: transaction structure and total consideration; target revenue, growth rate, and gross margin relative to peer benchmarks; customer concentration and the revenue percentage attributable to the top two or three accounts; technical debt assessment and estimated engineer-months to remediate; key-person dependencies identifying named individuals and the specific functions dependent on each; IP ownership status including patents filed versus granted and open-source licensing obligations; current cash position and runway at stated burn rate; and the final recommendation with stated conditions and contingencies.

Flag: customer concentration above forty percent in the top two accounts; gross margin below comparable SaaS peers suggesting structural COGS issues; key-person risk without a documented succession plan or retention package as a condition of close; GPL or AGPL licensed dependencies in commercial products requiring legal review; provisional patents not yet granted providing limited defensive value; and earn-out structures that create incentive misalignment post-close.

### Supply Chain and Vendor Documents

Identify: affected components and current lead times compared to prior periods; downstream impact on planned programs or deliverables; recommended mitigation actions with owners and deadlines; pricing changes and contract renewal risk; and secondary dependencies that amplify the primary constraint.

Flag: lead time extensions that create blocking dependencies on time-sensitive program milestones; single-source dependencies with no qualified alternative supplier; concurrent constraints across multiple input categories that compound overall supply risk; and pricing changes that materially alter unit economics or threaten margin commitments made in customer contracts.

### Human Resources and Policy Documents

Identify: the scope of the policy and the employee populations to which it applies; specific obligations, prohibitions, and required procedures; exception processes and approval authority; monitoring and enforcement mechanisms; effective dates, review cycles, and version history; and cross-references to other policies or legal requirements.

Flag: policies that create compliance exposure if not enforced consistently; approval processes with ambiguous ownership creating accountability gaps; monitoring provisions that may conflict with data protection law in specific employee jurisdictions; equipment and data handling obligations inconsistent with the organization's stated information security controls; and stipends or reimbursements that may create taxable income issues in certain jurisdictions.

### Clinical and Research Documents

Identify: the study design and primary endpoint; the patient population including inclusion and exclusion criteria; sample size and statistical power assumptions; safety monitoring protocols including stopping rules and review schedule; regulatory filing status and outstanding requirements; timeline and milestone schedule; and identified risks to study completion or endpoint achievement.

Flag: primary endpoints that are not pre-registered; stopping rules that are asymmetric in a way that biases toward continuation over safety; safety monitoring review intervals too infrequent given the risk profile of the intervention; sample size assumptions based on effect sizes larger than prior literature supports; and regulatory submissions in multiple jurisdictions where requirements may conflict.

## Response Quality Standards

**Accuracy**: Every factual claim must be directly traceable to the provided document. Do not invent figures, names, dates, percentages, or obligations. When you perform arithmetic, show the calculation inline. When a number is approximate or inferred, say so explicitly.

**Completeness**: When the task is extraction, your extraction must be exhaustive within the scope of the provided document. Do not silently omit material items. If the document is incomplete or ambiguous on a specific point, say so and specify what additional information would resolve the ambiguity.

**Calibrated confidence**: Use definitive language when the document is definitive. Use qualified language when the document is ambiguous. Do not hedge when certainty is warranted. Do not assert certainty when ambiguity exists.

**Actionability**: Identified problems should be paired with recommended responses when possible. A risk without a proposed mitigation is less useful than a risk with one, even if the recommendation is directional rather than specific.

**Conciseness**: Do not restate the question. Do not prefix with AI disclaimers. Do not summarize what you are about to say. Deliver the substance immediately and stop when the substance is delivered.

## Output Formatting

Use tables for multi-attribute comparisons and when the user requests tabular output. Use numbered lists when sequence or priority matters. Use bullet lists for unordered enumerations. Use headers for responses spanning multiple distinct sections. Use plain prose for short answers and narrative recommendations.

Bold key terms, figures, and deadlines on first reference in dense sections. Do not bold entire sentences. When the user specifies an output format, follow that specification exactly, even when it requires omitting detail you would otherwise include.
SYSTEM_EOF
)

  # Unique document excerpts — each ~800-1,000 tokens (realistic enterprise content).
  # Combined with RAG_SYSTEM_PROMPT: ~3,800-4,000 tokens per request.
  # shellcheck disable=SC2034
  local docs=(
    "Master Services Agreement — Section 4.2 Limitation of Liability: In no event shall either party be liable to the other for any indirect, incidental, special, exemplary, or consequential damages arising out of or related to this agreement, including but not limited to loss of revenue, loss of profits, loss of business, or loss of data, even if such party has been advised of the possibility of such damages. The aggregate liability of either party for direct damages shall not exceed the total fees paid or payable by customer in the twelve months preceding the claim. These limitations apply regardless of the form of action, whether in contract, tort, negligence, strict liability, or otherwise. Section 4.3 Indemnification: Each party shall indemnify, defend, and hold harmless the other party and its officers, directors, employees, and agents from and against any claims, damages, losses, and expenses arising out of or related to: (a) the indemnifying party's breach of this agreement; (b) the indemnifying party's negligence or willful misconduct; or (c) any third-party claims arising from the indemnifying party's products or services."

    "Q3 FY2025 Financial Results — Consolidated P&L (unaudited): Total revenue \$847.3M (+14.2% YoY). Cloud services \$412.1M (+31.0%). Professional services \$218.6M (+9.4%). License \$216.6M (-3.1%). Gross profit \$520.5M; gross margin 61.4% (-180bps YoY). R&D expense \$142.3M (16.8% of revenue). S&M expense \$198.7M (23.5% of revenue). G&A expense \$61.2M (7.2% of revenue). Operating income \$118.3M; operating margin 14.0%. Net income \$94.1M; diluted EPS \$1.42. Free cash flow \$112.0M (-40.7% YoY). Cash and equivalents \$1.04B. Q4 guidance: revenue \$880–\$910M, operating margin 8–10%, reflecting planned headcount additions and increased infrastructure spend for capacity expansion. Key risks: FX headwinds (-\$18M impact at current rates), competitive pricing pressure in enterprise segment, three large deals slipped from Q3 into Q4 with combined TCV of \$47M."

    "System Architecture Specification v2.3 — Inference Serving Layer: The production serving stack is composed of three tiers. Tier 1 (Edge): Two HAProxy load balancers in active-active configuration handle TLS 1.3 termination, connection pooling, and weighted round-robin routing. Health checks run every 2 seconds; unhealthy backends removed within 6 seconds. Tier 2 (Gateway): Stateless API gateway performs JWT validation, per-tenant rate limiting (1000 RPM default, configurable per contract), request logging, and payload size enforcement (max 512KB). Tier 3 (Inference): Pool of GPU nodes running vLLM 0.4.x with continuous batching. Each node serves a single model replica. Horizontal scaling is manual; autoscaling is not yet implemented. The scheduler uses FIFO ordering with preemption disabled. KV cache is pre-allocated at node startup; size is determined by available VRAM after model weights are loaded. Nodes expose Prometheus metrics at :8000/metrics scraped every 15 seconds by the central monitoring stack. Known limitation: no cross-node request routing; a request pinned to a node with high cache utilization will queue behind local traffic even if peer nodes are idle."

    "ISO 27001 Gap Assessment — Annual Audit Cycle Q3 2025: SCOPE: All production systems handling customer PII and financial data across three regions (US-East, EU-West, AP-Southeast). FINDINGS: A.9 Access Control — 34% of privileged accounts reviewed in past 12 months vs 100% quarterly requirement. 12 orphaned service accounts identified; 4 with active credentials. Remediation owner: IT-Ops. Deadline: 30 days. A.10 Cryptography — 3 internal microservices still accept TLS 1.1 connections. Certificate rotation for 2 externally-facing services overdue by 47 days. A.12 Operations Security — Centralized log retention enforced on 87% of systems; 6 legacy on-premises servers excluded from SIEM due to unsupported OS versions. Patch cadence for critical CVEs: mean 18 days vs 7-day SLA. A.15 Supplier Relationships — 2 Tier-1 vendors (payment processing, identity provider) do not have current SOC 2 Type II reports on file. Last reports dated 14 and 22 months ago respectively. A.16 Incident Management — Mean time to detect (MTTD) severity-1: 11 minutes. Mean time to contain (MTTC): 4.2 hours vs 2-hour SLA. 3 of 5 severity-1 incidents in the past 6 months exceeded SLA. Root cause: paging escalation tree not maintained after two on-call rotations changed personnel."

    "Engineering Roadmap H2 2025 — Inference Platform: Initiative: Cost Reduction. Target: 40% reduction in cost-per-token by Dec 31. Owner: Platform Engineering. Workstream 1 — Batching: Deploy continuous batching across all inference nodes. Expected gain: 15–20% throughput improvement. Status: In progress, 60% complete. Blocker: Integration testing on models with custom attention kernels. ETA: Week 34. Workstream 2 — Quantization: Roll out INT8 weight quantization to all production models. Expected gain: 2x memory efficiency, enabling larger batch sizes. Status: Pilot complete on 3 models; quality regression observed on math benchmarks (-2.1 points on MATH dataset). Decision pending: accept regression or switch to FP8. ETA: Week 38. Workstream 3 — Routing: Implement prompt-length-based routing to direct requests under 512 tokens to a smaller, cheaper model variant. Expected gain: 25% cost reduction on short-context traffic (estimated 40% of total volume). Status: Design review complete. Implementation not started. ETA: Week 42. Cross-cutting dependency: all three workstreams require updates to the metrics pipeline to track per-model cost attribution. Current pipeline does not support model-level cost breakdown."

    "M&A Due Diligence Report — Target: Meridian AI Labs (Series B, \$34M ARR): FINANCIAL: Revenue growth 112% YoY; however 68% of ARR concentrated in 3 customers, largest representing 31% alone. Gross margin 52% — below comparable SaaS peers (65–75%) due to high GPU infrastructure cost embedded in COGS. Burn rate \$2.1M/month; 14 months runway at current cash position. Deferred revenue \$8.4M suggests strong renewal pipeline but payment timing risk. TECHNICAL: Core differentiator is proprietary fine-tuning pipeline; reviewed codebase — significant technical debt in data preprocessing layer, estimated 3–4 engineer-months to bring to production standards. Model serving infrastructure built on open-source vLLM with light customization; easily portable. No custom hardware or silicon dependencies. IP review: 2 provisional patents filed, neither granted. 4 open-source dependencies with GPL-2.0 licensing require legal review for commercial use. PEOPLE: 23 employees; 14 engineers. Key-person risk: CTO holds primary relationships with 2 of 3 largest customers and authored 60% of core model training code. No documented succession plan. Attrition risk rated HIGH. RECOMMENDATION: Proceed to final bid with \$127M offer contingent on 24-month CTO retention package and customer contract assignment clauses."

    "Security Vulnerability Report — CVE-2025-3847 (Critical, CVSS 9.1): SUMMARY: Unauthenticated remote code execution in the model serving API endpoint /v1/generate when the request body contains a specially crafted JSON payload with a nested key depth exceeding 512 levels. The vulnerability exists in the JSON deserialization path shared by all inference endpoints. AFFECTED VERSIONS: vLLM 0.3.0 through 0.4.2. IMPACT: Full host compromise. An attacker can execute arbitrary code with the privileges of the inference server process, which in default deployments runs as root. EXPLOIT COMPLEXITY: Low. No authentication required. Proof-of-concept exploit publicly available as of 2025-08-14. AFFECTED SYSTEMS IN PRODUCTION: 12 inference nodes across 3 clusters. All running vLLM 0.4.1. PATCH AVAILABLE: vLLM 0.4.3 released 2025-08-15. RECOMMENDED ACTION: Emergency patch within 24 hours. If patching not immediately possible, mitigate by restricting /v1/generate to internal network only via firewall rules and adding JSON depth validation at the API gateway layer. Do not expose affected versions to public internet."

    "Board Meeting Minutes — Q3 2025 Audit Committee: ATTENDEES: 4 of 5 committee members present (quorum met). Management: CFO, General Counsel, VP Engineering. ITEM 1 — Financial Controls: CFO presented Q3 close results. No material weaknesses identified. One significant deficiency noted: revenue recognition timing for multi-element arrangements requires manual override in 3 of 47 enterprise contracts due to system limitation in billing platform. Remediation plan presented; system fix expected Q1 2026. ITEM 2 — Cybersecurity: VP Engineering briefed committee on the August severity-1 incident (47-minute outage, root cause: misconfigured deployment). Committee expressed concern about lack of pre-production load testing gate. Management committed to implementing mandatory staging validation before year-end. ITEM 3 — Vendor Risk: General Counsel reported two critical vendors without current SOC 2 reports. Committee directed management to obtain reports or initiate vendor transition within 90 days. ITEM 4 — Internal Audit Findings: External auditors presented ISO 27001 gap summary. Committee accepted findings, directed management to provide remediation status update at next meeting. NEXT MEETING: Scheduled Q4, date TBD pending earnings calendar."

    "Clinical Trial Protocol Summary — Phase 2b, NCT-2025-00847: STUDY TITLE: Efficacy and Safety of AI-Assisted Diagnostic Support in Emergency Radiology Settings. SPONSOR: Meridian Diagnostics Inc. INDICATION: Suspected intracranial hemorrhage in emergency department patients. PRIMARY ENDPOINT: Time from CT scan acquisition to radiologist sign-off, comparing AI-assisted workflow vs standard workflow. Non-inferiority margin: 5 minutes. Secondary endpoints: sensitivity and specificity for hemorrhage detection, false positive rate, radiologist workload metrics. POPULATION: 480 patients across 6 academic medical centers. Inclusion: adult patients presenting to ED with neurological symptoms and ordered CT head. Exclusion: known prior hemorrhage, transferred patients with existing imaging. SAFETY MONITORING: Independent DSMB reviews unblinded data after every 80 patients. Pre-specified stopping rules: if AI false negative rate for hemorrhage exceeds 5% at any interim analysis, study halts. REGULATORY: IND submitted and accepted. IRB approval obtained at all 6 sites. EU MDR Technical File in preparation for parallel regulatory submission."

    "Supply Chain Risk Notice — Semiconductor Lead Time Update Q3 2025: This notice summarizes current lead time status for critical components across our hardware supply chain. GPU (H100 SXM5): Current lead time 52 weeks, up from 34 weeks in Q2. Allocation secured through Q1 2026 via existing PO. Risk: any demand increase beyond current PO volume cannot be accommodated before Q2 2026 at earliest. Recommended action: freeze hardware expansion plans requiring H100 through Q1 2026 or identify alternative GPU sourcing. Networking (InfiniBand NDR 400G): Lead time 38 weeks. 2 of 4 planned cluster expansion switches delayed; cluster expansion timeline pushed from Week 40 to Week 52. Memory (HBM3e): Spot market pricing increased 23% MoM due to AI demand. Current contract pricing locked through Oct 31. Renegotiation scheduled Nov 1; expect 15–20% price increase on renewal. Power Infrastructure: Datacenter power upgrade for Building C delayed 6 weeks due to utility permitting. GPU rack deployment in Building C blocked until power upgrade completes, now estimated Week 48."

    "HR Policy Document — Remote Work and Equipment Policy v4.1: SCOPE: Applies to all full-time employees and contractors with system access. EQUIPMENT: Company-provided laptops are mandatory for all roles with access to production systems, customer data, or source code. Personal devices may not be used to access internal systems. Equipment refresh cycle: 3 years for standard roles, 2 years for engineering roles. Lost or stolen equipment must be reported within 4 hours to IT Security; device will be remotely wiped. REMOTE WORK: Employees may work remotely up to 3 days per week. Roles designated as office-required (listed in Appendix A) are exempt and require 5 days on-site. Remote work from outside the country of employment requires prior written approval from HR and Legal; approval process takes 10 business days minimum and is not guaranteed. HOME OFFICE STIPEND: \$75/month for internet reimbursement. One-time \$500 home office setup allowance for employees hired after Jan 1 2024. Receipts required within 60 days of purchase. MONITORING: Company reserves the right to monitor activity on company-issued devices and company network connections. Employees are notified of this policy at onboarding and annually."

    "Incident Post-Mortem — Severity 1, Duration 47 minutes, August 14 2025: TIMELINE: 14:23 UTC — Automated alert fires on elevated 5xx error rate (threshold: >1% over 5 minutes). 14:26 UTC — On-call engineer acknowledges page. 14:31 UTC — Root cause identified: rolling deployment pushed gpu-memory-utilization config from 0.85 to 0.98 across 8 of 12 inference nodes. 14:44 UTC — Decision made to roll back; rollback initiated. 15:10 UTC — Rollback complete across all nodes. 15:14 UTC — Error rate returns to baseline. IMPACT: 23% of requests failed during incident window. Estimated \$47K revenue impact. 3 enterprise customers exceeded their error SLA; customer success follow-up required. ROOT CAUSE: Config change was not covered by staging load test because staging cluster uses a different gpu-memory-utilization ceiling (0.90) than production (0.85). The regression was not caught in pre-deployment validation. CONTRIBUTING FACTORS: (1) No automated config diff between staging and production. (2) Canary deployment covered only 1% of traffic; insufficient to surface the issue before full rollout. (3) Runbook for KV cache OOM was outdated; on-call engineer lost 5 minutes consulting incorrect mitigation steps. ACTION ITEMS: [P0] Add mandatory gpu-memory-utilization validation to deployment pipeline — owner: Platform, due Week 34. [P0] Update KV cache OOM runbook — owner: On-call lead, due Week 33. [P1] Increase canary traffic to 10% — owner: Platform, due Week 36. [P2] Implement config drift detection between staging and production — owner: Infra, due Week 40."

    "Research Findings — LLM Serving Efficiency Study, 50-Node Production Cluster: We analyzed 90 days of production telemetry from a 50-node H100 cluster serving a mixture of enterprise RAG and chat workloads. Key findings: (1) GPU utilization averaged 34% across the observation period despite p99 TTFT exceeding SLA targets on 12% of days, indicating that low utilization and high latency coexist — the classic under-batching signature. (2) KV cache utilization exhibited a bimodal distribution: 71% of time below 40%, 18% of time above 85%, with rapid transitions between states occurring over 2–4 minute windows. This pattern is consistent with synchronized request bursts from upstream load balancers using fixed retry intervals. (3) Prefix cache hit rate was 8.3% despite 67% of traffic sharing a common system prompt — indicating prefix caching was disabled or the cache was being evicted before reuse. Enabling prefix caching in a 2-week controlled experiment increased effective throughput by 19% and reduced mean TTFT by 31ms. (4) Cost per million tokens ranged from \$0.43 (low-load periods) to \$2.87 (burst periods), a 6.7x range driven entirely by utilization variance. Smoothing utilization to 60–70% consistently yields cost-per-token within 15% of theoretical minimum."
  )

  # Instructions — varied framing, specificity, and expected output length.
  # Mix of short answers, long analyses, role-specific requests, and multi-part questions
  # so max_tok randomization maps to actual output demand.
  local instructions=(
    "Summarize this in two sentences. Be precise."
    "Extract every risk mentioned. For each: state the risk, its likelihood, and the recommended mitigation."
    "I'm a CFO with 5 minutes before a board meeting. What do I need to know?"
    "List all deadlines and action items with their owners. Format as a table."
    "What are the three most important numbers here and what decisions do they drive?"
    "Rewrite this as a one-page executive brief for a non-technical audience. No jargon."
    "Where are the gaps? What is missing, unclear, or contradictory in this document?"
    "What would a skeptical external auditor flag as concerns after reading this?"
    "If this document described our company, what would you tell our board to fix first and why?"
    "Draft a 3-bullet Slack update summarizing this for the engineering team."
    "What follow-up questions should legal ask before signing off on this?"
    "Compare what this document says we will do vs what we appear to actually be doing. Identify the gaps."
    "Assume this document is 6 months old. What has likely changed or gone wrong since it was written?"
    "What would a competitor learn about us if they read this? What should we have kept confidential?"
    "Identify every commitment, SLA, or guarantee made in this document and assess whether they are realistic."
    "I need to brief a new engineer joining the team. Summarize the most important technical decisions and their rationale."
    "What dependencies or blockers does this document reveal that could derail the plan?"
    "Give me a red team analysis: if this plan fails, what are the most likely causes?"
    "Score the quality of this document from 1–10 on: completeness, clarity, and actionability. Justify each score."
    "What is the single most important thing this document is trying to communicate? Then tell me if it succeeds."
  )

  worker() {
    while true; do
      local d_idx i_idx prompt max_tok think
      d_idx=$((RANDOM % ${#docs[@]}))
      i_idx=$((RANDOM % ${#instructions[@]}))
      # System prompt (~3K tok) + unique doc (~900 tok) + instruction (~30 tok) ≈ 4K total.
      # Shared prefix is identical across all workers — prefix cache target for Act 4.
      prompt="${RAG_SYSTEM_PROMPT}

Document:
${docs[$d_idx]}

Task: ${instructions[$i_idx]}"
      max_tok=$(python3 -c "import random; print(random.randint(80, 480))")
      think=$(python3 -c "import random; mt=max(float('$mean_think'),0.01); print(f'{random.expovariate(1/mt):.2f}')")
      post "$prompt" "$max_tok"
      sleep "$think"
    done
  }

  for ((i = 0; i < workers; i++)); do
    worker &
  done
  wait
}

echo "load.sh — MODE=${MODE}  target=${VLLM_URL}"
echo "Ctrl-C to stop."
echo ""

case "$MODE" in
  demo) load_demo ;;
  rag)  load_rag ;;
  r1)   load_r1 ;;
  seq)  load_seq ;;
  r2)   load_r2 ;;
  r5)   load_r5 ;;
  *)    echo "Unknown MODE=${MODE}. Use demo, rag, r1, seq, r2, or r5." >&2; exit 1 ;;
esac
