#!/usr/bin/env python3
"""Customer-support load generator for profile journeys.

Open-loop (QPS-driven) conversation arrivals on a Poisson clock with rate
jitter, so queues and bursts are real, not simulated. Traffic shape:

  - ~500-token shared support-policy system prompt (identical across requests;
    prefix cache target)
  - multi-turn conversations (default 1-4 turns); follow-ups carry the full
    history back, so prompts grow as sessions age
  - opening questions 50-300 tokens, topic first; follow-ups are about that
    ticket (not a generic bank)
  - bimodal answers: 80% short (100-300 tok), 20% long (1000-2000 tok). The
    long tail is the decode mass that keeps this opposite of SWE-bench
    (prefill-heavy coding agents)
  - QPS ramp in stages; within a stage the rate wanders (jitter) with
    occasional bursts, like real arrival traffic

Live knobs: the script re-reads --knobs (JSON) at every conversation start,
so traffic can be softened mid-journey without restarting, e.g.:

  echo '{"turns_min": 1, "turns_max": 2, "long_frac": 0.05, "qps_mult": 0.5}' \
    > support-load.knobs.json

Recognized keys: turns_min, turns_max, long_frac, jitter (0/1), qps_mult
(scales the current stage's arrival rate live; the AGENTS=N equivalent).

Model resolution matches load.sh: MODEL > SERVED_NAME > PROFILE_MODEL family
default (gemma -> gemma-4-26b-a4b, qwen -> Qwen3.6-27B, llama -> llama3).

Usage:
  PROFILE_MODEL=gemma ./support-load.py   # AMD default journey
  PROFILE_MODEL=qwen ./support-load.py --qps 0.3,1.3,5 --stage-secs 600

AMD Gemma (after ./start-gemma.sh): same command; needs the recipe chat
template so system-role messages do not 400 (ENABLE_GEMMA_TOOLS=1 default).
Soften live with knobs if the top stage overwhelms before profile can attach.

Default stages 0.3/1.3/5 conversations/s; at ~2.5 turns per conversation
that is ~0.8/3.2/12.8 requests/s (~16/64/256 expected in-flight at typical
service times). Top stage is meant to hurt. Requires aiohttp (present in any
vLLM environment). Ctrl-C stops.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time

try:
    import aiohttp
except ImportError:
    sys.exit("aiohttp required (pip install aiohttp); present in vLLM envs.")

FAMILY_DEFAULTS = {
    "gemma": "gemma-4-26b-a4b",
    "qwen": "Qwen3.6-27B",
    "llama": "llama3",
}

# ~500 tokens. Identical across all requests: prefix cache target.
SYSTEM_PROMPT = (
    "You are the customer support assistant for Meridian, a subscription "
    "software company selling project management tools to small businesses. "
    "Follow these policies in every reply. Refunds: full refund within 30 days "
    "of purchase, no questions asked; between 31 and 90 days, prorated refund "
    "if the customer reports a defect we confirmed; after 90 days, no refunds, "
    "offer account credit instead. Billing: annual plans bill on the signup "
    "anniversary, monthly plans on the calendar day of signup; failed payments "
    "retry three times over seven days before the account is paused; paused "
    "accounts keep their data for 180 days. Plan changes: upgrades apply "
    "immediately with prorated charges; downgrades apply at the next renewal; "
    "seat counts can change at any time and bill prorated. Cancellations: "
    "customers can cancel from the billing page or by asking us; cancellation "
    "stops renewal but service continues until the paid period ends; we do not "
    "cancel retroactively. Data: customers can export all their data as CSV or "
    "JSON from settings at any time, including after cancellation during the "
    "180-day retention window; deletion requests complete within 30 days and "
    "are irreversible. Security: never ask for passwords or full card numbers; "
    "verify identity by confirming the account email and the last four digits "
    "of the card on file; if verification fails, direct the customer to the "
    "self-service password reset. Outages: check the status page first; if an "
    "incident is open, share the status page link and the current update; do "
    "not speculate about causes or timelines. Escalation: billing disputes over "
    "500 dollars, legal threats, security reports, and press inquiries go to a "
    "human agent; tell the customer a specialist will reply within one business "
    "day. Tone: be direct, warm, and concise; answer the question first, then "
    "add necessary caveats; use plain language, no jargon; never blame the "
    "customer; if you do not know, say so and escalate rather than guess. If a "
    "request falls outside these policies, say what you can do instead of what "
    "you cannot. Always end by asking if anything else is needed."
)

# Each ticket owns its follow-ups so multi-turn stays on the same issue.
TICKETS = [
    {
        "opener": (
            "I was charged twice this month and I want to understand why before "
            "I dispute it with my bank."
        ),
        "follow_ups": [
            "The two charges are on the 3rd and the 5th for the same amount. Which one is the real renewal?",
            "If one is a duplicate, how long until the credit hits the card?",
            "Can you email a written confirmation that only one charge should stand?",
        ],
    },
    {
        "opener": (
            "How do I move from the monthly plan to the annual plan, and what "
            "happens to the days I already paid for?"
        ),
        "follow_ups": [
            "Will the unused days on monthly convert to credit on the annual invoice?",
            "If I switch today, does annual billing start immediately or at month end?",
            "Is there a discount code path for annual, or is the list price what I get?",
        ],
    },
    {
        "opener": (
            "Our payment failed while our card was being reissued and now the "
            "account says paused. What do we do?"
        ),
        "follow_ups": [
            "I updated the card already. How do I trigger a retry without waiting for the next automatic attempt?",
            "While paused, can the team still open existing projects read-only?",
            "If the retry fails again, how many days until data retention starts counting?",
        ],
    },
    {
        "opener": (
            "I need to export everything before we close the account at the end "
            "of the quarter."
        ),
        "follow_ups": [
            "Does the export include attachments and activity history, or only cards and comments?",
            "How long does a full-account export usually take for roughly two thousand projects?",
            "Can I keep exporting during the 180-day retention window after cancel?",
        ],
    },
    {
        "opener": (
            "Can I get a refund? We bought this six weeks ago and the timeline "
            "view has been broken for us since day one."
        ),
        "follow_ups": [
            "We filed a bug report on day three with screenshots. Does that count as a confirmed defect?",
            "If refund is prorated at six weeks, what dollar amount should I expect?",
            "If you will not refund, what account credit would you offer instead?",
        ],
    },
    {
        "opener": "We added five seats mid-cycle and the invoice looks wrong to me.",
        "follow_ups": [
            "The invoice shows a full month per seat. Should mid-cycle adds be prorated to the renewal date?",
            "Can you break out the line items for the five new seats versus the base plan?",
            "Who do I send a corrected PO to if finance needs the adjusted total?",
        ],
    },
    {
        "opener": (
            "The app has been down for our whole team since this morning. Is "
            "there an outage?"
        ),
        "follow_ups": [
            "Status page looked green twenty minutes ago. Is there a regional incident not listed yet?",
            "We are in us-east. Any known issue for that region right now?",
            "If this is on our side, what should we check first on corporate network or SSO?",
        ],
    },
    {
        "opener": (
            "I want to cancel but keep access until the period ends. How does "
            "that work exactly?"
        ),
        "follow_ups": [
            "If I cancel today on an annual plan, do we keep access until the anniversary date?",
            "Will anyone on the team get locked out early if I hit cancel in billing?",
            "After the period ends, how long is the data retained before deletion?",
        ],
    },
    {
        "opener": (
            "Someone on our team deleted a project by accident. Is there any "
            "way to recover it?"
        ),
        "follow_ups": [
            "It was deleted about six hours ago. Is there a trash or restore window?",
            "The project id was PRJ-18422 if that helps your side look it up.",
            "If restore is impossible, can you export whatever remnants still exist in backups?",
        ],
    },
    {
        "opener": (
            "We are a nonprofit. Do you offer any discount, and how do I apply "
            "it to an existing subscription?"
        ),
        "follow_ups": [
            "What documentation do you need to verify nonprofit status?",
            "Does the discount apply on the next renewal only, or can you adjust the current term?",
            "Is the nonprofit rate available on annual and monthly, or annual only?",
        ],
    },
    {
        "opener": (
            "My login stopped working after we changed our company email domain."
        ),
        "follow_ups": [
            "Old domain was acme-old.com, new is acme.com. Same mailbox local part.",
            "Do I need an admin to remap the user, or is there a self-serve domain change?",
            "SSO still redirects to the old IdP app. Is that a separate change on your side?",
        ],
    },
    {
        "opener": (
            "I asked for account deletion two weeks ago and I can still log in. "
            "When does it actually happen?"
        ),
        "follow_ups": [
            "I have the request confirmation email from the 12th. Can you check that ticket?",
            "Is login expected during the 30-day deletion window, or should access already be gone?",
            "How do I confirm deletion completed so our DPO can close the request?",
        ],
    },
    {
        "opener": (
            "The invoice needs our VAT number on it and I cannot find where to "
            "add it."
        ),
        "follow_ups": [
            "We are in Germany. Which tax ID field should we use in billing settings?",
            "Will past invoices regenerate with the VAT number, or only future ones?",
            "Finance needs a corrected PDF for last month. Can support reissue it?",
        ],
    },
    {
        "opener": (
            "We were promised a feature by your sales team that does not seem "
            "to exist. I want a refund or an answer."
        ),
        "follow_ups": [
            "The feature was custom workload views with shared filters across workspaces.",
            "I have the sales email thread. Where should I forward it for review?",
            "If the feature is on a roadmap, what is a realistic quarter, or do we cancel?",
        ],
    },
    {
        "opener": (
            "How do seat licenses work when an employee leaves and a new one "
            "joins the same week?"
        ),
        "follow_ups": [
            "Do we get charged for both seats that week, or can we transfer the license?",
            "Is deactivating the old user enough, or must we delete them?",
            "We use SCIM. Does removing them in the IdP free the seat automatically?",
        ],
    },
    {
        "opener": (
            "Why did my renewal price go up without any notice, and can you "
            "honor last year's rate?"
        ),
        "follow_ups": [
            "Last year was 12 dollars per seat. This renewal shows 15. When did pricing change?",
            "Was a notice sent to the billing email on file? Ours is ap@ourcompany.com.",
            "If you cannot honor the old rate, is there any loyalty credit you can apply?",
        ],
    },
    {
        "opener": (
            "Two of our workspaces got merged somehow and now permissions are "
            "a mess."
        ),
        "follow_ups": [
            "Workspace A was Marketing, B was Sales. They now show as one list of projects.",
            "Can support unmerge them, or do we have to rebuild permissions manually?",
            "Several guests lost access. Is there an audit log of who triggered the merge?",
        ],
    },
    {
        "opener": (
            "I keep getting a spreadsheet import error on a file that worked "
            "fine last month."
        ),
        "follow_ups": [
            "Error text is 'column mapping failed on sheet Tasks'. File is .xlsx about 4MB.",
            "Did the importer change required columns in the last release?",
            "Is there a sample template you want us to match against?",
        ],
    },
    {
        "opener": (
            "Does your product integrate with our calendar system, and is that "
            "included in our plan?"
        ),
        "follow_ups": [
            "We use Google Workspace calendars for due dates.",
            "Is calendar sync on Pro only, or available on our current plan?",
            "If it is an add-on, what is the per-seat cost and how do we enable it?",
        ],
    },
    {
        "opener": (
            "The mobile app logs me out every day and support articles have "
            "not helped."
        ),
        "follow_ups": [
            "iOS 18, app version from the App Store updated last week.",
            "It happens on both Wi-Fi and cellular, same Apple ID.",
            "Should we try removing the app and reinstalling, or is there a known SSO bug?",
        ],
    },
    {
        "opener": (
            "We need an invoice history for the last two years for an audit. "
            "Where do I find it?"
        ),
        "follow_ups": [
            "Billing page only shows twelve months. How do we get the older year?",
            "Can you send PDFs for January through December of last year to ap@ourcompany.com?",
            "Our auditors also need payment confirmation references for each invoice.",
        ],
    },
    {
        "opener": (
            "Can we get a copy of your security and compliance documentation "
            "for our vendor review?"
        ),
        "follow_ups": [
            "We specifically need SOC 2 type II and a current pen-test summary.",
            "Is there an NDA portal, or can you email the packet to security@ourcompany.com?",
            "What is the typical turnaround once we request access?",
        ],
    },
    {
        "opener": (
            "A former employee is still receiving account emails. How do we "
            "remove them completely?"
        ),
        "follow_ups": [
            "Their user still appears under Members as deactivated, email sam@oldco.com.",
            "They also get billing CC mail. Is that a separate billing contact to clear?",
            "After removal, how long until email stops, including digests?",
        ],
    },
    {
        "opener": (
            "The reports page shows different totals than the dashboard for "
            "the same date range."
        ),
        "follow_ups": [
            "Dashboard says 412 completed tasks last week; reports say 387.",
            "Timezone on the account is America/Chicago. Could that explain the gap?",
            "Which number should finance trust for the weekly ops review?",
        ],
    },
    {
        "opener": (
            "I upgraded by mistake and was charged immediately. Can this be "
            "reversed?"
        ),
        "follow_ups": [
            "Upgrade was about forty minutes ago to the Business plan.",
            "Can you downgrade now and refund the prorated upgrade charge?",
            "Will team permissions change again if you reverse the upgrade?",
        ],
    },
    {
        "opener": (
            "How do I transfer account ownership to another admin before I "
            "leave the company?"
        ),
        "follow_ups": [
            "The new owner is already an admin: jordan@ourcompany.com.",
            "Do I transfer first, or should billing contacts change first?",
            "After transfer, will my user keep access until I am removed?",
        ],
    },
    {
        "opener": (
            "Notifications stopped arriving for half our team after your "
            "last update."
        ),
        "follow_ups": [
            "Affected users are on email digest and Slack. In-app still works.",
            "Did the release change default notification settings?",
            "Is there a workspace-level toggle we should re-enable?",
        ],
    },
    {
        "opener": (
            "We hit the storage limit and I do not understand what is counting "
            "against it."
        ),
        "follow_ups": [
            "Settings shows 98% used but projects look small. Are file versions counted?",
            "Can you list the top storage consumers on the account?",
            "If we archive projects, does that free storage immediately?",
        ],
    },
    {
        "opener": (
            "Is there a way to lock certain projects so only two people can "
            "see them?"
        ),
        "follow_ups": [
            "We need private projects for HR and legal only.",
            "Does locking hide them from workspace search for everyone else?",
            "Can guests ever be invited into a locked project, or is that blocked?",
        ],
    },
    {
        "opener": (
            "Your status page says resolved but our team still cannot open "
            "any boards."
        ),
        "follow_ups": [
            "Error is a blank page with a 502 in the network tab.",
            "Cleared cache and tried another browser. Same result.",
            "Should we wait, or open an incident if status still says green?",
        ],
    },
    {
        "opener": (
            "I need to change the billing email to our accounts payable "
            "address."
        ),
        "follow_ups": [
            "New address should be ap@ourcompany.com; remove me from invoice CC.",
            "Will the next renewal notice go only to that address?",
            "Can past invoices be resent to AP for our records?",
        ],
    },
    {
        "opener": (
            "The trial ended before we finished evaluating. Can we get an "
            "extension?"
        ),
        "follow_ups": [
            "We need ten more days for two remaining teams to finish testing.",
            "If extension is not possible, can we pause billing for a week after signup?",
            "Who approves trial extensions on your side?",
        ],
    },
    {
        "opener": (
            "We were double-invoiced last quarter and the credit never "
            "appeared."
        ),
        "follow_ups": [
            "Invoice numbers were INV-20411 and INV-20419 for the same period.",
            "Finance says no credit memo landed in the portal.",
            "Can you confirm whether the credit was card refund or account credit?",
        ],
    },
    {
        "opener": (
            "How do I bulk-archive completed projects without deleting their "
            "history?"
        ),
        "follow_ups": [
            "We have about three hundred projects marked done from last year.",
            "Does archive keep search and export working for auditors?",
            "Is there an API or UI filter to archive all completed in one action?",
        ],
    },
    {
        "opener": (
            "Can you confirm whether our data is stored in the EU? Our legal "
            "team is asking."
        ),
        "follow_ups": [
            "We need region and subprocessors in writing for the DPA packet.",
            "If we are on US hosting, can we migrate to EU without rebuilding projects?",
            "Who is the right contact for signing the updated DPA?",
        ],
    },
    {
        "opener": (
            "The API tokens we generated last week stopped working this "
            "morning."
        ),
        "follow_ups": [
            "Calls return 401 with token prefix mrd_live_7f. No rotate on our side.",
            "Did a security rotation invalidate tokens overnight?",
            "How do we mint a replacement without downtime for our sync job?",
        ],
    },
    {
        "opener": (
            "What happens to shared guest access when we downgrade to the "
            "starter plan?"
        ),
        "follow_ups": [
            "We have fourteen guests today on Pro. Starter allows fewer.",
            "Will guests lose access immediately on downgrade, or at renewal?",
            "Which guests should we convert to full seats before switching?",
        ],
    },
    {
        "opener": (
            "My colleague and I both got admin removed and nobody knows who "
            "did it."
        ),
        "follow_ups": [
            "We need an audit of admin role changes in the last forty-eight hours.",
            "Can support restore admin for jordan@ and me without a remaining admin?",
            "Is there SSO group mapping that could have overwritten roles?",
        ],
    },
    {
        "opener": (
            "The export file is missing attachments. Is that expected or a "
            "bug?"
        ),
        "follow_ups": [
            "CSV has comments and fields, but linked files are absent.",
            "Is attachment export a separate job or a plan-gated feature?",
            "If it is a bug, what workaround do we use for our migration this week?",
        ],
    },
    {
        "opener": (
            "We want to consolidate three subscriptions into one account and "
            "one invoice."
        ),
        "follow_ups": [
            "Accounts are team-a, team-b, and team-c under the same company domain.",
            "Can projects move without breaking links and guest access?",
            "Will unused term value on the child accounts credit the surviving invoice?",
        ],
    },
]

FILLER = [
    "For context, we are a team of about a dozen people and have been paying customers for a while now.",
    "I already looked through the help center and could not find a clear answer to this exact situation.",
    "Our finance person needs a written confirmation of whatever you tell me for our records.",
    "This is fairly urgent on our side because it blocks the rest of the team from working normally.",
    "I tried logging out and back in, a different browser, and clearing the cache, and nothing changed.",
    "We renewed recently, so I am surprised this is coming up at all, and I want to avoid it next cycle.",
    "The account is under the same email I am writing from, and I can confirm card digits if needed.",
    "If this needs a specialist, that is fine, but please tell me the realistic timeline for a reply.",
    "We evaluated a competitor last month and stayed with you, so I would like this handled well.",
    "Last time we contacted support the answer took a week, which caused real problems for our billing run.",
    "I am the account owner, and I can provide any verification details you need to look into this.",
    "Please keep the explanation simple; I will be forwarding it to people who do not use the product.",
]

# Journey default: top stage should stress the server. Soften with knobs
# (qps_mult / long_frac) if needed; do not ship a quiet ramp.
DEFAULT_QPS = "0.3,1.3,5"
DEFAULT_KNOBS = {
    "turns_min": 1,
    "turns_max": 4,
    "long_frac": 0.2,
    "jitter": 1,
    "qps_mult": 1.0,
}


def read_knobs(path: str) -> dict:
    """Merge live knobs file over defaults. Missing/invalid file → defaults."""
    knobs = dict(DEFAULT_KNOBS)
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            for k in knobs:
                if k in loaded:
                    knobs[k] = loaded[k]
    except (OSError, ValueError):
        pass
    try:
        knobs["turns_min"] = max(1, int(knobs["turns_min"]))
        knobs["turns_max"] = max(knobs["turns_min"], int(knobs["turns_max"]))
        knobs["long_frac"] = min(1.0, max(0.0, float(knobs["long_frac"])))
        qps_mult = float(knobs["qps_mult"])
        if not 0.0 < qps_mult < float("inf"):
            raise ValueError("qps_mult must be finite and positive")
        knobs["qps_mult"] = max(0.01, qps_mult)
    except (TypeError, ValueError):
        return dict(DEFAULT_KNOBS)
    return knobs


def build_question(rng: random.Random, ticket: dict) -> str:
    """50-300 token opener: ticket topic first, then 2-8 context sentences."""
    fillers = rng.sample(FILLER, rng.randint(2, 8))
    rng.shuffle(fillers)
    return " ".join([ticket["opener"], *fillers])


def pick_follow_up(rng: random.Random, ticket: dict, used: set[str]) -> str:
    """Next user turn from this ticket's bank; fall back if all used."""
    choices = [f for f in ticket["follow_ups"] if f not in used]
    if not choices:
        choices = list(ticket["follow_ups"])
    follow = rng.choice(choices)
    used.add(follow)
    return follow


def max_tokens_for(rng: random.Random, long_frac: float) -> int:
    """Bimodal answers: short 100-300 tok, long 1000-2000 tok."""
    if rng.random() < long_frac:
        return rng.randint(1000, 2000)
    return rng.randint(100, 300)


class Stats:
    def __init__(self) -> None:
        self.convs = 0
        self.sent = 0
        self.done = 0
        self.errors = 0
        self.inflight = 0
        self.conv_active = 0
        self.rate_mult = 1.0


async def conversation(session: aiohttp.ClientSession, args: argparse.Namespace,
                       model: str, rng: random.Random, stats: Stats) -> None:
    knobs = read_knobs(args.knobs)
    turns = rng.randint(knobs["turns_min"], knobs["turns_max"])
    ticket = rng.choice(TICKETS)
    used_follow_ups: set[str] = set()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_question(rng, ticket)}]
    stats.convs += 1
    stats.conv_active += 1
    try:
        for turn in range(turns):
            body = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens_for(rng, knobs["long_frac"]),
                "temperature": 0.7,
            }
            stats.sent += 1
            stats.inflight += 1
            try:
                async with session.post(f"{args.url}/v1/chat/completions",
                                        json=body) as r:
                    data = await r.json(content_type=None)
                    if r.status != 200:
                        stats.errors += 1
                        return
                    stats.done += 1
            except Exception:
                stats.errors += 1
                return
            finally:
                stats.inflight -= 1
            if turn + 1 == turns:
                return
            reply = ""
            try:
                reply = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                return
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": pick_follow_up(rng, ticket, used_follow_ups),
            })
            # Customer think time between turns.
            await asyncio.sleep(rng.uniform(*args.think_secs))
    finally:
        stats.conv_active -= 1


async def jitter_loop(stats: Stats, args: argparse.Namespace,
                      rng: random.Random) -> None:
    """Resample the arrival-rate multiplier every 30s; occasional bursts.
    Live qps_mult knob scales on top (the AGENTS=N equivalent, no restart)."""
    while True:
        knobs = read_knobs(args.knobs)
        if knobs["jitter"]:
            if rng.random() < 0.10:
                stats.rate_mult = rng.uniform(2.0, 3.0)  # burst
            else:
                stats.rate_mult = rng.uniform(0.6, 1.5)
        else:
            stats.rate_mult = 1.0
        stats.rate_mult *= knobs["qps_mult"]
        await asyncio.sleep(30)


async def reporter(stats: Stats, stage_of: "list[str]") -> None:
    while True:
        await asyncio.sleep(10)
        print(
            f"[{time.strftime('%H:%M:%S')}] {stage_of[0]} x{stats.rate_mult:.2f} "
            f"convs {stats.convs} (active {stats.conv_active}) "
            f"sent {stats.sent} done {stats.done} errors {stats.errors} "
            f"in-flight {stats.inflight}",
            flush=True,
        )


async def run(args: argparse.Namespace, model: str) -> None:
    rng = random.Random(args.seed)
    stats = Stats()
    stage_label = ["stage 0"]
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        rep = asyncio.create_task(reporter(stats, stage_label))
        jit = asyncio.create_task(jitter_loop(stats, args, rng))
        tasks: set[asyncio.Task] = set()
        try:
            for i, qps in enumerate(args.qps, 1):
                stage_label[0] = f"stage {i}/{len(args.qps)} conv-qps {qps}"
                print(f"=== {stage_label[0]} for {args.stage_secs}s", flush=True)
                end = time.monotonic() + args.stage_secs
                while time.monotonic() < end:
                    # Poisson arrivals: exponential gaps at the jittered rate.
                    delay = rng.expovariate(qps * stats.rate_mult)
                    remaining = end - time.monotonic()
                    if delay >= remaining:
                        await asyncio.sleep(max(0.0, remaining))
                        break
                    await asyncio.sleep(delay)
                    t = asyncio.create_task(
                        conversation(session, args, model, rng, stats))
                    tasks.add(t)
                    t.add_done_callback(tasks.discard)
            print(f"=== ramp done; draining {len(tasks)} conversations", flush=True)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            rep.cancel()
            jit.cancel()
    print(f"final: convs {stats.convs} sent {stats.sent} done {stats.done} "
          f"errors {stats.errors}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--url", default=os.environ.get("VLLM_URL", "http://localhost:8000"))
    p.add_argument(
        "--model",
        default=None,
        help="request model id (default: MODEL, else SERVED_NAME, else PROFILE_MODEL family default)",
    )
    p.add_argument("--qps", default=DEFAULT_QPS,
                   help="comma-separated conversation arrival rates per stage "
                        f"(default {DEFAULT_QPS}; ~2.5 turns each → ~0.8/3.2/12.8 req/s)")
    p.add_argument("--stage-secs", type=int, default=600)
    p.add_argument("--knobs", default="support-load.knobs.json",
                   help="live-tunable JSON knobs file, re-read per conversation "
                        "(turns_min, turns_max, long_frac, jitter, qps_mult)")
    p.add_argument("--think-secs", default="5,20",
                   help="min,max seconds of customer think time between turns")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--request-timeout", type=float, default=600.0)
    args = p.parse_args()
    args.qps = [float(x) for x in args.qps.split(",") if x.strip()]
    if not args.qps or any(q <= 0 for q in args.qps):
        p.error("--qps needs positive comma-separated rates")
    lo, hi = (float(x) for x in args.think_secs.split(","))
    args.think_secs = (lo, hi)

    profile_model = os.environ.get("PROFILE_MODEL", "gemma")
    if profile_model not in FAMILY_DEFAULTS:
        p.error(f"PROFILE_MODEL must be one of {sorted(FAMILY_DEFAULTS)} (got: {profile_model})")
    model = (
        args.model
        or os.environ.get("MODEL")
        or os.environ.get("SERVED_NAME")
        or FAMILY_DEFAULTS[profile_model]
    )
    print(f"target {args.url} model {model} conv-qps stages {args.qps} "
          f"stage {args.stage_secs}s knobs {args.knobs} {read_knobs(args.knobs)}",
          flush=True)
    try:
        asyncio.run(run(args, model))
    except KeyboardInterrupt:
        print("stopped", flush=True)


if __name__ == "__main__":
    main()
