# Future LLM/Agentic Directions

Forward-looking companion to `docs/LLM_COMPONENT_OPTIONS.md` (the closed
investigation on this repo). That doc's verdict: this codebase is dense,
well-labeled tabular data, exactly the case where a tuned gradient booster
already wins and reading the same facts with an LLM has nothing to add. These
ideas are deliberately not more of that — each one is picked for having a
mechanism a tabular model structurally cannot provide: live, unstructured
input; no historical training corpus; a consequential action, not a number.

One in-repo idea, then three outside it, described in enough depth to take
further without re-deriving the shape from scratch.

---

## In-repo: a live pre-game news and market-reaction agent

**What it does.** Watches for breaking pre-game information in the hours
before tip-off — injury updates, lineup announcements, beat-reporter posts —
and judges whether Polymarket's current price already reflects it. When it
judges a real, unpriced gap, it flags the game with its reasoning.

**Why the mechanism actually fits, unlike everything just closed out.**
`docs/LLM_COMPONENT_OPTIONS.md`'s option 4 (news as a training feature) was
rejected because building a supervised feature needs a timestamped,
point-in-time news archive spanning years — expensive, and the retroactive
labeling is itself hard to get right. An agent that acts live, today, needs
none of that backfill. It only has to be good in the moment it runs. Same
data source, a use case the earlier rejection doesn't touch.

The input is genuinely unstructured (a tweet, a vague coach's quote, a
lineup graphic) and the judgment — is this credible, is it already priced
in — is not something the existing structured injury-PDF pipeline can
express as a feature at all.

**Shape.** A monitoring loop (poll a small set of sources for each of
today's games) → retrieval of that game's recent context (this repo's own
injury history, past price moves, this team's recent news) as grounding →
an LLM judgment call with tools (fetch the current Polymarket price, fetch
the source article) → a flag with stated reasoning and confidence, not an
autonomous trade.

**Evaluation, the part that keeps it honest.** Track flagged games forward:
did the price move in the flagged direction afterward, and by how much
relative to games it didn't flag. That is a real, checkable question,
independent of whether the underlying prediction model ever uses the
output. Start with human-reviewed flags before any auto-action.

**Honest caveats.** Source access (X/Twitter API, beat-reporter feeds) is
the main friction, not the agent logic. Small sample size per season limits
how confidently the evaluation can speak. This is a monitoring/alerting
tool, not a new model feature — its value is a faster or more careful read
of public information, not exclusive information, matching what
`docs/MARKET_EDGE.md` already found about this specific market.

---

## Outside this repo: agents against a live system, with consequences

The common shape across all three: input that cannot be reduced to a clean
label in advance, a decision with a real cost if wrong, and tools that reach
a live system rather than a static dataset. Pick based on which domain you
have the fastest feedback loop and the safest blast radius to experiment in.

### 1. Incident-response / SRE triage agent

**What it does.** Watches live signals from a real system — logs, metrics,
alert feed (e.g. Prometheus/CloudWatch/Datadog) — and for each anomaly,
decides: noise, a known and already-mitigating issue, or a real incident
needing attention. For a real incident, it correlates across services to
propose a likely root cause and drafts a bounded remediation (restart one
service, roll back the last deploy, scale a resource) for a human to
approve before it executes.

**Why it's a strong fit.** The ambiguity is real and stateful — the same
error-rate spike can be nothing or a five-alarm fire depending on
correlated signals a rule-based alert can't see. The decision space is
genuinely multi-step: gather more evidence, form a hypothesis, check it
against another data source, only then propose an action. It is a
recognized, actively-developing category ("AI SRE") right now, so there is
real prior art and infrastructure to build against, not a from-scratch
problem.

**Technical shape.** Tools to query the observability platform's API,
recent deploy history, and a runbook knowledge base (RAG over your own
past incidents and docs). Output is a structured incident summary plus a
proposed action, gated behind human approval to start. Evaluation is
concrete: precision on "was this actually an incident" against your
existing alert history, and whether the proposed root cause matched what a
human found.

**Scoping down for a first build.** Start read-only — draft the incident
summary and proposed action but never execute — on your own project or a
toy system with synthetic incidents you inject yourself, so you get a
controllable evaluation set without touching anything that matters.

### 2. Support/ticket triage agent on a real helpdesk

**What it does.** Reads incoming support tickets (a real system: Zendesk,
Linear, GitHub Issues, or your own), retrieves relevant material from a
knowledge base or past resolved tickets, and either drafts a direct answer,
escalates with a summary and suggested owner, or asks a clarifying question
— then actually updates the ticket's status, tags, and assignee via the
platform's API.

**Why it's a strong fit.** Classic RAG (grounding answers in your actual
docs, not invented ones) combined with a genuine judgment call (answer vs.
escalate vs. ask) and a real write action against a live system. The
evaluation is unusually clean for an agent project: resolution accuracy
and correct escalation are both measurable against how a human actually
handled the same tickets historically, if you have that history.

**Technical shape.** Retrieval index over your docs/knowledge base;
tool calls to the ticketing platform's read and write API; a routing
decision with an explicit confidence threshold below which it always
escalates rather than guesses. Good candidate for measuring the effect of
confidence calibration directly, since a wrong "answer directly" is more
costly than a wrong "escalate."

**Scoping down for a first build.** Point it at your own project's GitHub
Issues or a small real support inbox with low stakes, draft-only mode
first (propose the reply, don't send), and compare its proposed
routing against what actually happened for a real evaluation set.

### 3. Cloud cost / configuration drift agent

**What it does.** Periodically scans real cloud resources via the
provider's API, compares against a policy (tagging standards, expected
instance sizes, security-group rules, cost budget per team), and for a
genuine drift or anomaly, either opens a PR against your infrastructure-
as-code repo with the proposed fix, or files a ticket, rather than
silently alerting.

**Why it's a strong fit.** The judgment is genuinely ambiguous — an
oversized instance might be intentional for a launch this week, not a
mistake — so this cannot be a static rule engine without constant tuning
and false positives. It reaches a real, consequential external system
(your actual cloud account), and the output (a PR, not a raw alert) is a
concrete, checkable artifact: did the PR get merged, was it correct.

**Technical shape.** Read access to the cloud provider's resource API and
your IaC repo's current state; a policy document as grounding (RAG over
your own internal standards, which is exactly the kind of institutional
knowledge that's usually scattered across docs and tribal memory); tool
calls to open a PR or ticket rather than modify infrastructure directly on
a first pass. Evaluation: PR acceptance rate, and false-positive rate
against a manually reviewed sample.

**Scoping down for a first build.** Read-only reporting first (a weekly
digest of proposed changes, no PRs yet) against your own real cloud
account if you have one, or a deliberately-seeded test account with known
drift you introduce yourself for a controllable evaluation set.

---

## Picking one

If minimizing new infrastructure matters most, the support/ticket agent is
the fastest to a real evaluation, since GitHub Issues on a repo you already
own is enough to start. If the appeal is closer to what this repo's failed
attempts were reaching for — reading ambiguous signals and forming a
judgment under uncertainty — the incident-response agent is the closer
cousin, with the safety valve of running read-only against a system you
control. The cloud-cost agent is the strongest fit if the actual production
consequence (a merged PR against real infrastructure) is the part you want
the experience of building toward, not just prototyping.
