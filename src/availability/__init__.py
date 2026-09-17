"""
Availability estimation for players listed with an uncertain injury status
(Questionable / Doubtful). See docs/features/availability_agent_scope.md.

Initial scope: labels from box-score minutes, a point-in-time retrieval layer,
and non-LLM baselines. The LLM estimator is a later phase and must beat the
tabular baseline here before it is wired into the feature pipeline.
"""
