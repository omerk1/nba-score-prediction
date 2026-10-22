# Live injury feed fix (2026-10-22)

ESPN dropped `team.abbreviation` from its injuries feed (only `displayName`
remains), so every live row was dropped as an unknown team while the nightly
step still logged ok: no live injury data after 2026-04-12. Fixed in the
scraper (full-name map, ESPN's "LA Clippers" spelling); the nightly step now
raises if entries arrive but none map to a team. Also removed the dead
`impact_score` decay step, which crashed on the formula scorer's output.
Not recoverable: Oct 13-21 predictions were made with no injury data.

Known live/training mismatch (not changed): training rows come from the 11PM
report of the game date (avg 4.0 out / 0.37 questionable per team), live rows
from an ESPN snapshot at 16:00 local (~9am ET), and ESPN has no
"Questionable" status. Plan: collect live snapshots for 4-6 weeks, then
ablate live-format injury features against the current ones.
