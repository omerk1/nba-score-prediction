# Morning report

`scripts/morning_report.py`, LaunchAgent `com.omerkoren.nba-morning-report`
at 10:00 local (installed by `scripts/install_launchd.sh`). Refreshes final
scores, grades `outputs/predictions_log.csv`, sends one HTML page: last slate
day's strong bets (edge >= 5%) and game-by-game predicted vs. actual, overall
model error, and the strong-bet record. Marginal bets are not tracked.
Return = profit per bet from a flat 1-unit stake; units = total profit.

10:00 because the last West Coast games end ~08:00 Israel time and final
scores land in nba_api shortly after (Oct 21's 11 games were all present at
09:42).
