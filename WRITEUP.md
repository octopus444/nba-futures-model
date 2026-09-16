# Design Notes

## What the model does

It takes a league, a season and an as-of date, and returns fair values for four
contract types: to win the championship, to make the playoffs, to reach a given
round, and over/under on regular-season wins. Everything before the as-of date is
treated as fact and is never resimulated. Everything after it is generated thousands
of times, and the prices are the frequencies that come out. The line for the
win-total contract is the median of the simulated distribution, and the over and
under prices are the share of simulations landing either side of it.

Data is twenty-one seasons of NBA results, 2006 through 2026, from the official
stats API.

## Estimating strength

Strength is estimated by least squares on point margins. Each played game gives one
equation: home rating minus away rating plus a league-wide home advantage should
equal the observed margin. A full season gives 1230 equations against 31 unknowns.

The simplest alternative is to average each team's point differential over the games
it has played. That falls apart because it takes no account of who those games were
against. A team that has faced the weakest half of the league gets the same credit
for a plus-eight average as a team that has faced the strongest. In the regression
every game constrains two ratings at once, so beating a good team moves your rating
further than beating a bad one.

Ratings come out in points. A rating of 7.8 means beating an average opponent by 7.8
on a neutral floor. Home advantage is estimated from the same games and is not
assumed: it has fallen from 3.1 points in 2006 to 1.6 in 2026, and refitting it every
run means the model absorbs that without being told.

Margins are capped at 25 points. Overtime games are recorded as a draw.

## The prior

The brief asks for any as-of date, including opening night. A regression with no
games returns nothing, and with ten games per team it returns something worse than
nothing: barely more equations than unknowns, a fit that lands on the data and says
nothing about the teams. The validation below confirms it directly.

So we start from last season, and measure how much of it still holds. Regressing
every team's rating on its own rating a season earlier gives a slope of 0.62, across
447 team-seasons, being every team in every pair of consecutive seasons that were
both played to a normal schedule. A team that finished with a rating of 10 is
expected back at 6.2, so 38% of a team's strength does not survive the off-season.
The scatter around that line is +-3.47 points, which is how precisely a team is known
before a ball is bounced.

The prior enters the fit as one extra equation per team, carrying a weight. That
weight is the ratio of two quantities already measured: the noise of a single game,
12.52 points, divided by the width of the prior, 3.47, giving 3.61. Since information
accumulates as the *square* of the coefficients, the prior is worth about thirteen
games of evidence. At thirteen games played the prior and the season speak with equal
force, at forty the season carries three quarters, and at zero the rating is the
prior outright.

## Carrying uncertainty forward

A point estimate makes every game of a team independent of every other, which
understates how far a season can drift and systematically overprices favourites. Two
things are uncertain here, and both are modelled.

The first is where a team stands today, measured from a finite number of games. That
uncertainty falls out of the same fit as the ratings and comes to about 1.6 points in
midseason. It shifts a team's whole trajectory up or down.

The second is that strength moves. A title contract resolves in June, and the team
playing then is not the team playing now: trades, injuries, rest, shortened playoff
rotations. This is modelled as a random walk, calibrated by splitting past seasons in
half and removing the estimation error from the spread between the halves, which
gives 0.25 points per square root of a day for recent seasons. Because it
accumulates, the horizon falls out of the dates instead of being chosen: total
uncertainty about a team's strength runs at 1.6 points today, 2.8 by the end of the
regular season and 3.3 by the finals, all from the same draw.

The effect is large. For the strongest team on a January 2024 as-of date, the
championship price runs 48.9 percent on point estimates with no prior and 28.0
percent once both sources of uncertainty are carried. The spread of that team's final
win total goes from 2.59 wins to 3.55 over the same sequence.

As a rough external check, that team was quoted at +260 at the 2024 All-Star break,
about 28 percent before removing the overround. The two numbers sit in about the same
place, but a real comparison needs the full set of quotes across all thirty teams,
devigged, and that is not in this work.

## Where the league ends and the engine begins

`engine/` knows how to rate teams, play a season forward and turn simulated standings
into prices. It knows nothing about basketball. `leagues/` knows the sport.

The line was drawn by asking which rules change between leagues. The play-in
tournament looks like part of the postseason and belongs in the league file, because
it is an NBA rule: it is a function the config hands to the engine, and a league where
the top seeds simply qualify hands over nothing. Series format is stored per round,
not per league, because leagues change format as a postseason progresses. Byes and
reseeding are config flags, because the NFL has both and the NBA has neither.

`leagues/nfl.py` is included to show this. The NFL's seven-team field, its bye for the
top seed, its single-game rounds and its reseeding after every round all run through
the same bracket code, verified on a synthetic field. One thing it would still need
from the engine is seeding: the current implementation orders teams by wins and breaks
ties at random, which is acceptable for the NBA and wrong for the NFL, where division
winners are seeded above wild cards regardless of record. That needs a seeding hook
alongside the qualification hook that already exists, and it is the one engine change
a second league actually justifies.

## Validation / Sanity Checks

Three checks, all out of sample.

**Does the model predict games.** Walking forward through seventeen seasons (leaving
three atypical seasons aside), refitting at two-week intervals and scoring the next
two weeks, the ratings beat a naive rule by about 0.07 in log loss. The naive rule
ignores the teams entirely and gives every home side the league's average home win
rate, which scores 0.680. Early in a season the model without a prior scores 0.692,
*worse than the naive rule*; with the prior it scores 0.626. The gain decays to zero
at around forty games played, which is where thirteen games of prior weight should
stop mattering.

**Is the win distribution the right width.** Comparing every team's actual final win
total against the distribution the model produced for it in January, across 510
team-seasons, 79.6 percent of outcomes fall inside the model's eighty percent
interval. Without the drift term that figure is 75.7 percent, which is direct
evidence that a fixed-strength model is too confident.

**Are the prices internally consistent.** Championship probabilities sum to one,
playoff berths to sixteen, conference finals berths to four.

Two hypotheses were tested and rejected. Weighting games by recency, and discounting
games played once a team's playoff fate was already settled, both produced gains of
around 0.002 in log loss with standard errors of the same size. The noise floor for
this sample is around 0.002, measured from the stages where the prior is known to do
nothing. Neither was adopted.

## Pricing a season that has not started

The 2026-27 season has not begun, so the only input is the prior: last season's
ratings pulled toward the league mean. Run on opening night, the model gives:

| team | title | make playoffs | mean wins | sd wins |
|---|---|---|---|---|
| OKC | 13.4% | 92.1% | 53.2 | 9.0 |
| SAS | 10.1% | 87.9% | 51.2 | 9.4 |
| DET | 8.7% | 83.4% | 49.2 | 9.5 |
| BOS | 8.0% | 84.9% | 49.4 | 9.6 |
| HOU | 5.8% | 78.1% | 47.0 | 9.6 |
| NYK | 5.8% | 76.8% | 46.5 | 9.7 |
| DEN | 5.6% | 76.4% | 46.5 | 9.7 |
| CLE | 4.9% | 74.3% | 45.5 | 9.6 |
| CHA | 4.8% | 72.1% | 44.9 | 9.9 |
| TOR | 3.6% | 65.4% | 42.9 | 9.6 |

The market has Oklahoma City and San Antonio as co-favourites, quoted around +250
to +275 at the books and 21.5 and 20.5 percent on Polymarket, where there is no
overround to strip out. So the top of the model agrees with the top of the market
on nothing but last season.

The disagreements below that are more interesting, because each one names
something the model cannot see. Detroit won 60 games last year and the model has
them third; the market has them in a group at 25-1. Philadelphia is absent from
the model's top ten and fourth in the market, having signed LeBron James and
traded for Jaylen Brown over the summer. Boston is fourth in the model and far
lower in the market, being the team that gave Brown up. One transaction, two
teams, and the model is wrong in both directions.

The model is also flatter than the market: 13.4 percent on the favourite against
about 21. On a date where the only evidence is a season that ended in June, being
less certain than a market that knows the rosters is the right side to be wrong
on. The size of that gap is a fair measure of what roster information is worth,
and it is the strongest argument for the first two items under where this goes
next.

Two caveats. The schedule carries 80 of the 82 games per team, because the
fixtures replacing NBA Cup knockout games are not published yet, so win totals run
about two light. And home advantage cannot be estimated with nothing played, so a
league-level value stands in until games arrive.

## Assumptions

The model has no information about who is on the floor. It learns that a star is hurt
only from results, several games late, and a deadline trade reaches it the same way.
Any information about players is not incorporated.

Games within a playoff series are drawn independently, which overprices the favourite
over seven games. I decided not to calibrate the correlation with the historical
average as this aggregate in my opinion doesn't provide a signal about the underlying
process. Again, information about players, injuries, rosters should be used here.

The drift volatility is the parameter the championship price is most sensitive to.
The median across the sample is 0.18 and recent seasons sit near 0.25. Across that
range the title price for the strongest team moves from 31.6 percent down to 28.0. Recent
seasons were chosen because strength has moved roughly twice as fast over the sample,
and because 0.25 is also the value that makes the interval coverage above come out
right. It is still the number I would watch first on a live dashboard.
