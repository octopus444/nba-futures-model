# NBA Season Futures: Fair Value Model

A fair-value model for season-long futures markets. Given a league, a season and
an as-of date, it produces prices for four contract types:

- **To Win Championship**
- **To Make Playoffs**
- **To Reach Round X** (every round, from the first round to the final)
- **Over/Under Season Wins**

NBA is implemented end to end. NFL is included as a config to show where a second
league plugs in.

---

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

The processed dataset is committed at `data/processed/games.csv`, so nothing needs
downloading. Open `notebooks/pipeline.ipynb` and run it top to bottom.

Two inputs control everything, and they sit in one cell near the top:

```python
SEASON = 2024        # season ending in 2024, i.e. 2023-24
ASOF = "2024-01-15"  # everything before this date is fact
```

### Notebooks

| notebook | what it does | need to run it? |
|---|---|---|
| `pipeline.ipynb` | prices all four contract types for one as-of date | **yes, this is the model** |
| `calibration.ipynb` | derives the league constants and runs the validation | no, unless re-deriving |
| `data_fetch.ipynb` | pulls from the NBA stats API and builds the dataset | no, output is committed |

`calibration.ipynb` takes a few minutes and reproduces every number quoted in the
writeup. `data_fetch.ipynb` needs network access and caches raw responses to
`data/raw/`, which is gitignored.

---

## Where the line is drawn

A clear distinction between engine and league-specific stuff.

**`engine/` knows how to model a season. It knows nothing about basketball.**

| file | responsibility |
|---|---|
| `observations.py` | split a season at the as-of date, turn a scoreline into an observation |
| `ratings.py` | least-squares team strength, with prior and uncertainty |
| `carryover.py` | measure how much strength survives an off-season |
| `probabilities.py` | strength difference to win probability |
| `uncertainty.py` | sample strength trajectories from the posterior |
| `simulate.py` | play out a remaining regular season |
| `standings.py` | order teams into a seeded field |
| `playoffs.py` | series, byes, reseeding, recursion through rounds |
| `contracts.py` | run everything together, count outcomes into prices |
| `weights.py`, `motivation.py` | per-game weighting (tested, not enabled: see writeup) |

**`leagues/` knows the sport and nothing else.**

`leagues/nba.py` holds conferences and divisions, the postseason shape, the play-in
rules, and a small set of constants calibrated from the data. `leagues/nfl.py` holds
the same for the NFL.

The interesting cases are the ones that would tempt you to put a rule in the engine:

- **Qualification.** The play-in tournament is an NBA rule, not machinery, so it is a
  function living in `leagues/nba.py` that the config passes to the engine. A league
  where the top seeds simply qualify passes nothing.
- **Series format, per round.** Stored as one entry per round rather than one per
  league. The NBA plays best-of-seven throughout; the NFL plays single elimination,
  which is the same code with one win needed and a one-game home pattern. A league
  with different formats by round needs no new code.
- **Byes and reseeding.** The NFL gives its top seed a bye and re-pairs survivors by
  seed after every round; the NBA does neither. Both are flags on the config.

### Adding a league

Write one file in `leagues/`. Supply the group structure, the postseason shape, and
the calibration constants. Run `calibration.ipynb` against that league's data to
derive the constants rather than guessing them.

**What NFL would still need from the engine, honestly:** seeding. `standings.py`
orders teams by wins and breaks ties at random, which is an acceptable simplification
for the NBA and wrong for the NFL, where division winners are seeded above wild cards
regardless of record. That needs a seeding hook on the config, alongside the
qualification hook that already exists. It is the one engine change a second league
actually justifies, and it is noted in `leagues/nfl.py`.

---

## Design notes

Fuller reasoning is in `WRITEUP.md`. The short version:

- **Strength is estimated by regression on point margins**, not win rates and not Elo.
  Margins carry more information than results, and regression corrects for schedule
  strength in a way that averaging does not. Ratings come out in points, so a rating
  of 7.8 means beating an average opponent by 7.8 on a neutral floor.
- **A prior from last season makes early-season dates work.** Its weight is derived
  from two measured quantities rather than tuned, and comes to about twelve games of
  evidence.
- **Uncertainty is carried through the simulation.** Ratings are not treated as known
  exactly, and strength is allowed to move between the as-of date and the day a
  contract resolves. Both matter: holding strength fixed systematically overprices
  favourites on long-horizon contracts.
- **Validation is out-of-sample.** Walk-forward across seventeen seasons, plus a check
  that the width of the simulated win distribution matches how far real seasons
  actually land from their forecast.

---

## Where this goes next (extensions)

**Expected score instead of actual score.** A final margin is one random draw from a process: a team that took thirty good three-point looks and hit eight of them is not weaker than one that took twenty poor looks and hit ten, but the scoreline says otherwise. Shot-quality data lets you build the margin the game deserved, the basketball equivalent of expected goals, and feed that to the regression instead. Everything downstream is unchanged, the observation is simply less noisy. The rest of this model spends considerable effort managing noise that this would remove at the source.

**Roster and transaction data.** The biggest missing input. The model learns that a star is hurt only from results, several games late, and a trade deadline reshuffle reaches it the same slow way. Minutes and availability would let strength be attributed to players rather than to a team-shaped abstraction, which fixes injuries, trades and rest with one mechanism.

**Weighting games by how much they still describe the team.** Two versions were tested here and neither cleared the noise floor, but both were crude. The shape that makes sense is not monotone in recency: weight should hold up through the trade deadline, when rosters freeze, and fall off in the closing weeks for teams whose seed is already settled and not for teams still fighting for one. Fitting that needs more parameters than the postseason sample can identify, so it needs a target with more data behind it.

**Correlation within a series, informed rather than assumed.** Games in a series are drawn independently, which overprices the favourite over seven games. Fitting a single correlation parameter to the historical distribution of series lengths would be fitting to an aggregate that several other things also move. The version worth building uses what actually causes the correlation: an injury inside a series, a matchup a coach solves after two games, a rotation shortening. That is roster data again.

**Schedule congestion.** Strength of schedule is already handled, by the regression looking backward and by simulating the remaining fixtures one by one. What is not handled is fatigue: back-to-backs, long road trips, three games in four nights. These are known in advance from the schedule, so they cost nothing to include once calibrated.
