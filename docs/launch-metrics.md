# PrintForge Launch Metrics Targets

## Day 1 Targets (Launch Day)

| Metric | Target | Success | Failure |
|--------|--------|---------|---------|
| GitHub Stars | 100+ | >150 stars | <50 stars |
| Reddit Upvotes (r/3Dprinting) | 200+ | Front page, >500 | <50 upvotes |
| Hacker News Points | 50+ | Front page, >100 | <20 points |
| Product Hunt Upvotes | 100+ | Top 5 of the day | <30 upvotes |
| Demo video views | 1,000+ | >5,000 | <200 |

## Week 1 Targets

| Metric | Target | Success | Failure |
|--------|--------|---------|---------|
| pip installs | 500+ | >1,000 | <100 |
| Unique GitHub visitors | 2,000+ | >5,000 | <500 |
| Issues opened | 20+ | >50 (shows engagement) | <5 |
| Pull requests from community | 3+ | >10 | 0 |
| Docker pulls | 200+ | >500 | <50 |
| Discord/community members | 50+ | >200 | <10 |

## Month 1 Targets

| Metric | Target | Success | Failure |
|--------|--------|---------|---------|
| Monthly Active Users (MAU) | 500+ | >2,000 | <100 |
| Pro tier conversions | 20+ | >50 (10% conversion) | <5 |
| Monthly Recurring Revenue | $500+ | >$2,000 | <$100 |
| GitHub Stars (cumulative) | 1,000+ | >3,000 | <300 |
| pip installs (cumulative) | 5,000+ | >15,000 | <1,000 |
| Average quality score | 75+ | >85 | <60 |
| Models generated (total) | 10,000+ | >50,000 | <1,000 |

## Key Conversion Funnel

```
Visit GitHub/Website  →  Install  →  First Generation  →  Regular Use  →  Pro
    100%                   30%          60%                  25%           10%
```

## Success / Failure Criteria

### Green (Launch Success)
- Hit Week 1 pip install target
- Positive sentiment on Reddit/HN (>70% positive comments)
- At least 3 community PRs
- No critical bugs reported in first 48 hours

### Yellow (Needs Attention)
- Hit Day 1 but miss Week 1 targets
- Mixed sentiment (50-70% positive)
- 1-2 community PRs
- Non-critical bugs only

### Red (Launch Failure)
- Miss Day 1 targets across all channels
- Negative sentiment dominates
- Zero community contributions
- Critical bugs in core pipeline

## Tracking Tools

- **GitHub Insights** — stars, clones, traffic, referrers
- **PyPI Stats** — pip install counts (pypistats.org)
- **Google Analytics** — web UI visitors, session duration
- **Local Analytics** — printforge stats (built-in telemetry)
- **Social Listening** — Reddit, HN, Twitter/X mentions
