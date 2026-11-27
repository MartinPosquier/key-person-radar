# Key-Person Risk Radar

**How much of a company is one person?**

We measure how strongly a handful of public figures (CEO, founder, CFO) move their company's
stock price, independently of fundamentals, for about 60 large US and European companies
over 2015–2025. For each person and year we estimate:

- **S−**: sensitivity of the stock to bad news about the person (0–10);
- **S+**: sensitivity to good news (0–10);
- **V**: implied share of market value tied to the person (an estimate, with its interval).

Signals come from public attention data (Wikipedia pageviews, news volume and tone from
GDELT) combined with an event study on abnormal returns and machine learning models.

> Work in progress: ESILV machine learning project. Not investment advice.

## Status

Bootstrapping. See `PARALLEL_SESSIONS.md` for how the work is organised.
