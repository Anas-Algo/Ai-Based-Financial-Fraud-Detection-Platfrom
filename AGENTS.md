<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

- Flask API client lives in src/lib/fraud-api.ts (base URL from VITE_FLASK_API_URL); React Query options in src/lib/api-queries.ts. Why: one seam for the real backend, never return fake data.
- Demo data (src/lib/demo-data.ts) stays separate from API data. Why: easy removal once Flask is live.
