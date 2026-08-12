# Pre-enriched Candidate Upload — File Specification

For the enrichment team preparing candidate files outside the app (Claude/manual research).
Files in this format upload through **Talent Pool → Upload CSV / Excel** (or a role's Upload
button), load in the background, and the candidates are immediately usable in **AI Shortlist** —
no in-app "Fetch Contact" or further enrichment needed.

A ready-to-fill template with one example row: `docs/templates/candidate_upload_template.csv`.

## File basics

- **Format:** `.csv` (UTF-8) or `.xlsx`/`.xls` (first sheet only, headers in row 1)
- **One row = one candidate**
- Column names below are matched case-insensitively; punctuation/underscores are ignored
- Extra columns are never lost — anything unrecognized is stored on the candidate as
  searchable "uploaded fields" and AI Shortlist can match on them

## Required columns (upload is rejected without all five)

| Column | Rules |
|---|---|
| `firstName` | |
| `lastName` | |
| `linkedinPublicUrl` | Full URL, e.g. `https://www.linkedin.com/in/john-doe-123`. **This is the dedupe key** — rows without it fall back to email / name+company matching |
| `City` | City name, e.g. `Bengaluru` |
| `Title` | Current job title (e.g. `Senior BDR`). Stored as the candidate's headline. If omitted but a `headline` column exists, that is used instead |

## Contact columns (strongly recommended — this is the point of pre-enrichment)

| Column | Rules |
|---|---|
| `Email` | One address, `name@domain.tld` |
| `Mobile` | With country code, e.g. `+919876543210`. Stored as-is — clean formatting matters because the dialer uses it |

Note: if the same person already exists in the database **with** contact info, the existing
DB values win over the file (the file fills blanks; it does not overwrite).

## Profile columns (drive AI Shortlist quality)

| Column | Rules |
|---|---|
| `Profile Description` | The LinkedIn About/bio text. Searchable by AI Shortlist. IMPORTANT: do NOT name this column "about", "bio" or "summary" — those are currently dropped by the importer |
| `addressWithCountry` | Full location, e.g. `Bengaluru, Karnataka, India`. Feeds location filters |
| `Skills` | Comma/semicolon-separated list |
| `totalExperienceYears` | Number, e.g. `7.5`. Used as the authoritative value for "minimum X years experience" filters |

## Work history — repeat the block per job, `/0/` = most recent

```
experiences/0/companyName
experiences/0/title
experiences/0/jobStartedOn      ← "Jan 2023" or "2023-01"
experiences/0/jobEndedOn        ← empty = current job
experiences/0/jobDescription
experiences/0/companyIndustry
experiences/0/jobLocation
experiences/1/companyName
experiences/1/title
... up to experiences/19/...
```

These power tenure math ("min 2 years in current role", "avg tenure"), company/industry
filters, and the evidence AI Shortlist quotes. **The more complete, the better the
shortlisting.**

## Education — repeat per degree

```
educations/0/title                    ← institution
educations/0/subtitle                 ← degree, e.g. "B.Tech, Computer Science"
educations/0/period/startedOn/year
educations/0/period/endedOn/year
```

## Recruiter extras (recognized automatically, filterable in AI Shortlist)

`Recruiter Summary`, `Current CTC`, `Expected CTC`, `Notice Period`, `Preferred Location`,
`Focused Geography`, `Shift timings`, `Outbound Exp`, `Targets`, `CV` (link)

Any other column you invent is also kept, under its own name.

## Upload procedure (whoever loads the file)

1. Log in as a **recruiter** account (the upload endpoint rejects admin accounts) —
   or use a role's Upload button to also auto-assign candidates to that role.
2. Talent Pool → **Upload CSV / Excel** → confirm the column mapping the preview suggests
   (it is automatic for every column named as in this spec).
3. If the file has `experiences/...` columns, the **Verified profile** enrichment mode is
   selected automatically — keep it. It parses the work history into structured
   roles/companies/education and computes tenure numbers. No external calls needed when
   the file is complete.
4. Commit. The load runs in the background; progress is shown in the upload panel.
5. Done — candidates appear in Talent Pool and are immediately searchable in AI Shortlist.

## What you do NOT need to provide

- Embeddings / vectors — AI Shortlist does not require them
- `normalized_linkedin`, dedupe bookkeeping, status — computed automatically
- Company research (funding stage, size, etc.) — derived from the work-history block

## Dedupe behavior

Matching order: LinkedIn URL → email → (first + last + company). Re-uploading the same
person **updates** their record (file values fill blanks, extra fields merge); it never
creates a duplicate. To load a correction run, just upload the corrected file again.
