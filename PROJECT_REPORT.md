# Project Report

## Update 1 — Job-description skill extraction

**Date:** 2026-10-22

### Problem

The missing-skills list included ordinary job-description prose and partial phrases (for example, "and unstructured datasets for", "bachelor's or", and "VP of engineering key responsibilities") instead of only relevant skills.

### Cause

The resume-focused NER model was also used to identify skills in job descriptions. It often labels surrounding prose as a skill, and the existing phrase cleanup only filtered empty or oversized spans.

### Solution

- Added skill-specific cleanup that splits comma- and conjunction-separated candidates, trims grammatical words at phrase boundaries, and rejects obvious degree, job-title, section-heading, and generic-prose fragments.
- Kept the current model stack after testing a job-posting model that missed common explicit technical skills and still labeled ordinary action phrases as skills.
- Added a regression test covering the reported noisy examples while confirming that actual skills, titles, and degrees remain categorized.

### Validation

- The focused unit test verifies that noisy spans are filtered, comma/conjunction-separated skills remain separate, and title/degree extraction is unchanged.
- No additional model download or dependency is required for this fix.

### Follow-up

The first cleanup pass did not normalize whitespace around model-produced apostrophes and did not apply the responsibility-phrase check to non-skill labels. A follow-up adds entity normalization, filters incomplete degree fragments and responsibility-tailed titles across categories, and excludes generic standalone "model(s)" labels from skills. The regression test now includes the exact remaining examples reported after Update 1.
