# Retained result tables

Small synthetic result and sensitivity tables remain at their original paths and retain their original bytes. `matched_completion/` contains training and pair-evaluation summaries, including files explicitly named as smoke results. `sensitivity/` contains historical sensitivity summaries.

These files must be linked to their exact data, source revision, settings and checkpoints before supporting a revised paper claim. A CSV's presence is not evidence that the complete reproduction protocol has passed. Append-only merge behavior for existing matched-completion tables is unchanged.

Checkpoints, raw logs, per-example binary prediction files and routine generated outputs are ignored. Store large run artifacts separately and commit a reviewed manifest and small summary. Clinical patient-level predictions remain in credentialed storage.
