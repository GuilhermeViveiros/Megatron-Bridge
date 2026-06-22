---
name: feedback-src-edits
description: Ask before modifying src/ files unless the change is euro-vl related
metadata:
  type: feedback
---

Do not modify files under src/ without explicitly telling the user first and getting confirmation — unless the change is directly related to the euro-vl model (`src/megatron/bridge/models/euro_vl/`).

**Why:** User wants visibility and control over changes to the shared codebase.

**How to apply:** For any src/ edit outside euro-vl, explain the change and ask permission before touching the file.
