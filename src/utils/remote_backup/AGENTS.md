# src/utils/remote_backup

Checkpoint transport through SSH or W&B.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Strategies implement remote upload/download behind one interface. Builder initialization downloads resume artifacts before callbacks initialize run identity.

## File map

| File(s) | Responsibility |
| --- | --- |
| [base.py](base.py) | RemoteBackupStrategy interface. |
| [ssh_strategy.py](ssh_strategy.py) | SSH/SCP/rsync checkpoint and run-ID transport. |
| [wandb_strategy.py](wandb_strategy.py) | W&B artifact upload/download and checkpoint aliases. |
| [__init__.py](__init__.py) | Strategy exports. |

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
