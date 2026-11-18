# src/data_processor/downloaders

Dataset download backends.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

The preparation dispatcher instantiates a backend from dataset_config.yaml. Each backend downloads/extracts data; reorganization is a separate hook.

## File map

| File(s) | Responsibility |
| --- | --- |
| [kaggle_downloader.py](kaggle_downloader.py) | Authenticated Kaggle dataset retrieval. |
| [url_downloader.py](url_downloader.py) | Direct URL files and ZIP extraction. |
| [multi_part_archive_downloader.py](multi_part_archive_downloader.py) | Selectable multi-part archives, transfer, and extraction. |
| [__init__.py](__init__.py) | Downloader package exports. |

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
