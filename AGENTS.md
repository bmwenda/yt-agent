# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python agent for YouTube workflows. `app.py` contains the LangChain orchestration and demo entry point. `tools.py` defines the tool functions used by the agent. There is no dedicated `tests/` directory or packaging metadata yet, so keep new code close to these two modules unless the project grows.

## Build, Test, and Development Commands
There is no formal build system. Use these commands during development:

- `python app.py` runs the current demo flow.
- `python -m py_compile app.py tools.py` checks for syntax errors without calling external APIs.

If you add tests, prefer `pytest` and place them under `tests/`.

## Coding Style & Naming Conventions
Use standard Python style: 4-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes if any are added. Keep tool functions in `tools.py` and orchestration logic in `app.py`. Favor small, explicit functions over hidden control flow. Keep strings and comments concise, and use ASCII unless a non-ASCII character is already required.

## Testing Guidelines
No test framework is configured today. Add tests alongside new behavior, especially for helper functions like `extract_video_id` and error handling around transcript or metadata retrieval. Name tests after behavior, for example `test_extract_video_id_handles_watch_url`.

## Commit & Pull Request Guidelines
This checkout does not include a Git history to infer a commit style, so use short imperative commit messages such as `Add transcript fallback`. For pull requests, include a brief summary of the behavior change, any new environment variables or API dependencies, and manual verification steps. If output changes, note a representative example.

## Security & Configuration Tips
Keep secrets in `.env` and never commit them. `OPENAI_API_KEY` is required for the OpenAI-backed agent path, and YouTube-related libraries may hit live network services during runtime. Avoid logging raw secrets, full tokens, or private environment values.
