# Repository Guidelines

## Research Goal

The final research objective of this repository is to create an AI agent that operates as an FPS player in an unknown environment where enemies and other hazards are present. The agent must rely only on information that would be available to the player, evaluate risk and partial observations as it explores, and continually choose the best available action while progressing toward the goal. Development decisions should support this long-term objective rather than treating the project as a simple fully observable shortest-path problem.

## Project Structure & Module Organization

This is a Unity 6 (`6000.3.9f1`) project for training an ML-Agents path-search agent. Runtime C# code lives in `Assets/Scripts/`, with `ExplorerAgent.cs` implementing agent behavior and `MapGenerator.cs` building environments. Scenes are in `Assets/Scenes/`, reusable objects in `Assets/Prefab/`, materials and trained models in `Assets/Materials/` and `Assets/Models/`, and PPO settings in `Assets/config/trainer.yaml`. Unity package versions are pinned in `Packages/`; project-wide editor settings belong in `ProjectSettings/`. Treat `Library/`, `Temp/`, `Logs/`, `obj/`, and generated solution files as local artifacts. Training outputs under `results/` can be large; add them only when they are intentional deliverables.

## Build, Test, and Development Commands

- Open the repository with Unity Hub using Editor `6000.3.9f1`, then load `Assets/Scenes/SampleScene.unity` and press Play for local validation.
- `dotnet build PathSearchAgentTest.sln` performs a quick C# compilation check after Unity has generated the solution.
- `Unity.exe -batchmode -projectPath . -runTests -testPlatform EditMode -testResults test-results.xml -quit` runs Unity Edit Mode tests from a Developer PowerShell (adjust the executable path as needed).
- `mlagents-learn Assets/config/trainer.yaml --run-id=<name>` starts PPO training. Add `--resume` to continue a run or `--force` to replace one; never combine those flags casually.

## Coding Style & Naming Conventions

Use four-space indentation and standard Unity C# conventions: PascalCase for types, methods, and public members; camelCase for locals and private fields. Keep MonoBehaviour lifecycle methods grouped logically and use `[SerializeField] private` when Inspector access is needed without expanding the public API. Prefer focused components over adding unrelated responsibilities to `ExplorerAgent`. Commit every Unity asset together with its matching `.meta` file; do not hand-edit generated `.csproj` files.

## Testing Guidelines

The repository currently has no dedicated automated tests. Add new tests under `Assets/Tests/EditMode/` or `Assets/Tests/PlayMode/`, using filenames such as `MapGeneratorTests.cs`. Use the Unity Test Framework (`com.unity.test-framework`) and cover deterministic map generation, reward changes, episode termination, and collision behavior. Before submitting, run the relevant test platform and manually verify `SampleScene` in Play Mode.

## Commit & Pull Request Guidelines

Recent commits use short, descriptive Japanese summaries focused on behavior changes. Follow that established style, keep each commit scoped, and state the observed outcome (for example, `ゴール視認後の距離報酬を調整`). Pull requests should explain the gameplay or training impact, list validation performed, link related issues, and include screenshots or short captures for scene or Inspector changes. Call out modified seeds, hyperparameters, model files, and generated training results explicitly.
