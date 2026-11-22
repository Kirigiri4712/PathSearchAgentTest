using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using Unity.MLAgents.Policies;
using UnityEngine;

/// <summary>
/// Owns the shared training configuration and creates isolated training areas.
/// Place this component at the origin of the area layout.
/// </summary>
public class LearnManager : MonoBehaviour
{
    [Serializable]
    public sealed class InferenceFailureRecord
    {
        public int seed;
        public int areaIndex;
        public string agent;
        public string layout;
        public int mapWidth;
        public int mapHeight;
        public int waypointCount;
        public string endReason;
        public int steps;
        public Vector3 spawnPosition;
        public Vector3 goalPosition;
        public Vector3 finalPosition;
        public Vector3 finalVelocity;
        public int visitedCellCount;
        public string map;
        public List<ExplorerAgent.InferenceTraceSample> trace;
    }

    private enum OperationMode
    {
        Training,
        InferenceTest
    }

    [Header("Prefabs")]
    [SerializeField] private MapGenerator mapGeneratorPrefab;
    [Tooltip("Assign the complete player prefab root. The ExplorerAgent may be on a child object.")]
    [SerializeField] private GameObject explorerAgentPrefab;

    [Header("Operation Mode")]
    [SerializeField] private OperationMode operationMode = OperationMode.Training;
    [Tooltip("Required in Inference Test mode. Import the trained ONNX into Assets before assigning it.")]
    [SerializeField] private ModelAsset testModel;
    [Tooltip("Use deterministic model actions so repeated seed tests are reproducible.")]
    [SerializeField] private bool deterministicTestInference = true;
    [Tooltip("Map topology used only for single-agent inference tests. Training always uses the current corridor layout.")]
    [SerializeField] private MapLayoutMode inferenceTestMapLayout = MapLayoutMode.PathCorridor;
    [SerializeField, Min(1)] private int inferenceTestAreaCount = 4;
    [SerializeField, Min(1)] private int inferenceTestAreaColumns = 2;
    [SerializeField, Min(1f)] private float inferenceTestTimeScale = 5f;

    [Header("Initial Map Complexity")]
    [SerializeField, Min(5)] private int mapWidth = 5;
    [SerializeField, Min(5)] private int mapHeight = 5;
    [SerializeField, Min(0)] private int waypointCount;

    [Header("Random Seeds")]
    [Tooltip("0 selects random training seeds; non-zero keeps training seed generation reproducible.")]
    [SerializeField] private int randomSeed;
    [SerializeField, Min(1)] private int testSeedMinimum = 1;
    [SerializeField, Min(1)] private int testSeedMaximum = 1000;
    [Tooltip("Training debug only: repeat one reserved test seed instead of generating training maps.")]
    [SerializeField] private bool useFixedTestSeed;
    [SerializeField, Min(1)] private int fixedTestSeed = 1;

    [Header("Parallel Training Areas")]
    [SerializeField, Min(1)] private int trainingAreaCount = 2;
    [SerializeField, Min(1)] private int areaColumns = 2;
    [SerializeField, Min(5)] private int maximumMapWidth = 40;
    [SerializeField, Min(5)] private int maximumMapHeight = 40;
    [SerializeField, Min(0)] private int maximumWaypointCount = 20;
    [SerializeField, Min(0)] private float areaGap = 5f;

    [Header("Automatic Curriculum")]
    [SerializeField] private bool automaticCurriculum = true;
    [Tooltip("Each area keeps this many recent results at the current difficulty. All areas must fill the window before a curriculum check.")]
    [SerializeField, Min(1)] private int successRateWindow = 20;
    [Tooltip("Mean per-area success rate required to advance difficulty, expressed as successes in the configured window.")]
    [SerializeField, Min(1)] private int successesRequiredToAdvance = 18;
    [Tooltip("Mean per-area success rate at or below this number of successes in the configured window decreases difficulty by one level.")]
    [SerializeField, Min(0)] private int successesRequiredToRegress = 10;
    [SerializeField, Min(1)] private int mapWidthIncreasePerLevel = 2;
    [SerializeField, Min(1)] private int mapHeightIncreasePerLevel = 2;
    [Tooltip("Increase waypoints once per this many difficulty promotions.")]
    [SerializeField, Min(1)] private int waypointIncreaseEveryLevels = 3;
    [SerializeField, Min(1)] private int waypointIncreaseAmount = 1;

    private readonly List<TrainingArea> trainingAreas = new List<TrainingArea>();
    private readonly List<InferenceFailureRecord> inferenceFailures =
        new List<InferenceFailureRecord>();
    private readonly Dictionary<ExplorerAgent, TrainingArea> areaByAgent =
        new Dictionary<ExplorerAgent, TrainingArea>();
    private System.Random randomTrainingSeedGenerator;
    private int curriculumBaseMapWidth;
    private int curriculumBaseMapHeight;
    private int curriculumBaseWaypointCount;
    private int nextTestSeed;
    private bool testRunComplete;
    private float previousTimeScale = 1f;
    private bool changedTimeScale;
    private bool inferenceRuntimeErrorHandled;
    [SerializeField, HideInInspector] private bool captureInferenceTrace;

    public int CompletedEpisodes { get; private set; }
    public float RecentSuccessRate { get; private set; }
    public int CurriculumAreasReady { get; private set; }
    public int CurriculumLevel { get; private set; }
    public int TestEpisodesCompleted { get; private set; }
    public int TestSuccesses { get; private set; }
    public float TestSuccessRate { get; private set; }
    public bool IsTestRunComplete => testRunComplete;
    public bool IsInferenceTestMode => operationMode == OperationMode.InferenceTest;
    public bool CaptureInferenceTrace => captureInferenceTrace;
    public IReadOnlyList<InferenceFailureRecord> InferenceFailures => inferenceFailures;

#if UNITY_EDITOR
    public void ConfigureInferenceTestForEditor(
        ModelAsset model,
        MapLayoutMode mapLayout,
        int areaCount,
        int columns,
        float timeScale,
        int seedMinimum,
        int seedMaximum,
        int testMapWidth,
        int testMapHeight,
        bool captureActionTrace = false)
    {
        operationMode = OperationMode.InferenceTest;
        testModel = model;
        deterministicTestInference = true;
        inferenceTestMapLayout = mapLayout;
        inferenceTestAreaCount = areaCount;
        inferenceTestAreaColumns = columns;
        inferenceTestTimeScale = timeScale;
        testSeedMinimum = seedMinimum;
        testSeedMaximum = seedMaximum;
        mapWidth = testMapWidth;
        mapHeight = testMapHeight;
        captureInferenceTrace = captureActionTrace;
        automaticCurriculum = false;
    }
#endif

    private sealed class TrainingArea
    {
        public MapGenerator MapGenerator { get; }
        public ExplorerAgent Agent { get; }
        public int EpisodeCount { get; set; }
        public int ActiveEpisodeDifficultyLevel { get; set; }
        public int ActiveSeed { get; set; }
        public Queue<bool> RecentEpisodeResults { get; } = new Queue<bool>();
        public int RecentSuccesses { get; private set; }
        public float RecentSuccessRate => RecentEpisodeResults.Count == 0
            ? 0f
            : (float)RecentSuccesses / RecentEpisodeResults.Count;

        public TrainingArea(MapGenerator mapGenerator, ExplorerAgent agent)
        {
            MapGenerator = mapGenerator;
            Agent = agent;
        }

        public void RecordEpisodeResult(bool reachedGoal, int windowSize)
        {
            RecentEpisodeResults.Enqueue(reachedGoal);
            if (reachedGoal)
            {
                RecentSuccesses++;
            }

            while (RecentEpisodeResults.Count > windowSize)
            {
                if (RecentEpisodeResults.Dequeue())
                {
                    RecentSuccesses--;
                }
            }
        }

        public void ClearRecentResults()
        {
            RecentEpisodeResults.Clear();
            RecentSuccesses = 0;
        }
    }

    private void Awake()
    {
        nextTestSeed = testSeedMinimum;
        curriculumBaseMapWidth = mapWidth;
        curriculumBaseMapHeight = mapHeight;
        curriculumBaseWaypointCount = waypointCount;

        if (IsInferenceTestMode)
        {
            if (testModel == null)
            {
                Debug.LogError("Inference Test mode requires a trained ONNX model in the Test Model field.", this);
                enabled = false;
                return;
            }

            // Evaluation uses one shared policy, a fixed difficulty, and each
            // reserved seed exactly once. It must not advance the curriculum.
            automaticCurriculum = false;
            previousTimeScale = Time.timeScale;
            Time.timeScale = Mathf.Max(1f, inferenceTestTimeScale);
            changedTimeScale = true;
            Application.logMessageReceived += StopInferenceTestAfterRuntimeError;
        }

        if (randomSeed == 0)
        {
            randomTrainingSeedGenerator = new System.Random(Guid.NewGuid().GetHashCode());
        }

        CreateTrainingAreas();
    }

    /// <summary>
    /// Called by each agent when its episode begins. The current curriculum level
    /// is captured so episodes already in progress cannot affect the next level.
    /// </summary>
    public GeneratedMap BeginEpisode(ExplorerAgent agent)
    {
        if (!areaByAgent.TryGetValue(agent, out TrainingArea area))
        {
            throw new InvalidOperationException(
                $"Agent '{agent.name}' is not registered with this LearnManager.");
        }

        int seed = CreateEpisodeSeed(area);
        area.ActiveSeed = seed;
        area.ActiveEpisodeDifficultyLevel = CurriculumLevel;
        area.EpisodeCount++;

        if (!agent.TryGetVerticalSpawnAlignment(out float bodyWorldY, out float footWorldY))
        {
            throw new InvalidOperationException(
                $"Agent '{agent.name}' needs a non-trigger Collider attached to its Rigidbody to align the generated floor.");
        }

        Transform mapTransform = area.MapGenerator.transform;
        Vector3 mapOrigin = mapTransform.position;
        float playerLocalY = mapTransform.InverseTransformPoint(
            new Vector3(mapOrigin.x, bodyWorldY, mapOrigin.z)).y;
        float floorSurfaceLocalY = mapTransform.InverseTransformPoint(
            new Vector3(mapOrigin.x, footWorldY, mapOrigin.z)).y;

        MapLayoutMode layoutMode = IsInferenceTestMode
            ? inferenceTestMapLayout
            : MapLayoutMode.PathCorridor;

        return area.MapGenerator.Generate(
            mapWidth,
            mapHeight,
            waypointCount,
            seed,
            playerLocalY,
            floorSurfaceLocalY,
            layoutMode);
    }

    /// <summary>
    /// Entry point for curriculum logic to change the map difficulty.
    /// Changes take effect when each area's next episode starts.
    /// </summary>
    public void SetMapComplexity(int width, int height, int waypoints)
    {
        mapWidth = Mathf.Clamp(width, 5, maximumMapWidth);
        mapHeight = Mathf.Clamp(height, 5, maximumMapHeight);
        waypointCount = Mathf.Clamp(waypoints, 0, maximumWaypointCount);
        curriculumBaseMapWidth = mapWidth;
        curriculumBaseMapHeight = mapHeight;
        curriculumBaseWaypointCount = waypointCount;
        CurriculumLevel = 0;
        ClearCurriculumWindows();
    }

    public bool RecordEpisodeResult(
        ExplorerAgent agent,
        bool reachedGoal,
        string endReason,
        int steps,
        Vector3 spawnPosition,
        Vector3 goalPosition,
        Vector3 finalPosition,
        Vector3 finalVelocity,
        int visitedCellCount)
    {
        if (!areaByAgent.TryGetValue(agent, out TrainingArea area))
        {
            Debug.LogWarning($"Received an episode result from unregistered agent '{agent.name}'.", agent);
            return false;
        }

        CompletedEpisodes++;

        if (IsInferenceTestMode)
        {
            if (!reachedGoal)
            {
                int areaIndex = trainingAreas.IndexOf(area);
                MapGenerator mapGenerator = area.MapGenerator;
                InferenceFailureRecord failure = new InferenceFailureRecord
                {
                    seed = area.ActiveSeed,
                    areaIndex = areaIndex,
                    agent = agent.name,
                    layout = mapGenerator.LastLayoutMode.ToString(),
                    mapWidth = mapGenerator.Width,
                    mapHeight = mapGenerator.Height,
                    waypointCount = mapGenerator.WaypointCount,
                    endReason = endReason,
                    steps = steps,
                    spawnPosition = spawnPosition,
                    goalPosition = goalPosition,
                    finalPosition = finalPosition,
                    finalVelocity = finalVelocity,
                    visitedCellCount = visitedCellCount,
                    map = mapGenerator.CreateDiagnosticMapSnapshot(),
                    trace = captureInferenceTrace ? agent.GetInferenceTraceSnapshot() : null
                };
                inferenceFailures.Add(failure);
                Debug.LogWarning(
                    $"[Inference Test Failure] Seed={failure.seed}, layout={failure.layout}, " +
                    $"map={failure.mapWidth}x{failure.mapHeight}, waypoints={failure.waypointCount}, " +
                    $"reason={failure.endReason}, steps={failure.steps}, " +
                    $"finalPosition={failure.finalPosition}, velocity={failure.finalVelocity}, " +
                    $"visitedCells={failure.visitedCellCount}.\n" +
                    failure.map,
                    this);
            }

            TestEpisodesCompleted++;
            if (reachedGoal)
            {
                TestSuccesses++;
            }

            TestSuccessRate = (float)TestSuccesses / TestEpisodesCompleted;
            RecentSuccessRate = TestSuccessRate;
            int totalTestSeeds = testSeedMaximum - testSeedMinimum + 1;
            Debug.Log(
                $"[Inference Test] Seed {area.ActiveSeed}: {(reachedGoal ? "SUCCESS" : "FAIL")} " +
                $"reason={endReason}, steps={steps} " +
                $"({TestEpisodesCompleted}/{totalTestSeeds}), success rate {TestSuccessRate:P1}.",
                this);

            if (TestEpisodesCompleted >= totalTestSeeds)
            {
                testRunComplete = true;
                Debug.Log(
                    $"[Inference Test Complete] Seeds {testSeedMinimum}-{testSeedMaximum}: " +
                    $"{TestSuccesses}/{TestEpisodesCompleted} successes ({TestSuccessRate:P1}).",
                    this);
                RestoreTimeScale();
            }

            // Once every seed has been assigned, completed agents must stay
            // parked while the remaining in-flight seeds finish.
            return testRunComplete || nextTestSeed > testSeedMaximum;
        }

        // Episodes already in progress at the previous level must not affect the
        // per-area windows after the shared curriculum level changes.
        if (area.ActiveEpisodeDifficultyLevel != CurriculumLevel)
        {
            return false;
        }

        int previousReadyAreaCount = CurriculumAreasReady;
        area.RecordEpisodeResult(reachedGoal, successRateWindow);

        bool allAreaWindowsReady = TryCalculateMeanAreaSuccessRate(out float meanSuccessRate);
        if (CurriculumAreasReady > previousReadyAreaCount &&
            (CurriculumAreasReady == trainingAreas.Count ||
             CurriculumAreasReady % Mathf.Max(1, trainingAreas.Count / 10) == 0))
        {
            Debug.Log(
                $"Curriculum assessment windows ready: {CurriculumAreasReady}/{trainingAreas.Count} areas " +
                $"({successRateWindow} episodes per area).",
                this);
        }

        if (!allAreaWindowsReady)
        {
            return false;
        }

        RecentSuccessRate = meanSuccessRate;
        if (!automaticCurriculum || useFixedTestSeed)
        {
            return false;
        }

        float promotionThreshold = (float)successesRequiredToAdvance / successRateWindow;
        float regressionThreshold = (float)successesRequiredToRegress / successRateWindow;

        bool canIncreaseMapSize = mapWidth < maximumMapWidth || mapHeight < maximumMapHeight;
        bool canIncreaseWaypoints = waypointCount < maximumWaypointCount;
        if (RecentSuccessRate >= promotionThreshold && (canIncreaseMapSize || canIncreaseWaypoints))
        {
            AdvanceCurriculum(RecentSuccessRate);
        }
        else if (CurriculumLevel > 0 && RecentSuccessRate <= regressionThreshold)
        {
            RegressCurriculum(RecentSuccessRate);
        }

        return false;
    }

    private bool TryCalculateMeanAreaSuccessRate(out float meanSuccessRate)
    {
        meanSuccessRate = 0f;
        CurriculumAreasReady = 0;
        if (trainingAreas.Count == 0)
        {
            return false;
        }

        float totalAreaSuccessRate = 0f;
        foreach (TrainingArea area in trainingAreas)
        {
            if (area.RecentEpisodeResults.Count < successRateWindow)
            {
                continue;
            }

            CurriculumAreasReady++;
            totalAreaSuccessRate += area.RecentSuccessRate;
        }

        if (CurriculumAreasReady < trainingAreas.Count)
        {
            return false;
        }

        // Every area contributes exactly one rate, independent of how many
        // episodes it completes beyond its own rolling window.
        meanSuccessRate = totalAreaSuccessRate / trainingAreas.Count;
        return true;
    }

    private void CreateTrainingAreas()
    {
        if (mapGeneratorPrefab == null || explorerAgentPrefab == null)
        {
            Debug.LogError("LearnManager requires MapGenerator and ExplorerAgent prefabs.", this);
            enabled = false;
            return;
        }

        ExplorerAgent prefabAgent = explorerAgentPrefab.GetComponentInChildren<ExplorerAgent>(true);
        if (prefabAgent == null)
        {
            Debug.LogError("The assigned player prefab must contain an ExplorerAgent component.", this);
            enabled = false;
            return;
        }

        BehaviorParameters prefabBehavior =
            explorerAgentPrefab.GetComponentInChildren<BehaviorParameters>(true);
        if (IsInferenceTestMode && prefabBehavior == null)
        {
            Debug.LogError("The assigned ExplorerAgent prefab must contain Behavior Parameters for inference testing.", this);
            enabled = false;
            return;
        }

        int areaCount = Mathf.Max(1, IsInferenceTestMode ? inferenceTestAreaCount : trainingAreaCount);
        int columns = Mathf.Clamp(
            IsInferenceTestMode ? inferenceTestAreaColumns : areaColumns,
            1,
            areaCount);

        float sensorRange = Mathf.Max(prefabAgent.visionDistance, 15f);
        float isolationMargin = sensorRange * 2f + areaGap;
        float spacingX = maximumMapWidth + isolationMargin;
        float spacingZ = maximumMapHeight + isolationMargin;

        for (int i = 0; i < areaCount; i++)
        {
            int column = i % columns;
            int row = i / columns;
            Vector3 localOffset = new Vector3(column * spacingX, 0f, row * spacingZ);
            Vector3 areaOrigin = transform.TransformPoint(localOffset);

            MapGenerator mapGenerator = Instantiate(
                mapGeneratorPrefab, areaOrigin, transform.rotation, transform);

            // Instantiate the complete prefab hierarchy: the Rigidbody may be on
            // its root while ExplorerAgent is attached to a child object.
            GameObject agentRoot = Instantiate(
                explorerAgentPrefab, areaOrigin, transform.rotation, transform);
            agentRoot.SetActive(false);
            ExplorerAgent agent = agentRoot.GetComponentInChildren<ExplorerAgent>(true);
            if (agent == null)
            {
                Debug.LogError("The instantiated player prefab has no ExplorerAgent component.", agentRoot);
                Destroy(agentRoot);
                Destroy(mapGenerator.gameObject);
                enabled = false;
                return;
            }

            TrainingArea area = new TrainingArea(mapGenerator, agent);
            trainingAreas.Add(area);
            areaByAgent.Add(agent, area);

            agent.BindToLearnManager(this);

            if (IsInferenceTestMode)
            {
                ConfigureInferenceTestAgent(agent, agentRoot);
            }

            agentRoot.SetActive(true);
        }
    }

    private void ConfigureInferenceTestAgent(ExplorerAgent agent, GameObject agentRoot)
    {
        BehaviorParameters behavior = agentRoot.GetComponentInChildren<BehaviorParameters>(true);
        if (behavior == null)
        {
            Debug.LogError("The instantiated ExplorerAgent prefab has no Behavior Parameters component.", agentRoot);
            enabled = false;
            return;
        }

        // Set the model through Agent.SetModel (the ML-Agents runtime API), then
        // force inference so this mode cannot accidentally train via a Python trainer.
        behavior.BehaviorType = BehaviorType.HeuristicOnly;
        behavior.DeterministicInference = deterministicTestInference;
        agent.SetModel(behavior.BehaviorName, testModel, behavior.InferenceDevice);
        behavior.BehaviorType = BehaviorType.InferenceOnly;
    }

    private int CreateEpisodeSeed(TrainingArea area)
    {
        if (IsInferenceTestMode)
        {
            if (nextTestSeed > testSeedMaximum)
            {
                throw new InvalidOperationException("The reserved inference test seed range has been exhausted.");
            }

            return nextTestSeed++;
        }

        if (useFixedTestSeed)
        {
            return Mathf.Clamp(fixedTestSeed, testSeedMinimum, testSeedMaximum);
        }

        if (randomSeed == 0)
        {
            return randomTrainingSeedGenerator.Next(testSeedMaximum + 1, int.MaxValue);
        }

        int areaIndex = trainingAreas.IndexOf(area);
        uint value = unchecked((uint)randomSeed);
        value ^= unchecked((uint)(areaIndex + 1) * 0x9E3779B9u);
        value ^= unchecked((uint)(area.EpisodeCount + 1) * 0x85EBCA6Bu);
        value ^= value >> 16;
        value = unchecked(value * 0x7FEB352Du);
        value ^= value >> 15;
        value = unchecked(value * 0x846CA68Bu);
        value ^= value >> 16;

        uint trainingSeedRange = (uint)(int.MaxValue - testSeedMaximum);
        return testSeedMaximum + 1 + (int)(value % trainingSeedRange);
    }

    private void AdvanceCurriculum(float meanSuccessRate)
    {
        int nextLevel = CurriculumLevel + 1;
        int nextMapWidth = Mathf.Min(
            maximumMapWidth,
            curriculumBaseMapWidth + nextLevel * mapWidthIncreasePerLevel);
        int nextMapHeight = Mathf.Min(
            maximumMapHeight,
            curriculumBaseMapHeight + nextLevel * mapHeightIncreasePerLevel);
        int nextWaypointCount = Mathf.Min(
            maximumWaypointCount,
            curriculumBaseWaypointCount +
            (nextLevel / waypointIncreaseEveryLevels) * waypointIncreaseAmount);

        CurriculumLevel = nextLevel;
        mapWidth = nextMapWidth;
        mapHeight = nextMapHeight;
        waypointCount = nextWaypointCount;
        Debug.Log(
            $"Curriculum advanced to level {CurriculumLevel}: " +
            $"map {mapWidth}x{mapHeight}, waypoints {waypointCount} " +
            $"(mean area success {meanSuccessRate:P1}, {trainingAreas.Count} areas x {successRateWindow} episodes).",
            this);

        ClearCurriculumWindows();
    }

    private void RegressCurriculum(float meanSuccessRate)
    {
        CurriculumLevel--;
        mapWidth = Mathf.Min(
            maximumMapWidth,
            curriculumBaseMapWidth + CurriculumLevel * mapWidthIncreasePerLevel);
        mapHeight = Mathf.Min(
            maximumMapHeight,
            curriculumBaseMapHeight + CurriculumLevel * mapHeightIncreasePerLevel);
        waypointCount = Mathf.Min(
            maximumWaypointCount,
            curriculumBaseWaypointCount +
            (CurriculumLevel / waypointIncreaseEveryLevels) * waypointIncreaseAmount);

        Debug.Log(
            $"Curriculum decreased to level {CurriculumLevel}: " +
            $"map {mapWidth}x{mapHeight}, waypoints {waypointCount} " +
            $"(mean area success {meanSuccessRate:P1}, {trainingAreas.Count} areas x {successRateWindow} episodes).",
            this);

        ClearCurriculumWindows();
    }

    private void ClearCurriculumWindows()
    {
        foreach (TrainingArea area in trainingAreas)
        {
            area.ClearRecentResults();
        }

        RecentSuccessRate = 0f;
        CurriculumAreasReady = 0;
    }

    private void RestoreTimeScale()
    {
        if (!changedTimeScale)
        {
            return;
        }

        Time.timeScale = previousTimeScale;
        changedTimeScale = false;
    }

    private void OnDestroy()
    {
        Application.logMessageReceived -= StopInferenceTestAfterRuntimeError;
        RestoreTimeScale();
    }

    private void StopInferenceTestAfterRuntimeError(
        string condition,
        string stackTrace,
        LogType type)
    {
        if (!IsInferenceTestMode ||
            inferenceRuntimeErrorHandled ||
            (type != LogType.Error && type != LogType.Exception))
        {
            return;
        }

        inferenceRuntimeErrorHandled = true;
        // Prevent another FixedUpdate from appending observations to a failed
        // recurrent inference batch while the Editor exits Play Mode.
        Time.timeScale = 0f;
#if UNITY_EDITOR
        UnityEditor.EditorApplication.delayCall += StopEditorPlayModeAfterInferenceError;
#endif
    }

#if UNITY_EDITOR
    private static void StopEditorPlayModeAfterInferenceError()
    {
        if (UnityEditor.EditorApplication.isPlaying)
        {
            UnityEditor.EditorApplication.isPlaying = false;
        }
    }
#endif

    private void OnValidate()
    {
        mapWidth = Mathf.Max(5, mapWidth);
        mapHeight = Mathf.Max(5, mapHeight);
        waypointCount = Mathf.Max(0, waypointCount);
        trainingAreaCount = Mathf.Max(1, trainingAreaCount);
        areaColumns = Mathf.Clamp(areaColumns, 1, trainingAreaCount);
        maximumMapWidth = Mathf.Max(mapWidth, maximumMapWidth);
        maximumMapHeight = Mathf.Max(mapHeight, maximumMapHeight);
        maximumWaypointCount = Mathf.Max(waypointCount, maximumWaypointCount);
        areaGap = Mathf.Max(0f, areaGap);
        testSeedMinimum = Mathf.Clamp(testSeedMinimum, 1, int.MaxValue - 2);
        testSeedMaximum = Mathf.Clamp(testSeedMaximum, testSeedMinimum, int.MaxValue - 2);
        inferenceTestAreaCount = Mathf.Max(1, inferenceTestAreaCount);
        inferenceTestAreaColumns = Mathf.Clamp(inferenceTestAreaColumns, 1, inferenceTestAreaCount);
        inferenceTestTimeScale = Mathf.Max(1f, inferenceTestTimeScale);
        fixedTestSeed = Mathf.Clamp(fixedTestSeed, testSeedMinimum, testSeedMaximum);
        successRateWindow = Mathf.Max(1, successRateWindow);
        successesRequiredToAdvance = Mathf.Clamp(successesRequiredToAdvance, 1, successRateWindow);
        successesRequiredToRegress = Mathf.Clamp(
            successesRequiredToRegress,
            0,
            Mathf.Max(0, successesRequiredToAdvance - 1));
        mapWidthIncreasePerLevel = Mathf.Max(1, mapWidthIncreasePerLevel);
        mapHeightIncreasePerLevel = Mathf.Max(1, mapHeightIncreasePerLevel);
        waypointIncreaseEveryLevels = Mathf.Max(1, waypointIncreaseEveryLevels);
        waypointIncreaseAmount = Mathf.Max(1, waypointIncreaseAmount);
    }
}
