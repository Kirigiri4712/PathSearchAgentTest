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
    [Tooltip("Number of completed episodes at the current difficulty used for promotion checks.")]
    [SerializeField, Min(1)] private int successRateWindow = 20;
    [Tooltip("Required successes in the current window to advance difficulty.")]
    [SerializeField, Min(1)] private int successesRequiredToAdvance = 18;
    [SerializeField, Min(1)] private int mapWidthIncreasePerLevel = 2;
    [SerializeField, Min(1)] private int mapHeightIncreasePerLevel = 2;
    [Tooltip("Increase waypoints once per this many difficulty promotions.")]
    [SerializeField, Min(1)] private int waypointIncreaseEveryLevels = 3;
    [SerializeField, Min(1)] private int waypointIncreaseAmount = 1;

    private readonly List<TrainingArea> trainingAreas = new List<TrainingArea>();
    private readonly Dictionary<ExplorerAgent, TrainingArea> areaByAgent =
        new Dictionary<ExplorerAgent, TrainingArea>();
    private readonly Queue<bool> recentEpisodeResults = new Queue<bool>();
    private System.Random randomTrainingSeedGenerator;
    private int nextTestSeed;
    private bool testRunComplete;

    public int CompletedEpisodes { get; private set; }
    public float RecentSuccessRate { get; private set; }
    public int CurriculumLevel { get; private set; }
    public int TestEpisodesCompleted { get; private set; }
    public int TestSuccesses { get; private set; }
    public float TestSuccessRate { get; private set; }
    public bool IsTestRunComplete => testRunComplete;
    public bool IsInferenceTestMode => operationMode == OperationMode.InferenceTest;

    private sealed class TrainingArea
    {
        public MapGenerator MapGenerator { get; }
        public ExplorerAgent Agent { get; }
        public int EpisodeCount { get; set; }
        public int ActiveEpisodeDifficultyLevel { get; set; }
        public int ActiveSeed { get; set; }

        public TrainingArea(MapGenerator mapGenerator, ExplorerAgent agent)
        {
            MapGenerator = mapGenerator;
            Agent = agent;
        }
    }

    private void Awake()
    {
        nextTestSeed = testSeedMinimum;

        if (IsInferenceTestMode)
        {
            if (testModel == null)
            {
                Debug.LogError("Inference Test mode requires a trained ONNX model in the Test Model field.", this);
                enabled = false;
                return;
            }

            // Evaluation uses one policy instance, a fixed difficulty, and each
            // reserved seed exactly once. It must not advance the training curriculum.
            trainingAreaCount = 1;
            areaColumns = 1;
            automaticCurriculum = false;
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
        return area.MapGenerator.Generate(mapWidth, mapHeight, waypointCount, seed);
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
    }

    public bool RecordEpisodeResult(ExplorerAgent agent, bool reachedGoal)
    {
        if (!areaByAgent.TryGetValue(agent, out TrainingArea area))
        {
            Debug.LogWarning($"Received an episode result from unregistered agent '{agent.name}'.", agent);
            return false;
        }

        CompletedEpisodes++;

        if (IsInferenceTestMode)
        {
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
                $"({TestEpisodesCompleted}/{totalTestSeeds}), success rate {TestSuccessRate:P1}.",
                this);

            if (TestEpisodesCompleted >= totalTestSeeds)
            {
                testRunComplete = true;
                Debug.Log(
                    $"[Inference Test Complete] Seeds {testSeedMinimum}-{testSeedMaximum}: " +
                    $"{TestSuccesses}/{TestEpisodesCompleted} successes ({TestSuccessRate:P1}).",
                    this);
                return true;
            }

            return false;
        }

        // Episodes already in progress at the previous level must not affect the
        // success window for the new level after another area triggers promotion.
        if (area.ActiveEpisodeDifficultyLevel != CurriculumLevel)
        {
            return false;
        }

        recentEpisodeResults.Enqueue(reachedGoal);

        while (recentEpisodeResults.Count > successRateWindow)
        {
            recentEpisodeResults.Dequeue();
        }

        int successes = 0;
        foreach (bool result in recentEpisodeResults)
        {
            if (result) successes++;
        }

        RecentSuccessRate = (float)successes / recentEpisodeResults.Count;

        if (automaticCurriculum && !useFixedTestSeed &&
            recentEpisodeResults.Count >= successRateWindow &&
            successes >= successesRequiredToAdvance)
        {
            AdvanceCurriculum(successes);
        }

        return false;
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

        trainingAreaCount = Mathf.Max(1, trainingAreaCount);
        areaColumns = Mathf.Clamp(areaColumns, 1, trainingAreaCount);

        float sensorRange = Mathf.Max(prefabAgent.visionDistance, 15f);
        float isolationMargin = sensorRange * 2f + areaGap;
        float spacingX = maximumMapWidth + isolationMargin;
        float spacingZ = maximumMapHeight + isolationMargin;

        for (int i = 0; i < trainingAreaCount; i++)
        {
            int column = i % areaColumns;
            int row = i / areaColumns;
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

    private void AdvanceCurriculum(int successes)
    {
        int nextLevel = CurriculumLevel + 1;
        bool increaseWaypoints = nextLevel % waypointIncreaseEveryLevels == 0;
        bool canIncreaseMapSize = mapWidth < maximumMapWidth || mapHeight < maximumMapHeight;
        bool canIncreaseWaypoints = waypointCount < maximumWaypointCount;

        if (!canIncreaseMapSize && !canIncreaseWaypoints)
        {
            automaticCurriculum = false;
            Debug.Log("Automatic curriculum reached its configured maximum difficulty.", this);
            return;
        }

        mapWidth = Mathf.Min(maximumMapWidth, mapWidth + mapWidthIncreasePerLevel);
        mapHeight = Mathf.Min(maximumMapHeight, mapHeight + mapHeightIncreasePerLevel);
        if (increaseWaypoints)
        {
            waypointCount = Mathf.Min(maximumWaypointCount, waypointCount + waypointIncreaseAmount);
        }

        CurriculumLevel = nextLevel;
        Debug.Log(
            $"Curriculum advanced to level {CurriculumLevel}: " +
            $"map {mapWidth}x{mapHeight}, waypoints {waypointCount} " +
            $"({successes}/{successRateWindow} successes).",
            this);

        recentEpisodeResults.Clear();
        RecentSuccessRate = 0f;
    }

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
        fixedTestSeed = Mathf.Clamp(fixedTestSeed, testSeedMinimum, testSeedMaximum);
        successRateWindow = Mathf.Max(1, successRateWindow);
        successesRequiredToAdvance = Mathf.Clamp(successesRequiredToAdvance, 1, successRateWindow);
        mapWidthIncreasePerLevel = Mathf.Max(1, mapWidthIncreasePerLevel);
        mapHeightIncreasePerLevel = Mathf.Max(1, mapHeightIncreasePerLevel);
        waypointIncreaseEveryLevels = Mathf.Max(1, waypointIncreaseEveryLevels);
        waypointIncreaseAmount = Mathf.Max(1, waypointIncreaseAmount);
    }
}
