using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Owns the shared training configuration and creates isolated training areas.
/// Place this component at the origin of the area layout.
/// </summary>
public class LearnManager : MonoBehaviour
{
    [Header("Prefabs")]
    [SerializeField] private MapGenerator mapGeneratorPrefab;
    [Tooltip("Assign the complete player prefab root. The ExplorerAgent may be on a child object.")]
    [SerializeField] private GameObject explorerAgentPrefab;

    [Header("Initial Map Complexity")]
    [SerializeField, Min(5)] private int mapWidth = 5;
    [SerializeField, Min(5)] private int mapHeight = 5;
    [SerializeField, Min(0)] private int waypointCount;
    [SerializeField] private int randomSeed;

    [Header("Parallel Training Areas")]
    [SerializeField, Min(1)] private int trainingAreaCount = 2;
    [SerializeField, Min(1)] private int areaColumns = 2;
    [SerializeField, Min(5)] private int maximumMapWidth = 40;
    [SerializeField, Min(5)] private int maximumMapHeight = 40;
    [SerializeField, Min(0)] private float areaGap = 5f;
    [SerializeField, Min(1)] private int successRateWindow = 100;

    private readonly List<TrainingArea> trainingAreas = new List<TrainingArea>();
    private readonly Dictionary<ExplorerAgent, TrainingArea> areaByAgent =
        new Dictionary<ExplorerAgent, TrainingArea>();
    private readonly Queue<bool> recentEpisodeResults = new Queue<bool>();

    public int CompletedEpisodes { get; private set; }
    public float RecentSuccessRate { get; private set; }

    private sealed class TrainingArea
    {
        public MapGenerator MapGenerator { get; }
        public ExplorerAgent Agent { get; }
        public int EpisodeCount { get; set; }

        public TrainingArea(MapGenerator mapGenerator, ExplorerAgent agent)
        {
            MapGenerator = mapGenerator;
            Agent = agent;
        }
    }

    private void Awake()
    {
        CreateTrainingAreas();
    }

    /// <summary>
    /// Called by each agent when its episode begins. Map complexity is read here
    /// so a future curriculum can update the shared values between episodes.
    /// </summary>
    public GeneratedMap BeginEpisode(ExplorerAgent agent)
    {
        if (!areaByAgent.TryGetValue(agent, out TrainingArea area))
        {
            throw new InvalidOperationException(
                $"Agent '{agent.name}' is not registered with this LearnManager.");
        }

        int seed = CreateEpisodeSeed(area);
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
        waypointCount = Mathf.Max(0, waypoints);
    }

    public void RecordEpisodeResult(bool reachedGoal)
    {
        CompletedEpisodes++;
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
            agentRoot.SetActive(true);
        }
    }

    private int CreateEpisodeSeed(TrainingArea area)
    {
        if (randomSeed == 0)
        {
            return Guid.NewGuid().GetHashCode();
        }

        int areaIndex = trainingAreas.IndexOf(area);
        return unchecked(randomSeed + areaIndex * 1000003 + area.EpisodeCount * 9176);
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
        areaGap = Mathf.Max(0f, areaGap);
        successRateWindow = Mathf.Max(1, successRateWindow);
    }
}
