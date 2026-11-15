#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.SceneManagement;

[InitializeOnLoad]
public static class InferenceTestBatchRunner
{
    private const string ActiveKey = "PathSearch.InferenceTest.Active";
    private const string LayoutKey = "PathSearch.InferenceTest.Layout";
    private const string StartedUtcKey = "PathSearch.InferenceTest.StartedUtcTicks";
    private const string LastCompletedKey = "PathSearch.InferenceTest.LastCompleted";
    private const string LastProgressUtcKey = "PathSearch.InferenceTest.LastProgressUtcTicks";
    private const string ScenePath = "Assets/Scenes/SampleScene.unity";
    private const string ModelPath = "Assets/Models/ExplorerAgent-100000237.onnx";
    private const int TestAreaCount = 4;
    private const int TestAreaColumns = 2;
    private const int TestSeedMinimum = 1;
    private const int TestSeedMaximum = 1000;
    private const float TestTimeScale = 5f;
    private const double NoProgressTimeoutSeconds = 600d;

    static InferenceTestBatchRunner()
    {
        if (SessionState.GetBool(ActiveKey, false))
        {
            EditorApplication.update -= PollTest;
            EditorApplication.update += PollTest;
        }
    }

    public static void RunPathCorridorTest()
    {
        StartTest(MapLayoutMode.PathCorridor);
    }

    public static void RunRoomsAndCorridorsTest()
    {
        StartTest(MapLayoutMode.RoomsAndCorridors);
    }

    private static void StartTest(MapLayoutMode layout)
    {
        try
        {
            if (!Application.isBatchMode)
            {
                throw new InvalidOperationException("InferenceTestBatchRunner must run in Unity batch mode.");
            }

            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            ModelAsset model = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelPath);
            if (model == null)
            {
                throw new InvalidOperationException($"Could not import or load the model asset at {ModelPath}.");
            }

            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            LearnManager manager = UnityEngine.Object.FindFirstObjectByType<LearnManager>();
            if (manager == null)
            {
                throw new InvalidOperationException($"No LearnManager was found in {ScenePath}.");
            }

            manager.ConfigureInferenceTestForEditor(
                model,
                layout,
                TestAreaCount,
                TestAreaColumns,
                TestTimeScale,
                TestSeedMinimum,
                TestSeedMaximum,
                20,
                20);
            EditorUtility.SetDirty(manager);
            EditorSceneManager.MarkSceneDirty(manager.gameObject.scene);
            if (!EditorSceneManager.SaveScene(manager.gameObject.scene))
            {
                throw new InvalidOperationException("Could not save SampleScene with the selected test model and settings.");
            }
            string modelGuid = AssetDatabase.AssetPathToGUID(ModelPath);
            string savedSceneText = File.ReadAllText(ScenePath);
            if (string.IsNullOrEmpty(modelGuid) || !savedSceneText.Contains(modelGuid))
            {
                throw new InvalidOperationException("Saved SampleScene does not reference the requested test model.");
            }

            long now = DateTime.UtcNow.Ticks;
            SessionState.SetBool(ActiveKey, true);
            SessionState.SetInt(LayoutKey, (int)layout);
            SessionState.SetInt(LastCompletedKey, 0);
            SessionState.SetString(StartedUtcKey, now.ToString());
            SessionState.SetString(LastProgressUtcKey, now.ToString());
            EditorApplication.update -= PollTest;
            EditorApplication.update += PollTest;

            Debug.Log($"[InferenceTestRunner] Starting {layout}; model={ModelPath}; " +
                      $"areas={TestAreaCount}; seeds={TestSeedMinimum}-{TestSeedMaximum}; " +
                      $"timeScale={TestTimeScale}.");
            EditorApplication.EnterPlaymode();
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            SessionState.SetBool(ActiveKey, false);
            EditorApplication.Exit(1);
        }
    }

    private static void PollTest()
    {
        if (!SessionState.GetBool(ActiveKey, false) || !EditorApplication.isPlaying)
        {
            return;
        }

        LearnManager manager = UnityEngine.Object.FindFirstObjectByType<LearnManager>();
        if (manager == null)
        {
            return;
        }

        int previousCompleted = SessionState.GetInt(LastCompletedKey, 0);
        if (manager.TestEpisodesCompleted > previousCompleted)
        {
            SessionState.SetInt(LastCompletedKey, manager.TestEpisodesCompleted);
            SessionState.SetString(LastProgressUtcKey, DateTime.UtcNow.Ticks.ToString());
        }

        long lastProgressTicks = ParseTicks(SessionState.GetString(LastProgressUtcKey, "0"));
        if ((DateTime.UtcNow - new DateTime(lastProgressTicks, DateTimeKind.Utc)).TotalSeconds >
            NoProgressTimeoutSeconds)
        {
            Fail($"No inference-test episode completed for {NoProgressTimeoutSeconds:F0} seconds. " +
                 $"Completed {manager.TestEpisodesCompleted} seeds so far.");
            return;
        }

        if (!manager.IsTestRunComplete)
        {
            return;
        }

        WriteSummary(manager, (MapLayoutMode)SessionState.GetInt(LayoutKey, 0));
        SessionState.SetBool(ActiveKey, false);
        EditorApplication.update -= PollTest;
        Debug.Log("[InferenceTestRunner] Test completed; exiting Play Mode.");
        EditorApplication.isPlaying = false;
        EditorApplication.delayCall += () => EditorApplication.Exit(0);
    }

    private static void WriteSummary(LearnManager manager, MapLayoutMode layout)
    {
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;
        string outputDirectory = Path.Combine(projectRoot, "results", "inference-tests");
        Directory.CreateDirectory(outputDirectory);

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string outputPath = Path.Combine(
            outputDirectory,
            $"ExplorerAgent-100000237_{layout}_{TestSeedMinimum}-{TestSeedMaximum}_{timestamp}.json");
        var summary = new InferenceTestSummary
        {
            model = ModelPath,
            layout = layout.ToString(),
            seedMinimum = TestSeedMinimum,
            seedMaximum = TestSeedMaximum,
            episodes = manager.TestEpisodesCompleted,
            successes = manager.TestSuccesses,
            successRate = manager.TestSuccessRate,
            parallelAreas = TestAreaCount,
            timeScale = TestTimeScale,
            completedUtc = DateTime.UtcNow.ToString("O")
        };
        File.WriteAllText(outputPath, JsonUtility.ToJson(summary, true));
        Debug.Log($"[InferenceTestRunner] Summary saved to {outputPath}");

        if (manager.InferenceFailures.Count > 0)
        {
            string failurePath = Path.Combine(
                outputDirectory,
                $"ExplorerAgent-100000237_{layout}_{TestSeedMinimum}-{TestSeedMaximum}_{timestamp}_failures.json");
            var failureReport = new InferenceFailureReport
            {
                model = ModelPath,
                layout = layout.ToString(),
                episodes = manager.TestEpisodesCompleted,
                failureCount = manager.InferenceFailures.Count,
                completedUtc = DateTime.UtcNow.ToString("O"),
                records = new List<LearnManager.InferenceFailureRecord>(manager.InferenceFailures)
            };
            File.WriteAllText(failurePath, JsonUtility.ToJson(failureReport, true));
            Debug.Log($"[InferenceTestRunner] Failure maps saved to {failurePath}");
        }
    }

    private static void Fail(string message)
    {
        Debug.LogError($"[InferenceTestRunner] {message}");
        SessionState.SetBool(ActiveKey, false);
        EditorApplication.update -= PollTest;
        EditorApplication.isPlaying = false;
        EditorApplication.delayCall += () => EditorApplication.Exit(1);
    }

    private static long ParseTicks(string value)
    {
        return long.TryParse(value, out long ticks) && ticks > 0 ? ticks : DateTime.UtcNow.Ticks;
    }

    [Serializable]
    private sealed class InferenceTestSummary
    {
        public string model;
        public string layout;
        public int seedMinimum;
        public int seedMaximum;
        public int episodes;
        public int successes;
        public float successRate;
        public int parallelAreas;
        public float timeScale;
        public string completedUtc;
    }

    [Serializable]
    private sealed class InferenceFailureReport
    {
        public string model;
        public string layout;
        public int episodes;
        public int failureCount;
        public string completedUtc;
        public List<LearnManager.InferenceFailureRecord> records;
    }
}
#endif
