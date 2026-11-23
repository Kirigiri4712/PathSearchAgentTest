#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

[InitializeOnLoad]
public static class TrainingBatchRunner
{
    private const string ActiveKey = "PathSearch.TrainingBatch.Active";
    private const string DefaultScenePath = "Assets/Scenes/SampleScene.unity";
    private const string StopFileName = "stop-visible-tile-training.flag";
    private static bool exitScheduled;

    static TrainingBatchRunner()
    {
        if (SessionState.GetBool(ActiveKey, false))
        {
            EditorApplication.update -= PollForStopRequest;
            EditorApplication.update += PollForStopRequest;
        }
    }

    public static void StartVisibleTileTraining()
    {
        try
        {
            if (!Application.isBatchMode)
            {
                throw new InvalidOperationException(
                    "TrainingBatchRunner must run in Unity batch mode.");
            }

            string stopFilePath = GetStopFilePath();
            if (File.Exists(stopFilePath))
            {
                File.Delete(stopFilePath);
            }

            string scenePath = GetScenePath();
            EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);
            LearnManager manager = UnityEngine.Object.FindFirstObjectByType<LearnManager>();
            if (manager == null)
            {
                throw new InvalidOperationException(
                    $"No LearnManager was found in {scenePath}.");
            }
            if (manager.IsInferenceTestMode)
            {
                throw new InvalidOperationException(
                    "SampleScene must use Training mode before starting PPO training.");
            }

            SessionState.SetBool(ActiveKey, true);
            EditorApplication.update -= PollForStopRequest;
            EditorApplication.update += PollForStopRequest;
            Debug.Log(
                "[TrainingBatchRunner] Starting headless training from SampleScene.");
            EditorApplication.EnterPlaymode();
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            SessionState.SetBool(ActiveKey, false);
            EditorApplication.Exit(1);
        }
    }

    private static void PollForStopRequest()
    {
        if (!SessionState.GetBool(ActiveKey, false) || exitScheduled ||
            !File.Exists(GetStopFilePath()))
        {
            return;
        }

        SessionState.SetBool(ActiveKey, false);
        EditorApplication.update -= PollForStopRequest;
        exitScheduled = true;
        Debug.Log("[TrainingBatchRunner] Stop request received; exiting Unity.");
        EditorApplication.isPlaying = false;
        EditorApplication.delayCall += () => EditorApplication.Exit(0);
    }

    // Optional "-trainingScene <path>" selects another scene, e.g. a profiling copy.
    private static string GetScenePath()
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == "-trainingScene")
            {
                return args[i + 1];
            }
        }
        return DefaultScenePath;
    }

    private static string GetStopFilePath()
    {
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;
        return Path.Combine(projectRoot, "Temp", StopFileName);
    }
}
#endif
