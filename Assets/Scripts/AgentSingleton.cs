using UnityEngine;

public class AgentSingleton : MonoBehaviour
{
    public static AgentSingleton instance;

    public MapGenerator mapGenerator;
    public Vector3 SpawnPos;
    public Vector3 GoalPos;

    private void Awake()
    {
        if (instance != null && instance != this)
        {
            Destroy(gameObject);
            return;
        }
        instance = this;
        DontDestroyOnLoad(gameObject);
    }
}
