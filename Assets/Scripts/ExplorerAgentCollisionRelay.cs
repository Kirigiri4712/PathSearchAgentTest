using UnityEngine;

[DisallowMultipleComponent, RequireComponent(typeof(Rigidbody))]
public sealed class ExplorerAgentCollisionRelay : MonoBehaviour
{
    private ExplorerAgent explorerAgent;

    public void Bind(ExplorerAgent agent)
    {
        explorerAgent = agent;
    }

    private void Awake()
    {
        if (explorerAgent == null)
        {
            explorerAgent = GetComponentInParent<ExplorerAgent>();
        }

        if (explorerAgent == null)
        {
            explorerAgent = GetComponentInChildren<ExplorerAgent>(true);
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        explorerAgent?.HandleCollisionEnter(collision);
    }

    private void OnCollisionStay(Collision collision)
    {
        explorerAgent?.HandleCollisionStay(collision);
    }

    private void OnCollisionExit(Collision collision)
    {
        explorerAgent?.HandleCollisionExit(collision);
    }
}