using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// 未知マップ探索用Agent
/// ・RayPerceptionSensor3D前提
/// ・内部マップ情報へアクセスしない
/// ・Physicsベース
/// ・探索重視
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class ExplorerAgent : Agent
{
    [Header("Movement")]
    public float moveSpeed = 3f;
    public float rotateSpeed = 180f;

    [Header("Rewards")]
    public float stepPenalty = -0.0005f;
    public float goalReward = 10f;
    public float goalVisibleReward = 0.2f;

    [Header("ActionRewards")]
    public float exploreReward = 0.02f;
    public float exploreRewardAfterGoalVisible = 0.01f;
    public float wallPenalty = -0.01f;
    public float rotatePenalty = -0.001f;

    [Header("Exploration")]
    public float visitedCellSize = 2f;

    [Header("Vision")]
    public float visionDistance = 15f;
    public float visionAngle = 60f;

    private bool foundGoal = false;

    private int stepCount;
    public int maxStep = 5000;

    private Rigidbody rb;

    // 探索済み位置
    private HashSet<Vector2Int> visitedCells =
        new HashSet<Vector2Int>();

    // スタック防止
    private Vector3 lastPosition;
    private float stuckTimer;

    private bool canGoal = false;

    private MapGenerator mapGenerator;

    private void Awake()
    {
        rb = GetComponent<Rigidbody>();
    }

    private void Update()
    {
        if (mapGenerator == null) mapGenerator = AgentSingleton.instance.mapGenerator;
        if (transform.position.y < -10f)
        {
            AddReward(-10000f);
            EndThisEpisode();
        }
    }
    /// <summary>
    /// ゴール視認判定
    /// </summary>
    private void CheckGoalVisible()
    {
        if (foundGoal)
            return;

        Collider[] hits =
            Physics.OverlapSphere(
                transform.position,
                visionDistance);

        foreach (Collider hit in hits)
        {
            if (!hit.CompareTag("Goal"))
                continue;

            Vector3 dir =
                (hit.transform.position - transform.position)
                .normalized;

            float angle =
                Vector3.Angle(transform.forward, dir);

            // 視野角内か
            if (angle > visionAngle * 0.5f)
                continue;

            // 壁遮蔽チェック
            if (Physics.Raycast(
                transform.position + Vector3.up * 0.5f,
                dir,
                out RaycastHit rayHit,
                visionDistance))
            {
                if (rayHit.collider.CompareTag("Goal"))
                {
                    foundGoal = true;

                    AddReward(goalVisibleReward);

                    Debug.Log("Goal Found!");
                }
            }
        }
    }

    public override void OnEpisodeBegin()
    {
        canGoal = false;
        StartCoroutine(InstantGoalLimit());
        mapGenerator?.Generate();

        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        transform.position = AgentSingleton.instance.SpawnPos + Vector3.up * 0.5f;

        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        transform.rotation = Quaternion.Euler(0, Random.Range(0f, 360f), 0);

        visitedCells.Clear();

        stuckTimer = 0f;

        stepCount = 0;

        lastPosition = transform.position;

        RegisterVisited(transform.position);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 速度（ローカル）
        Vector3 localVelocity =
            transform.InverseTransformDirection(rb.linearVelocity);

        sensor.AddObservation(localVelocity.x);
        sensor.AddObservation(localVelocity.z);

        // 前方方向
        sensor.AddObservation(transform.forward.x);
        sensor.AddObservation(transform.forward.z);
    }

    public override void OnActionReceived(
        ActionBuffers actions)
    {
        int moveAction = actions.DiscreteActions[0];
        int rotateAction = actions.DiscreteActions[1];

        // ====================
        // 回転
        // ====================

        float rotate = 0f;

        switch (rotateAction)
        {
            case 1:
                rotate = -1f;
                break;

            case 2:
                rotate = 1f;
                break;
        }

        Quaternion deltaRotation = Quaternion.Euler(0f, rotate * rotateSpeed * Time.fixedDeltaTime, 0f);

        rb.MoveRotation(rb.rotation * deltaRotation);
        if (rotateAction != 0) AddReward(rotatePenalty);

        // ====================
        // 移動
        // ====================

        Vector3 move = Vector3.zero;

        switch (moveAction)
        {
            case 1:
                move = transform.forward;
                break;
            case 2:
                move = -transform.forward;
                break;
        }

        Vector3 targetVelocity = move * moveSpeed;

        Vector3 velocityChange = targetVelocity - new Vector3(rb.linearVelocity.x, 0f, rb.linearVelocity.z);

        rb.AddForce(velocityChange, ForceMode.VelocityChange);

        if (Physics.Raycast(transform.position + Vector3.up * 0.5f, transform.forward, 1.0f))
        {
            AddReward(-0.001f);
        }
        if (!Physics.Raycast(transform.position + Vector3.up * 0.5f, transform.forward, 2.0f))
        {
            AddReward(0.001f);
        }

            // ====================
            // 常時ペナルティ
            // ====================

            AddReward(stepPenalty);

        // ====================
        // 新規探索報酬
        // ====================

        Vector2Int cell = WorldToCell(transform.position);

        if (!visitedCells.Contains(cell))
        {
            visitedCells.Add(cell);

            AddReward(foundGoal ? exploreRewardAfterGoalVisible : exploreReward);
        }

        // ====================
        // スタック検知
        // ====================

        //float moved =
        //    Vector3.Distance(
        //        transform.position,
        //        lastPosition);

        //if (moved < 0.05f)
        //{
        //    stuckTimer += Time.fixedDeltaTime;

        //    if (stuckTimer > 3f)
        //    {
        //        AddReward(-1f);
        //        EndThisEpisode();
        //    }
        //}
        //else
        //{
        //    stuckTimer = 0f;
        //}

        CheckGoalVisible();

        lastPosition = transform.position;

        stepCount++;

        if (stepCount >= maxStep)
        {
            AddReward(-1f);
            EndThisEpisode();
        }
    }

    public override void Heuristic(
        in ActionBuffers actionsOut)
    {
        var discrete = actionsOut.DiscreteActions;

        // Move
        if (Input.GetKey(KeyCode.W))
            discrete[0] = 1;
        else if (Input.GetKey(KeyCode.S))
            discrete[0] = 2;
        else
            discrete[0] = 0;

        // Rotate
        if (Input.GetKey(KeyCode.A))
            discrete[1] = 1;
        else if (Input.GetKey(KeyCode.D))
            discrete[1] = 2;
        else
            discrete[1] = 0;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.CompareTag("Wall"))
        {
            AddReward(wallPenalty);
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (!canGoal) return;
        if (other.CompareTag("Goal"))
        {
            AddReward(goalReward);
            Debug.Log("Goal!" + other.GetHashCode());
            EndThisEpisode();
        }
    }

    private void RegisterVisited(Vector3 pos)
    {
        visitedCells.Add(WorldToCell(pos));
    }

    private Vector2Int WorldToCell(Vector3 pos)
    {
        return new Vector2Int(
            Mathf.FloorToInt(pos.x / visitedCellSize),
            Mathf.FloorToInt(pos.z / visitedCellSize)
        );
    
    }

    private IEnumerator InstantGoalLimit()
    {
        canGoal = false;
        yield return new WaitForSeconds(1);
        canGoal = true;
    }

    private void EndThisEpisode()
    {
        transform.position = Vector3.up * 10f;
        Debug.Log(GetCumulativeReward());
        EndEpisode();
    }
}